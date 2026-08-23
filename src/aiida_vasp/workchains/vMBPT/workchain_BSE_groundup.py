# pylint: disable=too-many-arguments
from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Float
from aiida.engine import WorkChain, ToContext

from aiida_vasp.workchains.vMBPT.workchain_G0W0_groundup import VaspG0W0GroundUpWorkChain
from aiida_vasp.workchains.vMBPT.workchain_atomic_BSE import VaspAtomicBSEWorkChain

# The VaspWorkChain-family ports shared identically by both steps (structure/code/POTCAR/k-mesh/scheduler
# options and the various VaspWorkChain behavior knobs) - forwarded once from this class's own top-level
# exposure into the BSE step's inputs (see run_bse()), so a caller-supplied value applies consistently to every
# phase rather than only to the G0W0/DFT ones. Kept as one named tuple (rather than duplicated inline in both
# define() and run_bse()) so the exclude-list used to namespace VaspAtomicBSEWorkChain's own inputs under 'bse'
# and the copy-loop that forwards them stay in sync automatically.
_SHARED_VASP_PORTS = (
    'structure', 'code', 'potential_family', 'potential_mapping', 'potential', 'kpoints', 'kpoints_spacing',
    'options', 'metadata', 'clean_workdir', 'verify', 'monitors', 'handler_overrides', 'max_iterations',
    'auto_parallel', 'charge_density', 'keep_last_workdir', 'ldau_mapping', 'magmom_mapping', 'remote_folder',
    'site_magnetization', 'vdw_kernel', 'verbose', 'wavefunctions', 'calc', 'dynamics',
)


class VaspBSEGroundUpWorkChain(WorkChain):
    """Run a full DFT -> G0W0 -> BSE/optical pipeline, by composing VaspG0W0GroundUpWorkChain with one
    VaspAtomicBSEWorkChain step on top.

    Purpose of workchain:
    1) Get the DFT ground state (and, when needed, a real G0W0 calculation) out of the way by submitting
       VaspG0W0GroundUpWorkChain as a single child workchain, then run one BSE/optical calculation using
       whichever of its outputs the chosen optical.algo actually needs as a restart.
    2) Auto-fill the BSE step's scissor shift from the G0W0 step's own QP gap correction (gaps_QPc) when a real
       G0W0 calculation ran and the caller hasn't already supplied one explicitly.
    3) Re-expose both steps' outputs under one combined output namespace.

    Design rationale:
    1) Composition, not a shared phase engine: unlike VaspG0W0GroundUpWorkChain (which owns a multi-phase
       retry/skip FSM because it manages 3 real phases with real dependencies between them), this class only
       ever runs two steps in a fixed sequence, each of which already owns its own internal retry logic
       (VaspG0W0GroundUpWorkChain's phase-level retries; VaspAtomicBSEWorkChain's own process handlers) - so a
       plain linear outline is enough, with no phase-engine machinery of its own.
    2) Which child class runs the first step is a single class attribute (_G0W0_WORKCHAIN_CLASS), not hardcoded
       inline - a future ML/interpolation-enabled variant of this class can override just that one attribute to
       swap in a surrogate WorkChain satisfying VaspG0W0GroundUpWorkChain's output contract (RemoteData_DFT/
       RemoteData_G0W0/gaps/gaps_QPc/...), without touching anything else here.
    3) optical.algo decides the restart source, not a separate flag: 'BSE' (screening read from files) needs a
       real G0W0 calculation to have produced WFULL*/W0* files, so it restarts from RemoteData_G0W0; 'modelBSE'
       and 'IPA' need no G0W0 output at all (model dielectric screening, or no screening at all respectively),
       so they restart directly from RemoteData_DFT - whether a real G0W0 calculation even ran is entirely the
       caller's choice, via VaspG0W0GroundUpWorkChain's own exposed ns_option.run_2DFTvo_3G0W0 flag.
    4) optimization.kpar/optimization.lreal have different sensible defaults for a G0W0 calculation than for a
       BSE calculation (see VaspAtomicG0W0WorkChain/VaspAtomicBSEWorkChain's own defaults) - so this class does
       NOT merge the two steps' inputs into one flat namespace. Only the genuinely shared VaspWorkChain-family
       ports (_SHARED_VASP_PORTS) are forwarded automatically; everything BSE-specific (optical.*, screening.*,
       encut, nbands, encut_chi, magnetic_moment_onsite, optimization.*, extraresources_fallback_options) lives
       under its own 'bse' input namespace, set independently of the G0W0 step's equivalents.
    """

    _G0W0_WORKCHAIN_CLASS = VaspG0W0GroundUpWorkChain

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(cls._G0W0_WORKCHAIN_CLASS)
        spec.expose_inputs(VaspAtomicBSEWorkChain, namespace='bse',
                           exclude=('parameters', 'settings', 'restart_folder') + _SHARED_VASP_PORTS)

        spec.expose_outputs(cls._G0W0_WORKCHAIN_CLASS)
        spec.expose_outputs(VaspAtomicBSEWorkChain, namespace='bse')

        spec.exit_code(420, 'ERROR_G0W0_GROUNDUP_FAILED',        message='The G0W0 groundup sub-workchain did not finish successfully.')
        spec.exit_code(421, 'ERROR_MISSING_G0W0_OUTPUT_FOR_BSE', message="bse.optical.algo == 'BSE' but the G0W0 groundup sub-workchain produced no RemoteData_G0W0 output (was ns_option.run_2DFTvo_3G0W0 disabled?).")
        spec.exit_code(422, 'ERROR_BSE_STEP_FAILED',              message='The BSE/optical sub-workchain did not finish successfully.')

        spec.outline(
            cls.run_g0w0_groundup,
            cls.run_bse,
            cls.elaborate_results,
        )

    def run_g0w0_groundup(self):
        """Submit the DFT/G0W0 pipeline as a single child workchain."""
        inputs = self.exposed_inputs(self._G0W0_WORKCHAIN_CLASS)
        return ToContext(g0w0_groundup=self.submit(self._G0W0_WORKCHAIN_CLASS, **inputs))

    def run_bse(self):
        """Submit the BSE/optical step, restarting from whichever of the G0W0-groundup step's outputs
        bse.optical.algo actually needs."""
        g0w0 = self.ctx.g0w0_groundup
        if not g0w0.is_finished_ok:
            return self.exit_codes.ERROR_G0W0_GROUNDUP_FAILED

        algo = self.inputs.bse.optical.algo.value
        if algo == 'BSE':
            if 'RemoteData_G0W0' not in g0w0.outputs:
                return self.exit_codes.ERROR_MISSING_G0W0_OUTPUT_FOR_BSE
            restart_folder = g0w0.outputs.RemoteData_G0W0
        else:
            restart_folder = g0w0.outputs.RemoteData_DFT

        #[1] Shared VaspWorkChain-family ports - same structure/code/POTCAR/k-mesh/scheduler options as the
        # G0W0/DFT phases.
        inputs = AttributeDict()
        for key in _SHARED_VASP_PORTS:
            if key in self.inputs:
                inputs[key] = self.inputs[key]

        #[2] BSE-specific inputs, namespaced under 'bse' (see the class docstring's Design rationale [4]).
        inputs.update(self.exposed_inputs(VaspAtomicBSEWorkChain, namespace='bse'))
        inputs.restart_folder = restart_folder

        #[3] Auto-fill the scissor shift from the G0W0 step's own QP gap correction, unless the caller already
        # set one explicitly (any nonzero bse.optical.scissor value is left untouched) or no real G0W0 step ran.
        if algo != 'IPA' and 'gaps_QPc' in g0w0.outputs and inputs.optical.scissor.value == 0.0:
            gaps_QPc = g0w0.outputs.gaps_QPc.get_dict()
            inputs.optical.scissor = Float(gaps_QPc['spinUp']['Dir'])
            self.report(f"INFO: auto-filled optical.scissor={inputs.optical.scissor.value} from the G0W0 step's own gaps_QPc.")

        return ToContext(bse=self.submit(VaspAtomicBSEWorkChain, **inputs))

    def elaborate_results(self):
        """Re-expose both steps' outputs under one combined output namespace."""
        if not self.ctx.bse.is_finished_ok:
            return self.exit_codes.ERROR_BSE_STEP_FAILED
        self.out_many(self.exposed_outputs(self.ctx.g0w0_groundup, self._G0W0_WORKCHAIN_CLASS))
        self.out_many(self.exposed_outputs(self.ctx.bse, VaspAtomicBSEWorkChain, namespace='bse'))
        return None
