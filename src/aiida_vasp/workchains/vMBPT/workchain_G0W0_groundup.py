    # pylint: disable=too-many-arguments
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple

from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Int, Float, Str, Dict, Bool, RemoteData, ArrayData, BandsData, KpointsData
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import WorkChain, ToContext, while_

from aiida_vasp.utils.workchains import prepare_process_inputs
from aiida_vasp.workchains.vMBPT.utils_helpers_extrapolation import input_magnetic_moment_tomagmom
from aiida_vasp.workchains.vMBPT.workchain_atomic_G0W0 import VaspAtomicG0W0WorkChain


class PhaseStatus(Enum):
    """Execution status of the currently active phase. Meaningful only once ctx.phase_idx is in range."""
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'


class TerminalState(Enum):
    """Final state of the phase loop once it stops iterating."""
    COMPLETE = 'COMPLETE'
    FAILED = 'FAILED'


@dataclass(frozen=True)
class WorkflowPhase:
    """Describes one phase in self.ctx.phases generically - the FSM engine (initialize/update_state/
    validate_step/prepare_step/execute_step/elaborate_results) never branches on phase *name*, only on
    phase.<hook>(self) / phase.<hook>(self, node), so a subclass can override or replace individual WorkflowPhase
    entries without touching the engine itself."""
    key: str
    build_inputs: Callable = staticmethod(lambda self: AttributeDict())
    capture_outputs: Callable = staticmethod(lambda self, node: None)
    get_restart_folder: Callable = staticmethod(lambda self: None)
    required_files: Tuple[str, ...] = ()
    missing_files_exit_code_name: str = None
    skip_if: Callable = staticmethod(lambda self: False)
    seed_restart_if_skipped: Callable = staticmethod(lambda self: None)


class VaspG0W0GroundUpWorkChain(WorkChain):
    """Run a full DFT(gr) -> DFT(vo) -> G0W0 pipeline from scratch (or from a supplied DFT ground-state restart).

    Purpose of workchain:
    1) Orchestrate three phases as VaspWorkChain-shaped child processes: '1DFTgr' (DFT ground state, optional),
       '2DFTvo' (single-iteration DFT with all virtual orbitals, producing the WAVECAR/WAVEDER the G0W0 phase
       restarts from), '3G0W0' (delegates entirely to VaspAtomicG0W0WorkChain - no local INCAR construction for
       this phase, unlike '1DFTgr'/'2DFTvo').
    2) Own retry/skip bookkeeping generically over an ordered list of phases (self.ctx.phases, built in
       initialize()), so a subclass can extend or replace individual phases without touching the FSM engine
       itself (see [2] below).
    3) Assemble the DFT/G0W0 band structures into gaps/QP-correction outputs (gaps, gaps_QPc, bands_QPc).

    Design rationale:
    1) Not a VaspWorkChain subclass - a plain WorkChain orchestrator submitting vasp.vasp/VaspAtomicG0W0WorkChain
       as children. It builds its own INCAR for the '1DFTgr'/'2DFTvo' phases (see _prepare_inputs_DFT), but
       delegates INCAR construction and restart-folder validation for the '3G0W0' phase entirely to
       VaspAtomicG0W0WorkChain (see _prepare_inputs_G0W0). VaspBSEGroundUpWorkChain composes this class (submits
       it as a child workchain) rather than duplicating its phases, and adds an optical/BSE phase on top.
    2) Built-in extensibility for a future ML/interpolation surrogate: a subclass can keep all 3 phase keys, make
       '2DFTvo' a permanently-skipped placeholder (skip_if=lambda self: True, with seed_restart_if_skipped
       forwarding '1DFTgr's real output straight through to the '3G0W0' restart slot), and point '3G0W0' at a
       surrogate WorkChain class instead of VaspAtomicG0W0WorkChain - '1DFTgr' (real DFT ground state) stays the
       only phase that actually runs VASP. elaborate_results()'s node lookup below is written generically (reads
       state_WC.submitted presence via __last_wc_node, not the ns_option.* input flags) specifically so this
       requires no override in such a subclass.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(WorkflowFactory('vasp.vasp'), exclude=('parameters', 'settings', 'restart_folder'))
        spec.expose_inputs(VaspAtomicG0W0WorkChain,       exclude=('parameters', 'settings', 'restart_folder'))
        # Both exposures deliberately exclude restart_folder (every phase's build_inputs overwrites it anyway,
        # from ctx.state_WC.restart_folders - an exposed top-level restart_folder port would be inert/confusing).
        # This gives, for free from the VaspAtomicG0W0WorkChain exposure: encut, nbands, encut_chi, nbandsgw,
        # nomega, magnetic_moment_onsite, optimization.kpar/lreal/set_PRECFOCK_to_Fast,
        # extraresources_fallback_options - the exact same names used both by the DFT phases below and by the
        # G0W0 phase, so no namespace-translation glue is needed anywhere in build_inputs.

        spec.input('optimization.npar', valid_type=Int, required=False, default=lambda: Int(1),
                   help='NPAR value used only by the 1DFTgr/2DFTvo phases (the G0W0 atomic wrapper has no npar port).')

        spec.input('kpoints', valid_type=KpointsData,
                   help='K-mesh used for VASP G0W0 and DFT runs; get_kpoints_mesh() must work.')

        spec.input('ns_reference.starting_RemoteData', valid_type=RemoteData, required=False,
                   help='The DFT ground state wavefunction (WAVECAR) and CHGCAR will be copied from this RemoteData folder as a starting point.')

        spec.input('ns_option.maximum_iterations', valid_type=Int, required=False, default=lambda: Int(1),
                   help='Maximum number of times the workchain will restart a crashed calculation, per phase.')
        spec.input('ns_option.run_1DFTgr',         valid_type=Bool, required=False, default=lambda: Bool(False),
                   help='If True, run a DFT calculation to get the ground state (WAVECAR and CHGCAR) as a starting point for following calculations. If False, starting_RemoteData must be provided as input.')
        spec.input('ns_option.run_2DFTvo_3G0W0',   valid_type=Bool, required=False, default=lambda: Bool(True),
                   help='Run the single-iteration DFT with all unoccupied bands included (DFTvo) and G0W0 with the same encut/nbands and the DFT wavefunctions/energies as starting point.')
        spec.input('ns_option.calculation_label',  valid_type=Str,  required=False, default=lambda: Str(""),
                   help='The summary printed at the end of the workflow will be labeled with this string.')

        spec.output('NGarray',    valid_type=ArrayData, required=False, help='FFT grid used.')
        spec.output('ENMAXarray', valid_type=ArrayData, required=False, help='Array containing the ENMAX of all employed POTCARs.')
        spec.output('kpoints',    valid_type=DataFactory('core.array.kpoints'), help='The actual k-mesh used for VASP G0W0 and DFT runs.')

        spec.output('RemoteData_G0W0', valid_type=RemoteData, required=False, help='RemoteData for the G0W0 calculation node.')
        spec.output('RemoteData_DFT',  valid_type=RemoteData, required=True,  help='RemoteData for the DFT calculation node.')
        spec.output('bands_G0W0',      valid_type=BandsData,  required=False, help='BandsData for the G0W0 calculation node.')
        spec.output('bands_DFT',       valid_type=BandsData,  required=True,  help='BandsData for the DFT calculation node.')
        spec.output('gaps',            valid_type=Dict,       required=True,  help='Direct and indirect gaps for the DFT and G0W0 nodes.')

        spec.output('gaps_QPc',  valid_type=Dict,      required=False, help='QP HOMO/LUMO correction at direct gap kpt.')
        spec.output('bands_QPc', valid_type=BandsData, required=False)
        # Output structure:
        #   gaps = { 'DFT': {spin_label: {Dir,Ind,Gam}}, 'G0W0': {spin_label: {Dir,Ind,Gam}} }
        #   gaps_QPc = {spin_label: {Dir,Ind,Gam}}   (only if G0W0 ran - QP correction = G0W0 - DFT)
        #   bands_QPc : BandsData, E_QP(k,n) = E_G0W0(k,n) - E_DFT(k,n)   (only if G0W0 ran)
        #   spin_label in {'spinUp'} (non spin-polarized) or {'spinUp','spinDw'} (spin-polarized)

        spec.exit_code(401, 'REACHED_MAXIMUM_TRY_NUMBER',           message='The workflow reached the maximum number of tries.')
        spec.exit_code(402, 'NO_STARTING_WAVECAR_forDFTvo',         message='No starting WAVECAR found for DFTvo calculation.')
        spec.exit_code(403, 'NO_STARTING_WAVECAR_WAVEDER_forG0W0',  message='Cannot start G0W0 as DFTvo failed.')

        spec.outline(
            cls.initialize,
            while_(cls.should_wc_continue)(
                cls.update_state,
                cls.validate_step,
                cls.prepare_step,
                cls.correct_previous_errors,
                cls.execute_step,
            ),
            cls.elaborate_results,
        )

    def initialize(self):
        """Initialize workflow context: the generic phase-index FSM position, the ordered phase list itself
        (self.ctx.phases) together with the class each phase submits (self.ctx._next_workchain), and the
        workflow state container (state_WC). No submission or input-preparation logic here.

        self.ctx.phases and self.ctx._next_workchain are built together here as ctx state, so a subclass can
        override this single method (call super().initialize(), then patch self.ctx.phases/
        self.ctx._next_workchain) to change "what runs" without touching any other part of the FSM engine. See
        the class docstring's Design rationale [2] for the intended future use (an ML/interpolation surrogate
        overriding just the '2DFTvo'/'3G0W0' entries)."""
        #[1] Generic phase-index FSM position (see WorkflowPhase above)
        self.ctx.phase_idx = -1        # -1 = not yet started
        self.ctx.phase_status = None   # meaningful only once phase_idx is in range
        self.ctx.terminal = None       # None while iterating; TerminalState once done

        #[2] Ordered phase list - DFT/GW only. A subclass wanting a different '3G0W0' child class (e.g. a future
        # ML/interpolation surrogate) overrides initialize(), calls super().initialize() first, then replaces the
        # relevant entries of self.ctx.phases (and self.ctx._next_workchain[key] below) afterward.
        self.ctx.phases = (
            WorkflowPhase(
                key='1DFTgr',
                get_restart_folder=lambda self: self.ctx.state_WC.starting_RemoteData,
                build_inputs=lambda self: self._prepare_inputs_DFT(
                    restart_folder=self.ctx.state_WC.starting_RemoteData, calc_type='1DFTgr'),
                capture_outputs=lambda self, node: self.ctx.state_WC.restart_folders.__setitem__(
                    'for_2DFTvo', node.outputs.remote_folder),
                skip_if=lambda self: not self.inputs.ns_option.run_1DFTgr.value,
                seed_restart_if_skipped=lambda self: self.ctx.state_WC.restart_folders.__setitem__(
                    'for_2DFTvo', self.ctx.state_WC.starting_RemoteData),
            ),
            WorkflowPhase(
                key='2DFTvo',
                get_restart_folder=lambda self: self.ctx.state_WC.restart_folders.for_2DFTvo,
                required_files=('WAVECAR',),
                missing_files_exit_code_name='NO_STARTING_WAVECAR_forDFTvo',
                build_inputs=lambda self: self._prepare_inputs_DFT(
                    restart_folder=self.ctx.state_WC.restart_folders.for_2DFTvo, calc_type='2DFTvo'),
                capture_outputs=lambda self, node: self.ctx.state_WC.restart_folders.__setitem__(
                    'for_3G0W0', node.outputs.remote_folder),
                skip_if=lambda self: not self.inputs.ns_option.run_2DFTvo_3G0W0.value,
            ),
            WorkflowPhase(
                key='3G0W0',
                get_restart_folder=lambda self: self.ctx.state_WC.restart_folders.for_3G0W0,
                required_files=('WAVECAR', 'WAVEDER'),
                missing_files_exit_code_name='NO_STARTING_WAVECAR_WAVEDER_forG0W0',
                build_inputs=lambda self: self._prepare_inputs_G0W0(
                    restart_folder=self.ctx.state_WC.restart_folders.for_3G0W0),
                capture_outputs=lambda self, node: None,
                skip_if=lambda self: not self.inputs.ns_option.run_2DFTvo_3G0W0.value,
            ),
        )
        # NOTE: the '3G0W0' required_files check is redundant with VaspAtomicG0W0WorkChain's own internal
        # restart-folder validation, but harmless to keep as defense-in-depth (fails fast in validate_step, before
        # a submission is even attempted).

        #[3] Which workchain class should be submitted for each phase - grouped right here with self.ctx.phases so
        # a subclass overriding "what actually runs" only has one place to look.
        self.ctx._next_workchain = {
            '1DFTgr': WorkflowFactory('vasp.vasp'),
            '2DFTvo': WorkflowFactory('vasp.vasp'),
            '3G0W0':  VaspAtomicG0W0WorkChain,
        }

        #[4] Workflow state container. submitted/retries are seeded generically from self.ctx.phases (rather than
        # hardcoded phase-name literals) so a subclass overriding self.ctx.phases needs no changes here at all.
        self.ctx.state_WC = AttributeDict({
            'starting_RemoteData': None,
            'restart_folders': AttributeDict({
                'for_2DFTvo': None,   # RemoteData with WAVECAR + CHGCAR
                'for_3G0W0':  None,   # RemoteData with WAVECAR + WAVEDER
            }),
            'submitted': AttributeDict({phase.key: [] for phase in self.ctx.phases}),
            'retries':   AttributeDict({phase.key: 0  for phase in self.ctx.phases}),
        })

        #[5] Store optional external starting RemoteData
        try:
            self.ctx.state_WC.starting_RemoteData = self.inputs.ns_reference.starting_RemoteData
        except Exception:
            self.ctx.state_WC.starting_RemoteData = None

        #[6] Scratch space used later by prepare/execute
        self.ctx.inputs_finalized = None

        #[7] Spin polarization: magnetic_moment_onsite is a top-level exposed port (from VaspAtomicG0W0WorkChain's
        # namespace).
        self.ctx.is_spinpol  = ('magnetic_moment_onsite' in self.inputs)
        self.ctx.spin_labels = ('spinUp', 'spinDw') if self.ctx.is_spinpol else ('spinUp',)

    def should_wc_continue(self) -> bool:
        """Determine whether the workflow should continue looping."""
        return self.ctx.terminal is None

    def _advance_to_first_runnable_phase(self):
        """Move ctx.phase_idx forward past any phase (leading, or reached mid-loop after a previous phase
        completes) whose skip_if(self) is True, running each skipped phase's seed_restart_if_skipped hook once.
        Called both at INIT (phase_idx -1 -> 0) and after each phase completes. Sets ctx.terminal=COMPLETE once
        every remaining phase is skipped/done."""
        phases = self.ctx.phases
        while self.ctx.phase_idx < len(phases) and phases[self.ctx.phase_idx].skip_if(self):
            phases[self.ctx.phase_idx].seed_restart_if_skipped(self)
            self.ctx.phase_idx += 1
        if self.ctx.phase_idx >= len(phases):
            self.ctx.terminal = TerminalState.COMPLETE
        else:
            self.ctx.phase_status = PhaseStatus.PENDING

    def update_state(self):
        """Update the FSM execution state: inspect the current phase_idx/phase_status, inspect the last submitted
        workchain for the active phase (if any), decide the next phase_idx/phase_status/terminal. Performs no
        submissions and no INCAR/input preparation. Generic over self.ctx.phases."""
        #[1] Not yet started -> pick the first runnable phase
        if self.ctx.phase_idx == -1:
            self.ctx.phase_idx = 0
            self.report(f"[<{self.inputs.ns_option.calculation_label.value}> update_state] starting phase loop over {[p.key for p in self.ctx.phases]}")
            self._advance_to_first_runnable_phase()
            return

        phase = self.ctx.phases[self.ctx.phase_idx]

        #[2] PENDING -> nothing to decide yet, execute_step will submit it
        if self.ctx.phase_status != PhaseStatus.RUNNING:
            return

        #[3] RUNNING -> DONE (advance) / retry / FAIL
        node = self.__last_wc_node(self.ctx, phase.key)
        if node is None or not node.is_finished:
            return
        if node.is_finished_ok:
            phase.capture_outputs(self, node)
            self.report(f"[<{self.inputs.ns_option.calculation_label.value}> update_state] {phase.key} finished successfully")
            self.ctx.phase_idx += 1
            self._advance_to_first_runnable_phase()
        else:
            return_error = self.__handle_failure_with_retry(step=phase.key)
            if return_error is not None:
                return return_error
        return

    def validate_step(self):
        """Validate prerequisites for the next PENDING step. Only performs checks (and, on failure, sets
        terminal=FAILED); does NOT modify any other state. Generic over self.ctx.phases."""
        if self.ctx.phase_idx == -1 or self.ctx.phase_status != PhaseStatus.PENDING:
            return
        phase = self.ctx.phases[self.ctx.phase_idx]
        if not phase.required_files:
            return
        ok = self.__validate_remote_has_required_files(
            remote=phase.get_restart_folder(self),
            required=list(phase.required_files),
            label=f'{phase.key} restart',
        )
        if not ok:
            self.ctx.terminal = TerminalState.FAILED
            return getattr(self.exit_codes, phase.missing_files_exit_code_name)

    def correct_previous_errors(self):
        """Attempt to correct errors from a previous failed calculation before resubmitting. Currently a NO-OP;
        intended to be extended in the future. MUST NOT submit calculations."""
        return

    def execute_step(self):
        """Submit the active phase's calculation, if it is PENDING. Generic over self.ctx.phases."""
        if self.ctx.phase_idx == -1 or self.ctx.phase_status != PhaseStatus.PENDING:
            return
        phase = self.ctx.phases[self.ctx.phase_idx]

        #[1] Submit
        running_wc = self.submit(self.ctx._next_workchain[phase.key], **self.ctx.inputs_finalized)

        #[2] Bump retry counter (this submission attempt)
        self.ctx.state_WC.retries[phase.key] += 1
        tmp_attempt_num = self.ctx.state_WC.retries[phase.key]

        #[3] Record submission into state_WC
        self.ctx.state_WC.submitted[phase.key].append(running_wc)

        #[4] Update execution state
        self.ctx.phase_status = PhaseStatus.RUNNING

        #[5] Log
        self.__report_compact_submission(running_wc, tmp_attempt_num, phase.key)

        #[6] Register dependency for engine
        return ToContext(**{f'calc_{phase.key}': running_wc})

    def prepare_step(self):
        """Prepare inputs for the active phase, if it is PENDING. Builds inputs_finalized; does NOT submit and
        does NOT change execution state. Generic over self.ctx.phases."""
        if self.ctx.phase_idx == -1 or self.ctx.phase_status != PhaseStatus.PENDING:
            return
        phase = self.ctx.phases[self.ctx.phase_idx]
        self.ctx.inputs_finalized = phase.build_inputs(self)

    def elaborate_results(self):
        """Final post-processing and output assembly:
        1. Identifies the last successful DFT and G0W0 nodes
        2. Exposes raw outputs (bands, kpoints, RemoteData)
        3. Computes spin-resolved gaps and band extrema using elaborate_single_spin_component
        4. Builds nested Dict outputs with a stable, documented structure
        5. Computes quasiparticle (QP) corrections if G0W0 is available
        Output dictionary structure (conceptual):
            gaps = { 'DFT': {spin_label: {Dir, Ind, Gam}}, 'G0W0': {spin_label: {Dir, Ind, Gam}} }
            gaps_QPc = {spin_label: {Dir, Ind, Gam}}
        where spin_label in {'spinUp', 'spinDw'}. All numerical values are floats in eV."""

        # ---------------------------------------------------------
        #[1] Determine last successful DFT and G0W0 nodes. Generic: reads state_WC.submitted presence (via
        # __last_wc_node, which returns None when a phase was never submitted) rather than re-deriving "did this
        # phase run" from the ns_option.* input flags - this is what lets a subclass with a differently-skipped
        # '2DFTvo' (e.g. an ML surrogate, see the class docstring) work here with zero override: the later of
        # '1DFTgr'/'2DFTvo' that actually produced a node wins as the DFT baseline, and '3G0W0' is looked up
        # unconditionally regardless of *why* an earlier phase was or wasn't skipped.
        last_node_DFT = None
        for key in ('1DFTgr', '2DFTvo'):
            node = self.__last_wc_node(self.ctx, key)
            if node is not None:
                last_node_DFT = node
        last_node_G0W0 = self.__last_wc_node(self.ctx, '3G0W0')

        #[2] Expose node outputs that do not need further elaboration
        if last_node_DFT:
            self.out('RemoteData_DFT', last_node_DFT.outputs.remote_folder)
            self.out('NGarray',        last_node_DFT.outputs.NGarray)
            self.out('ENMAXarray',     last_node_DFT.outputs.ENMAXarray)
            self.out('kpoints',        last_node_DFT.outputs.kpoints)
            self.out('bands_DFT',      last_node_DFT.outputs.bands)

        if last_node_G0W0:
            self.out('RemoteData_G0W0', last_node_G0W0.outputs.remote_folder)
            self.out('bands_G0W0',      last_node_G0W0.outputs.bands)

        #[3] spin handling - magnetic_moment_onsite is a top-level port here (see initialize()'s comment [7]).
        spin_channels = ([0, 1] if "magnetic_moment_onsite" in self.inputs else [None])

        #[4] Elaborate DFT and G0W0 bands
        gaps = AttributeDict();      bnd_extrema = AttributeDict()
        gaps_DFT = AttributeDict();  bnd_extrema_DFT = AttributeDict()
        gaps_G0W0 = AttributeDict(); bnd_extrema_G0W0 = AttributeDict()
        for sp_comp in spin_channels:
            sp_label = "spinUp" if sp_comp in (None, 0) else "spinDw"
            bd_DFT_trimmed_sp_comp,  bnd_extrema_DFT_sp_comp,  gap_DFT_sp_comp  = \
                self.elaborate_single_spin_component(last_node_DFT,  sp_comp)
            bd_G0W0_trimmed_sp_comp, bnd_extrema_G0W0_sp_comp, gap_G0W0_sp_comp = \
                self.elaborate_single_spin_component(last_node_G0W0, sp_comp)
            gaps_DFT[sp_label]         = deepcopy(gap_DFT_sp_comp)
            bnd_extrema_DFT[sp_label]  = deepcopy(bnd_extrema_DFT_sp_comp)
            gaps_G0W0[sp_label]        = deepcopy(gap_G0W0_sp_comp)
            bnd_extrema_G0W0[sp_label] = deepcopy(bnd_extrema_G0W0_sp_comp)

        if len(gaps_G0W0):        gaps.G0W0 = gaps_G0W0
        if len(bnd_extrema_G0W0): bnd_extrema.G0W0 = bnd_extrema_G0W0
        if len(gaps_DFT):         gaps.DFT = gaps_DFT
        if len(bnd_extrema_DFT):  bnd_extrema.DFT = bnd_extrema_DFT
        gaps = Dict(dict=gaps).store()
        self.out('gaps', gaps)

        #[5] QP Corrections
        if last_node_G0W0:
            gaps_QPc = AttributeDict(); bnd_extrema_QPc = AttributeDict()
            qp_bands_list = []; qp_occ_list = []; qp_kpoints = None

            for sp_label in bnd_extrema_DFT.keys():
                sp_comp = 0 if sp_label == "spinUp" and self.ctx.is_spinpol else (None if not self.ctx.is_spinpol else 1)

                # Recompute trimmed bands (cheap, consistent)
                bd_DFT_trimmed,  bnd_DFT,  _ = self.elaborate_single_spin_component(last_node_DFT,  sp_comp)
                bd_G0W0_trimmed, bnd_G0W0, _ = self.elaborate_single_spin_component(last_node_G0W0, sp_comp)

                # QP extrema (HOMO/LUMO) ---
                QPc = AttributeDict()
                QPc["HOMO"] = bnd_G0W0["HOMO"] - bnd_DFT["HOMO"]
                QPc["LUMO"] = bnd_G0W0["LUMO"] - bnd_DFT["LUMO"]
                bnd_extrema_QPc[sp_label] = QPc

                # QP gap corrections ---
                gap_QPc = AttributeDict()
                gap_QPc["Dir"] = Float(np.min(QPc["LUMO"] - QPc["HOMO"]))
                gap_QPc["Ind"] = Float(np.min(QPc["LUMO"]) - np.max(QPc["HOMO"]))
                gap_QPc["Gam"] = Float(QPc["LUMO"][0] - QPc["HOMO"][0])
                gaps_QPc[sp_label] = gap_QPc

                # QP bands ---
                qp_bands_list.append(bd_G0W0_trimmed.get_array("bands") - bd_DFT_trimmed.get_array("bands"))
                qp_occ_list.append(bd_DFT_trimmed.get_array("occupations"))
                if qp_kpoints is None:
                    qp_kpoints = bd_DFT_trimmed.get_kpoints()

            # Assemble BandsData for QP bands ---
            bd_QPcorr = DataFactory("core.array.bands")()
            bd_QPcorr.set_kpoints(qp_kpoints)

            if len(qp_bands_list) == 1:
                bd_QPcorr.set_bands(qp_bands_list[0], occupations=qp_occ_list[0])
            else:
                bd_QPcorr.set_bands(np.stack(qp_bands_list), occupations=np.stack(qp_occ_list))

            # --- Attach to outputs ---
            bd_QPcorr.store()
            gaps_QPc = Dict(dict=gaps_QPc).store()
            self.out("bands_QPc", bd_QPcorr)
            self.out("gaps_QPc",  gaps_QPc)

            self.__report_compact_results(last_node_DFT=last_node_DFT, last_node_G0W0=last_node_G0W0,
                                           gaps_dict=gaps, gaps_qpc_dict=gaps_QPc,
                                           bnd_extrema_DFT=bnd_extrema.DFT, bnd_extrema_G0W0=bnd_extrema.G0W0)

    ##[HELPER FUNCTIONS for elaborate_results]
    @staticmethod
    def elaborate_single_spin_component(last_node, spin_index=None, OCCUPATION_THRESHOLD=0.45):
        """Compute gaps and quasiparticle corrections for a single spin component.
        Parameters
        last_node : WorkChainNode
        spin_index : int or None   None -> non spin-polarized; 0/1 -> spin-polarized component
        OCCUPATION_THRESHOLD : float, occupation below which a state is considered unoccupied.
        Returns (bands_trimmed, bnd_extrema, gap); all None if last_node is None."""
        #[0] Guard against last_node as None
        if last_node is None:
            return (None, None, None)

        #[1] Extract bands and occupations ---
        if spin_index is None:
            bands   = last_node.outputs.bands.get_array("bands")
            bnd_occ = last_node.outputs.bands.get_array("occupations")
        else:
            bands   = last_node.outputs.bands.get_array("bands")[spin_index]
            bnd_occ = last_node.outputs.bands.get_array("occupations")[spin_index]

        #[1] HOMO / LUMO detection ---
        occ = bnd_occ < OCCUPATION_THRESHOLD
        c_kptNum = occ.shape[0]
        bndIdx_HOMOar = []; bndIdx_LUMOar = []
        for kptIdx in range(c_kptNum):
            occ_crossings = np.where(occ[kptIdx, 1:] != occ[kptIdx, :-1])[0]
            if len(occ_crossings) == 0:
                raise ValueError("No HOMO/LUMO crossing found (metallic system?)")
            bndIdx_HOMOar.append(occ_crossings[0])
            bndIdx_LUMOar.append(occ_crossings[0] + 1)

        #[2] Energy extraction ---
        bnd_extrema = AttributeDict()
        bnd_extrema['HOMO'] = np.array([bands[k, i] for k, i in enumerate(bndIdx_HOMOar)])
        bnd_extrema['LUMO'] = np.array([bands[k, i] for k, i in enumerate(bndIdx_LUMOar)])

        #[3] Gaps ---
        gap = AttributeDict()
        gap['Dir'] = np.min(bnd_extrema['LUMO'] - bnd_extrema['HOMO'])
        gap['Ind'] = np.min(bnd_extrema['LUMO']) - np.max(bnd_extrema['HOMO'])
        gap['Gam'] = bnd_extrema['LUMO'][0] - bnd_extrema['HOMO'][0]

        #[4] Trim -1 bands (AiiDA can pad bands arrays with -1; keep only bands without -1 entries) ---
        bands_wocc = np.array(last_node.outputs.bands.get_bands(also_occupations=True))
        c_kpt_num = bands_wocc.shape[1]
        c_bnd_num = bands_wocc.shape[2]
        try:
            firstBnd_toTrim = min(np.nonzero(np.in1d(bands_wocc[0, i, :], [-1]))[0][0] for i in range(c_kpt_num))
        except Exception:
            firstBnd_toTrim = c_bnd_num

        bands_trim = bands_wocc[0, :, :firstBnd_toTrim]
        occup_trim = bands_wocc[1, :, :firstBnd_toTrim]
        bands_trimmed = DataFactory('core.array.bands')()
        bands_trimmed.set_kpoints(last_node.outputs.bands.get_kpoints())
        bands_trimmed.set_bands(bands_trim, occupations=occup_trim)
        return (bands_trimmed, bnd_extrema, gap)

    ##[HELPER FUNCTIONS for prepare_step]
    def _prepare_inputs_DFT(self, restart_folder, calc_type):
        """Prepare inputs for a DFT calculation (DFTgr or DFTvo). `restart_folder` is saved directly into
        inputs.restart_folder without checking its validity here (if it's None, None is saved) - all checks MUST
        be performed earlier (validate_step)."""
        #[1] Base
        inputs = AttributeDict()
        inputs.update(self.exposed_inputs(self.ctx._next_workchain[calc_type]))
        inputs.clean_workdir = Bool(False)
        inputs.restart_folder = restart_folder

        #[2] Parser settings
        inputs.settings = AttributeDict({'parser_settings': {'include_node': ['bands', 'kpoints', 'structure', 'NGarray', 'maximum_number_pw']}})

        #[3] INCAR
        incar = {'incar': {'ediff': 1E-7, 'algo': "Normal", 'nelm': 200,'loptics': '.TRUE.',
                            'ismear': 0, 'sigma': 0.02,
                            'prec': 'Accurate', 'lmaxmix': 4, 'lorbit': 11}}
        if ('encut' in self.inputs):
            incar['incar']['encut'] = self.inputs.encut.value
        if self.ctx.is_spinpol:
            _, incar['incar']['magmom'] = input_magnetic_moment_tomagmom(self.inputs.structure, self.inputs.magnetic_moment_onsite.get_dict())
            incar['incar']['ispin'] = 2;    incar['incar']['icharg'] = 1
            incar['incar']['amix_mag'] = 0.8; incar['incar']['bmix_mag'] = 0.00001
            incar['incar']['amix'] = 0.2;     incar['incar']['bmix'] = 0.00001
        else:
            incar['incar']['ispin'] = 1

        # Parallelization settings: kpar/npar/lreal from the (now shared, top-level) optimization namespace.
        if ('kpar' in self.inputs.optimization):    incar['incar']['kpar'] = self.inputs.optimization.kpar.value
        if ('npar' in self.inputs.optimization):    incar['incar']['npar'] = self.inputs.optimization.npar.value
        if (self.inputs.optimization.lreal.value == True):  incar['incar']['lreal'] = 'Auto'
        else:   incar['incar']['lreal'] = '.FALSE.'

        if (calc_type == '2DFTvo'):
            if ('nbands' in self.inputs): incar['incar']['nbands'] = self.inputs.nbands.value
            incar['incar']['algo'] = "Exact"
            incar['incar']['nelm'] = 1

        inputs.parameters = Dict(dict=incar)
        prepared_inputs = prepare_process_inputs(inputs, namespaces=['calc', 'dynamics', 'verify'])
        return prepared_inputs

    def _prepare_inputs_G0W0(self, restart_folder):
        """Prepare inputs for the G0W0 phase by delegating entirely to VaspAtomicG0W0WorkChain's own input
        namespace - unlike _prepare_inputs_DFT above, NO local INCAR construction happens here: the atomic
        wrapper builds its own INCAR internally, inside its own init_inputs()/__build_parameters(). No
        `inputs.settings` is set either - `settings` was excluded from this file's expose_inputs(
        VaspAtomicG0W0WorkChain, ...) call in define(), so VaspAtomicG0W0WorkChain.init_inputs() sees an
        empty/absent settings and applies its own default (bands/kpoints/structure inclusion + WAVEDER copy),
        which is exactly what elaborate_results() above needs."""
        inputs = AttributeDict()
        inputs.update(self.exposed_inputs(VaspAtomicG0W0WorkChain))
        inputs.clean_workdir = Bool(False)
        inputs.restart_folder = restart_folder
        prepared_inputs = prepare_process_inputs(inputs, namespaces=['calc', 'dynamics', 'verify'])
        return prepared_inputs

    ##[HELPER FUNCTIONS for update_state and validate_step]
    @staticmethod
    def __last_wc_node(ctx, step: str):
        """Return last submitted child WC node for a given submitted list key: '1DFTgr' | '2DFTvo' | '3G0W0'."""
        lst = ctx.state_WC.submitted.get(step, [])
        return lst[-1] if lst else None

    def __handle_failure_with_retry(self, step: str):
        """Handle a failed step with retry logic. Returns an ExitCode if retries are exhausted, otherwise None
        (and resets ctx.phase_status back to PENDING for the same ctx.phase_idx)."""
        max_iter = self.inputs.ns_option.maximum_iterations.value
        retries  = self.ctx.state_WC.retries[step]
        if retries < max_iter:
            self.report(f"[update_state] {step} failed, retrying (attempt {retries+1}/{max_iter})")
            self.ctx.phase_status = PhaseStatus.PENDING
            return None

        self.report(f"[update_state] {step} failed AND maximum retries reached -> ABORT")
        self.ctx.terminal = TerminalState.FAILED
        return self.exit_codes.REACHED_MAXIMUM_TRY_NUMBER

    def __validate_remote_has_required_files(self, remote, required, label: str):
        """Validate that RemoteData exists and contains the required files. Returns True/False; logs missing
        files (and listdir failures) via self.report."""
        if remote is None:
            self.report(f"[validate_step] {label} restart folder is None")
            return False
        try:
            files = set(remote.listdir())
        except Exception as exc:
            self.report(f"[update_state] Could not list {label} folder contents: {exc}")
            return False
        missing = [fname for fname in required if fname not in files]
        if missing:
            self.report(f"[update_state] {label} missing required files: {missing}")
            return False
        return True

    ##[HELPER FUNCTIONS for execute_step / elaborate_results reporting]
    @staticmethod
    def __fmt_float(x, nd=3):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    @staticmethod
    def __generate_compact_submission_string(wc_node, prefix="  > "):
        """Emit a compact input summary right before/after submitting a calculation."""
        def _get_incar_par(aiida_dict, key):
            try:
                return aiida_dict.get_dict()['incar'][key]
            except Exception:
                return None
        encut     = VaspG0W0GroundUpWorkChain.__fmt_float(_get_incar_par(wc_node.inputs.parameters, "encut"))
        nbands    = _get_incar_par(wc_node.inputs.parameters, "nbands")
        encut_chi = _get_incar_par(wc_node.inputs.parameters, "encutgw")
        nomega    = _get_incar_par(wc_node.inputs.parameters, "nomega")

        mesh = offset = nkpts = None
        try:
            mesh, offset = wc_node.inputs.kpoints.get_kpoints_mesh()
            nkpts = int(mesh[0]) * int(mesh[1]) * int(mesh[2])
        except Exception:
            try:
                kpts = wc_node.inputs.kpoints.get_kpoints()
                nkpts = int(len(kpts))
                mesh = "explicit"
            except Exception:
                mesh = "unknown"

        try:
            kpar = _get_incar_par(wc_node.inputs.parameters.get_dict(), "kpar")
        except Exception:
            kpar = None

        pot_family = None
        pot_mapping = None
        try:
            pot_family = wc_node.inputs.potential_family.value
        except Exception:
            pass
        try:
            pot_mapping = wc_node.inputs.potential_mapping.get_dict()
        except Exception:
            pass

        lines = [f"{prefix}nbands={nbands}  encut={encut}  encut_chi={encut_chi}  nomega={nomega}  kpar={kpar}", "\n",
                 f"{prefix}kpts_mesh={mesh} nkpts={nkpts}", "\n",
                 f"{prefix}potcars_family={pot_family}  potcars_mapping={pot_mapping}"]
        return "".join(lines)

    def __report_compact_submission(self, running_wc, tmp_attempt, calc_type: str):
        """Emit a compact input summary right after submitting a calculation.

        NOTE (accepted cosmetic gap): for the '3G0W0' phase, running_wc.inputs.parameters is just
        VaspAtomicG0W0WorkChain's inert default ({} - 'parameters' is excluded from this file's own exposure, and
        _prepare_inputs_G0W0 never sets it), so encut/nbands/encut_chi/nomega will print as None for G0W0
        submissions specifically. Harmless (already wrapped in try/except above); an easy future improvement
        would read self.inputs.encut/.nbands/etc. directly for the '3G0W0' case instead."""
        try:
            label = self.inputs.ns_option.calculation_label.value
        except Exception:
            label = ""
        prolog = (f"[<{label}> execute_step] launching {calc_type} pk={running_wc.pk} "
                  f"(attempt num={tmp_attempt}) -> state updated to={calc_type} {self.ctx.phase_status.name}")

        msg = prolog + "\n" + self.__generate_compact_submission_string(running_wc) + "\n"
        self.report(msg)

    def __report_compact_results(self, last_node_DFT, last_node_G0W0,
                                  gaps_dict, gaps_qpc_dict=None,
                                  bnd_extrema_DFT=None, bnd_extrema_G0W0=None):
        """Emit a compact result summary at the end of elaborate_results.

        Assumptions
        -----------
        gaps_dict has: gaps_dict["DFT"][spin]["Dir"|"Ind"|"Gam"], gaps_dict["G0W0"][spin]["Dir"|"Ind"|"Gam"]
        (only if G0W0 ran). gaps_qpc_dict (optional) has: gaps_qpc_dict[spin]["Dir"|"Ind"|"Gam"].
        Also reuses __generate_compact_submission_string to print the same compact input summary for the
        finished child nodes."""
        #[1.1] Preliminary: Helpers
        def __sec_to_hms(sec):
            try:
                sec = float(sec)
            except Exception:
                return None
            if sec < 0:
                return None
            h = int(sec // 3600)
            m = int((sec - 3600 * h) // 60)
            s = int(sec - 3600 * h - 60 * m)
            return f"{h:02d}:{m:02d}:{s:02d}"

        def __bytes_to_gib(x):
            try:
                return float(x) / (1024.0 ** 3)
            except Exception:
                return None

        def __extract_perf(node):
            out = {"elapsed_s": None, "elapsed_hms": None, "max_mem_gib": None, "n_mpi_ranks": None}
            if node is None:
                return out
            try:
                misc = node.outputs.misc.get_dict()
                out["elapsed_s"] = misc['run_stats']['elapsed_time']
                out["elapsed_hms"] = __sec_to_hms(out["elapsed_s"]) if out["elapsed_s"] is not None else None
                mem_b = misc.get("maximum_memory_used", None)
                out["max_mem_gib"] = __bytes_to_gib(mem_b) if mem_b is not None else None
            except Exception:
                pass
            try:
                opts = node.inputs.options.get_dict()
                res = opts.get("resources", {})
                out["n_mpi_ranks"] = int(res["num_machines"]) * int(res["num_mpiprocs_per_machine"])
            except Exception:
                pass
            return out

        def __fmt_arr(x):
            x = np.array(x)
            return np.array2string(x, precision=4, separator=" ", max_line_width=10**9)

        #[1.2] Preliminary: Initial Label
        try:
            label = self.inputs.ns_option.calculation_label.value
        except Exception:
            label = ""
        lines = []

        #[1.3] Preliminary: allow passing a Dict instead of a dict
        if hasattr(gaps_dict, "get_dict"):
            gaps_dict = gaps_dict.get_dict()
        if gaps_qpc_dict is not None and hasattr(gaps_qpc_dict, "get_dict"):
            gaps_qpc_dict = gaps_qpc_dict.get_dict()

        #[1.4] Preliminary: Define base keys
        keys = ("Dir", "Ind", "Gam")

        #[2] Performance notes
        if last_node_DFT is not None:
            p = __extract_perf(last_node_DFT)
            lines += [f"\n  [1] DFT perf: elapsed={p['elapsed_hms'] or p['elapsed_s']}  -  mpi_ranks={p['n_mpi_ranks']}", "\n"]

        if last_node_G0W0 is not None:
            p = __extract_perf(last_node_G0W0)
            lines += [f"  [1] G0W0 perf: elapsed={p['elapsed_hms'] or p['elapsed_s']}  -  mpi_ranks={p['n_mpi_ranks']}", "\n"]

        #[3] Input strings
        if last_node_DFT is not None:
            lines += ["  [2] DFT inputs:\n"]
            lines += [self.__generate_compact_submission_string(wc_node=last_node_DFT, prefix="  [2] "), "\n"]
        if last_node_G0W0 is not None:
            lines += ["  [2]G0W0 inputs:\n"]
            lines += [self.__generate_compact_submission_string(wc_node=last_node_G0W0, prefix="  [2] "), "\n"]

        #[4]/[5] HOMO/LUMO eigenvalue arrays + gaps
        for sp in self.ctx.spin_labels:
            lines += [f"  [3] HOMO/LUMO - Spin component : {sp}\n"]
            if bnd_extrema_DFT is not None:
                lines += [f"  [3] HOMO DFT eigenvalues: {__fmt_arr(bnd_extrema_DFT[sp]['HOMO'])}\n"]
                lines += [f"  [3] LUMO DFT eigenvalues: {__fmt_arr(bnd_extrema_DFT[sp]['LUMO'])}\n"]
            if bnd_extrema_G0W0 is not None:
                lines += [f"  [3] HOMO GW eigenvalues : {__fmt_arr(bnd_extrema_G0W0[sp]['HOMO'])}\n"]
                lines += [f"  [3] LUMO GW eigenvalues : {__fmt_arr(bnd_extrema_G0W0[sp]['LUMO'])}\n"]

            lines += [f"  [4] Gaps - Spin component : {sp}\n"]
            if ("DFT" in gaps_dict) and (sp in gaps_dict["DFT"]):
                for k in keys:
                    lines += [f"  [4] gap_DFT_{k}{Float(gaps_dict['DFT'][sp][k])}\n"]
            if ("G0W0" in gaps_dict) and (sp in gaps_dict["G0W0"]):
                for k in keys:
                    lines += [f"  [4] gap_G0W0_{k}{Float(gaps_dict['G0W0'][sp][k])}\n"]
            if (gaps_qpc_dict is not None) and (sp in gaps_qpc_dict):
                qpc_sp = gaps_qpc_dict[sp]
                lines += [f"  [5] QPc gaps {sp}: Dir={VaspG0W0GroundUpWorkChain.__fmt_float(qpc_sp['Dir'])}  "
                          f"Ind={VaspG0W0GroundUpWorkChain.__fmt_float(qpc_sp['Ind'])}  "
                          f"Gam={VaspG0W0GroundUpWorkChain.__fmt_float(qpc_sp['Gam'])}", "\n"]

        self.report("".join([f"[<{label}> elaborate_results] Summary"] + lines))
