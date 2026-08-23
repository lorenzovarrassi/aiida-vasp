# pylint: disable=too-many-arguments

from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.engine import process_handler, ProcessHandlerReport

from aiida_vasp.workchains.v2.vasp import VaspWorkChain
from aiida_vasp.workchains.vMBPT.utils_helpers_extrapolation import input_magnetic_moment_tomagmom

PRECFOCK_VOLUME_THRESHOLD = 250


class VaspAtomicG0W0WorkChain(VaspWorkChain):
    """Run exactly one G0W0 VASP calculation.
    Purpose of workchain:
    1) Hide the complexity of G0W0 INCAR construction from the user, which is managed by the workchain.
    2) Validate the inputs for consistency and completeness (encut/encut_chi warnings, restart-folder file check).
    3) Introduce specific error handling for the G0W0 case: excepted-GW retry and exit-700 resource fallback.

    Design rationale:
    1) This workchain is a single-step workchain, not a multi-step workflow. It is not intended to manage a sequence of calculations.
       It's essentially a wrapper around a single VaspCalculation, with the added value of input validation, INCAR construction, and error handling.
       Thus it expects a restart_folder input, which is the output of a previous VASP calculation (usually a DFT calculation) that produced
       the necessary WAVECAR/WAVEDER files.
    2) Subclass VaspWorkChain:
       spec.outline() is inherited unchanged from VaspWorkChain.define() via super().define(spec) and not redefined here.
       spec.outline() is:
        setup -> init_inputs
              -> if_(run_auto_parallel)(prepare_inputs, perform_autoparallel)
              -> while_(should_run_process)(prepare_inputs, run_process, inspect_process)
              -> results
        Of those steps, init_inputs() and inspect_process() are overridden below - init_inputs() performs both
        input validation and INCAR construction; inspect_process() adds the excepted-GW retry (see [3] below).
        The other steps (setup, prepare_inputs, run_process, results) are inherited as-is from VaspWorkChain.
    3) Two failure-recovery fallbacks:
         [1] Excepted-GW retry: too low OMEGATL parameter (which by default is automatically set by VASP) 
             cause a abrupt crash with an MPI_ABORT error. This would normally be an excepted node, which would 
             abort the workchain with ERROR_SUB_PROCESS_EXCEPTED. Instead, this workchain intercepts that excepted node and 
             retries once with OMEGATL=16000. If that retry also excepts, the workchain aborts with ERROR_EXCEPTED_RETRY_FAILED.
         [2] Exit-700 resource fallback: ERROR_DID_NOT_FINISH may be cause by either OoM or simply run reached walltime.
             As GW calculations are memory-intensive, the workchain allows a single retry with extraresources_fallback_options (if supplied) to request more resources.
    4) WFULL/W0* files are always written by VASP for a G0W0 run - no toggle is exposed for this; downstream
       consumers (e.g. VaspAtomicBSEWorkChain in 'BSE'/readFromFiles mode) pick them up directly from this
       workchain's remote_folder output.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('parameters',     valid_type=orm.Dict,       required=False, default=lambda: orm.Dict(dict={}),
                   help='Ignored - INCAR is built internally by this workchain from its own scalar inputs.')
        spec.input('restart_folder', valid_type=orm.RemoteData, required=True,
                   help='Output of a previous DFT calculation, providing WAVECAR/WAVEDER for the G0W0 restart.')
        spec.input('encut',          valid_type=orm.Float, required=False, serializer=to_aiida_type,
                   help='ENCUT (eV). If omitted, VASP infers it from the restarted WAVECAR (a warning is logged).')
        spec.input('nbands',         valid_type=orm.Int,   required=False, serializer=to_aiida_type,
                   help='NBANDS. If omitted, VASP infers it from the restarted WAVECAR (a warning is logged).')
        spec.input('encut_chi',      valid_type=orm.Float, required=False, serializer=to_aiida_type,
                   help='ENCUTGW/ENCUTGWSOFT (eV). Independent of encut - if omitted, VASP determines automatically.')
        spec.input('nbandsgw',       valid_type=orm.Int,   required=False, serializer=to_aiida_type)
        spec.input('nomega',         valid_type=orm.Int,   required=False, default=lambda: orm.Int(200), serializer=to_aiida_type)
        spec.input('magnetic_moment_onsite', valid_type=orm.Dict, required=False, serializer=to_aiida_type,
                   help="Per-site magnetic moments, e.g. {'Cr1':3.0,'Cr2':-3.0}. Presence activates ISPIN=2.")

        spec.input('optimization.kpar',  valid_type=orm.Int,  required=False, default=lambda: orm.Int(4),      serializer=to_aiida_type)
        spec.input('optimization.lreal', valid_type=orm.Bool, required=False, default=lambda: orm.Bool(False), serializer=to_aiida_type)
        spec.input('optimization.set_PRECFOCK_to_Fast', valid_type=orm.Bool, required=False,
                   default=lambda: orm.Bool(True), serializer=to_aiida_type)

        spec.input('extraresources_fallback_options', valid_type=orm.Dict, required=False, serializer=to_aiida_type,
                   help=("Alternative scheduler options Dict (same shape as 'options'), used ONLY for the single "
                         "retry that handler_unfinished_calc_generic grants after an ERROR_DID_NOT_FINISH (exit "
                         "700) failure. If not supplied, the retry resubmits with the original options unchanged."))

        spec.exit_code(404, 'ERROR_EXCEPTED_RETRY_FAILED', message='Handler handle_gw_exception did not solve the exception.')
        spec.exit_code(406, 'ERROR_MISSING_RESTART_FILES',  message='restart_folder is missing one or more required files (WAVECAR, WAVEDER).')

    def __validate_remote_has_required_files(self, remote, required, or_groups=()):
        """Check that the given remote folder contains all required files, and at least one file from each of the
        or_groups. A G0W0 restart only ever needs the exact WAVECAR/WAVEDER names, so or_groups is unused here -
        kept for signature parity with VaspAtomicBSEWorkChain's own copy, which does need it (WFULL*/W0* case).
        If any required files are missing, or if any of the or_groups are not satisfied, report an error and
        return None. Return the folder's file listing on success (not currently used by any caller here, unlike
        the BSE copy, but kept for the same reason as or_groups)."""
        if remote is None:
            self.report('restart_folder is None')
            return None
        try:
            files = set(remote.listdir())
        except Exception as exc:  # pylint: disable=broad-except
            self.report(f'Could not list restart_folder contents: {exc}')
            return None
        missing = [fname for fname in required if fname not in files]
        if missing:
            self.report(f'restart_folder missing required files: {missing}')
            return None
        for group in or_groups:
            found = any(
                (pattern.endswith('*') and any(f.startswith(pattern[:-1]) for f in files)) or pattern in files
                for pattern in group
            )
            if not found:
                self.report(f'restart_folder missing at least one of: {group}')
                return None
        return files

    def __validate_inputs_entries(self):
        #[1] Encut / nbands related warnings: VASP infers encut/nbands from the restarted WAVECAR if they are not
        # supplied, but this is a potential source of confusion for users -> log an explicit warning.
        if 'encut' not in self.inputs:  self.report('WARNING: encut not supplied - VASP will infer ENCUT from the restarted WAVECAR.')
        if 'nbands' not in self.inputs: self.report('WARNING: nbands not supplied - VASP will infer NBANDS from the restarted WAVECAR.')

        # encut and encut_chi are independent - no auto-derivation, no hard requirement between them.
        if ('encut_chi' in self.inputs) and ('encut' not in self.inputs):
            self.report('WARNING: encut_chi supplied without encut - ENCUT will be inferred by VASP from the '
                        'restarted WAVECAR while ENCUTGW/ENCUTGWSOFT are set explicitly; double-check this is intended.')
        elif 'encut' in self.inputs and 'encut_chi' not in self.inputs:
            self.report('INFO: encut_chi not supplied - this lets VASP automatically determine ENCUTGW/ENCUTGWSOFT itself.')

    def __build_parameters(self):
        """Build the flat INCAR-tag dict expected directly by VaspCalculation (NOT nested under an 'incar' key -
        that nesting is only needed when handing parameters to ANOTHER VaspWorkChain's own ParametersMassage,
        which this class, being itself the VaspWorkChain, bypasses)."""
        #[1] G0W0-specific parameters, common to every run of this workchain.
        incar = {
            'nelm': 1, 'algo': 'EVGW0', 'ismear': 0, 'sigma': 0.02,
            'prec': 'Accurate', 'lmaxmix': 4, 'lorbit': 11,
            'nomega': self.inputs.nomega.value,
        }
        if 'encut' in self.inputs:
            incar['encut'] = self.inputs.encut.value
        if 'nbands' in self.inputs:
            incar['nbands'] = self.inputs.nbands.value
        if 'nbandsgw' in self.inputs:
            incar['nbandsgw'] = self.inputs.nbandsgw.value
        if 'encut_chi' in self.inputs:
            incar['encutgw'] = self.inputs.encut_chi.value
            incar['encutgwsoft'] = self.inputs.encut_chi.value

        #[2] Magnetism: MAGMOM/ISPIN. Unlike a phase inside an FSM-style chain (which can rely on ICHARG-based
        # restart continuity from an earlier DFT phase), this class must set MAGMOM itself: it's meant to be
        # usable standalone, with no guarantee of such a preceding phase.
        if 'magnetic_moment_onsite' in self.inputs:
            _, incar['magmom'] = input_magnetic_moment_tomagmom(self.inputs.structure, self.inputs.magnetic_moment_onsite.get_dict())
            incar['ispin'] = 2
            incar['icharg'] = 1
            #incar['amix_mag'] = 0.8
            #incar['bmix_mag'] = 0.00001
            #incar['amix'] = 0.2
            #incar['bmix'] = 0.00001
        else:
            incar['ispin'] = 1

        #[3] Optimization related parameters.
        incar['kpar']  = self.inputs.optimization.kpar.value
        incar['lreal'] = 'Auto' if self.inputs.optimization.lreal.value else '.FALSE.'

        volume = self.inputs.structure.get_cell_volume()
        if volume > PRECFOCK_VOLUME_THRESHOLD or self.inputs.optimization.set_PRECFOCK_to_Fast.value:
            incar['precfock'] = 'Fast'

        return incar

    def __build_settings(self):
        """Build the 'settings' dict passed to VaspCalculation, merging into whatever the caller/exposing-parent
        already put in self.ctx.inputs.settings (set by super().init_inputs() from self.inputs.settings) rather
        than overwriting it - a caller composing this class into a larger workflow (e.g. a GroundUp chain) may
        already have settings of its own. Always ensures 'bands'/'kpoints'/'structure' are parsed (this is a G0W0
        wrapper - its whole point is producing bands, so these should never be silently unavailable) and that
        WAVEDER is remote-copied for the restart, on top of whatever else was already requested."""
        settings = dict(self.ctx.inputs.get('settings') or {})

        parser_settings = dict(settings.get('parser_settings', {}))
        include_node = list(parser_settings.get('include_node', []))
        for name in ('bands', 'kpoints', 'structure'):
            if name not in include_node:
                include_node.append(name)
        parser_settings['include_node'] = include_node
        settings['parser_settings'] = parser_settings

        copy_list = list(settings.get('ADDITIONAL_REMOTE_COPY_LIST', []))
        if 'WAVEDER' not in copy_list:
            copy_list.append('WAVEDER')
        settings['ADDITIONAL_REMOTE_COPY_LIST'] = copy_list

        return settings

    def init_inputs(self):
        # Perform the superclass input initialization first, so that the inputs namespace is fully populated and ready for validation.
        exit_code = super().init_inputs()
        if exit_code is not None: return exit_code

        #[1] Validate the inputs entries for consistency and completeness.
        exit_code = self.__validate_inputs_entries()
        if exit_code is not None: return exit_code

        self.ctx.inputs.parameters = self.__build_parameters()
        self.ctx.inputs.settings = self.__build_settings()

        if self.__validate_remote_has_required_files(self.inputs.restart_folder, ('WAVECAR', 'WAVEDER')) is None:
            return self.exit_codes.ERROR_MISSING_RESTART_FILES
        return None

    # =====================================================================
    # Retry: excepted-GW branch requires an inspect_process() override (an
    # excepted node never reaches @process_handler dispatch); the exit-700
    # fallback stays a @process_handler, reached via super() below.
    #
    # super() is called in exactly the not-excepted branch, never in the
    # excepted branch (neither before nor after handling it):
    # BaseRestartWorkChain.inspect_process() starts with "if excepted ->
    # abort ERROR_SUB_PROCESS_EXCEPTED" - exactly the outcome being
    # overridden here to retry instead. Calling super() anywhere in the
    # excepted branch would just re-trigger that abort.
    # =====================================================================

    def inspect_process(self):
        node = self.ctx.children[self.ctx.iteration - 1]

        if not node.is_excepted:
            return super().inspect_process()

        self.report(f'Detected excepted G0W0 calculation {node.pk}')

        if self.ctx.get('gw_excepted_retry_launched', False):
            self.report('Recovery already attempted once; aborting')
            return self.exit_codes.ERROR_EXCEPTED_RETRY_FAILED

        self.ctx.gw_excepted_retry_launched = True

        report = self.handle_gw_exception(node)
        if report:
            return report.exit_code  # usually 0 -> restart
        return None

    @process_handler
    def handle_gw_exception(self, node):
        """Escalate OMEGATL if the previous GW calculation excepted.

        Called directly as a plain method from inspect_process() above, not
        via the @process_handler dispatch loop (excepted nodes never reach
        it). Decorator kept for introspection/`verdi process report` only.
        """
        if not node.is_excepted:
            return None

        self.ctx.gw_iteration = self.ctx.get('gw_iteration', 0) + 1

        incar = dict(self.ctx.inputs.parameters)
        incar['omegatl'] = 16000
        self.ctx.inputs.parameters = incar

        self.report(f'[G0W0 restart] iter {self.ctx.gw_iteration}: excepted calc {node.pk}; set OMEGATL={incar["omegatl"]}')
        return ProcessHandlerReport()  # restart

    @process_handler(priority=900)
    def handler_unfinished_calc_generic(self, node):
        """Exit-700 (OOM/walltime) resource fallback. Reached via
        super().inspect_process() for any non-excepted failure - including a
        retry of a calc that previously excepted.

        Defers to VaspWorkChain's generic handler for ERROR_DID_NOT_FINISH
        (exit 700) for all of its existing logic (single retry, then abort
        on a second consecutive failure). The only addition: when that
        handler grants the retry and extraresources_fallback_options was
        supplied, swap it in for ctx.inputs.metadata['options'] before the
        retry.
        """
        report = super().handler_unfinished_calc_generic(node)
        if report is not None and report.exit_code.status == 0 and 'extraresources_fallback_options' in self.inputs:
            self.report('Calculation did not finish (exit 700) - retrying with extraresources_fallback_options.')
            self.ctx.inputs.metadata['options'] = self.inputs.extraresources_fallback_options.get_dict()
        return report
