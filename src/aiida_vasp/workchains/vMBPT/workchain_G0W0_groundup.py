    # pylint: disable=too-many-arguments
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Int, Float, Str, Dict, Bool, RemoteData, ArrayData, BandsData, KpointsData
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import WorkChain, ToContext, while_

from aiida_vasp.utils.workchains import prepare_process_inputs
from aiida_vasp.workchains.vMBPT.utils_helpers_extrapolation import input_magnetic_moment_tomagmom
from aiida_vasp.workchains.vMBPT.workchain_atomic_G0W0 import VaspAtomicG0W0WorkChain


class PhaseStatus(Enum):
    """Per-phase execution status. 
    - self.ctx.state.phase_statuses holds one of these per phase, index-aligned with __get_all_phases() 
      (phase_statuses[i] describes phases[i]). 
    - Exactly one phase is ever active (i.e. anything other than NOT_STARTED/SKIPPED/COMPLETED/FAILED) at a time:
      the one with index self.ctx.state.phase_idx inside self.ctx.state.phase_statuses
    - Lifecycle for a single phase:
        NOT_STARTED -> SKIPPED                                            (check_skip decided to skip it)
        NOT_STARTED -> PENDING -> PREPAREDINPUTS -> RUNNING -> COMPLETED  (ran successfully)
                                                             -> PENDING    (failed, retrying)
                                                             -> FAILED     (failed, retries exhausted)
    SKIPPED/COMPLETED/FAILED are terminal for that phase - once set, that entry never changes again. Exactly
    one phase is ever "in flight" ("""
    NOT_STARTED    = 'NOT_STARTED'
    PENDING        = 'PENDING'
    PREPAREDINPUTS = 'PREPAREDINPUTS'
    RUNNING        = 'RUNNING'
    COMPLETED      = 'COMPLETED'
    SKIPPED        = 'SKIPPED'
    FAILED         = 'FAILED'

class WorkflowTerminalState(Enum):
    """Final state of the whole phase loop once it stops iterating (self.ctx.state.terminal)."""
    COMPLETE = 'COMPLETE'
    FAILED   = 'FAILED'


@dataclass(frozen=True)
class WorkflowPhase:
    """Describes one phase in __get_all_phases() generically.
    - Deliberately never stored anywhere under self.ctx - it embeds live callables (build_inputs/skip_if)
      and a process class, neither of which is data any serializer can represent -> would causes SerializationErrors.
      Solution is to rebuild (deterministically, from self.inputs) on every call instead of being persisted.
    - Its callables are bound to the owning workchain instance by _build_phases() (see there), so a
      WorkflowPhase tuple belongs to the instance that built it and must not be applied to another one -
      one more reason it is rebuilt per access rather than cached/persisted anywhere.
    - Components:
      -- key: phase identifier name.
      -- build_inputs: callable(restart_folder) -> AttributeDict;
                        Constructs the AttributeDict used as inputs for the phase's workchain. Already
                        bound to the workchain instance by _build_phases() - it is that phase's own bound
                        self._prepare_inputs_<phase> method (one per phase, all with this same signature),
                        so the engine supplies only the restart_folder.
                        That restart_folder is resolved by the FSM engine (prepare_step, via
                        __get_restart_remote_folder_for_current_phase()) and handed in, so a phase never has
                        to reach back for its own description to find out where it restarts from. Used as:
                        self.ctx.state.inputs_finalized = phase.build_inputs(restart_folder)
      -- process_class: the WorkChain/CalcJob class submitted for this phase. Used as:
                        running_wc = self.submit(phase.process_class, **self.ctx.state.inputs_finalized)
      -- key_to_use_as_restartData: Optional[str], key of the phase this one restarts from ('1DFTgr' for '2DFTvo',
                        '2DFTvo' for '3G0W0'), or None when the phase has no predecessor at all ('1DFTgr').
                        This is the ONLY place a cross-phase dependency is declared, and it is plain data
                        rather than a callable: __get_restart_remote_folder_for_current_phase() resolves it
                        by walking self.ctx.state.nodes for that phase's last node (via
                        __get_last_node_by_key) and reading its outputs.remote_folder - no intermediate
                        captured/cached state at all, and nothing declared on the predecessor's side.
      -- allow_fallback_to_startingRemoteData: bool, whether this phase may fall back to the externally-supplied
                        self.inputs.ns_reference.starting_RemoteData when the predecessor yielded no usable
                        node (never ran, was skipped, or finished without a remote_folder output). True for
                        the DFT phases: '1DFTgr' has nothing else to start from, and '2DFTvo' must still be
                        able to run against an externally-supplied ground state when '1DFTgr' is skipped.
                        False for '3G0W0' - a DFT ground-state folder is not a valid G0W0 restart point (no
                        WAVEDER), so substituting it would only turn a clear "2DFTvo produced nothing" into
                        a confusing downstream VASP failure. When the fallback is off (or unavailable) the
                        resolution returns None, which validate_step's required_files check already turns
                        into a clean, reported exit code - no restart-folder-specific error handling needed.
      -- required_files: tuple of str, names of files that must exist in the restart_folder for this phase to run.
        --              Es: ('WAVECAR', 'WAVEDER') for the G0W0 phase, ('WAVECAR',) for the 2DFTvo phase.
                        Empty tuple () means "nothing to check" and validate_step skips the check entirely.
                        A phase does NOT name its own exit code: every failure of this check returns the single
                        generic ERROR_MISSING_RESTART_FILES, whose templated message is filled in at return time
                        with this phase's key and the actually-missing files (see validate_step). That keeps the
                        per-case detail in the persisted node.exit_message while leaving _build_phases() free to
                        add phases without also having to declare an exit code in define().
      -- skip_if: callable() -> bool; True means this phase is skipped. Bound to the workchain instance by
                        _build_phases() the same way build_inputs is (a closure over the flags it reads),
                        so it too takes no arguments. Consulted ONLY by check_skip - see there.
    """
    key: str
    process_class:      Optional[type] = None
    # Plain lambdas, not staticmethod(...): a dataclass writes a field default into the INSTANCE dict, and
    # instance-dict lookups never go through the descriptor protocol, so `phase.build_inputs` returns the
    # raw function either way - the staticmethod wrapper these used to carry protected against nothing.
    build_inputs:                         Callable = lambda restart_folder: AttributeDict()
    key_to_use_as_restartData:            Optional[str] = None
    allow_fallback_to_startingRemoteData: bool = True
    required_files:                       Tuple[str, ...] = ()
    skip_if:                              Callable = lambda: False



class VaspG0W0GroundUpWorkChain(WorkChain):
    """Run a full DFT(gr) -> DFT(vo) -> G0W0 pipeline from scratch (or from a supplied DFT ground-state restart).

    Purpose of workchain:
    1) Orchestrate three phases as VaspWorkChain-shaped child processes: 
        '1DFTgr' (DFT ground state, optional),
        '2DFTvo' (single-iteration DFT with all virtual orbitals, producing the WAVECAR/WAVEDER the G0W0 phase restarts from), 
        '3G0W0'  (Run the G0W0).
        
    3) Assemble the DFT/G0W0 band structures into gaps/QP-correction outputs (gaps, gaps_QPc, bands_QPc).

    Architecture: this is a finite-state-machine (FSM). It is made of two deliberately separate halves:
    - Static description of "what phases exist" (list of WorkflowPhase objects , which is constructed by _build_phases() 
      and accessed via __get_all_phases()). 
      -- This is defined once per class, and never changes during a run. 
    - The actual dynamic state of the run (self.ctx.state).
    - This is only an orchestrator; it submits vasp.vasp for DFT calcs , and VaspAtomicG0W0WorkChain for G0W0 ; this is why
      it's a subclass of Workchain and not VaspWorkchain.
    
    Note: difference between vasp.vasp and VaspAtomicG0W0WorkChain calls:
        - for the '1DFTgr'/'2DFTvo' the full input.parameters AttributeDict is constructed inside _prepare_inputs_DFTgr/_prepare_inputs_DFTvo
          (both on top of the shared _build_inputs_DFT) and then passed to VaspWorkChain.
        - For '3G0W0' INCAR construction and restart-folder validation for that phase are entirely delegated to VaspAtomicG0W0WorkChain, 
          and _prepare_inputs_G0W0 only builds the inputs dict for that child.

    Main components:
    * WorkflowPhase - the static part: "what phases exist and what each one does". 
    - One frozen (immutable) WorkflowPhase per phase, defining the process_class and the Callable objects the FSM engine uses
      (those callables are bound to this workchain instance when _build_phases() constructs them - see there).
      This part is descriptive, and is never updated/mutated during a run (not a run state).
    - It is never stored on self.ctx (it embeds live callables and a process class, which can't be serialized for checkpoint,
      and that would cause SerializationError whenever AiiDA engine constructs a checkpoint).
    - Methods to access/construct the WorkflowPhase tuple:
        -- Constructed by _build_phases(): 
           returns the ordered tuple ('1DFTgr', '2DFTvo', '3G0W0') for this class. 
           A subclass wanting a different pipeline overrides only this one method.
        -- Accessed by __get_all_phases(): 
           Every step should use this method to read the phase list (it just calls _build_phases()). 

    * self.ctx.state - the dynamic part: which phase is active, its progress, and the handoff data between phases. Fields: 
    - phase_idx (int): 
        -- Index which defines the current active phase among self.__get_all_phases() 
        -- Exactly one phase is ever active at any moment; every other phases entry is either not yet reached (NOT_STARTED) 
           or already resolved (SKIPPED/COMPLETED/FAILED). 
           phases[phase_idx] (via __get_all_phases()) is the active WorkflowPhase, and phase_statuses[phase_idx] is its current PhaseStatus         
        -- phase_idx==-1 means no phase has been started yet (the FSM is still in initialize()) ; during workflow run, phase_idx is always 
           in [0, len(__get_all_phases())-1] (and  terminal is None).
        -- Related methods:
           update_step_phaseidx : updates the current phase index based on the workflow's progress.
           __get_current_phase()/__get_status_current_phase() : return the WorkflowPhase and PhaseStatus for the current phase_idx 
             (does not return directly phase_idx)
           __set_status_current_phase() : sets the PhaseStatus for the current phase_idx.
            
    - phase_statuses (List[PhaseStatus]): 
        -- index-aligned with phases - phase_statuses[i] describes phases[i].
    - nodes (List[List[Node]]): 
        -- index-aligned with phases - nodes[i] accumulates one entry per submission attempt for phase i (empty until phase i is first submitted).
    - retries (List[int]): index-aligned with phases - retry counters, one per phase.
    - terminal (Optional[WorkflowTerminalState]): 
        None while the FSM is still iterating (see should_wc_continue()); 
        set to COMPLETE or FAILED once the while_() loop should stop.
    - inputs_finalized (Optional[AttributeDict]): 
        contains inputs for whichever phase is currently being submitted, constructed inside prepare_step and consumed in execute_step.

            """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(WorkflowFactory('vasp.vasp'), exclude=('parameters', 'settings', 'restart_folder'))
        spec.expose_inputs(VaspAtomicG0W0WorkChain,      exclude=('parameters', 'settings', 'restart_folder'))

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
        spec.output('gaps',      valid_type=Dict,      required=True,  help='Direct and indirect gaps for the DFT and G0W0 nodes.')
        spec.output('gaps_QPc',  valid_type=Dict,      required=False, help='QP HOMO/LUMO correction at direct gap kpt.')
        spec.output('bands_QPc', valid_type=BandsData, required=False)
        # Output structure:
        #   gaps      = { 'DFT': {spin_label: {Dir,Ind,Gam}}, 'G0W0': {spin_label: {Dir,Ind,Gam}} }
        #   gaps_QPc  = {spin_label: {Dir,Ind,Gam}}   (only if G0W0 ran - QP correction = G0W0 - DFT)
        #   bands_QPc = BandsData (containing E_QP(k,n) = E_G0W0(k,n) - E_DFT(k,n)  )
        #   spin_label in {'spinUp'} (non spin-polarized) or {'spinUp','spinDw'} (spin-polarized)

        spec.exit_code(401, 'REACHED_MAXIMUM_TRY_NUMBER',           message='The workflow reached the maximum number of tries.')
        # One generic code for every phase's restart-folder check (same convention as VaspAtomicG0W0WorkChain /
        # VaspAtomicBSEWorkChain, which both use 406 ERROR_MISSING_RESTART_FILES). The phase and the actually-missing
        # files are interpolated into the message via ExitCode.format() at return time - see validate_step - so the
        # detail lands in the persisted, queryable node.exit_message instead of only in the report log.
        spec.exit_code(406, 'ERROR_MISSING_RESTART_FILES',
                       message='Phase {phase}: restart folder is missing required file(s): {missing}.')

        spec.outline(
            cls.initialize,
            while_(cls.should_wc_continue)(
                cls.update_step_phaseidx,
                cls.check_skip,
                cls.validate_step,
                cls.prepare_step,
                cls.correct_previous_errors,
                cls.execute_step,
            ),
            cls.elaborate_results,
        )

    def _build_phases(self) -> Tuple[WorkflowPhase, ...]:
        """Build the ordered phase list - DFT/GW only. Called on every access (see `__get_all_phases()`
        above); a subclass wanting a different pipeline overrides this method instead of initialize().

        Every callable stored on a WorkflowPhase is bound to THIS instance here, at build time, rather than
        taking `self` as a parameter the FSM engine passes back in later:
        - build_inputs is simply that phase's own bound `self._prepare_inputs_<phase>` method - there is one
          per phase and they all have the exact same (restart_folder) signature, so nothing has to be
          partially applied or wrapped in a lambda to reach it;
        - skip_if is a closure over the `self` of this very method.
        So the engine only ever supplies genuinely per-call data - `phase.build_inputs(restart_folder)`,
        `phase.skip_if()` - and can no longer hand a phase the wrong instance. The lambdas these replaced
        declared their own `self` parameter, which shadowed this method's `self` and read as circular
        (`lambda self, restart_folder: self._prepare_inputs_G0W0(restart_folder=restart_folder)`).

        Note that the phases returned are therefore instance-bound: `_build_phases()` must be called on a
        real instance (or a stub carrying the _prepare_inputs_* attributes), never on the class with a dummy
        `self`, since the bound methods are resolved eagerly, right here."""
        return (
            WorkflowPhase(
                key='1DFTgr',
                process_class=WorkflowFactory('vasp.vasp'),
                key_to_use_as_restartData=None,
                required_files=(),
                # No key_to_use_as_restartData: this is the entry point, so it restarts from
                # self.inputs.ns_reference.starting_RemoteData if one was supplied, and from nothing
                # (restart_folder=None) if not.
                build_inputs=self._prepare_inputs_DFTgr,
                skip_if=lambda: not self.inputs.ns_option.run_1DFTgr.value,
            ),
            WorkflowPhase(
                key='2DFTvo',
                process_class=WorkflowFactory('vasp.vasp'),
                key_to_use_as_restartData='1DFTgr',
                required_files=('WAVECAR',),
                build_inputs=self._prepare_inputs_DFTvo,
                skip_if=lambda: not self.inputs.ns_option.run_2DFTvo_3G0W0.value,
            ),
            WorkflowPhase(
                key='3G0W0',
                process_class=VaspAtomicG0W0WorkChain,
                key_to_use_as_restartData='2DFTvo',
                allow_fallback_to_startingRemoteData=False,  # a DFT ground state has no WAVEDER - see WorkflowPhase
                required_files=('WAVECAR', 'WAVEDER'),
                build_inputs=self._prepare_inputs_G0W0,
                skip_if=lambda: not self.inputs.ns_option.run_2DFTvo_3G0W0.value,
            ),
        )
        # NOTE: the '3G0W0' required_files check is redundant with VaspAtomicG0W0WorkChain's own internal
        # restart-folder validation, but harmless to keep as defense-in-depth (fails fast in validate_step, before
        # a submission is even attempted).

    def __get_all_phases(self) -> Tuple[WorkflowPhase, ...]:
        """Return the ordered tuple of WorkflowPhase objects for this instance.
        Always calls _build_phases() on every call, rebuilding fresh rather than caching
        (This is done to avoid serialization issues with WorkflowPhase objects, which contain live callables and process 
        classes that cannot be serialized; trying to save inside self.ctx would cause SerializationErrors on checkpoint saving)."""
        return self._build_phases()



    ##[HELPER FUNCTIONS for the FSM engine - "current phase"/"current status" bookkeeping
    def __get_current_phase(self) -> WorkflowPhase:
        """Return the WorkflowPhase object at index ctx.state.phase_idx.
        as phase_idx must index a valid phase inside __get_all_phases(), the function raises a RuntimeError if phase_idx is out of bounds."""
        phasesList = self.__get_all_phases()
        if self.ctx.state.phase_idx < 0 or self.ctx.state.phase_idx >= len(phasesList):
            raise RuntimeError(f"Invalid phase_idx={self.ctx.state.phase_idx} for phases {phasesList} of length {len(phasesList)}")
        return self.__get_all_phases()[self.ctx.state.phase_idx]

    def __get_status_current_phase(self) -> PhaseStatus:
        """Return the PhaseStatus for the WorkflowPhase at index ctx.state.phase_idx."""
        return self.ctx.state.phase_statuses[self.ctx.state.phase_idx]

    def __set_status_current_phase(self, value: PhaseStatus) -> None:
        """Set the PhaseStatus for the WorkflowPhase at index ctx.state.phase_idx."""
        self.ctx.state.phase_statuses[self.ctx.state.phase_idx] = value

    def __record_node(self, node) -> None:
        """Append a freshly-submitted node for the active phase (called once per submission attempt).
        - Note that this is not a "phase completion" record - the node may still be running or even fail; 
          the PhaseStatus for the active phase's node finished state  is updated separately by update_step_phaseidx.
        - for each phase, self.ctx.state.nodes[phase_idx] is a list that accumulates one entry per submission attempt."""
        self.ctx.state.nodes[self.ctx.state.phase_idx].append(node)

    def __get_last_node_by_key(self, key: str):
        """Return the last submitted node for the phase with this key, 
        or None if it was never submitted (still NOT_STARTED, or SKIPPED) - or if no phase has this key at all."""
        #The function walks through the phases in order (phases retrieved from __get_all_phases()).
        #For each phase (of index idx inside __get_all_phases()) check if the phase.key matches the requested key
        #If match is positive  it retrieves the list of nodes for that phase (self.ctx.state.nodes[idx]) 
        #and returns the last node in that list (if any), or None if the list is empty.
        for idx, phase in enumerate(self.__get_all_phases()):
            if phase.key == key:
                lst = self.ctx.state.nodes[idx]
                return lst[-1] if lst else None
        return None

    def __get_restart_remote_folder_for_current_phase(self):
        """Get the RemoteData folder to use as the restart_folder for the active phase.

        The single resolution point for "where does this phase restart from" - both validate_step (to run the
        required_files check) and prepare_step (to hand the value to build_inputs) call this, and neither
        needs to know anything about phase dependencies: those are declared as plain data on the active
        WorkflowPhase (key_to_use_as_restartData / allow_fallback_to_startingRemoteData, see that class's docstring).

        The RemoteData is determined as:
        - if the active phase names a predecessor (key_to_use_as_restartData), look up that phase's last submitted node
          via __get_last_node_by_key(key_to_use_as_restartData) and, if it actually produced one, return its
          outputs.remote_folder. Both halves must be checked: __get_last_node_by_key returns None when the
          predecessor never ran or was skipped, and even a node that DID run is not guaranteed to carry a
          remote_folder output (e.g. it excepted, or was killed before the calcjob attached its outputs) -
          hence the explicit `'remote_folder' in node.outputs` membership test rather than an unguarded
          attribute access, which would otherwise raise instead of degrading to the fallback below.
        - otherwise fall back to the externally-supplied self.inputs.ns_reference.starting_RemoteData, but
          ONLY if the active phase allows it (allow_fallback_to_startingRemoteData) - '3G0W0' sets this False
          because a DFT ground-state folder is not a valid G0W0 restart point.
        - if neither is available, return None. No need to raise here: validate_step's required_files check
          already turns a None restart folder into a clean, reported exit code (see
          __validate_remote_has_required_files)."""
        phase = self.__get_current_phase()
        if phase.key_to_use_as_restartData is not None:
            node = self.__get_last_node_by_key(phase.key_to_use_as_restartData)
            if node is not None and 'remote_folder' in node.outputs:
                return node.outputs.remote_folder
        if not phase.allow_fallback_to_startingRemoteData:
            return None
        try:
            return self.inputs.ns_reference.starting_RemoteData
        except Exception:
            return None

    def __validate_remote_has_required_files(self, remote, required, label: str):
        """Validate that RemoteData (pointed by argument remote) exists and contains the required files.

        Returns the list of MISSING file names - so an empty list [] means the check passed. It returns the
        missing names rather than a bool because the caller (validate_step) interpolates them straight into
        the ERROR_MISSING_RESTART_FILES message, so this is the only place that has to know which ones failed.

        Always reports REQUIRED/PRESENT/MISSING via self.report, on success as well as on failure: a passing
        check used to be completely silent, which made "did it even look?" unanswerable from the report log.
        Only the required names are listed, not the whole remote listing (a VASP run folder is large and would
        drown the log) - the full listing is dumped only when something is actually missing, where it is what
        one needs to see.

        The two "cannot even look" cases (no restart folder at all, or an unreachable/unlistable one) report
        their own reason and then return the full required list: from validate_step's point of view nothing is
        present, which is exactly the right outcome, and the preceding report line says why."""
        required = list(required)
        if remote is None:
            self.report(f"[validate_step] {label}: restart folder is None -> "
                        f"none of the required files {required} can be present")
            return required
        try:
            files = set(remote.listdir())
        except Exception as exc:
            self.report(f"[validate_step] {label}: could not list folder contents ({exc}) -> "
                        f"treating all required files {required} as missing")
            return required
        present = [fname for fname in required if fname in files]
        missing = [fname for fname in required if fname not in files]
        self.report(f"[validate_step] {label}: REQUIRED={required} PRESENT={present} MISSING={missing}")
        if missing:
            self.report(f"[validate_step] {label}: full remote folder listing: {sorted(files)}")
        return missing

    ##[HELPER FUNCTIONS for update_step_phaseidx and validate_step]
    def __handle_failure_with_retry(self):
        """Handle a failed active phase with retry logic. Returns an ExitCode if retries are exhausted,
        otherwise None (and resets the active phase's status back to PENDING for the same phase_idx)."""
        state = self.ctx.state
        phase = self.__get_current_phase()
        max_iter = self.inputs.ns_option.maximum_iterations.value
        retries = state.retries[state.phase_idx]
        if retries < max_iter:
            self.report(f"[update_step_phaseidx] {phase.key} failed, retrying (attempt {retries+1}/{max_iter})")
            self.__set_status_current_phase(PhaseStatus.PENDING)
            return None

        self.report(f"[update_step_phaseidx] {phase.key} failed AND maximum retries reached -> ABORT")
        self.__set_status_current_phase(PhaseStatus.FAILED)
        state.terminal = WorkflowTerminalState.FAILED
        return self.exit_codes.REACHED_MAXIMUM_TRY_NUMBER


    def initialize(self):
        """Initialize workflow context: self.ctx.state (a plain AttributeDict - see the comment above
        WorkflowPhase for why it isn't a dataclass), sized to __get_all_phases() (see that method above),
        plus any externally-supplied starting RemoteData. No submission or input-preparation logic here.

        A subclass wanting a different pipeline overrides _build_phases() (or `__get_all_phases()` more
        directly) instead of this method, to change "what runs" without touching any other part of the FSM
        engine. See the class docstring's Design rationale [2] for the intended future use (an
        ML/interpolation surrogate overriding just the '2DFTvo'/'3G0W0' entries)."""
        #[1] The whole FSM state machine (see the comment above WorkflowPhase for the full shape/rationale).
        # No starting_RemoteData field here - it's immutable process input
        # (self.inputs.ns_reference.starting_RemoteData), not run state, so
        # __get_restart_remote_folder_for_current_phase() reads it straight from self.inputs instead of a copy
        # being kept in self.ctx.state. Likewise no restart-folder field: each phase's restart point is
        # resolved on demand from ctx.state.nodes + its own declared key_to_use_as_restartData (see WorkflowPhase).
        phases = self.__get_all_phases()
        self.ctx.state = AttributeDict({
            'phase_statuses': [PhaseStatus.NOT_STARTED] * len(phases),
            'nodes':     [[] for _ in phases],
            'retries':   [0] * len(phases),
            'phase_idx': -1,
            'terminal':  None,
            'inputs_finalized': None,
        })

        #[2] Spin polarization: magnetic_moment_onsite is a top-level exposed port (from VaspAtomicG0W0WorkChain's
        # namespace).
        self.ctx.is_spinpol  = ('magnetic_moment_onsite' in self.inputs)
        self.ctx.spin_labels = ('spinUp', 'spinDw') if self.ctx.is_spinpol else ('spinUp',)

    def should_wc_continue(self) -> bool:
        """Determine whether the workflow should continue looping."""
        return self.ctx.state.terminal is None

    def check_skip(self):
        """The only place any phase's skip_if is ever consulted. Runs every while_() iteration, immediately
        after update_step_phaseidx: while the phase at the current phase_idx is still NOT_STARTED and its
        skip_if() is True, marks it SKIPPED and advances phase_idx - repeating until it lands on a phase
        that should actually run, or runs out of phases (ctx.state.terminal = COMPLETE). No separate
        skip-time hook needed: __get_restart_remote_folder_for_current_phase() already falls back to
        starting_RemoteData whenever the active phase's named predecessor has no recorded node (and the phase
        allows that fallback), which is exactly what a skipped phase leaves behind. Marks whatever phase it lands on PENDING, so
        validate_step onward can assume they are only ever looking at a confirmed non-skipped, PENDING phase."""
        state = self.ctx.state
        if state.terminal is not None:
            return  # already decided to stop
        if state.phase_idx >= len(self.__get_all_phases()):
            # update_step_phaseidx just advanced phase_idx past the last phase (that phase finished
            # successfully) - nothing left to skip-check; __get_status_current_phase() is a
            # phase_statuses[phase_idx] lookup and would IndexError if evaluated below with phase_idx out of range.
            state.terminal = WorkflowTerminalState.COMPLETE
            return
        if self.__get_status_current_phase() != PhaseStatus.NOT_STARTED:
            return  # this phase's skip decision was already made - normal for most iterations
        phases = self.__get_all_phases()
        while state.phase_idx < len(phases) and phases[state.phase_idx].skip_if():
            self.__set_status_current_phase(PhaseStatus.SKIPPED)
            self.report(f"[<{self.inputs.ns_option.calculation_label.value}> check_skip] {phases[state.phase_idx].key} skipped")
            state.phase_idx += 1
        if state.phase_idx >= len(phases):
            state.terminal = WorkflowTerminalState.COMPLETE
        else:
            self.__set_status_current_phase(PhaseStatus.PENDING)

    def update_step_phaseidx(self):
        """(renamed from update_state) React to the active phase's submitted child node, if it is RUNNING:
        still running (no-op) / finished successfully (record COMPLETED, advance phase_idx by 1) / finished
        with a failure (retry, or FAILED if retries are exhausted). Performs no submissions and no INCAR/input
        preparation, and contains NO skip logic at all - see check_skip, which runs immediately after this in
        the same while_() iteration and is the only place any phase's skip_if is ever consulted. Generic over
        __get_all_phases()."""
        state = self.ctx.state

        #[1] Not yet started -> point phase_idx at the first phase; check_skip (next step) decides whether it
        # is actually runnable.
        if state.phase_idx == -1:
            state.phase_idx = 0
            self.report(f"[<{self.inputs.ns_option.calculation_label.value}> update_step_phaseidx] starting phase loop over {[p.key for p in self.__get_all_phases()]}")
            return

        #[2] Not RUNNING -> nothing submitted for the active phase yet this iteration, nothing to react to
        if self.__get_status_current_phase() != PhaseStatus.RUNNING:
            return

        #[3] RUNNING -> DONE (advance) / retry / FAIL
        phase = self.__get_current_phase()
        node  = self.__get_last_node_by_key(phase.key)
        if node is None or not node.is_finished:
            return
        if node.is_finished_ok:
            self.report(f"[<{self.inputs.ns_option.calculation_label.value}> update_step_phaseidx] {phase.key} finished successfully")
            self.__set_status_current_phase(PhaseStatus.COMPLETED)
            state.phase_idx += 1
        else:
            return self.__handle_failure_with_retry()
        return

    def validate_step(self):
        """Validate prerequisites for the next PENDING step. Only performs checks (and, on failure, sets
        terminal=FAILED and marks the active phase FAILED); does NOT modify any other state. Generic over
        __get_all_phases().

        update_step_phaseidx/check_skip always run immediately before this in the same while_() iteration, and
        either leave ctx.state.terminal set (workflow already decided to stop - e.g. every remaining phase got
        skipped) or have already put the active phase into PENDING - so if terminal is still None, anything
        other than PENDING here means that invariant broke (e.g. a subclass overriding one of those steps
        incorrectly): raise rather than silently doing nothing, since that would otherwise hang the workflow."""
        state = self.ctx.state
        if state.terminal is not None:
            return  # a previous step this same iteration already decided to stop
        if self.__get_status_current_phase() != PhaseStatus.PENDING:
            raise RuntimeError(
                f"validate_step reached with phase_idx={state.phase_idx}, "
                f"current_status={self.__get_status_current_phase()} "
                "while ctx.state.terminal is still None - expected PENDING. This should never happen: "
                "update_step_phaseidx/check_skip run immediately before this step and must have already "
                "established it.")
        phase = self.__get_current_phase()
        if not phase.required_files:
            return
        missing = self.__validate_remote_has_required_files(
            remote=self.__get_restart_remote_folder_for_current_phase(),
            required=list(phase.required_files),
            label=f'{phase.key} restart',
        )
        if missing:
            self.__set_status_current_phase(PhaseStatus.FAILED)
            state.terminal = WorkflowTerminalState.FAILED
            # A single exit code for every phase; ExitCode.format() clones it with the template message filled
            # in, so which phase and which files failed is persisted on the node, not just logged above.
            return self.exit_codes.ERROR_MISSING_RESTART_FILES.format(
                phase=phase.key, missing=', '.join(missing))

    def correct_previous_errors(self):
        """Attempt to correct errors from a previous failed calculation before resubmitting. Currently a NO-OP;
        intended to be extended in the future. MUST NOT submit calculations.

        This is a workchain-level, cross-phase complement to each child process's own `@process_handler`-based
        error handling (e.g. a VaspWorkChain's BaseRestartWorkChain handlers) - those inspect and react to a
        single failed child node from the inside; this method would instead sit here, between prepare_step and
        execute_step, with the ability to patch ctx.state.inputs_finalized (built by prepare_step, for the
        phase at ctx.state.phase_idx) before execute_step submits it - e.g. reacting to a pattern only visible
        across retries/phases, which no single child process handler would see on its own."""
        return

    def prepare_step(self):
        """Build inputs for the active phase, if it is PENDING. Sets ctx.state.inputs_finalized and advances
        current_status to PREPAREDINPUTS; does NOT submit. Generic over __get_all_phases().

        validate_step always runs immediately before this in the same while_() iteration; see its docstring
        for why anything other than PENDING here (while ctx.state.terminal is still None) is a broken
        invariant, not a normal condition to silently return on."""
        state = self.ctx.state
        if state.terminal is not None:
            return  # a previous step this same iteration already decided to stop
        if self.__get_status_current_phase() != PhaseStatus.PENDING:
            raise RuntimeError(
                f"prepare_step reached with phase_idx={state.phase_idx}, "
                f"current_status={self.__get_status_current_phase()} "
                "while ctx.state.terminal is still None - expected PENDING. This should never happen.")
        phase = self.__get_current_phase()  # ctx.state.phase_idx is the sole source of truth for "which phase is active"
        state.inputs_finalized = phase.build_inputs(self.__get_restart_remote_folder_for_current_phase())
        self.__set_status_current_phase(PhaseStatus.PREPAREDINPUTS)

    def execute_step(self):
        """Submit the active phase's calculation, if its inputs were just built (PREPAREDINPUTS) this
        iteration. Generic over __get_all_phases().

        prepare_step always runs immediately before this in the same while_() iteration; see validate_step's
        docstring for why anything other than PREPAREDINPUTS here (while ctx.state.terminal is still None) is
        a broken invariant, not a normal condition to silently return on."""
        state = self.ctx.state
        if state.terminal is not None:
            return  # a previous step this same iteration already decided to stop
        if self.__get_status_current_phase() != PhaseStatus.PREPAREDINPUTS:
            raise RuntimeError(
                f"execute_step reached with phase_idx={state.phase_idx}, "
                f"current_status={self.__get_status_current_phase()} "
                "while ctx.state.terminal is still None - expected PREPAREDINPUTS. This should never happen.")
        phase = self.__get_current_phase()  # ctx.state.phase_idx is the sole source of truth for "which phase is active"

        #[1] Submit
        running_wc = self.submit(phase.process_class, **state.inputs_finalized)

        #[2] Bump retry counter (this submission attempt) and record the node
        state.retries[state.phase_idx] += 1
        tmp_attempt_num = state.retries[state.phase_idx]
        self.__record_node(running_wc)

        #[3] Update execution state
        self.__set_status_current_phase(PhaseStatus.RUNNING)

        #[4] Log
        self.__report_compact_submission(running_wc, tmp_attempt_num, phase.key)

        #[5] Register dependency for engine
        return ToContext(**{f'calc_{phase.key}': running_wc})

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
        #[1] Determine last successful DFT and G0W0 nodes. Generic: reads ctx.state.nodes presence (via
        # last_node_by_key, which returns None when a phase was never submitted) rather than re-deriving "did
        # this phase run" from the ns_option.* input flags - this is what lets a subclass with a differently-
        # skipped '2DFTvo' (e.g. an ML surrogate, see the class docstring) work here with zero override: the
        # later of '1DFTgr'/'2DFTvo' that actually produced a node wins as the DFT baseline, and '3G0W0' is
        # looked up unconditionally regardless of *why* an earlier phase was or wasn't skipped.
        last_node_DFT = None
        for key in ('1DFTgr', '2DFTvo'):
            node = self.__get_last_node_by_key(key)
            if node is not None:
                last_node_DFT = node
        last_node_G0W0 = self.__get_last_node_by_key('3G0W0')

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

        #[3] spin handling - magnetic_moment_onsite is a top-level port here (see initialize()'s comment [4]).
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
    def _build_inputs_DFT(self, restart_folder):
        """Build the input set shared by BOTH DFT phases ('1DFTgr' and '2DFTvo').

        Deliberately stops one step short of a submittable input set: the returned `inputs.parameters` is
        still a PLAIN dict (not yet a Dict node) and prepare_process_inputs() has NOT been run on it yet.
        That is exactly what lets _prepare_inputs_DFTvo patch its handful of INCAR keys on top without
        having to unwrap and re-wrap a Dict node - prepare_process_inputs() converts a raw 'parameters'
        dict itself (it is in that function's own `no_dict` list, then wrapped at the end), so each phase's
        builder below runs it exactly once, as its last step.

        Reads nothing from ctx.state: which phase is active is irrelevant to building this input set (see
        the note on exposed_inputs below), so both callers can build their inputs at any FSM position.

        `restart_folder` is saved directly into inputs.restart_folder without checking its validity here (if
        it's None, None is saved). Assumption: all checks MUST be performed earlier (validate_step)."""
        #[1] Base. The class handed to exposed_inputs() is only a lookup key into what define() exposed
        # (spec-time data), so it is named here directly rather than read back off the active phase via
        # __get_current_phase().process_class: that would make this builder depend on ctx.state.phase_idx
        # pointing at its own phase, and a wrong class fails SILENTLY - aiida's spec._exposed_inputs is a
        # defaultdict, so an unexposed class yields [] and this update() would quietly add nothing at all.
        # Both DFT phases declare exactly this process_class in _build_phases; _prepare_inputs_G0W0 below
        # names VaspAtomicG0W0WorkChain the same way.
        inputs = AttributeDict()
        inputs.update(self.exposed_inputs(WorkflowFactory('vasp.vasp')))
        inputs.clean_workdir = Bool(False)
        inputs.restart_folder = restart_folder

        #[2] Parser settings
        inputs.settings = AttributeDict({'parser_settings': {'include_node': ['bands', 'kpoints', 'structure', 'NGarray', 'maximum_number_pw']}})

        #[3] INCAR - the part common to both DFT phases ('2DFTvo' overrides nbands/algo/nelm on top of it,
        # see _prepare_inputs_DFTvo below).
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

        inputs.parameters = incar  # PLAIN dict on purpose - each caller wraps it via prepare_process_inputs
        return inputs

    def _prepare_inputs_DFTgr(self, restart_folder):
        """Prepare inputs for the '1DFTgr' phase (DFT ground state): the shared DFT input set, unchanged."""
        return prepare_process_inputs(self._build_inputs_DFT(restart_folder),
                                      namespaces=['calc', 'dynamics', 'verify'])

    def _prepare_inputs_DFTvo(self, restart_folder):
        """Prepare inputs for the '2DFTvo' phase: the shared DFT input set (see _build_inputs_DFT above) plus
        the three INCAR keys that turn it into the single-iteration, all-virtual-orbitals run whose
        WAVECAR/WAVEDER the '3G0W0' phase restarts from - nbands (include all the unoccupied bands),
        algo='Exact' (exact diagonalisation) and nelm=1 (a single electronic iteration: the ground-state
        wavefunctions come in through restart_folder, they are not re-converged here)."""
        inputs = self._build_inputs_DFT(restart_folder)
        if ('nbands' in self.inputs): inputs.parameters['incar']['nbands'] = self.inputs.nbands.value
        inputs.parameters['incar']['algo'] = "Exact"
        inputs.parameters['incar']['nelm'] = 1
        return prepare_process_inputs(inputs, namespaces=['calc', 'dynamics', 'verify'])

    def _prepare_inputs_G0W0(self, restart_folder):
        """Prepare inputs for the G0W0 phase by delegating entirely to VaspAtomicG0W0WorkChain's own input
        namespace - unlike the _prepare_inputs_DFT* pair above, NO local INCAR construction happens here: the atomic
        wrapper builds its own INCAR internally, inside its own init_inputs()/__build_parameters(). No
        `inputs.settings` is set either - `settings` was excluded from this file's expose_inputs(
        VaspAtomicG0W0WorkChain, ...) call in define(), so VaspAtomicG0W0WorkChain.init_inputs() sees an
        empty/absent settings and applies its own default (bands/kpoints/structure inclusion + WAVEDER copy),
        which is exactly what elaborate_results() above needs."""
        inputs = AttributeDict()
        inputs.update(self.exposed_inputs(VaspAtomicG0W0WorkChain))
        inputs.clean_workdir = Bool(False)
        inputs.restart_folder = restart_folder
        # 'optimization.npar' is declared by this class (define(), above) into the SAME shared 'optimization'
        # namespace also exposed from VaspAtomicG0W0WorkChain - so self.exposed_inputs(VaspAtomicG0W0WorkChain)
        # picks it up too, even though VaspAtomicG0W0WorkChain's own 'optimization' sub-spec has no 'npar' port
        # (it's a non-dynamic namespace: kpar/lreal/set_PRECFOCK_to_Fast only, see its docstring/define()) and
        # only the 1DFTgr/2DFTvo phases (_build_inputs_DFT) actually read it. Must be stripped here, or
        # submission fails with "Unexpected ports {'npar': ...}, for a non dynamic namespace".
        inputs.optimization = AttributeDict({k: v for k, v in inputs.optimization.items() if k != 'npar'})
        prepared_inputs = prepare_process_inputs(inputs, namespaces=['calc', 'dynamics', 'verify', 'optimization'])
        return prepared_inputs

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

    def __report_compact_submission(self, running_wc, tmp_attempt, phase_key: str):
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
        prolog = (f"[<{label}> execute_step] launching {phase_key} pk={running_wc.pk} "
                  f"(attempt num={tmp_attempt}) -> state updated to={phase_key} {self.__get_status_current_phase().name}")
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
