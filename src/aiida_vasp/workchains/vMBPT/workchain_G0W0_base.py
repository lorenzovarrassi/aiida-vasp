# pylint: disable=too-many-arguments
import numpy as np
from copy import deepcopy
from aiida.common.extendeddicts import AttributeDict
from aiida import orm
from aiida.orm import Code, Int, Float, Str, Dict, Bool , List , RemoteData , ArrayData , BandsData , XyData , KpointsData
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import WorkChain, calcfunction , ToContext , append_ , submit, while_
from aiida_vasp.utils.workchains import prepare_process_inputs
import warnings
from sklearn.linear_model import LinearRegression
from aiida_vasp.utils.workchains import site_magnetization_to_magmom
from enum import Enum, auto

from .utils_helpers_extrapolation import  input_magnetic_moment_tomagmom
from .workchain_wrapper_VaspWorkchain_G0W0 import VaspGWWorkChain

    # General idea:
    #[Loop Iteration 1]
    #  ├─ update_state()
    #  │   - execution_state at entry: INIT
    #  │
    #  │   Initial branching logic:
    #  │   1) If ns_option.run_1DFTgr == True:
    #  │      - Always schedule a DFT ground-state calculation (DFTgr)
    #  │      1.a) If ns_reference.starting_RemoteData is provided:
    #  │           • use it as restart_folder for DFTgr
    #  │           • emit INFO log: "Using starting_RemoteData as restart for DFTgr"
    #  │      1.b)  Else:
    #  │           • run DFTgr from scratch
    #  │           • emit INFO log: "No starting_RemoteData provided; running fresh DFTgr"
    #  │      - Set execution_state → DFTGR_PENDING
    #  │
    #  │   2) Else (ns_option.run_1DFTgr == False):
    #  │      2.a) If ns_reference.starting_RemoteData is provided:
    #  │           • assume it represents a valid DFT ground state
    #  │           • store it as restart_folders.for_2DFTvo
    #  │           • skip DFTgr entirely
    #  │           • If ns_option.run_2DFTvo_3G0W0 == True:
    #  │                 set execution_state → DFTVO_PENDING
    #  │             Else:
    #  │                 set execution_state → COMPLETE
    #  │
    #  │      2.b) Else (run_1DFTgr == False AND starting_RemoteData is NOT provided):
    #  │           • This is an unrecoverable configuration error
    #  │           • No way to obtain a WAVECAR for DFTvo
    #  │           • Set execution_state → FAILED
    #  │           • Return exit_code NO_STARTING_WAVECAR_forDFTvo
    #  │
    #  │   (No submission happens in INIT; only state selection)
    #  │
    #  └─ Loop continues if execution_state ∉ {COMPLETE, FAILED}
    #  [ Assume ns_option.run_1DFTgr == False for simplicity ]
    #  ├─ validate_step()
    #  │   - Validate prerequisites for DFTvo:
    #  │       • restart_folders.for_2DFTvo must exist
    #  │       • restart folder must contain a WAVECAR
    #  │   - If validation fails:
    #  │       • execution_state → FAILED
    #  │       • Return exit_code NO_STARTING_WAVECAR_forDFTvo
    #  ├─ prepare_step()
    #  │   - Build inputs for DFTvo:
    #  │       • INCAR (nbands, optics, exact diagonalization, etc.)
    #  │       • kpoints
    #  │       • settings
    #  │       • restart_folder = restart_folders.for_2DFTvo
    #  │
    #  └─ execute_step()
    #      - Submit DFTvo workchain
    #      - Record submission in state_WC.submitted['2DFTvo']
    #      - Increment retry counter state_WC.retries['2DFTvo']
    #      - Update execution_state → DFTVO_RUNNING
    #      - return ToContext(calc_2DFTvo = wc)
    #
    #[Loop Iteration 2]
    #  ├─ update_state()
    #  │   - execution_state at entry: DFTVO_RUNNING
    #  │   - Inspect last submitted DFTvo workchain:
    #  │
    #  │   1) If DFTvo is NOT finished:
    #  │        • Do nothing
    #  │        • execution_state remains DFTVO_RUNNING
    #  │
    #  │   2) If DFTvo is finished AND is_finished_ok:
    #  │        • Store its remote_folder as restart_folders.for_3G0W0
    #  │        • Update execution_state → DFTVO_DONE
    #  │
    #  │   3) If DFTvo is finished BUT failed:
    #  │        • If retries['2DFTvo'] < ns_option.maximum_iterations:
    #  │              – execution_state → DFTVO_PENDING
    #  │              – (retry DFTvo in next loop)
    #  │          Else:
    #  │              – execution_state → FAILED
    #  │              – Return exit_code REACHED_MAXIMUM_TRY_NUMBER
    #  │
    #  │   After DFTVO_DONE:
    #  │   - If ns_option.run_2DFTvo_3G0W0 == True:
    #  │         execution_state → G0W0_PENDING
    #  │     Else:
    #  │         execution_state → COMPLETE
    #  │
    #  ├─ validate_step()
    #  │   - (only meaningful if execution_state == G0W0_PENDING)
    #  ├─ prepare_step()
    #  └─ execute_step()
    # Note: errors are handled using process_handler inside the various wrapper. 

class state_execution_enum(Enum):
        INIT = auto()
        DFTGR_PENDING = auto()    # Need to run DFTgr
        DFTGR_RUNNING = auto()    # DFTgr submitted, waiting
        DFTGR_DONE    = auto()

        DFTVO_PENDING = auto()    # Need to run DFTvo
        DFTVO_RUNNING = auto()    # DFTvo submitted, waiting
        DFTVO_DONE    = auto()

        G0W0_PENDING  = auto()    # Need to run G0W0
        G0W0_RUNNING  = auto()    # G0W0 submitted, waiting
        COMPLETE = auto()         # All done successfully
        FAILED   = auto()         # Unrecoverable error

        #Explicit recovery state, which is currently unused
        # but template for eventual extensions
        RECOVERY = auto()   # Attempt to fix previous failure

class VaspDFTGWWorkChain(WorkChain):
        @classmethod
        def define(cls, spec):
                super().define(spec)

                spec.expose_inputs(WorkflowFactory('vasp.vasp') , exclude=('parameters', 'settings'))
                spec.expose_inputs(VaspGWWorkChain              , exclude=('parameters', 'settings'))               

                spec.input('ns_parameters.encut'                  , valid_type=Float       , required=False , help='cutoff energy for the wavefunction in eV. encut variable in VASP.')  #ns stands for namespace
                spec.input('ns_parameters.nbands'                 , valid_type=Int         , required=False , help='total number of bands included in the DFT and G0W0 runs. nbands variable in VASP.'  )   
                #Note magnetic_moment_onsite assumes that the calculation is spin-polarized; Spin-Orbit calculations are currently not supported. 
                #If magnetic_moment_onsite  is not passed, the calculation is instead assumed spin non-polarized.
                spec.input('ns_parameters.magnetic_moment_onsite' , valid_type=Dict        , required=False , help='Starting collinear on-site magnetic moment ; Syntax is {ElName:value}')
                spec.input('ns_parameters.nomega'                 , valid_type=Int         , required=False , default=lambda: Int(200) , help='number of frequency points for the chi and sigma calculation in G0W0 runs. Default is 1 (COHSEX).') 
                spec.input('ns_parameters.nbandsgw'               , valid_type=Int         , required=False , help='number of bands for which QP energies are calculated - nbandsGW variable in VASP')  
                spec.input('ns_parameters.encut_chi'              , valid_type=Float       , required=False , help='cutoff energy for the response function in eV - encutGW variable in VASP') 

                spec.input('kpoints'                              , valid_type=KpointsData , help='K-mesh used for VASP G0W0 and DFT runs; get_kpoints_mesh() must work.' )     

                spec.input('ns_parallelization.kpar'              , valid_type=Int         , required=False , default=lambda: Int(4)      , help='kpar value to be used in G0W0 calculations')
                spec.input('ns_parallelization.npar'              , valid_type=Int         , required=False , default=lambda: Int(1)      , help='NPAR value to be used in G0W0 calculations')
                spec.input('ns_parallelization.lreal'             , valid_type=Bool        , required=False , default=lambda: Bool(False) , help='lreal value to be used in all calculations. If True sets to Auto, otherwise False') 

                spec.input('ns_reference.starting_RemoteData'     , valid_type=RemoteData  , required=False , help='the DFT ground state wavefunction (WAVECAR) and CHGCAR will be copied from this RemoteData folder as a starting point' )
                
                spec.input('ns_option.maximum_iterations'         , valid_type=Int  , required=False , default=lambda: Int(1)      , help='maximum number of times the workchain will restart a crashed G0W0 runs.')
                spec.input("ns_option.run_1DFTgr"                 , valid_type=Bool , required=False , default=lambda: Bool(False) , help='If True, run a DFT calculation to get the ground state (WAVECAR and CHGCAR) to be used as a starting point for following calculations. If False, no DFT ground state calculation is performed; in this case, starting_RemoteData must be provided as input.')
                spec.input('ns_option.run_2DFTvo_3G0W0'           , valid_type=Bool , required=False , default=lambda: Bool(True)  , help='Run the single-iteration DFT with all unoccupied bands included (DFTvo, vo stands for virtual orbital) and G0W0 with same encut and number of bands and the DFT wavefunctions and energies as starting point.')
                spec.input('ns_option.calculation_label'          , valid_type=Str  , required=False , default=lambda: Str("")     , help='The summary printed at the end will be labeled with this string.')

                spec.output('NGarray'         , valid_type=ArrayData  , required=False , help='FFT grid used.')
                spec.output('ENMAXarray'      , valid_type=ArrayData  , required=False , help='Array containing the ENMAX of all employed POTCARs.')           
                spec.output('kpoints'         , valid_type=DataFactory('core.array.kpoints') , help='The actual k-mesh used for VASP G0W0 and DFT runs' )
                
                spec.output('RemoteData_G0W0' , valid_type=RemoteData , required=False , help='RemoteData for the G0W0 calculation node.' )
                spec.output('RemoteData_DFT'  , valid_type=RemoteData , required=True , help= 'RemoteData for the DFT calculation node.')
                spec.output('bands_G0W0'      , valid_type=BandsData  , required=False , help='BandsData for the G0W0 calculation node.' )
                spec.output('bands_DFT'       , valid_type=BandsData  , required=True , help= 'BandsData for the DFT calculation node.')
                spec.output('gaps'            , valid_type=Dict       , required=True , help= 'Direct and indirect gaps for the DFT and G0W0 nodes.' )
                
                spec.output('gaps_QPc'        , valid_type=Dict       , required=False , help='QP HOMO correction at direct gap kpt')
                spec.output('bands_QPc'       , valid_type=BandsData  , required=False  ) 
                # Workflow outputs created:
                #   gaps : Dict with nested structure:
                #       { 'DFT':  { 'spinUp'|'spinDw': {'Dir': float , 'Ind': float ,  'Gam': float}},
                #         'G0W0': { 'spinUp'|'spinDw': {'Dir': float , 'Ind': float ,  'Gam': float}}, }
                #   gaps_QPc : Dict with nested structure (only if G0W0 was run - it corresponds to Quasiparticle gap corrections (G0W0 − DFT)
                #       { 'spinUp'|'spinDw': {'Dir': float , 'Ind': float , 'Gam': float }}
                #   bands_QPc : BandsData   (only if G0W0 was run)
                #       Quasiparticle band corrections, i.e. E_QP(k, n) = E_G0W0(k, n) − E_DFT(k, n)
                # Additionally exposed without modification:
                #   - bands_DFT, bands_G0W0
                #   - RemoteData_DFT, RemoteData_G0W0
                #   - kpoints, NGarray, ENMAXarray
                # Spin handling:
                #   * Non spin-polarized → only 'spinUp' key is present
                #   * Spin-polarized     → both 'spinUp' and 'spinDw'              
                                
                # spec.expose_outputs(cls._next_workchain) 

                spec.exit_code(401,'REACHED_MAXIMUM_TRY_NUMBER'          , message='The workflow reached the maximum number of tries.')
                spec.exit_code(402,'NO_STARTING_WAVECAR_forDFTvo'        , message='No starting WAVECAR found execution_statefor DFTvo calculation.')
                spec.exit_code(403,'NO_STARTING_WAVECAR_WAVEDER_forG0W0' , message='Cannot start G0W0 as DFTvo failed.')


                spec.outline(
                    cls.initialize,
                # Run it.
                    while_(cls.should_wc_continue)(   # 
                        cls.update_state,             # 
                        cls.validate_step,            # 
                        cls.prepare_step,             # 
                        cls.correct_previous_errors,  
                        cls.execute_step,                     
                        ),
                    cls.elaborate_results,
                    #cls.clean_remoteFolder_DFT,
                )

        def initialize(self):
            """ Initialize workflow context.
                This sets up: - the explicit Finite-state-machine like execution state
                            - the workflow state container (state_WC)
                No logic is executed here.     """

            #[1] Explicit workflow execution state (FSM)
            self.ctx.state_execution = state_execution_enum.INIT

            #[2] Workflow state container
            self.ctx.state_WC = AttributeDict({
                'starting_RemoteData': None,   # External restart folder (if provided)
                'restart_folders': AttributeDict({
                    'for_2DFTvo': None,             # RemoteData with WAVECAR + CHGCAR
                    'for_3G0W0':  None,         }), # RemoteData with WAVECAR + WAVEDER 
                'submitted': AttributeDict({'1DFTgr':[], '2DFTvo':[], '3G0W0':[], }),  # Submitted DFTgr/DFTvo/G0W0 workchains
                'retries':   AttributeDict({'1DFTgr':0, '2DFTvo':0, '3G0W0': 0,   }),     
                })

            #[3] Store optional external starting RemoteData
            try:
                self.ctx.state_WC.starting_RemoteData = self.inputs.ns_reference.starting_RemoteData
            except Exception:
                self.ctx.state_WC.starting_RemoteData = None

            # [4] Scratch space used later by prepare/execute
            self.ctx.inputs_finalized = None

            # [5] Which workchain should be called 
            self.ctx._next_workchain = { '1DFTgr': WorkflowFactory( 'vasp.vasp' )  ,
                                         '2DFTvo': WorkflowFactory( 'vasp.vasp' )  ,
                                         '3G0W0' : VaspGWWorkChain                 ,}
                                        #'3G0W0':  WorkflowFactory( 'vasp.vasp' ) ,  }
            
            # [6] Regarding spin polarization
            self.ctx.is_spinpol  = ("magnetic_moment_onsite" in self.inputs.ns_parameters)
            self.ctx.spin_labels = ("spinUp", "spinDw") if self.ctx.is_spinpol else ("spinUp",)
            
            
        def should_wc_continue(self) -> bool:
            """ Determine whether the workflow should continue looping. """
            return self.ctx.state_execution not in { state_execution_enum.COMPLETE, state_execution_enum.FAILED, }

        def update_state(self):
            """ Update the Finite-State-Machine execution state.
            This method:
            - inspects the current execution_state
            - inspects last submitted workchains (if any)
            - decides the next execution_state
            - performs no submissions
            - performs no INCAR / input preparation
            """
            #### ============================================================
            #[1] INIT → decide initial path
            if self.ctx.state_execution == state_execution_enum.INIT:

                if self.inputs.ns_option.run_1DFTgr.value :
                    # Always run DFTgr if explicitly requested
                    self.ctx.state_execution = state_execution_enum.DFTGR_PENDING
                    self.report(f"[<{self.inputs.ns_option.calculation_label.value}> update_state] run_1DFTgr=True: run 1DFTgr, using starting_RemoteData as restart for DFTgr if provided.")
                    return

                # run_1DFTgr == False
                if self.ctx.state_WC.starting_RemoteData is not None:
                    # Assume starting_RemoteData is a valid DFT ground state
                    self.report(f"[<{self.inputs.ns_option.calculation_label.value}> update_state] run_1DFTgr=False: using starting_RemoteData as DFTgr replacement")
                    self.ctx.state_WC.restart_folders.for_2DFTvo = self.ctx.state_WC.starting_RemoteData
                    if self.inputs.ns_option.run_2DFTvo_3G0W0.value: 
                        self.ctx.state_execution = state_execution_enum.DFTVO_PENDING
                        self.report(f"[<{self.inputs.ns_option.calculation_label.value}> update_state] DFTvo to be prepared&executed  -> state updated to={self.ctx.state_execution.name}")  
                    else:                                            
                        self.ctx.state_execution = state_execution_enum.COMPLETE
                    return

                # Neither DFTgr nor starting WAVECAR available
                self.report(f"[<{self.inputs.ns_option.calculation_label.value}> update_state] ERROR: no way to obtain a DFT ground state (no DFTgr run enabled, no starting_RemoteData)")
                self.ctx.state_execution = state_execution_enum.FAILED
                return self.exit_codes.NO_STARTING_WAVECAR_forDFTvo

            #### ============================================================
            #[2] DFTGR_RUNNING → DFTGR_DONE / retry / FAIL
            if self.ctx.state_execution == state_execution_enum.DFTGR_RUNNING:
                node = self.__last_wc_node(self.ctx,'1DFTgr')
                if node is None or not node.is_finished: return  #Guard against 
                if node.is_finished_ok:
                    self.ctx.state_WC.restart_folders.for_2DFTvo = node.outputs.remote_folder
                    self.ctx.state_execution = state_execution_enum.DFTGR_DONE
                    self.report(f"[<{self.inputs.ns_option.calculation_label.value}> update_state] DFTgr finished successfully  -> state updated to={self.ctx.state_execution.name}")
                else:
                    return_error = self.__handle_failure_with_retry( step='1DFTgr', pending_state=state_execution_enum.DFTGR_PENDING )
                    if return_error is not None: return return_error

                return

            #### ============================================================
            #[3] DFTGR_DONE → DFTVO_PENDING or COMPLETE
            if self.ctx.state_execution == state_execution_enum.DFTGR_DONE:
                if self.inputs.ns_option.run_2DFTvo_3G0W0.value:
                    self.ctx.state_execution = state_execution_enum.DFTVO_PENDING
                    self.report(f"[<{self.inputs.ns_option.calculation_label.value}> update_state] DFTgr finished successfully; DFTvo to be prepared&executed  -> state updated to={self.ctx.state_execution.name}")
                else:
                    self.ctx.state_execution = state_execution_enum.COMPLETE
                    self.report(f"[<{self.inputs.ns_option.calculation_label.value}> update_state] DFTgr finished successfully; DFTvo IS skipped  -> state updated to={self.ctx.state_execution.name}")
                return

            #### ============================================================
            #[4] DFTVO_RUNNING → DFTVO_DONE / retry / FAIL
            if self.ctx.state_execution == state_execution_enum.DFTVO_RUNNING:
                node = self.__last_wc_node(self.ctx,'2DFTvo')
                if node.is_finished_ok:
                    self.ctx.state_WC.restart_folders.for_3G0W0 = node.outputs.remote_folder
                    self.ctx.state_execution = state_execution_enum.DFTVO_DONE
                    self.report(f"[<{self.inputs.ns_option.calculation_label.value}> update_state] DFTvo finished successfully  -> state updated to={self.ctx.state_execution.name}")
                else:
                    return_error = self.__handle_failure_with_retry( step='2DFTvo', pending_state=state_execution_enum.DFTVO_PENDING )
                    if return_error is not None: return return_error
                return

            #### ============================================================
            #[5] DFTVO_DONE → G0W0_PENDING or COMPLETE
            if self.ctx.state_execution == state_execution_enum.DFTVO_DONE:
                if self.inputs.ns_option.run_2DFTvo_3G0W0.value:
                    self.ctx.state_execution = state_execution_enum.G0W0_PENDING
                    self.report(f"[<{self.inputs.ns_option.calculation_label.value}> update_state] DFTvo finished successfully; G0W0 to be prepared&executed  -> state updated to={self.ctx.state_execution.name}")
                else:
                    self.ctx.state_execution = state_execution_enum.COMPLETE
                return

            #### ============================================================
            #[6] G0W0_RUNNING →  COMPLETE / retry / FAIL
            if self.ctx.state_execution == state_execution_enum.G0W0_RUNNING:
                node = self.__last_wc_node(self.ctx,'3G0W0')
                if node.is_finished_ok:
                    self.ctx.state_execution = state_execution_enum.COMPLETE
                    self.report(f"[<{self.inputs.ns_option.calculation_label.value}> update_state] G0W0 finished successfully  -> state updated to={self.ctx.state_execution.name}")
                else:
                    return_error = self.__handle_failure_with_retry( step='3G0W0', pending_state=state_execution_enum.G0W0_PENDING )
                    if return_error is not None: return return_error
                return

        def validate_step(self):
            """  Validate prerequisites for the next PENDING step. 
                Does only perform checks and eventually throw errors, but NOT modify state (EXCEPT FOR FAILURE).  """

            state = self.ctx.state_execution
            if state == state_execution_enum.DFTVO_PENDING:
                ok = self.__validate_remote_has_required_files( remote=self.ctx.state_WC.restart_folders.for_2DFTvo , 
                                                            required=['WAVECAR'], label='DFTvo restart'         )
                if not ok:
                    self.ctx.state_execution = state_execution_enum.FAILED
                    return self.exit_codes.NO_STARTING_WAVECAR_forDFTvo

            if state == state_execution_enum.G0W0_PENDING:
                ok = self.__validate_remote_has_required_files( remote=self.ctx.state_WC.restart_folders.for_3G0W0    ,
                                                            required=['WAVECAR', 'WAVEDER'], label='G0W0 restart' )
                if not ok:
                    self.ctx.state_execution = state_execution_enum.FAILED
                    return self.exit_codes.NO_STARTING_WAVECAR_WAVEDER_forG0W0

        def correct_previous_errors(self):
            """ Attempt to correct errors from a previous failed calculation before resubmitting.

            Currently a NO-OP.
            Intended to be extended in the future, possibly driven by:
            - process_handler outcomes in child workchains
            - inspection of scheduler stderr
            - adaptive input modification (NBANDS, ENCUT, NCORE, etc.)
            This method MUST NOT submit calculations.
            """
            if self.ctx.state_execution != state_execution_enum.RECOVERY: return
            return

        def execute_step(self):
            #[1] Determine which calc to submit from state, i.e.
            mapping_enum_to_calc_type_torun = {	state_execution_enum.DFTGR_PENDING: '1DFTgr',
                                                 state_execution_enum.DFTVO_PENDING: '2DFTvo',
                                                 state_execution_enum.G0W0_PENDING:  '3G0W0' , }
            calc_type = mapping_enum_to_calc_type_torun.get(self.ctx.state_execution)
            if calc_type is None:   return

            #[2] Submit
            running_wc = self.submit( self.ctx._next_workchain[calc_type] , **self.ctx.inputs_finalized)

            #[3] Bump retry counter (this submission attempt)
            self.ctx.state_WC.retries[calc_type] += 1
            tmp_attempt_num = self.ctx.state_WC.retries[calc_type]

            #[4] Record submission into state_WC
            self.ctx.state_WC.submitted[calc_type].append(running_wc)		

            #[5] Update execution state
            mapping_calctype_to_state = {  '1DFTgr': state_execution_enum.DFTGR_RUNNING,
                                           '2DFTvo': state_execution_enum.DFTVO_RUNNING,
                                           '3G0W0':  state_execution_enum.G0W0_RUNNING , }
            self.ctx.state_execution = mapping_calctype_to_state[calc_type]

            #[6] Log
            self.__report_compact_submission(running_wc, tmp_attempt_num, calc_type)

            #[7] Register dependency for engine
            return ToContext(**{f'calc_{calc_type}': running_wc})

        def prepare_step(self):
            """ Prepare inputs for the next calculation based on execution_state.
            This method:    - builds inputs_finalized
                            - does NOT submit
                            - does NOT change execution_state   """

            if self.ctx.state_execution == state_execution_enum.DFTGR_PENDING:
                self.ctx.inputs_finalized = self._prepare_inputs_DFT( restart_folder=self.ctx.state_WC.starting_RemoteData,
                                                                      calc_type='1DFTgr' )
                return
            if self.ctx.state_execution == state_execution_enum.DFTVO_PENDING:
                self.ctx.inputs_finalized = self._prepare_inputs_DFT( restart_folder=self.ctx.state_WC.restart_folders.for_2DFTvo,
                                                                      calc_type='2DFTvo' )
                return
            if self.ctx.state_execution == state_execution_enum.G0W0_PENDING:
                self.ctx.inputs_finalized = self._prepare_inputs_G0W0( restart_folder=self.ctx.state_WC.restart_folders.for_3G0W0 )
                return

        def elaborate_results(self):
           """ Final post-processing and output assembly.
           This method:
               1. Identifies the last successful DFT and G0W0 WorkChains
               2. Exposes raw outputs (bands, kpoints, RemoteData)
               3. Computes spin-resolved gaps and band extrema using `elaborate_single_spin_component`
               4. Builds nested Dict outputs with a stable, documented structure
               5. Computes quasiparticle (QP) corrections if G0W0 is available
       Output dictionary structure (conceptual):
               gaps = { 'DFT':  { spin_label: {Dir, Ind, Gam} },
                        'G0W0': { spin_label: {Dir, Ind, Gam} } }
               gaps_QPc = { spin_label: {Dir, Ind, Gam} }
       where:  spin_label in {'spinUp', 'spinDw'}
       All numerical values are floats in eV. """
         
           # ---------------------------------------------------------
           #[1] Determine last successful DFT node and G0W0 nodes
           last_node_DFT  = None
           last_node_G0W0 = None
           if self.inputs.ns_option.run_2DFTvo_3G0W0.value:
               last_node_DFT  = self.__last_wc_node(self.ctx,'2DFTvo')
               last_node_G0W0 = self.__last_wc_node(self.ctx,'3G0W0')               
           elif self.inputs.ns_option.run_1DFTgr.value:
               last_node_DFT = self.__last_wc_node(self.ctx,'1DFTgr')
                 
                 
           #[2] Expose node outputs that do not need further elaboration
           if last_node_DFT:
               self.out('RemoteData_DFT' , last_node_DFT.outputs.remote_folder )
               self.out('NGarray'        , last_node_DFT.outputs.NGarray       )
               self.out('ENMAXarray'     , last_node_DFT.outputs.ENMAXarray    )
               self.out('kpoints'        , last_node_DFT.outputs.kpoints       )
               self.out('bands_DFT'     , last_node_DFT.outputs.bands)

           if last_node_G0W0:
               self.out('RemoteData_G0W0', last_node_G0W0.outputs.remote_folder)
               self.out('bands_G0W0'     , last_node_G0W0.outputs.bands)

           #[3] spin handling
           spin_channels = ( [0, 1] if "magnetic_moment_onsite" in self.inputs.ns_parameters else [None] )
               
           #[4] Elaborate DFT and G0W0 bands
           gaps     = AttributeDict()  ; bnd_extrema      = AttributeDict()
           gaps_DFT = AttributeDict()  ; bnd_extrema_DFT  = AttributeDict()
           gaps_G0W0 = AttributeDict() ; bnd_extrema_G0W0 = AttributeDict()
           for sp_comp in spin_channels:
               sp_label = "spinUp" if sp_comp in (None, 0) else "spinDw"
               bd_DFT_trimmed_sp_comp,  bnd_extrema_DFT_sp_comp,  gap_DFT_sp_comp  =\
                   self.elaborate_single_spin_component(last_node_DFT,  sp_comp) 
               bd_G0W0_trimmed_sp_comp, bnd_extrema_G0W0_sp_comp, gap_G0W0_sp_comp =\
                   self.elaborate_single_spin_component(last_node_G0W0, sp_comp)
               gaps_DFT[sp_label]          = deepcopy( gap_DFT_sp_comp )
               bnd_extrema_DFT[sp_label]   = deepcopy( bnd_extrema_DFT_sp_comp )
               gaps_G0W0[sp_label]         = deepcopy( gap_G0W0_sp_comp )
               bnd_extrema_G0W0[sp_label]  = deepcopy( bnd_extrema_G0W0_sp_comp )
                              
           if len( gaps_G0W0 ) : gaps.G0W0 = gaps_G0W0
           if len( bnd_extrema_G0W0 ) : bnd_extrema.G0W0 = bnd_extrema_G0W0
           if len( gaps_DFT ) :  gaps.DFT = gaps_DFT
           if len( bnd_extrema_DFT ) :  bnd_extrema.DFT = bnd_extrema_DFT
           gaps = Dict(dict=gaps).store()
           #bnd_extrema = Dict(dict=bnd_extrema).store()
           self.out('gaps', gaps)
           #self.out('bnd_extrema',bnd_extrema)
           
           #[5] QP Corrections
           if last_node_G0W0:
               gaps_QPc = AttributeDict(); bnd_extrema_QPc = AttributeDict()
               qp_bands_list = []; qp_occ_list = [] ; qp_kpoints = None

               for sp_label in bnd_extrema_DFT.keys():
                   sp_label = "spinUp" if sp_comp in (None, 0) else "spinDw"
                   
                         
                 # Recompute trimmed bands (cheap, consistent)
                   bd_DFT_trimmed,  bnd_DFT,  _ = \
                       self.elaborate_single_spin_component(last_node_DFT,  sp_comp)
                   bd_G0W0_trimmed, bnd_G0W0, _ = \
                       self.elaborate_single_spin_component(last_node_G0W0, sp_comp)
           
                   #QP extrema (i.e. HOMO/LUMO) ---
                   QPc = AttributeDict()
                   QPc["HOMO"] = bnd_G0W0["HOMO"] - bnd_DFT["HOMO"]
                   QPc["LUMO"] = bnd_G0W0["LUMO"] - bnd_DFT["LUMO"]
                   bnd_extrema_QPc[sp_label] = QPc
           
                   #QP gap corrections ---
                   gap_QPc = AttributeDict()
                   gap_QPc["Dir"] = Float(np.min(QPc["LUMO"] - QPc["HOMO"]))
                   gap_QPc["Ind"] = Float(np.min(QPc["LUMO"]) - np.max(QPc["HOMO"]))
                   gap_QPc["Gam"] = Float(QPc["LUMO"][0] - QPc["HOMO"][0])
                   gaps_QPc[sp_label] = gap_QPc
           
                   #QP bands ---
                   qp_bands_list.append( bd_G0W0_trimmed.get_array("bands")
                                         - bd_DFT_trimmed.get_array("bands")   )
                   qp_occ_list.append( bd_DFT_trimmed.get_array("occupations") )
                   if qp_kpoints is None:
                       qp_kpoints = bd_DFT_trimmed.get_kpoints()
           
               #Assemble BandsData for QP bands ---
               bd_QPcorr =  DataFactory("core.array.bands")()
               bd_QPcorr.set_kpoints(qp_kpoints)
           
               if len(qp_bands_list) == 1:
                   bd_QPcorr.set_bands( qp_bands_list[0],
                                         occupations=qp_occ_list[0], )
               else:
                   bd_QPcorr.set_bands( np.stack(qp_bands_list),
                                        occupations=np.stack(qp_occ_list),  )
           
               # --- Attach to outputs ---
               bd_QPcorr.store()
               gaps_QPc        = Dict(dict=gaps_QPc).store()
               #bnd_extrema_QPc = Dict(dict=bnd_extrema_QPc).store()
               self.out("bands_QPc", bd_QPcorr)
               self.out("gaps_QPc" , gaps_QPc)
               #self.out("bnd_extrema_QPc", bnd_extrema_QPc)
 
               self.__report_compact_results( last_node_DFT=last_node_DFT, last_node_G0W0=last_node_G0W0,
                                              gaps_dict=gaps, gaps_qpc_dict=gaps_QPc,     
                                              bnd_extrema_DFT=bnd_extrema.DFT, bnd_extrema_G0W0=bnd_extrema.G0W0 )


        ##[HELPER FUNCTIONS for elaborate_results] 
        @staticmethod
        def elaborate_single_spin_component(last_node, spin_index=None, OCCUPATION_THRESHOLD = 0.45):
           """ Compute gaps and quasiparticle corrections for a single spin component.
           Parameters
           last_bode : WorkChainNode       
           spin_index : int or None  None  -> non spin-polarized calculation
                                     0/1   -> spin-polarized component
           OCCUPATION_THRESHOLD : float
                        Occupation value below which a state is considered unoccupied.
           Returns
           #   bands_trimmed : BandsData
           #       - Same k-points as input bands
           #       - Bands and occupations trimmed to remove trailing "-1" padding
           #       - Shape:
           #           * non-spin-polarized : (nkpts, nbands_trimmed)
           #           * spin-polarized     : (nkpts, nbands_trimmed) for the spin selected by spin_index
           #   bnd_extrema : AttributeDict
           #       Keys: HOMO' : np.ndarray, shape (nkpts,)  :  HOMO energy at each k-point
           #            'LUMO' : np.ndarray, shape (nkpts,)  :  LUMO energy at each k-point
           #   gap : AttributeDict
           #       Keys: 'Dir' : float  :   Minimum direct gap over all k-points
           #             'Ind' : float  :   Indirect gap = min(LUMO) - max(HOMO)
           #             'Gam' : float  :   Direct gap at Γ (assumed k-point index 0)
           # If last_node is None, all three returned values are None.
           """
           #[0] Guard against last_node as None
           if last_node is None:
               return (None, None, None)

           #[1] Extract bands and occupations ---
           if spin_index is None:
               bands     = last_node.outputs.bands.get_array("bands")
               bnd_occ  = last_node.outputs.bands.get_array("occupations")
           else:
               bands    = last_node.outputs.bands.get_array("bands")[spin_index]
               bnd_occ  = last_node.outputs.bands.get_array("occupations")[spin_index]

           #[1] HOMO / LUMO detection ---
           occ = bnd_occ < OCCUPATION_THRESHOLD
           c_kptNum = occ.shape[0] #c_kptNum represents the total number of k-points in the Irreducible Brillouin Zone.
           #bndIdx_HOMOar and bndIdx_LUMOar are arrays (dimension = c_kptNum) containing the band indexes of the highest occupied / lowest unoccupied bands at each k-point.
           bndIdx_HOMOar = [] ; bndIdx_LUMOar = []
           for kptIdx in range(c_kptNum):
               # Identify the HOMO/LUMO crossing at this k-point:
               # occ[k, :] is a boolean array where True = unoccupied, False = occupied.
               # We look for the first band index where the occupation changes(occupied → unoccupied). 
               # # np.where(...) returns all indices of such changes; # [0][0] selects the first one (highest occupied band).
               #This index corresponds to the HOMO, # and the following band to the LUMO.
               occ_crossings = np.where(occ[kptIdx, 1:] != occ[kptIdx, :-1])[0]
               if len(occ_crossings) == 0: raise ValueError("No HOMO/LUMO crossing found (metallic system?)")
               bndIdx_HOMOar.append( occ_crossings[0] )
               bndIdx_LUMOar.append( occ_crossings[0]  + 1 )

           #[2] Energy extraction ---
           bnd_extrema = AttributeDict()
           bnd_extrema['HOMO'] = np.array([bands[k, i] for k, i in enumerate(bndIdx_HOMOar)])
           bnd_extrema['LUMO'] = np.array([bands[k, i] for k, i in enumerate(bndIdx_LUMOar)])

           #[3] Gaps ---
           gap = AttributeDict()
           gap['Dir'] = (np.min(bnd_extrema['LUMO'] - bnd_extrema['HOMO']))
           gap['Ind'] = (np.min(bnd_extrema['LUMO']) - np.max(bnd_extrema['HOMO']))
           gap['Gam'] = (bnd_extrema['LUMO'][0] - bnd_extrema['HOMO'][0])


           #[4] Trim -1 bands (unchanged logic) ---
           # Aiida can append -1 to the bands arrays - We want to retrun the DFT and G0W0 bands without these -1 values in the last band indexes.
           #In order to do this we first find the last band index without -1 entries (of indexes firstBnd_toTrim -1 ) and then we keep only those bands.
           #bnd_DFTvo_toTrim and bnd_G0W0_toTrim contain the -1 (not already trimmed).
           bands_wocc = np.array(last_node.outputs.bands.get_bands(also_occupations=True))
           c_kpt_num = bands_wocc.shape[1]
           c_bnd_num = bands_wocc.shape[2]
           try:
               firstBnd_toTrim = min( np.nonzero(np.in1d(bands_wocc[0, i, :], [-1]))[0][0]
                                      for i in range(c_kpt_num)    )
           except Exception:
                   firstBnd_toTrim = c_bnd_num

           bands_trim  = bands_wocc[0, :, :firstBnd_toTrim]
           occup_trim  = bands_wocc[1, :, :firstBnd_toTrim]
           bands_trimmed = DataFactory('core.array.bands')()
           bands_trimmed.set_kpoints(last_node.outputs.bands.get_kpoints())
           bands_trimmed.set_bands(bands_trim, occupations=occup_trim)
           return ( bands_trimmed, bnd_extrema, gap )


        ##[HELPER FUNCTIONS for prepare_step]
        def _prepare_inputs_DFT(self, restart_folder, calc_type):
            """ Prepare inputs for a DFT calculation (DFTgr or DFTvo). Reorganized from original prepare_DFT.  
                NOTE: `restart_folder` is saved inside input.restart_folder but its validity
                (existence and required files) is NOT checked in this method; if it's = None a None is saved inside inputs.restart_folder
                All checks MUST be performed earlier"""
            #[1] Base
            inputs = AttributeDict()
            inputs.update(self.exposed_inputs(self.ctx._next_workchain[calc_type]))
            inputs.clean_workdir=Bool(False)
            inputs.restart_folder = restart_folder

            #[2] Parser settings
            inputs.settings = AttributeDict({'parser_settings': {'include_node': ['bands','kpoints','structure','NGarray','maximum_number_pw']}})          

            #[4] INCAR 
            incar = {'incar': {'ediff':1E-7 , 'algo':"Normal"   , 'nelm':200 ,
                                'ismear':0   , 'sigma':0.02      , 
                                'prec':'Accurate' ,  'lmaxmix':4 , 'lorbit':11 }}
            if ( 'encut' in self.inputs.ns_parameters ):  incar['incar']['encut']  = self.inputs.ns_parameters.encut.value
            if self.ctx.is_spinpol :       
                    _ , incar['incar']['magmom'] = input_magnetic_moment_tomagmom(self.inputs.structure , self.inputs.ns_parameters.magnetic_moment_onsite.get_dict())
                    incar['incar']['ispin']  = 2     ; incar['incar']['icharg'] = 1
                    incar['incar']['amix_mag'] = 0.8 ; incar['incar']['bmix_mag'] = 0.00001
                    incar['incar']['amix'] = 0.2     ; incar['incar']['bmix'] = 0.00001
            else: 
                incar['incar']['ispin'] = 1

            # Parallelization settings : for now we report kpar and npar as defined from the inputs.ns_parallelization namespace.
            if ('kpar' in self.inputs.ns_parallelization ): incar['incar']['kpar'] = self.inputs.ns_parallelization.kpar.value
            if ('npar' in self.inputs.ns_parallelization ): incar['incar']['npar'] = self.inputs.ns_parallelization.npar.value
            if (self.inputs.ns_parallelization.lreal.value == True):
                incar['incar']['lreal'] = 'Auto'
            else:
                incar['incar']['lreal'] = '.FALSE.'
            
            if (self.ctx.state_execution == state_execution_enum.DFTVO_PENDING) :
                if ('nbands' in self.inputs.ns_parameters ): incar['incar']['nbands'] = self.inputs.ns_parameters.nbands.value
                incar['incar']['loptics'] = '.TRUE.'  
                incar['incar']['algo']    = "Exact"
                incar['incar']['nelm']    = 1    
                        
            inputs.parameters = Dict(dict=incar) 
            prepared_inputs = prepare_process_inputs(inputs , namespaces=['dynamics','verify'])
            return prepared_inputs

        def _prepare_inputs_G0W0(self, restart_folder):
            """ Prepare inputs for a G0W0 calculation.
                NOTE: `restart_folder` is saved inside input.restart_folder but its validity
                (existence and required files) is NOT checked in this method; if it's = None a None is saved inside inputs.restart_folder
                All checks MUST be performed earlier"""
            #[1] Base
            inputs = AttributeDict()
            inputs.update(self.exposed_inputs(self.ctx._next_workchain["3G0W0"]))
            inputs.clean_workdir=Bool(False)
            inputs.restart_folder = restart_folder

            #[2] Parser settings
            inputs.settings = AttributeDict({'parser_settings': {'include_node': ['bands','kpoints','structure']}})          
            inputs.settings['ADDITIONAL_REMOTE_COPY_LIST'] = ['WAVEDER'] 
            #[4] INCAR 
            incar = {'incar': {'nelm':1 , 'algo':'EVGW0' , 
                               'ismear':0 , 'sigma':0.02 , 
                               'prec':'Accurate', 'lmaxmix':4 , 'lorbit':11    ,
                               'nomega':self.inputs.ns_parameters.nomega.value , 
                               'kpar':self.inputs.ns_parallelization.kpar      }}
            if ('encut'  in self.inputs.ns_parameters ):  incar['incar']['encut'] = self.inputs.ns_parameters.encut.value
            if ('nbands' in self.inputs.ns_parameters ): incar['incar']['nbands'] = self.inputs.ns_parameters.nbands.value            
            #else: incar['incar']['nbands'] =  np.shape(self.ctx.WC_record_2DFTvo[-1].outputs.bands.get_bands())[1]   #In altenrnativa : self.ctx.WC_record_DFT[-1].outputs.get_dict()['run_status']['nbands']
            if ('nbandsgw'  in self.inputs.ns_parameters ): incar['incar']['nbandsgw'] = self.inputs.ns_parameters.nbandsgw.value
            if ('encut_chi' in self.inputs.ns_parameters ):  
                incar['incar']['encutgw']      = self.inputs.ns_parameters.encut_chi.value                                                      
                incar['incar']['encutgwsoft']  = self.inputs.ns_parameters.encut_chi.value 
            if self.ctx.is_spinpol :       
                    _ , incar['incar']['magmom'] = input_magnetic_moment_tomagmom(self.inputs.structure , self.inputs.ns_parameters.magnetic_moment_onsite.get_dict())
                    incar['incar']['ispin']  = 2     ; incar['incar']['icharg'] = 1
                    incar['incar']['amix_mag'] = 0.8 ; incar['incar']['bmix_mag'] = 0.00001
                    incar['incar']['amix'] = 0.2     ; incar['incar']['bmix'] = 0.00001
            else: 
                incar['incar']['ispin'] = 1

            # Parallelization settings : for now we report kpar and npar as defined from the inputs.ns_parallelization namespace.
            if ('kpar' in self.inputs.ns_parallelization ): incar['incar']['kpar'] = self.inputs.ns_parallelization.kpar.value
            if (self.inputs.ns_parallelization.lreal.value == True):
                incar['incar']['lreal'] = 'Auto'
            else:
                incar['incar']['lreal'] = '.FALSE.'
                                
            inputs.parameters = Dict(dict=incar) 
            prepared_inputs = prepare_process_inputs(inputs , namespaces=['dynamics','verify'])
            return prepared_inputs
	
        ##[HELPER FUNCTIONS for update_state and validate_step] - should be kept here as it uses self
        @staticmethod
        def __last_wc_node(ctx, step: str):
            """ Return last submitted child WC node for a given submitted list key: '1DFTgr' | '2DFTvo' | '3G0W0' """
            lst = ctx.state_WC.submitted.get(step, [])
            return lst[-1] if lst else None

        def __handle_failure_with_retry(self, step: str, pending_state):
            """ Handle a failed step with retry logic.
            Parameters
            ----------
            step : str
                One of '1DFTgr', '2DFTvo', '3G0W0'
            pending_state : state_execution_enum
                State to transition to if a retry is allowed

            Returns
            -------
            None or ExitCode
                Returns an ExitCode if retries are exhausted, otherwise None.         """

            max_iter = self.inputs.ns_option.maximum_iterations.value
            retries  = self.ctx.state_WC.retries[step]
            if retries < max_iter:
                self.report(f"[update_state] {step} failed, retrying (attempt {retries+1}/{max_iter})")
                self.ctx.state_execution = pending_state
                return None

            self.report(f"[update_state] {step} failed AND maximum retries reached -> ABORT")
            self.ctx.state_execution = state_execution_enum.FAILED
            return self.exit_codes.REACHED_MAXIMUM_TRY_NUMBER

        def __validate_remote_has_required_files(self, remote, required, label: str):
            """ Validate that RemoteData exists contains required files.
                Returns:	bool: True if all required files are present, False otherwise.
                Side effects:	Logs missing files (and listdir failures) via self.report.  """
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

        ##[HELPER FUNCTION FOR execute_step]
        @staticmethod
        def __fmt_float(x, nd=3):
            try:              return f"{float(x):.{nd}f}"
            except Exception: return str(x)
        
        @staticmethod
        def __generate_compact_submission_string( wc_node , prefix="  > " ):
            """Emit a compact input summary right before submitting a calculation. """
            # --- basic electronic parameters ---
            def _get_incar_par(aiida_dict, key):
                try:
                    return aiida_dict.get_dict()['incar'][key]
                except Exception:
                    return None
            encut     = VaspDFTGWWorkChain.__fmt_float( _get_incar_par(wc_node.inputs.parameters, "encut") )
            nbands    = _get_incar_par(wc_node.inputs.parameters, "nbands")
            encut_chi = _get_incar_par(wc_node.inputs.parameters, "encutgw")
            nomega    = _get_incar_par(wc_node.inputs.parameters, "nomega")

            # --- kpoints ---
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

            # --- parallelization ---
            try:
                kpar = _get_incar_par(wc_node.inputs.parameters.get_dict(), "kpar")  
            except Exception:
                kpar = None

            # --- potentials ---
            pot_family  = None
            pot_mapping = None
            try:
                pot_family = wc_node.inputs.potential_family.value
            except Exception:
                pass
            try:
                pot_mapping = wc_node.inputs.potential_mapping.get_dict()
            except Exception:
                pass

            lines = [ f"{prefix}nbands={nbands}  encut={encut}  encut_chi={encut_chi}  nomega={nomega}  kpar={kpar}","\n",
                      f"{prefix}kpts_mesh={mesh} nkpts={nkpts}","\n",
                      f"{prefix}potcars_family={pot_family}  potcars_mapping={pot_mapping}",  ]
            return ("".join(lines))
        
        def __report_compact_submission(self, running_wc, tmp_attempt, calc_type: str):
            """Emit a compact input summary right before/after submitting a calculation."""
            try:
                label = self.inputs.ns_option.calculation_label.value
            except Exception:
                label = ""
            prolog = ( f"[<{label}> execute_step] launching {calc_type} pk={running_wc.pk} "
                       f"(attempt num={tmp_attempt}) → state updated to={self.ctx.state_execution.name}" )
        
            msg = prolog +"\n"+ self.__generate_compact_submission_string(running_wc)+"\n"
            self.report(msg)

        def __report_compact_results( self, last_node_DFT, last_node_G0W0,
                                     gaps_dict, gaps_qpc_dict=None,     
                                     bnd_extrema_DFT=None, bnd_extrema_G0W0=None):   
            """ Emit a compact result summary at the end of elaborate_results.
        
            Assumptions (NEW schema only)
            -----------------------------
            gaps_dict has:
                gaps_dict["DFT"][spin]["Dir"|"Ind"|"Gam"]
                gaps_dict["G0W0"][spin]["Dir"|"Ind"|"Gam"]   (only if G0W0 ran)
            gaps_qpc_dict (optional) has:
                gaps_qpc_dict[spin]["Dir"|"Ind"|"Gam"]
        
            Also reuses __generate_compact_submission_string(node.inputs, prolog=...) to print the
            same compact input summary for the finished child nodes.        """
            #[1.1] Preliminary : Helpers ------------ ------------ ------------
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
                """  Returns dict with keys: elapsed_s, elapsed_hms, max_mem_gib, n_mpi_ranks - Missing values are None. """
                out = {"elapsed_s": None, "elapsed_hms": None, "max_mem_gib": None, "n_mpi_ranks": None}
                if node is None: return out   
                try: #time and memory from output.misc
                    misc = node.outputs.misc.get_dict()
                    out["elapsed_s"] = misc['run_stats']['elapsed_time']
                    out["elapsed_hms"] = __sec_to_hms(out["elapsed_s"]) if out["elapsed_s"] is not None else None
        
                    mem_b = misc.get("maximum_memory_used", None)
                    out["max_mem_gib"] = __bytes_to_gib(mem_b) if mem_b is not None else None
                except Exception:
                    pass
                try: # mpi ranks from inputs.options.resources
                    opts = node.inputs.options.get_dict()
                    res = opts.get("resources", {})
                    out["n_mpi_ranks"] = int(res["num_machines"]) * int(res["num_mpiprocs_per_machine"])
                except Exception:
                    pass
                return out            
 
            def __fmt_arr(x):
                x = np.array(x)
                return np.array2string(x, precision=4, separator=" ", max_line_width=10**9)
    
            #[1.2] Preliminary : Initial Label ------------ ------------ ------
            try:
                label = self.inputs.ns_option.calculation_label.value
            except Exception:
                label = ""
            prolog = [f"[<{label}> elaborate_results] Summary"]
            lines  = []
            
            #[1.3] Preliminary : allows passings Dict instead of dicts --------
            if hasattr(gaps_dict, "get_dict"):
                gaps_dict = gaps_dict.get_dict()
            if gaps_qpc_dict is not None and hasattr(gaps_qpc_dict, "get_dict"):
                gaps_qpc_dict = gaps_qpc_dict.get_dict()
 
            #[1.4] Preliminary : Define base keys ------------ ------------ ---
            keys   = ("Dir", "Ind", "Gam") 
           
            #[2] Performance notes ------------ ------------ ------------ -----
            if last_node_DFT is not None:
                p = __extract_perf(last_node_DFT)
                lines+= [ f"\n  [1] DFT perf: elapsed={p['elapsed_hms'] or p['elapsed_s']}  -  mpi_ranks={p['n_mpi_ranks']}","\n"]
                     
            if last_node_G0W0 is not None:
                p = __extract_perf(last_node_G0W0)
                lines+= [ f"  [1] G0W0 perf: elapsed={p['elapsed_hms'] or p['elapsed_s']}  -  mpi_ranks={p['n_mpi_ranks']}","\n"]

            #[3] Input strings ------------ ------------ ------------ ---------
            if last_node_DFT is not None:
                lines += ["  [2] DFT inputs:\n"]
                lines += [ self.__generate_compact_submission_string(wc_node=last_node_DFT, prefix="  [2] ") ,"\n" ]
            if last_node_G0W0 is not None:
                lines += ["  [2]G0W0 inputs:\n"]
                lines += [self.__generate_compact_submission_string(wc_node=last_node_G0W0, prefix="  [2] ") ,"\n" ]
                

            #[4] HOMO/LUMO eigenvalue arrays
            for sp in self.ctx.spin_labels:
                lines += [f"  [3] HOMO/LUMO - Spin component : {sp}\n"]
                if bnd_extrema_DFT is not None:
                    lines += [f"  [3] HOMO DFT eigenvalues: {__fmt_arr(bnd_extrema_DFT[sp]['HOMO'])}\n"]
                    lines += [f"  [3] LUMO DFT eigenvalues: {__fmt_arr(bnd_extrema_DFT[sp]['LUMO'])}\n"]
                if bnd_extrema_G0W0 is not None:
                    lines += [f"  [3] HOMO GW eigenvalues : {__fmt_arr(bnd_extrema_G0W0[sp]['HOMO'])}\n"]
                    lines += [f"  [3] LUMO GW eigenvalues : {__fmt_arr(bnd_extrema_G0W0[sp]['LUMO'])}\n"]

            #[5] gaps 
                for sp in self.ctx.spin_labels:
                    lines += [f"  [4] Gaps - Spin component : {sp}\n"]
                    if ("DFT" in gaps_dict) and (sp in gaps_dict["DFT"]):
                        for k in keys:
                            lines += [f"  [4] gap_DFT_{k}{Float(gaps_dict['DFT'][sp][k])}\n"]

                    if ("G0W0" in gaps_dict) and (sp in gaps_dict["G0W0"]):
                        for k in keys:
                            lines += [f"  [4] gap_G0W0_{k}{Float(gaps_dict['G0W0'][sp][k])}\n"]
                    if (gaps_qpc_dict is not None) and (sp in gaps_qpc_dict):
                        qpc_sp = gaps_qpc_dict[sp]
                        lines+= [ f"  [5] QPc gaps {sp}: Dir={VaspDFTGWWorkChain.__fmt_float(qpc_sp['Dir'])}  Ind={VaspDFTGWWorkChain.__fmt_float(qpc_sp['Ind'])}  Gam={VaspDFTGWWorkChain.__fmt_float(qpc_sp['Gam'])}","\n"]
                        
            self.report("".join(lines))
                
                
