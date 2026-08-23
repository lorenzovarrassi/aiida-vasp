import numpy as np
from copy import deepcopy
from aiida import orm
import os.path
from enum import Enum, auto
from aiida.orm import Code, Int, Float, Str, Dict, Bool , List , RemoteData , ArrayData , BandsData , XyData
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine  import WorkChain, calcfunction , ToContext , append_ , submit, while_ , if_ , submit, run
from aiida_vasp.utils.workchains  import prepare_process_inputs
from aiida.common.extendeddicts   import AttributeDict
from aiida_vasp.utils.workchains  import site_magnetization_to_magmom
from .workchain_wrapper_VaspWorkchain_fallbacks import VaspWorkChainWithFallbacks
from .utils_helpers_mBSE import _determine_BSE_parameters
from .utils_helpers_extrapolation import  input_magnetic_moment_tomagmom


from aiida import load_profile
load_profile()




class MbseState(Enum):
    INIT = auto()
    DFT_PENDING = auto()
    DFT_RUNNING = auto()
    DFT_DONE = auto()
    MBSE_PENDING = auto()
    MBSE_RUNNING = auto()
    COMPLETE = auto()
    FAILED = auto()   # optional, but helpful


class VaspmBSEInitScriptWorkChain(WorkChain):
    """ [1] OVERVIEW OF THE PURPOSE
    High-level workflow to execute a complete mBSE (model Bethe–Salpeter Equation)
    calculation using:
    - a preceding DFT ground-state calculation (NSP or SP - run internally by the workchain)
    - analytic GW-based diagonal screening
    - optional quasiparticle corrections via an interpolation script or scisso
    - a final BSE run (TDHF/ALGO=TDHF in VASP)

    This workchain wraps two sub-workchains:
    1) vasp.vasp                          (DFT ground-state)
    2) VaspWorkChainWithFallbacks  (BSE)

    [2]INPUTS/OUTPUTS: OVERVIEW
    Top-level inputs:    
      code                 : Code          → used by vasp.vasp and init script step
      options              : Dict          → SLURM options, prepend_text, etc.
      structure            : StructureData
      kpoints              : KpointsData
      potential_family     : Str
      potential_mapping    : Dict({element: POTCAR})

    Namespace ns_parameters:
        encut                  : Float (optional)
        nbands                 : Int  (optional)
        magnetic_moment_onsite : Dict (optional)
        ibse                   : Int (optional - default=2)
        kpar                   : Int (optional - default=1 / from gpu if used)
    
    Note on QP correction: this workchain no longer performs WAVECAR/GW
    interpolation itself - it consumes whatever restart_folder RemoteData it
    is given for the mBSE step (already QP-corrected upstream, e.g. by
    aiida-vasp-qpcorrection's WavefunEigenCorrectCalculation, or not) and
    submits it via a plain vasp.vasp-style calculation
    (VaspWorkChainWithFallbacks). If no external QP correction is
    supplied, set ns_BSE.use_scissor=True to apply an internal SCISSOR
    approximation instead (see _determine_BSE_parameters).
    (Formerly this was driven by an ns_interpolation.* input namespace with
    its own local/remote GW-reference-folder branches and an
    __prepare_inputs_G0W0interpolation step; that logic moved out - see
    handoff.md's decisions log. Even more formerly, the BSE step ran through
    VaspInitScriptWorkChain/Vasp2wInitScriptCalculation, a generic
    prepend-script injection mechanism [local_init_script /
    local_files_to_copy_to_remote_submission_folder / init_script_call_command]
    - retired 2026-08-22 as unused once QP correction moved to a dedicated
    upstream CalcJob; see handoff.md.)

    Namespace ns_BSE:
        static_inverse_diel       : Float  (REQUIRED)
        screening_parameter       : Float  (REQUIRED)
        G0W0_gap                  : Float  (optional but recommended; otherwise DFT gap is used)
        optical_energy_window     : Float  (optional)
        OMEGAMAX                  : Float  (optional override)
        NBANDSV, NBANDSO          : Int    (optional overrides)

    Outputs created by the workchain:
        dielectrics        : ArrayData
        opticaltransitions : ArrayData
        Additionally, all retrieved files of the final BSE run are copied locally.

    [3]STEP-BY-STEP LOGIC
    Step 1: Run DFT ground-state calculation (non-spin-polarized)
            (optionally: run spin-polarized version — not implemented)
    Done by prepare_run_DFTground_NSP:
        - Builds inputs for vasp.vasp
        - Extracts INCAR from ns_parameters
        - Uses loptics=True to store WAVEDER for mBSE
        - Stores output remote_folder, bands, kpoints, structure
        - Returns ToContext(finishedWC_DFTgr_NSP)
   
    Step 2: Consume whatever restart_folder is produced by DFT (already
            QP-corrected upstream, or not) OR apply a SCISSOR approximation
            if ns_BSE.use_scissor is set
    Step 3: Launch a BSE calculation with model screening parameters
              (AEXX, HFSCREEN) and with NBANDSV/NBANDSO built automatically
              via _determine_BSE_parameters()
    Done by prepare_run_interpolation_BSE:
        - Builds inputs for VaspWorkChainWithFallbacks
        - Defines parser settings (retrieve BSEFATBAND, vaspout.h5)
        - Determines restart_folder from the DFT step
        - Builds INCAR for BSE run:
             algo=TDHF, lmodelhf, nbseeig, ismear, prec
        - Fills AEXX, HFSCREEN
        - Calls _determine_BSE_parameters() to compute NBANDSV/O:
             1. NBANDSV/O determined as the mininum nunmber of v/c bands
                required to include all IPA transitions below optical_energy_window
             2. DFT bands are used; if G0W0_gap is passed a scissor is applied to
                DFT bands before determining all IPA transitions
        - Injects SCISSOR if ns_BSE.use_scissor is set; requires G0W0_gap
        - Submits workchain
      
    Step 4: Retrieve dielectric function and optical transitions and copy
            full retrieved folder into:  ./3.1_mBSE_<kmesh>_id<pid>
    Done by elaborate_results:
        - Exports dielectrics and optical transitions as outputs
        - Copies retrieved folder into a labeled directory in CWD

        
    [4]NOTES AND LIMITATIONS
        - Spin-polarized ground-state workflow present but NOT implemented.
    """
    
    
    
    _vasp_workchain = WorkflowFactory('vasp.vasp')
    _vasp_mbse_workchain = VaspWorkChainWithFallbacks

    @classmethod
    def define(cls, spec):
            super(VaspmBSEInitScriptWorkChain, cls).define(spec) 

            spec.expose_inputs( cls._vasp_workchain            , exclude=('parameters','settings','options'))
            spec.expose_inputs( cls._vasp_mbse_workchain , exclude=('parameters','settings','options','extraresources_fallback_options'))


            spec.input('ns_parameters.encut'                  , valid_type=Float , required=False , help='Cutoff energy for the wavefunction in eV. ENCUT variable in VASP.')  #ns stands for namespace
            spec.input('ns_parameters.nbands'                 , valid_type=Int   , required=False , help='Total number of bands included in the DFT and G0W0 runs. NBANDS variable in VASP.'  )   
            spec.input('ns_parameters.magnetic_moment_onsite' , valid_type=Dict  , required=False , help='Starting collinear on-site magnetic moment ; Syntax is {ElName:value}')
            spec.input("ns_parameters.ibse"                   , valid_type=Int   , required=False , default=lambda: Int(2),  help="Controls the BSE integration scheme. See https://vasp.at/wiki/IBSE")
            spec.input('ns_parameters.kpar'                   , valid_type=Int   , required=False , default=lambda: Int(1),  help="Parallelization across k-points. Defaults = number of GPUs if available.")          
            spec.input('ns_parameters.nbseeig'                , valid_type=Int   , required=False , default=lambda: Int(50), help="Number of BSE eigenvectors written to BSEFATBAND.")          


            spec.input('ns_optimization.lreal'                , valid_type=Bool  , required=False , default=lambda: Bool(True) , help='lreal value to be used in all calculations. If True sets to Auto, otherwise False') 
            spec.input("ns_optimization.set_PRECFOCK_to_Fast" , valid_type=Bool  , required=False , default=lambda: Bool(True) , help=('The use of Precfock=Fast depends on the cell dimension, Precfock=Fast is set if volume>350'
                                                                                                                                        'If True set PRECFOCK=Fast in the mBSE calculation; if False, always set it to default.') )
            
            spec.input("ns_BSE.static_inverse_diel"  , valid_type=Float , required=True  , help='Required for analytic diagonal screening in mBSE.' )
            spec.input("ns_BSE.screening_parameter"  , valid_type=Float , required=True  , help='Required for analytic diagonal screening in mBSE.' )
            spec.input("ns_BSE.G0W0_gap"             , valid_type=Float , required=False , help="G0W0 gap; required to determine SCISSOR")
            spec.input("ns_BSE.optical_energy_window", valid_type=Float , required=False , help="Required for the automatic determination of the NBANDSV/NBANDSO given a target energy window")
            spec.input("ns_BSE.OMEGAMAX"             , valid_type=Float , required=False , help='Required for analytic diagonal screening in mBSE.' )
            spec.input("ns_BSE.NBANDSV"              , valid_type=Int   , required=False , help=('Force NBANDSV value in the INCAR; override the determination of NBANDSV via target energy window.' 
                                                                                                 'NBANDSV/NBANDSO should be passed together; cannot define only one of those two.')        )
            spec.input("ns_BSE.NBANDSO"              , valid_type=Int   , required=False , help=('Force NBANDSO value in the INCAR; override the determination of NBANDSO via target energy window.'
                                                                                                 'NBANDSV/NBANDSO should be passed together; cannot define only one of those two.')        )
            spec.input("ns_BSE.use_scissor"          , valid_type=Bool  , required=False , default=lambda: Bool(False),
                                                                       help=('Apply a SCISSOR approximation (from _determine_BSE_parameters) instead of relying on an '
                                                                         +'externally QP-corrected restart_folder. Leave False when the restart_folder already carries '
                                                                         +'QP-corrected eigenvalues (e.g. from an upstream WAVECAR interpolation/GP-prediction step).') )

            spec.input("options" , valid_type=Dict , required=True )
            spec.input("extraresources_fallback_options", valid_type=Dict, required=False,
                       help=("Optional larger scheduler-options profile. Used ONLY for the mBSE BSE "
                             "stage's single automatic retry after an ERROR_DID_NOT_FINISH (exit 700) failure "
                             "(e.g. OOM) - every other calculation, and the first attempt of this one, still "
                             "uses 'options'. If not supplied, that retry behaves exactly as before (same "
                             "options, unchanged)."))
            spec.input("ns_option.copy_result_locally"    , valid_type=Bool       , required=False , default=lambda:Bool(True) )
            spec.input('ns_option.calculation_label'      , valid_type=Str        , required=False , default=lambda: Str("")     , help='The summary printed at the end will be labeled with this string.')
            spec.input('ns_option.calculation_tag'        , valid_type=Str        , required=False , default=lambda: Str("")     , help='Short suffix appended to the locally-copied result folder name (see copy_result_locally), e.g. "_KPTSconv"/"_NBANDSVOconv"/"_final", to distinguish which stage produced it.')
            spec.input('ns_option.copy_result_locally_path', valid_type=Str        , required=False , default=lambda: Str("")     , help='Absolute path to copy results into (see copy_result_locally). If empty, falls back to os.getcwd() at the time this step runs - which is unreliable when the workchain step executes inside a daemon worker (its cwd need not match the directory the submit script was launched from). Submit scripts should set this explicitly, e.g. Str(os.getcwd()) captured before submission.')
            spec.input('ns_reference.starting_RemoteData' , valid_type=RemoteData , required=False , help='the DFT ground state wavefunction (WAVECAR) and CHGCAR will be copied from this RemoteData folder as a starting point' )
            spec.input('ns_reference.use_hdf5'            , valid_type=Bool       , required=False , default=lambda: Bool(False) , help='set LH5 and LWAVEH5 to true, i.e. use preferentially HF5 instead of WAVECAR.')






            spec.output("dielectrics"        , valid_type=ArrayData )
            spec.output("opticaltransitions" , valid_type=ArrayData , required=False )
            spec.exit_code(402,'MAGN_NOT_IMPLEMENTED', message='_determine_BSE_parameters and reading CHGCAR magnetic non implemented.')
            
            spec.outline(
                cls.initialize,
                while_(cls.should_wc_continue)(
                    cls.update_state,
                    cls.prepare_step,
                    cls.execute_step,
                ),
                cls.elaborate_results,
            )

    def initialize(self):
        """Initialize workflow context in the same style as G0W0 base."""
    
        #[1] Explicit workflow execution state (FSM)
        self.ctx.state_execution = MbseState.INIT
    
        #[2] Workflow state container (G0W0-style naming)
        self.ctx.state_WC = AttributeDict({
            "starting_RemoteData": None,  # optional external restart (if you later add it)
            "restart_folders": AttributeDict({
                "for_DFT": None,                # RemoteData used as restart for DFT (rare; usually None)
                "for_MBSE": None,           }), # RemoteData produced by DFT (WAVECAR/WAVEDER etc) used by MBSE
            "submitted": AttributeDict({  "DFT": [], "MBSE": [], }),
            "retries":   AttributeDict({ "DFT": -1,   "MBSE": -1, }), #-1 because the first is zer0?
            })
    
        #[3] Store optional external starting RemoteData
        try:
            self.ctx.state_WC.starting_RemoteData = self.inputs.ns_reference.starting_RemoteData
        except Exception:
            self.ctx.state_WC.starting_RemoteData = None
    
        #[4] Scratch space used later by prepare/execute
        self.ctx.inputs_finalized = None
        
        #[5] Regarding spin polarization (keep identical)
        self.ctx.is_spinpol  = ("magnetic_moment_onsite" in self.inputs.ns_parameters)
        self.ctx.spin_labels = ("spinUp", "spinDw") if self.ctx.is_spinpol else ("spinUp",)

        self.ctx._next_workchain = { "DFT": WorkflowFactory("vasp.vasp") ,
                                    "MBSE": VaspWorkChainWithFallbacks ,   }

    def should_wc_continue(self) -> bool:
        return self.ctx.state_execution not in {MbseState.COMPLETE, MbseState.FAILED}

    def update_state(self):
        """Advance FSM by inspecting last submitted nodes. No submission here."""
        label = self.inputs.ns_option.calculation_label.value

        # INIT -> DFT_PENDING
        if self.ctx.state_execution == MbseState.INIT:
            self.ctx.state_execution = MbseState.DFT_PENDING
            self.report(f"[<{label}> update_state] INIT -> DFT_PENDING")
            return

        # DFT_RUNNING -> DFT_DONE / FAILED
        if self.ctx.state_execution == MbseState.DFT_RUNNING:
            node = self._last_wc_node("DFT")
            if node is None or not node.is_finished:
                return
            if not node.is_finished_ok:
                self.ctx.state_execution = MbseState.FAILED
                self.report(f"[<{label}> update_state] DFT failed pk={node.pk} -> FAILED")
                return

            # success: store restart folder for MBSE
            self.ctx.state_WC.restart_folders.for_MBSE = node.outputs.remote_folder
            self.ctx.state_execution = MbseState.DFT_DONE
            self.report(f"[<{label}> update_state] DFT ok pk={node.pk} -> DFT_DONE")
            return

        # DFT_DONE -> MBSE_PENDING
        if self.ctx.state_execution == MbseState.DFT_DONE:
            self.ctx.state_execution = MbseState.MBSE_PENDING
            self.report(f"[<{label}> update_state] DFT_DONE -> MBSE_PENDING")
            return

        # MBSE_RUNNING -> COMPLETE / FAILED
        if self.ctx.state_execution == MbseState.MBSE_RUNNING:
            node = self._last_wc_node("MBSE")
            if node is None or not node.is_finished:
                return
            if not node.is_finished_ok:
                self.ctx.state_execution = MbseState.FAILED
                self.report(f"[<{label}> update_state] mBSE failed pk={node.pk} -> FAILED")
                return

            self.ctx.state_execution = MbseState.COMPLETE
            self.report(f"[<{label}> update_state] mBSE ok pk={node.pk} -> COMPLETE")
            return

    def execute_step(self):
        """Submit exactly one workchain for the current PENDING state."""
        label = self.inputs.ns_option.calculation_label.value
        state = self.ctx.state_execution
        
        #[1] Determine which calc to submit from state, i.e.
        mapping_enum_to_calc_type_torun = { MbseState.DFT_PENDING: "DFT",
                                            MbseState.MBSE_PENDING: "MBSE",     }
        calc_type = mapping_enum_to_calc_type_torun.get(state)
        
        #Some error checking
        if calc_type is None: return
        if self.ctx.inputs_finalized is None:
            self.report(f"[<{label}> execute_step] ERROR: inputs_finalized is None for {calc_type}")
            self.ctx.state_execution = MbseState.FAILED
            return

        #[2] Submit
        running_wc = self.submit(self.ctx._next_workchain[calc_type], **self.ctx.inputs_finalized)

        #[3] Bump retry counter (this submission attempt)
        self.ctx.state_WC.retries[calc_type] += 1
        attempt = self.ctx.state_WC.retries[calc_type]
        
        #[4] Record submission into state_WC
        self.ctx.state_WC.submitted[calc_type].append(running_wc)

        #[5] Update execution state
        if calc_type == "DFT":     self.ctx.state_execution = MbseState.DFT_RUNNING
        elif calc_type == "MBSE":  self.ctx.state_execution = MbseState.MBSE_RUNNING

        #[6] Log
        include_bse = (calc_type == "MBSE")
        msg = (  f"[<{label}> execute_step] submit {calc_type} attempt={attempt} pk={running_wc.pk}\n"
                 + self.__generate_compact_submission_string(running_wc, prefix="  > ", include_BSE_parameters=include_bse)
                 + "\n" )
        self.report(msg)

        #[7] Register dependency for engine
        return ToContext(**{f"wk_{calc_type}": running_wc})

    def prepare_step(self):
        """Prepare ctx.inputs_finalized for the next PENDING state. No submission here."""
        state = self.ctx.state_execution
    
        # default
        self.ctx.inputs_finalized = None
    
        if state == MbseState.DFT_PENDING:
            self.ctx.inputs_finalized = self.__prepare_inputs_DFT()
            return
    
        if state == MbseState.MBSE_PENDING:
            inputs = self.__prepare_inputs_mBSE_base()

            # Add INCAR (mutates `inputs.parameters`)
            inputs = self.__add_inputs_mBSE_incar(inputs)
    
            # Normalize namespaces expected by aiida-vasp wrappers
            self.ctx.inputs_finalized = prepare_process_inputs( inputs, namespaces=["calc", "dynamics", "verify"],  )
            return
        # any other state: nothing to prepare
        return

    def __build_options_entry(self, prepend_text="", input_dict=None):
        """helper to construct a Python dict (and not AiiDA Dict) for the scheduler options
        starting from `input_dict` (an AiiDA Dict), defaulting to self.inputs.options."""
        input_opts = (input_dict if input_dict is not None else self.inputs.options).get_dict()
        out = {}
        for k in ("account", "qos", "resources", "queue_name", "max_memory_kb", "max_wallclock_seconds"):
            if k in input_opts:
                out[k] = input_opts[k]
        if prepend_text:                               out["prepend_text"] = prepend_text
        if "custom_scheduler_commands" in input_opts:  out["custom_scheduler_commands"] = input_opts["custom_scheduler_commands"]
        return out
            
    def __prepare_inputs_DFT(self):
        """Prepare inputs for DFT ground state (your old prepare_run_DFTground_NSP, but no submit)."""
        inputs = AttributeDict()
        #[1] Base settings (for potential_mapping , potential_family , kpoints )
        inputs.update(self.exposed_inputs(self._vasp_workchain))
        inputs.clean_workdir = Bool(False)

        #[2] Incar parameters
        incar = {"incar": {"ediff": 1e-7, "algo": "Normal","nelm": 200,
                           "ismear": 0  , "sigma": 0.02,
                           "prec": "Accurate",
                           "lmaxmix": 4,
                           "loptics": ".TRUE.",             }        }
        if "encut" in self.inputs.ns_parameters:    incar["incar"]["encut"] = self.inputs.ns_parameters.encut.value
        if "nbands" in self.inputs.ns_parameters:   incar["incar"]["nbands"] = self.inputs.ns_parameters.nbands.value
        if self.ctx.is_spinpol:
            # kept from your code; MBSE SP path still not implemented
            _,incar["incar"]["magmom"] = input_magnetic_moment_tomagmom( self.inputs.structure, self.inputs.ns_parameters.magnetic_moment_onsite.get_dict(), )
            incar["incar"]["ispin"] = 2
            incar["incar"]["icharg"] = 1
            incar["incar"]["lorbit"] = 11
            incar["incar"]["amix_mag"] = 0.8
            incar["incar"]["bmix_mag"] = 1e-5
            incar["incar"]["amix"] = 0.2
            incar["incar"]["bmix"] = 1e-5
        
        #[2.1] parameters regarding HDF5 use
        if ("use_hdf5" in self.inputs.ns_reference) and self.inputs.ns_reference.use_hdf5.value :
            incar["incar"]["lh5"]      = ".TRUE."
            incar["incar"]["lwaveh5"]  = ".TRUE."
            incar["incar"]["lchargh5"] = ".TRUE."
            incar["incar"]["lwave"]    = ".FALSE."
            incar["incar"]["lcharg"]   = ".FALSE."
        else:
            incar["incar"]["lh5"]      = ".FALSE."
            incar["incar"]["lwaveh5"]  = ".FALSE."
            incar["incar"]["lchargh5"] = ".FALSE."
            incar["incar"]["lwave"]    = ".TRUE."
            incar["incar"]["lcharg"]   = ".TRUE."

        #[2.2] Finalize incar
        inputs.parameters = incar

        #[3] Parser settings 
        settings = AttributeDict({ "parser_settings": {"include_node": ["bands", "kpoints", "structure", "maximum_number_pw"]} })
        inputs.settings = settings
        
        #[4] Scheduler options
        inputs.options = self.__build_options_entry()
        return inputs
        
    def __prepare_inputs_mBSE_base(self):
        """helper to construct a Python dict (and not AiiDA Dict) for the settings + scheduler options"""
        
        #[0] starting checks
        if self.ctx.is_spinpol:
            raise ValueError("MAGN_NOT_IMPLEMENTED")
        if self.ctx.state_WC.restart_folders.for_MBSE is None:
            raise RuntimeError("MBSE pending but restart folder is None (DFT did not produce remote_folder?)")

        #[1] Base settings (for potential_mapping , potential_family , kpoints )
        inputs = AttributeDict()
        inputs.update(self.exposed_inputs(self._vasp_mbse_workchain))
        inputs.clean_workdir     = Bool(False)
        inputs.keep_last_workdir = Bool(True)


        #[2] Parser settings
        inputs.settings = AttributeDict()
        inputs.settings["parser_settings"] = { "include_node": ["kpoints", "dielectrics", "opticaltransitions"] ,
                                               "exclude_node": ["bands"],  }
        #[2.1] Additional settings for the RETRIEVE_LIST
        inputs.settings["ADDITIONAL_RETRIEVE_LIST"] = [  "BSEFATBAND", "vaspout.h5", "_aiidasubmit.sh", "POSCAR", "POTCAR", "KPOINTS",
            "INCAR", ]

        #[2.2] Setting for the Restart folder : the mBSE should restart from the WAVEDER / WAVECAR (or equivalenty WAVEDER+vaspwave) 
        inputs.restart_folder = self.ctx.state_WC.restart_folders.for_MBSE
        inputs.settings["ADDITIONAL_REMOTE_COPY_LIST"] = ["CONTCAR","CHGCAR","WAVECAR","WAVEDER"]
        if ("use_hdf5" in self.inputs.ns_reference) and self.inputs.ns_reference.use_hdf5.value :
            inputs.settings["ADDITIONAL_REMOTE_COPY_LIST"].extend(["vaspwave.h5"])

        #[3] Options
        inputs.options = self.__build_options_entry()
        if 'extraresources_fallback_options' in self.inputs:
            inputs.extraresources_fallback_options = self.__build_options_entry(input_dict=self.inputs.extraresources_fallback_options)
        return inputs

    def __add_inputs_mBSE_incar(self, inputs):
        """Add mBSE INCAR (TDHF + screening + band window logic) and GPU heuristics."""
        #[Preliminary-1] Reconstruct DFT node from stored submitted list
        dft_node = self._last_wc_node("DFT")
        if dft_node is None or not dft_node.is_finished_ok:
            raise RuntimeError("Cannot prepare mBSE: no successful DFT node found.")
        
        #[Preliminary-2] Are GPU used for this run? Several optimization options (and options for the BSE matrix)
        # later change heavily based on this
        num_GPU_perNode = 0
        opts = self.inputs.options.get_dict()
        if "custom_scheduler_commands" in opts:
            tokens = opts["custom_scheduler_commands"].replace("=", ":").split(":")
            if "gpu" in tokens:
                i = tokens.index("gpu")
                num_GPU_perNode = int(tokens[i + 1])
        
        #[1] Base incar
        incar = {"incar": { "ismear": 0, "sigma": 0.02,
                            "prec": "NORMAL",
                            "algo": "TDHF", "antires": 0,  "lmodelhf": ".TRUE.", }    }

        #[2] Magnetic stuff
        incar["incar"]["ispin"]  = 2 if self.ctx.is_spinpol else 1
        
        #[3] Nbands-Encut stuff
        incar["incar"]["nbands"] = np.shape(dft_node.outputs.bands.get_bands())[1]
        if "encut" in self.inputs.ns_parameters:
            incar["incar"]["encut"] = self.inputs.ns_parameters.encut.value

        #[4] Screening from inputs
        incar["incar"]["aexx"]     = self.inputs.ns_BSE.static_inverse_diel.value
        incar["incar"]["hfscreen"] = self.inputs.ns_BSE.screening_parameter.value

        #[5.1] BSE: base
        incar["incar"]["ibse"] = self.inputs.ns_parameters.ibse.value
        if "nbseeig" in self.inputs.ns_parameters:
            incar["incar"]["nbseeig"] = self.inputs.ns_parameters.nbseeig.value
        else:
            incar["incar"]["nbseeig"] = 20


        #[5.2] BSE: manage the parameters used to construct the BSE matrix : NBANDSV, NBANDSO, OMEGAMAX, 
        # We first start from an internal estimation via _determine_BSE_parameters
        # _determine_BSE_parameters takes the bands DFT and determine how many valence/conduction bands must be included to consider
        # all transitions (at IPA level) below energy_window_goal ; G0W0_gap is also included to approximate G0W0 bands
        input_G0W0_gap      = self.inputs.ns_BSE.G0W0_gap.value if "G0W0_gap" in self.inputs.ns_BSE else None
        input_optical_enwin = self.inputs.ns_BSE.optical_energy_window.value if "optical_energy_window" in self.inputs.ns_BSE else None
        BSE_params_estimated = _determine_BSE_parameters(
                                    bandsdata=dft_node.outputs.bands       ,
                                    energy_window_goal=input_optical_enwin ,
                                    G0W0_gap=input_G0W0_gap                )
        self.ctx.log = BSE_params_estimated["log"]
        incar["incar"]["nbandso"] = BSE_params_estimated["NBANDSO"]
        incar["incar"]["nbandsv"] = BSE_params_estimated["NBANDSV"]
        #Regarding OMEGAMAX: https://www.vasp.at/wiki/index.php/Category:Bethe-Salpeter_equations
        #When running BSE calculations on GPUs, we recommend not setting OMEGAMAX or setting it to a larger value so that 
        #all the bands selected in NBANDSV and NBANDSO are included in the kernel. Otherwise, additional data transfers 
        #between CPU and GPU might be required, which leads to a serious performance degradation on GPUs. 
        if num_GPU_perNode == 0:
            incar["incar"]["omegamax"] = BSE_params_estimated["OMEGAMAX"]
        
        #Now let's manage the overrides/optimization
        self.ctx.log +=  ("\n"+ "    [Override/Optimization section]")
        if "OMEGAMAX" in self.inputs.get("ns_BSE", {}):
            incar["incar"]["omegamax"] = self.inputs.ns_BSE.OMEGAMAX.value
            if num_GPU_perNode > 0:
                self.ctx.log += ("\n\nBIG WARNING: it's adviced to avoid setting OMEGAMAX (or setting to a value that includes all transitions defined by"
                                 "NBANDSV/NBANDSO) when GPU are used, see https://www.vasp.at/wiki/index.php/Category:Bethe-Salpeter_equations ."
                                 "I will continue, BUT I HOPE YOU KNOW WHAT ARE YOU DOING!\n\n")
        if ("NBANDSV" in self.inputs.ns_BSE) or ("NBANDSO" in self.inputs.ns_BSE):
            if ("NBANDSV" in self.inputs.ns_BSE) and ("NBANDSO" in self.inputs.ns_BSE):
                incar["incar"]["nbandso"] = self.inputs.ns_BSE.NBANDSO.value
                incar["incar"]["nbandsv"] = self.inputs.ns_BSE.NBANDSV.value
                self.ctx.log +=  ("\n"+f"     > Override: NBANDSO/V from workchain input : {incar["incar"]["nbandso"]}/{incar["incar"]["nbandsv"]}" )
            else:
                raise ValueError("ns_BSE.NBANDSV and ns_BSE.NBANDSO must be both set or both unset.")

        #[5.3] BSE: Another important flag involved in the construction of the BSE matrix : PRECFOCK
        # Default behavior follows https://vasp.at/wiki/Best_practices_for_Bethe-Salpeter_calculations
        #In large cells, the FFTs may take up the majority of the time in the calculation of the matrix elements, and 
        #reducing the FFT grid can largely speed up the calculation. For the large cells, even low precision can be found
        #sufficiently accurate, but the convergence with PRECFOCK must be investigated for each system.
        #where threshold for considering a system "large" is threshold_cell_volume_for_PRECFOCK.
        #set_PRECFOCK_to_Fast overrides this default: if it is set_PRECFOCK_to_Fast=True it's always set to Fast, if it's
        #False it's always left to Normal (and thus not defined)
        #In internal tests PRECFOCK is almost always useful with negligible cost in term of precision also for smaller cell,
        # but let's stick to the wiki
        threshold_cell_volume_for_PRECFOCK = 250
        if self.inputs.structure.get_cell_volume() > threshold_cell_volume_for_PRECFOCK :
             incar['incar']['precfock'] = "Fast"
             self.ctx.log +=  ("\n"+f"     > Optimization:  Cell volume is > threshold : {self.inputs.structure.get_cell_volume()} > {threshold_cell_volume_for_PRECFOCK} : automatically set precfock to fast!")
        #Now let's manage the override
        if ("set_PRECFOCK_to_Fast" in self.inputs.ns_optimization) and self.inputs.ns_optimization.set_PRECFOCK_to_Fast.value : 
            incar['incar']['precfock'] = "Fast" 
            self.ctx.log +=  ("\n"+f"     > Override: precfock flag from workchain input : set precfock to fast!")

        #[5.4] BSE : optimization options
        if ("lreal" in self.inputs.ns_optimization) and self.inputs.ns_optimization.lreal.value : 
            incar["incar"]["lreal"] = "Auto"
            self.ctx.log +=  ("\n"+f"     > Optimization: Setting Lreal=Auto; this may help reduce the memory space occupied by projectors.")
        
        #This follows the advice on https://vasp.at/wiki/Best_practices_for_Bethe-Salpeter_calculations
        # i.e. KPAR=num of GPUs. 
        #Given that NCCL for VASP imposes #mpiranks = #gpus, this means that all wavefunctions are stored on every MPI rank, 
        #which eliminates the need to send/receive the orbitals during the calculation of the matrix elements.
        if num_GPU_perNode > 0:
            # KPAR = num GPUs per node * num nodes
            num_nodes = inputs.options.get_dict()["resources"]["num_machines"]
            incar["incar"]["kpar"] = num_GPU_perNode * num_nodes
            self.ctx.log +=  ("\n"+f"     > Optimization: Using a total of {num_GPU_perNode * num_nodes} GPUs : automatically se KPAR to #(total GPUs)")

        #[5.5] BSE scissor, if explicitly requested (no external QP correction supplied)
        if self.inputs.ns_BSE.use_scissor.value:
            incar["incar"]["scissor"] = BSE_params_estimated["SCISSOR"]
            self.ctx.log +=  ("\n"+f"     > Override: ns_BSE.use_scissor is set, setting SCISSOR to {incar["incar"]["scissor"]}")
        self.report(self.ctx.log)
        
        #[6] parameters regarding HDF5 use
        if ("use_hdf5" in self.inputs.ns_reference) and self.inputs.ns_reference.use_hdf5.value :
            incar["incar"]["lh5"]      = ".TRUE."
            incar["incar"]["lwaveh5"]  = ".TRUE."
            incar["incar"]["lchargh5"] = ".TRUE."
            incar["incar"]["lwave"]    = ".FALSE."
            incar["incar"]["lcharg"]   = ".FALSE."
        else:
            incar["incar"]["lh5"]      = ".FALSE."
            incar["incar"]["lwaveh5"]  = ".FALSE."
            incar["incar"]["lchargh5"] = ".FALSE."
            incar["incar"]["lwave"]    = ".TRUE."
            incar["incar"]["lcharg"]   = ".TRUE."

        #[7] Finalize incar
        inputs.parameters = incar
        return inputs

    def elaborate_results(self):
        """Collect outputs from the final mBSE workchain and optionally copy files locally."""
        mbse_node = self._last_wc_node("MBSE")
        if mbse_node is None or not mbse_node.is_finished_ok:
            raise RuntimeError("Cannot elaborate results: no successful MBSE workchain found: (mBSE node).is_finished_ok is FALSE!")

        #expose outputs ---
        self.out("dielectrics" , mbse_node.outputs.dielectrics  )
        
        #Optional output (depends on IBSE)
        #We add an if because calculations determined with iterative methods (IBSE=1 and IBSE=3)
        if "opticaltransitions" in mbse_node.outputs: 
            self.out("opticaltransitions" , mbse_node.outputs.opticaltransitions )

        # --- copy retrieved folder locally ---
        if ("copy_result_locally" in self.inputs.ns_option) and self.inputs.ns_option.copy_result_locally.value:
            kmesh     = mbse_node.inputs.kpoints.get_kpoints_mesh()[0]
            kmesh_str = "".join(str(k) for k in kmesh)
            tag = str(self.inputs.ns_option.calculation_tag.value or "").strip() if "calculation_tag" in self.inputs.ns_option else ""
            foldername = f"3.1_mBSE_k{kmesh_str}_id{self.pid}{tag}"

            target_dir = ""
            if "copy_result_locally_path" in self.inputs.ns_option:
                target_dir = str(self.inputs.ns_option.copy_result_locally_path.value or "").strip()
            if not target_dir:
                target_dir = os.getcwd()
                self.report(
                    "WARNING: ns_option.copy_result_locally_path is not set - falling back to "
                    f"os.getcwd()={target_dir!r}. This is unreliable if this step runs inside a "
                    "daemon worker (its cwd need not match the directory the submit script was "
                    "launched from). Set ns_option.copy_result_locally_path explicitly in the "
                    "submit script to avoid results landing somewhere unexpected."
                )
            full_foldername = os.path.join(target_dir, foldername)
            os.makedirs(full_foldername, exist_ok=True)
    
            mbse_node.outputs.retrieved.copy_tree(full_foldername)
    
    @staticmethod 
    def __generate_compact_submission_string( wc_node , prefix="  > " , include_BSE_parameters=False ):
           """Emit a compact input summary right before submitting a calculation. """
           # --- basic electronic parameters ---
           def _get_incar_par(aiida_dict, key):
               try:
                   return aiida_dict.get_dict()['incar'][key]
               except Exception:
                   return None
           def __fmt_float(x, nd=3):
               try:              return f"{float(x):.{nd}f}"
               except Exception: return str(x) 
               
           encut     = __fmt_float( _get_incar_par(wc_node.inputs.parameters, "encut") )
           nbands    = _get_incar_par(wc_node.inputs.parameters, "nbands")
           encut_chi = _get_incar_par(wc_node.inputs.parameters, "encutgw")
           kpar      = _get_incar_par(wc_node.inputs.parameters, "kpar")        
           if include_BSE_parameters :
               ibse     =  _get_incar_par(wc_node.inputs.parameters, "ibse")
               bseprec  =  _get_incar_par(wc_node.inputs.parameters, "bseprec")
               nbandso  =  _get_incar_par(wc_node.inputs.parameters, "nbandso")
               nbandsv  =  _get_incar_par(wc_node.inputs.parameters, "nbandsv")
               omegamax = _get_incar_par(wc_node.inputs.parameters, "omegamax") 
               precfock = _get_incar_par(wc_node.inputs.parameters, "precfock") 
               scissor  =  _get_incar_par(wc_node.inputs.parameters, "scissor")
               aexx     =  _get_incar_par(wc_node.inputs.parameters, "aexx")
               hfscreen =  _get_incar_par(wc_node.inputs.parameters, "hfscreen")
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
           lines =  [ f"{prefix}nbands={nbands}  encut={encut}  encut_chi={encut_chi}  kpar={kpar}","\n" ]
           lines += [ f"{prefix}kpts_mesh={mesh} : nkpts={nkpts}","\n",
                      f"{prefix}potcars_family={pot_family}  potcars_mapping={pot_mapping}",  ]
           if include_BSE_parameters :
               lines +=  [ f"{prefix}mBSE specific parameters:","\n",
                            "  Reminder of call order : VaspmBSEInitScriptWorkChain -> VaspWorkChainWithFallbacks -> VaspCalculation","\n"
                           f"{prefix}ibse={ibse}  nbandso={nbandso}  nbandsv={nbandsv}  omegamax={omegamax}  bseprec={bseprec}","\n",
                           f"{prefix}precfock={precfock}  kpar={kpar}","\n",
                           f"{prefix}screening approximation w/ model diel.function : aexx={aexx}  hfscreen={hfscreen}  ","\n",
                           f"{prefix}QPcorrection : is scissor approximation used? scissor={scissor}  ","\n",
                           ]
           return ("".join(lines))
             
    def _last_wc_node(self, calc_type):
        """Return last submitted node for a given calc_type ('DFT' or 'MBSE')."""
        try:
            return self.ctx.state_WC.submitted[calc_type][-1]
        except Exception:
            return None
