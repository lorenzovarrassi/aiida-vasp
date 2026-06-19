import importlib
import numpy as np
from copy import deepcopy
from aiida import orm
import os.path
from enum import Enum, auto
from aiida.orm import Code, Int, Float, Str, Dict, Bool , List , RemoteData , ArrayData , BandsData , XyData , SinglefileData
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine  import WorkChain, calcfunction , ToContext , append_ , submit, while_ , if_ , submit, run
from aiida_vasp.utils.workchains  import prepare_process_inputs
from aiida.common.extendeddicts   import AttributeDict
from aiida_vasp.utils.workchains  import site_magnetization_to_magmom
from .workchain_wrapper_VaspWorkchain_initscript import VaspInitScriptWorkChain
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
    1) vasp.vasp                     (DFT ground-state)
    2) VaspInitScriptWorkChain       (interpolation + BSE)

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
    
    Namespace ns_interpolation:
        G0W0_reference              : RemoteData     (REQUIRED)
        nbandsgw_to_interpolate     : Int            (optional but critical if interpolation used)
                Number of GW bands used for interpolation.
        local_initscript            : SinglefileData (optional)
                Python script copied into the remote sandbox as 'script_init.py'.
                Executed BEFORE VASP to perform interpolation of GW corrections.
        gw_reference_filename      : Str
                Filename ofhte OUTCAR inside either FolderData or RemoteData that 
                will be passed to the interpolation script.
        [ GW reference source (two mutually exclusive branches)] 
        remote_gw_reference_folder    : RemoteData (optional)
                Path on the FODLER ON THE REMOTE COMPUTER where the GW results reside.
                This folder must already contain the OUTCAR / vasprun used for QP data.
                The interpolation script will read them from that remote folder without copying.
        local_gw_reference_folder     : Str (optional)
                Absolute path on LOCAL computer (where the AiiDA daemon runs) containing 
                OUTCAR / vasprun. These files will be copied into the remote sandbox 
                before execution and passed to the interpolation from that.
  
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
   
    Step 2: Interpolate GW corrections into WAVECAR OR apply a SCISSOR
            using a python script placed into the remote job folder (`script_init.py`)
    Step 3: Launch a BSE calculation with model screening parameters
              (AEXX, HFSCREEN) and with NBANDSV/NBANDSO built automatically
              via _determine_BSE_parameters()
    Done by prepare_run_interpolation_BSE:
        - Builds inputs for VaspInitScriptWorkChain
        - Defines parser settings (retrieve BSEFATBAND, vaspout.h5)
        - Determines restart_folder from the DFT step
        - Prepares interpolation arguments:
             The filename of the GW files (OUTCAR/WAVECAR) which will be used as reference
             The path where those filename reside (either locally or remotely)
             nbandsgw_dense : number of bands to interpolate QP corrections and apply
             interpolation_script (default: script_init.py)
        - Builds INCAR for BSE run:
             algo=TDHF, lmodelhf, nbseeig, ismear, prec
        - Fills AEXX, HFSCREEN
        - Calls _determine_BSE_parameters() to compute NBANDSV/O:
             1. NBANDSV/O determined as the mininum nunmber of v/c bands
                required to include all IPA transitions below optical_energy_window
             2. DFT bands are used; if G0W0_gap is passed a scissor is applied to
                DFT bands before determining all IPA transitions
        - Injects SCISSOR if interpolation script is disabled; require G0W0_gap
        - Passes prepend_text with python script call
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
    _vasp_initscript_workchain = VaspInitScriptWorkChain

    @classmethod
    def define(cls, spec):
            super(VaspmBSEInitScriptWorkChain, cls).define(spec) 

            spec.expose_inputs( cls._vasp_workchain            , exclude=('parameters','settings','options')) 
            spec.expose_inputs( cls._vasp_initscript_workchain , exclude=('parameters','settings','options')) 


            spec.input('ns_parameters.encut'                  , valid_type=Float , required=False , help='Cutoff energy for the wavefunction in eV. ENCUT variable in VASP.')  #ns stands for namespace
            spec.input('ns_parameters.nbands'                 , valid_type=Int   , required=False , help='Total number of bands included in the DFT and G0W0 runs. NBANDS variable in VASP.'  )   
            spec.input('ns_parameters.magnetic_moment_onsite' , valid_type=Dict  , required=False , help='Starting collinear on-site magnetic moment ; Syntax is {ElName:value}')
            spec.input("ns_parameters.ibse"                   , valid_type=Int   , required=False , default=lambda: Int(2),  help="Controls the BSE integration scheme. See https://vasp.at/wiki/IBSE")
            spec.input('ns_parameters.kpar'                   , valid_type=Int   , required=False , default=lambda: Int(1),  help="Parallelization across k-points. Defaults = number of GPUs if available.")          
            spec.input('ns_parameters.nbseeig'                , valid_type=Int   , required=False , default=lambda: Int(50), help="Number of BSE eigenvectors written to BSEFATBAND.")          


            path_interpolationscript_default = os.path.join( importlib.import_module('aiida_vasp').__path__[0] , "workchains/vMBPT/utils_interpolationclasses.v2.py")
            SFData_default = SinglefileData( file=path_interpolationscript_default ) 
            spec.input("ns_interpolation.local_initscript"           , valid_type=SinglefileData , required=False , default=lambda:SFData_default ,
                                                                       help=("A SinglefileData containing the interpolation python script - Copied to remote sandbox as script_init.py"
                                                                        +" - Default provided via SFData_default"))
            spec.input("ns_interpolation.use_interpolation"          , valid_type=Bool       , required=True , default=lambda:Bool(True) )
            spec.input("ns_interpolation.nbandsgw_to_interpolate"    , valid_type=Int        , required=False )
            spec.input("ns_interpolation.remote_gw_reference_folder" , valid_type=RemoteData , required=False ,
                                                                       help=("The GW OUTCAR / vasprun.xml references are inside a single folder on the remote machine " 
                                                                         +"- the RemoteData arguments points to that existing folder - Interpolation script will read from that location.") )
            spec.input("ns_interpolation.local_gw_reference_folder"  , valid_type=Str        , required=False ,
                                                                       help=("The GW OUTCAR / vasprun.xml references are inside a single folder on the local machine - the Str argument is "
                                                                          +"the absolute path of that existing folder - The file will be copied into the remote sandbox folder.") )
            spec.input("ns_interpolation.gw_reference_filename"      , valid_type=Str        , required=False  ,  
                                                                       help="The filename inside the RemoteData or FolderData that will be supplied to the interpolation script as reference for the QP corrections (must be an OUTCAR of a G0W0 calculation).")
            spec.input("ns_interpolation.python_sourcing_env_command", valid_type=Str        , required=True  , default=lambda:Str("source activate aiida-vasp"),
                                                                       help="Command which will be added to the jobscript - should load a venv/conda env which contains numpy - scipy - pymatgen - spglib")
            
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

            spec.input("options" , valid_type=Dict , required=True )
            spec.input("ns_option.copy_result_locally"    , valid_type=Bool       , required=False , default=lambda:Bool(True) )
            spec.input('ns_option.calculation_label'      , valid_type=Str        , required=False , default=lambda: Str("")     , help='The summary printed at the end will be labeled with this string.')
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
                                    "MBSE": VaspInitScriptWorkChain      ,   }

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
    
            # Add interpolation stage (mutates `inputs`, sets local_initscript, files, options.prepend_text)
            inputs = self.__prepare_inputs_G0W0interpolation(inputs)
    
            # Add INCAR (mutates `inputs.parameters`)
            inputs = self.__add_inputs_mBSE_incar(inputs)
    
            # Normalize namespaces expected by aiida-vasp wrappers
            self.ctx.inputs_finalized = prepare_process_inputs( inputs, namespaces=["calc", "dynamics", "verify", "local_files_to_copy_to_remote_submission_folder"],  )
            return
        # any other state: nothing to prepare
        return

    def __build_options_entry(self, prepend_text=""):
        """helper to construct a Python dict (and not AiiDA Dict) for the scheduler options
        starting from self.inputs.options."""
        input_opts = self.inputs.options.get_dict()
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
        inputs.update(self.exposed_inputs(self._vasp_initscript_workchain))
        inputs.clean_workdir     = Bool(False)
        inputs.keep_last_workdir = Bool(True)


        #[2] Parser settings
        inputs.settings = AttributeDict()
        inputs.settings["parser_settings"] = { "include_node": ["kpoints", "dielectrics", "opticaltransitions"] ,
                                               "exclude_node": ["bands"],  }
        #[2.1] Additional settings for the RETRIEVE_LIST
        inputs.settings["ADDITIONAL_RETRIEVE_LIST"] = [  "BSEFATBAND", "vaspout.h5", "_aiidasubmit.sh", "POSCAR", "POTCAR", "KPOINTS",
            "script_init.py", "INCAR", "CopiedFromLocal_OUTCAR_3", ]

        #[2.2] Setting for the Restart folder : the mBSE should restart from the WAVEDER / WAVECAR (or equivalenty WAVEDER+vaspwave) 
        inputs.restart_folder = self.ctx.state_WC.restart_folders.for_MBSE
        inputs.settings["ADDITIONAL_REMOTE_COPY_LIST"] = ["CONTCAR","CHGCAR","WAVECAR","WAVEDER"]
        if ("use_hdf5" in self.inputs.ns_reference) and self.inputs.ns_reference.use_hdf5.value :
            inputs.settings["ADDITIONAL_REMOTE_COPY_LIST"].extend(["vaspwave.h5"])

        #[3] Options
        inputs.options = self.__build_options_entry()
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

        #[5.5] BSE scissor if interpolation disabled
        if not self.inputs.ns_interpolation.use_interpolation.value:
            incar["incar"]["scissor"] = BSE_params_estimated["SCISSOR"]
            self.ctx.log +=  ("\n"+f"     > Override: Interpolation disabled from workchain input, setting SCISSOR to {incar["incar"]["scissor"]}")
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

    def __prepare_inputs_G0W0interpolation(self, inputs):
        """ This functions manages the interpolation related arguments.
        The interpolation script will be inserted inside the jobscript (via options.prepend_text ) and run just before the VASP executable
        - The first part of the function manages to copy the script file inside the remote folder where the job will be run 
        ( and renamed script_init.py ). 
        Internally it uses the input.settings['ADDITIONAL_LOCAL_COPY_LIST'] or input.settings['ADDITIONAL_REMOTE_COPY_LIST'] 
        to copy the interpolation script inside the folder on the remote cluster where the calculation will be run.
        - The second part manages the input to interpolation script.
        The interpolation stage requires two mandatory argument and one optional:
        1]path_dense_DFT_toInterp : path to the REMOTE folder with the WAVECAR which will be modified in place by adding
                                    interpolated QP correction to its eigenvalues (without modifying the orbitals);
                                    It's used because we want to apply to the current WAVECAR before launching VASP.
        2]path_sparse_GW :          path on the REMOTE CLUSTER where the folder with GW POSCAR, OUTCAR resides.
        Thus this second part will set the inputs and the flag to copy the files via 
          - local_initscript
          - optional local_files_to_copy_to_remote_submission_folder
        """
        flag_use_interp = self.inputs.ns_interpolation.use_interpolation.value
        flag_has_local  = "local_gw_reference_folder"  in self.inputs.ns_interpolation
        flag_has_remote = "remote_gw_reference_folder" in self.inputs.ns_interpolation

        #[Preliminary] Initial checks
        # First for what regards the Branch selection logic; the two branches are mutually excelusive
        if flag_use_interp and flag_has_local and flag_has_remote:
            raise ValueError("local_gw_reference_folder and remote_gw_reference_folder are mutually exclusive.")

        #[Preliminary] Creates the required dict
        args_interpolation = AttributeDict()
        # Ensure the dynamic namespace exists (even if we don’t use it)
        if "local_files_to_copy_to_remote_submission_folder" not in inputs:
            inputs.local_files_to_copy_to_remote_submission_folder = AttributeDict()


        #[1]the interpolation script
        #The workchain _vasp_initscript_workchain is simply a wrapper to Vasp2wInitScriptCalculation.
        #First thing, 'interpolation_script_remote_filename' is the name of the script inside the remote folder
        # convention from your CalcJob: local_init_script is staged as script_init.py
        args_interpolation['interpolation_script_remote_filename']= "script_init.py"
        #Then the SinglefileData inputs.local_initscript defines the script file on the local machine (i.e. the machine where
        #AiiDA and this workchain actually runs) which will be copied by the AiIDA-transport inside the remote folder.
        #self.inputs.ns_interpolation.local_initscript has a default value is the interpolation script
        #localed in aiida_vasp.workchains.vMBPT.utils_interpolationclasses.v2.py  
        #We simply pass that; if needed, it could be overriden.
        inputs.local_init_script = self.inputs.ns_interpolation.local_initscript

        #[1] nbandsgw - 1st argument for aiida_vasp.workchains.vMBPT.utils_interpolationclasses.v2.py  
        args_interpolation['nbandsgw_to_interpolate'] = (
                    self.inputs.ns_interpolation.nbandsgw_to_interpolate.value
                    if "nbandsgw_to_interpolate" in self.inputs.ns_interpolation else None )
        
        #[2]  #BRANCH A – LOCAL GW FOLDER (preferred)
        # Create a dynamic namespace: each key becomes a remote filename
        inputs.local_SinglefileData_tocopy_toremote = AttributeDict()

        if flag_use_interp and flag_has_local : 
            #The interpolation script extracts the QP correction from the OUTCAR of a G0W0 run (conventionally on a sparse k-mesh)
            #sparse_local_filename  is the name of the file on the local machine (i.e. the machine where
            #AiiDA and this workchain actually runs)
            
            local_gw_reference_file_name = ( self.inputs.ns_interpolation.gw_reference_filename.value
                                            if "gw_reference_filename" in self.inputs.ns_interpolation else 'OUTCAR.3'  )
            local_gw_reference_file_path  = self.inputs.ns_interpolation.local_gw_reference_folder.value
            local_gw_reference_file_abspath = os.path.join(local_gw_reference_file_path , local_gw_reference_file_name )
            #This to sanitize - and will be the name that the file has in the remote folder
            remote_file_name = "CopiedFromLocal_" + local_gw_reference_file_name.replace(".", "_")
            #Initializing a SinglefileData validates the existence of the file, this we do not need to do manually
            inputs.local_files_to_copy_to_remote_submission_folder[remote_file_name] = SinglefileData(file=local_gw_reference_file_abspath)
            args_interpolation['path_sparse_GW_remote_file_path'] = "./"
            args_interpolation['path_sparse_GW_remote_file_name'] = remote_file_name

        elif flag_use_interp and flag_has_remote:
            remote_folder_node    = self.inputs.ns_interpolation.remote_gw_reference_folder
            args_interpolation['path_sparse_GW_remote_file_path'] = remote_folder_node.get_remote_path()
            args_interpolation['path_sparse_GW_remote_file_name']  = ( self.inputs.ns_interpolation.gw_reference_filename.value
                                                                       if "gw_reference_filename" in self.inputs.ns_interpolation else 'OUTCAR'  )

        #The command TO RUN THE INTERPOLATION SCRIPT is added to inputs.options.prepend_text = str_prepend_command
        if flag_use_interp :
            sourcing_cmd = str(self.inputs.ns_interpolation.python_sourcing_env_command.value or "").strip()
            str_launch_command =( f"{sourcing_cmd}"+"\n"
                                   "python3 "                   +str(args_interpolation['interpolation_script_remote_filename'])+"  "
                                   "--path_sparse_GW "          +str(args_interpolation['path_sparse_GW_remote_file_name'])     +"  "
                                   "--sparse_GW_filename "      +str(args_interpolation['path_sparse_GW_remote_file_path'])     +"  "     
                                   "--path_dense_DFT_toInterp " +str("./")                                                      +"  "
                                   "--nbandsgw_dense "          +str(args_interpolation['nbandsgw_to_interpolate']) ) 
        else: str_launch_command = ""
        inputs.init_script_call_command = str_launch_command
        # Optional: store for later reporting/debugging
        self.ctx.args_interpolation = args_interpolation
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
            foldername = f"3.1_mBSE_k{kmesh_str}_id{self.pid}"
            full_foldername = os.path.join(os.getcwd(), foldername)
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
               prepend_text = "\n"+wc_node.inputs.init_script_call_command.value
               prepend_text = prepend_text.replace("\n","\n       ")
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
                            "  Reminder of call order : VaspmBSEInitScriptWorkChain -> VaspInitScriptWorkChain -> Vasp2wInitScriptCalculation","\n"    
                           f"{prefix}ibse={ibse}  nbandso={nbandso}  nbandsv={nbandsv}  omegamax={omegamax}  bseprec={bseprec}","\n",
                           f"{prefix}precfock={precfock}  kpar={kpar}","\n",
                           f"{prefix}screening approximation w/ model diel.function : aexx={aexx}  hfscreen={hfscreen}  ","\n",
                           f"{prefix}QPcorrection : is scissor approximation used? scissor={scissor}  ","\n",
                           f"{prefix}QPcorrection : prepend text for interpolation? {prepend_text}"
                           ]
           return ("".join(lines))
             
    def _last_wc_node(self, calc_type):
        """Return last submitted node for a given calc_type ('DFT' or 'MBSE')."""
        try:
            return self.ctx.state_WC.submitted[calc_type][-1]
        except Exception:
            return None
