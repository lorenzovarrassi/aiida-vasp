import importlib
import numpy as np
from copy import deepcopy
from aiida import orm
import os.path
from aiida.orm import Code, Int, Float, Str, Dict, Bool , List , RemoteData , ArrayData , BandsData , XyData , SinglefileData
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine  import WorkChain, calcfunction , ToContext , append_ , submit, while_ , if_ , submit, run
from aiida_vasp.utils.workchains  import prepare_process_inputs
from aiida.common.extendeddicts   import AttributeDict
from aiida_vasp.utils.workchains  import site_magnetization_to_magmom
from .workchain_wrapper_VaspWorkchain_initscript import VaspInitScriptWorkChain
from .utils_helpers_mBSE import determine_BSE_parameters
from .utils_calcfunctions import  input_magnetic_moment_tomagmom


from aiida import load_profile
load_profile()



class VaspmBSEInitScriptWorkChain(WorkChain):
    """
    [1] OVERVIEW OF THE PURPOSE
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
              via determine_BSE_parameters()
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
        - Calls determine_BSE_parameters() to compute NBANDSV/O:
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

            spec.expose_inputs( cls._vasp_workchain            , exclude=('parameters','settings',)) 
            spec.expose_inputs(cls._vasp_initscript_workchain  , exclude=('parameters','settings',)) 



            spec.input('ns_parameters.encut'                  , valid_type=Float , required=False , help='Cutoff energy for the wavefunction in eV. ENCUT variable in VASP.')  #ns stands for namespace
            spec.input('ns_parameters.nbands'                 , valid_type=Int   , required=False , help='Total number of bands included in the DFT and G0W0 runs. NBANDS variable in VASP.'  )   
            spec.input('ns_parameters.magnetic_moment_onsite' , valid_type=Dict  , required=False , help='Starting collinear on-site magnetic moment ; Syntax is {ElName:value}')
            spec.input("ns_parameters.ibse"                   , valid_type=Int   , required=False , default=lambda: Int(2),  help="Controls the BSE integration scheme. See https://vasp.at/wiki/IBSE")
            spec.input('ns_parameters.kpar'                   , valid_type=Int   , required=False , default=lambda: Int(1),  help="Parallelization across k-points. Defaults = number of GPUs if available.")          
            spec.input('ns_parameters.nbseeig'                , valid_type=Int   , required=False , default=lambda: Int(50), help="Number of BSE eigenvectors written to BSEFATBAND.")          

            spec.input("options" , valid_type=Dict)
            spec.input("copy_result_locally" , valid_type=Bool, required=False, default=lambda:Bool(True) )


            path_interpolationscript_default = os.path.join( importlib.import_module('aiida_vasp').__path__[0] , "workchains/vMBPT/utils_interpolationclasses.v2.py")
            #SinglefileData = DataFactory('core.singlefile')  #Calling directly SinglefileData had errors
            SFData_default = SinglefileData( file=path_interpolationscript_default ) 
            spec.input("ns_interpolation.local_initscript"         ,   valid_type=SinglefileData , required=False , default=lambda:SFData_default ,
                                                                       help=("A SinglefileData containing the interpolation python script - Copied to remote sandbox as script_init.py"
                                                                        +" - Default provided via SFData_default"))
            spec.input("ns_interpolation.use_interpolation"         ,  valid_type=Bool       , required=True , default=lambda:Bool(True) )
            spec.input("ns_interpolation.nbandsgw_to_interpolate"   ,  valid_type=Int        , required=False )
            spec.input("ns_interpolation.remote_gw_reference_folder",  valid_type=RemoteData , required=False ,
                                                                       help=("The GW OUTCAR / vasprun.xml references are inside a single folder on the remote machine " 
                                                                         +"- the RemoteData arguments points to that existing folder - Interpolation script will read from that location.") )
            spec.input("ns_interpolation.local_gw_reference_folder"  , valid_type=Str        , required=False ,
                                                                       help=("The GW OUTCAR / vasprun.xml references are inside a single folder on the local machine - the Str argument is "
                                                                          +"the absolute path of that existing folder - The file will be copied into the remote sandbox folder.") )
            spec.input("ns_interpolation.gw_reference_filename"      , valid_type=Str        , required=False  ,  
                                                                       help="A list of filename inside the RemoteData or FolderDat that will be supplied to the interpolation script.")
            spec.input("ns_interpolation.python_sourcing_env_command", valid_type=Str        , required=True  , default=lambda:Str("source activate aiida-vasp"),
                                                                       help="Command which will be added to the jobscript - should load a venv/conda env which contains numpy - scipy - pymatgen - spglib")
            
            spec.input("ns_BSE.static_inverse_diel" , valid_type=Float , required=True  , help='Required for analytic diagonal screening in mBSE.' )
            spec.input("ns_BSE.screening_parameter" , valid_type=Float , required=True  , help='Required for analytic diagonal screening in mBSE.' )

            spec.input("ns_BSE.G0W0_gap"            , valid_type=Float , required=False , help="G0W0 gap; required to determine SCISSOR")

            spec.input("ns_BSE.optical_energy_window" , valid_type=Float , required=False , help="Required for the automatic determination of the NBANDSV/NBANDSO given a target energy window")
            spec.input("ns_BSE.OMEGAMAX"              , valid_type=Float , required=False , help='Required for analytic diagonal screening in mBSE.' )
            spec.input("ns_BSE.NBANDSV"               , valid_type=Int   , required=False , help=('Force NBANDSV value in the INCAR; override the determination of NBANDSV via target energy window.' 
                                                                                                  'NBANDSV/NBANDSO should be passed together; cannot define only one of those two.'))
            spec.input("ns_BSE.NBANDSO"               , valid_type=Int   , required=False , help=('Force NBANDSO value in the INCAR; override the determination of NBANDSO via target energy window.' 
                                                                                                  'NBANDSV/NBANDSO should be passed together; cannot define only one of those two.'))
            spec.input("ns_BSE.set_PRECFOCK_to"       , valid_type=Bool  , required=False , help=('The use of Precfock=Fast depends on the cell dimension, Precfock=Fast is set if volume>350'
                                                                                                  'If True set PRECFOCK=Fast in the mBSE calculation; if False, always set it to default.'))

            spec.output("dielectrics"        , valid_type=ArrayData )
            spec.output("opticaltransitions" , valid_type=ArrayData )
            spec.exit_code(402,'MAGN_NOT_IMPLEMENTED', message='determine_BSE_parameters and reading CHGCAR magnetic non implemented.')


            spec.outline(
               cls.prepare_run_DFTground_NSP      ,
               #cls.prepare_run_DFTground_SP      ,
               cls.prepare_run_interpolation_BSE  ,
               cls.elaborate_results    ,
            )
            

    
    def prepare_run_DFTground_NSP(self):
            ##[PARTE 1: The Non-Spin-Polarized DFT ground-state]
            self.ctx.inputs_DFTgr_NSP = AttributeDict()
            self.ctx.inputs_DFTgr_NSP.update(self.exposed_inputs(self._vasp_workchain))
            self.ctx.inputs_DFTgr_NSP.clean_workdir=Bool(False)

            ##[Part 3][Defining INCAR]
            input_params = {'incar': {'ediff':1E-7 ,  'algo':"Normal" , 'ismear':0 , 'sigma':0.02 , 'prec':'Accurate' , 'nelm':200 , 'lmaxmix':4 , 'loptics':'.TRUE.'}}
            if ('encut'  in self.inputs['ns_parameters']):  input_params['incar']['encut']  = self.inputs.ns_parameters.encut.value
            if ('nbands' in self.inputs['ns_parameters']):  input_params['incar']['nbands'] = self.inputs.ns_parameters.nbands.value
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                _ , input_params['incar']['magmom'] = input_magnetic_moment_tomagmom(self.inputs.structure , self.inputs['ns_parameters']['magnetic_moment_onsite'].get_dict())
                input_params['incar']['ispin']  = 2
                input_params['incar']['icharg'] = 1
                input_params['incar']['lorbit'] = 11
                input_params['incar']['amix_mag'] = 0.8 ; input_params['incar']['bmix_mag'] = 0.00001
                input_params['incar']['amix'] = 0.2     ; input_params['incar']['bmix'] = 0.00001       
            self.ctx.inputs_DFTgr_NSP.parameters =  Dict( input_params )

            ##[Part 2: Defining settings ]
            self.ctx.inputs_DFTgr_NSP.settings = AttributeDict({'parser_settings': {'include_node': ['bands','kpoints','structure','maximum_number_pw']}})

            inputs_options = AttributeDict()
            inputs_options.account = self.inputs.options.get_dict()['account']
            inputs_options.qos     = self.inputs.options.get_dict()['qos']
            inputs_options.resources     = self.inputs.options.get_dict()['resources']
            inputs_options.queue_name    = self.inputs.options.get_dict()['queue_name']
            inputs_options.max_memory_kb = self.inputs.options.get_dict()['max_memory_kb']
            inputs_options.max_wallclock_seconds = self.inputs.options.get_dict()['max_wallclock_seconds']
            self.ctx.inputs_DFTgr_NSP.options = Dict( inputs_options )

            runningWC_DFTgr_NSP = self.submit( self._vasp_workchain , **self.ctx.inputs_DFTgr_NSP) 
            self.report('\n [Ground-State-1] launching DFT-groundState - NonSpinPolarized vasp.vasp workchain <{}> \n\n'.format(runningWC_DFTgr_NSP.pk))
            return ToContext( finishedWC_DFTgr_NSP=append_(runningWC_DFTgr_NSP) )            

    # def prepare_run_DFTground_SP(self):
    #     ##[PARTE 2: The Non-Spin-Polarized DFT ground-state]
    #     if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
    #         inputs_DFTgr_SP = Attri uteDict()
    #         inputs_DFTgr_SP.ns_option , inputs_DFTgr_SP.ns_parameters , inputs_DFTgr_SP.ns_reference = AttributeDict() , AttributeDict() , AttributeDict()
    #         inputs_DFTgr_SP.update(self.exposed_inputs(self._next_workchain))
    #         inputs_DFTgr_SP.clean_workdir=Bool(False)
    #     
    #         inputs_DFTgr_SP.ns_option.compute_dipole_transition_mat = Bool(True)  
    #         inputs_DFTgr_SP.ns_option.select_single_iteration = Bool(False) 
    #         inputs_DFTgr_SP.ns_option.select_algo_Exact       = Bool(False)  
    #         inputs_DFTgr_SP.ns_option.run_G0W0 = Bool(False)
    #
    #         inputs_DFTgr_SP.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite
    #
    #         inputs_DFTgr_SP.ns_reference.DFTgr_RemoteData = self.ctx.finishedWC_DFTgr_NSP[-1].outputs.RemoteData_DFT
    #
    #         dict_entry_options = AttributeDict()
    #         dict_entry_options.account = self.inputs.options.get_dict()['account']
    #         dict_entry_options.qos     = self.inputs.options.get_dict()['qos']
    #         dict_entry_options.resources     = self.inputs.options.get_dict()['resources']
    #         dict_entry_options.queue_name    = self.inputs.options.get_dict()['queue_name']
    #         dict_entry_options.max_memory_kb = self.inputs.options.get_dict()['max_memory_kb']
    #         dict_entry_options.max_wallclock_seconds = self.inputs.options.get_dict()['max_wallclock_seconds']
    #         inputs_DFTgr_SP.options = Dict( dict_entry_options )
    #
    #         self.ctx.inputs_DFTgr_SP = inputs_DFTgr_SP
    #         runningWC_DFTgr_SP = self.submit( WorkflowFactory('vasp.vasp') , **self.ctx.inputs_DFTgr_SP) 
    #         self.report('\n [Ground-State-1] launching DFT-groundState - SpinPolarized vasp.vasp workchain <{}> \n\n'.format(runningWC_DFTgr_SP.pk))
    #         return ToContext(finishedWC_DFTgr_SP=append_(runningWC_DFTgr_SP))                



 
    def prepare_run_interpolation_BSE(self):
            inputs = AttributeDict()
            inputs.clean_workdir=Bool(False)
            inputs.update(self.exposed_inputs( self._vasp_initscript_workchain))
            #inputs.update(self.exposed_inputs(self._next_workchain))

            #[1][ Define input.settings and restart from continuation ]## ------ # ------ # ------ # ------ # - ###      
            #     and stuff required for continuation from DFT ]
            inputs.settings = Dict()
            inputs.settings['parser_settings'] = {'include_node': ['kpoints','dielectrics','opticaltransitions'] ,
                                                  'exclude_node': ['bands'] }
            inputs.settings['ADDITIONAL_REMOTE_COPY_LIST'] = ['WAVEDER','CONTCAR'] 
            inputs.settings['ADDITIONAL_RETRIEVE_LIST']    = ['BSEFATBAND','vaspout.h5','_aiidasubmit.sh','POSCAR','POTCAR','KPOINTS','script_init.py','INCAR','CopiedFromLocal_OUTCAR_3'] 
            # PERSONALIZED
            # we also add the .h5 file; if not present, aiida will simply not retrieve it without errors

            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                DFT_lastWorkchain_node = self.ctx.finishedWC_DFTgr_SP[-1]
                return self.exit_codes.MAGN_NOT_IMPLEMENTED
            else:
                DFT_lastWorkchain_node = self.ctx.finishedWC_DFTgr_NSP[-1]
            inputs.restart_folder = DFT_lastWorkchain_node.outputs.remote_folder



            #[2][Define QP correction VIA Interpolation-related stuff OR SCISSOR ]## ------ # ------ # ---- ## -- ### 
            ##Internally it uses the input.settings['ADDITIONAL_LOCAL_COPY_LIST'] orinput.settings['ADDITIONAL_REMOTE_COPY_LIST'] 
            ##To copy the interpolation script inside the folder on the remote cluster where the calculation will be run.
            # The interpolation stage requires two mandatory argument and one optional:
            # 1]path_dense_DFT_toInterp : path to the REMOTE folder with the WAVECAR which will be modified in place by adding
            #                             interpolated QP correction to its eigenvalues (without modifying the orbitals);
            #                             It's used because we want to apply to the current WAVECAR before launching VASP.
            # 2]path_sparse_GW :          path on the REMOTE CLUSTER where the folder with GW POSCAR, OUTCAR resides.
            args_interpolation = AttributeDict()

            # Common parameter — always needed
            args_interpolation['nbandsgw_to_interpolate'] = (
                    self.inputs.ns_interpolation.nbandsgw_to_interpolate.value
                    if "nbandsgw_to_interpolate" in self.inputs.ns_interpolation else None   )
            args_interpolation['filename'] = (
                    self.inputs.ns_interpolation.gw_reference_filename.value
                    if "gw_reference_filename" in self.inputs.ns_interpolation else 'OUTCAR.3'  )
                    
            #BRANCH SELECTION LOGIC:  Mutually exclusive
            flag_use_interp = self.inputs.ns_interpolation.use_interpolation.value
            flag_has_local  = "local_gw_reference_folder"  in self.inputs.ns_interpolation
            flag_has_remote = "remote_gw_reference_folder" in self.inputs.ns_interpolation
            if flag_use_interp and flag_has_local and flag_has_remote: 
                raise ValueError( "ns_interpolation.local_gw_reference_folder/remote_gw_reference_folder are mutually exclusive." )
    
            #BRANCH A – LOCAL GW FOLDER (preferred)
            #   → user gives a local path + list of filename
            #   → files copied to remote sandbox
            #   → script uses:    --path_sparse_GW ./   and --gw_files "f1 f2 f3"
            if flag_use_interp and flag_has_local : 
                local_folder_path = self.inputs.ns_interpolation.local_gw_reference_folder.value
                
                # Create a dynamic namespace: each key becomes a remote filename
                inputs.local_files_tocopy_toremote = AttributeDict()
                
                # IMPORTANT: filename may be a string; treat it as a single-element list
                # otherwise a for over a single string may cycle over its characters.
                gw_files = [args_interpolation['filename']] 
                
                for fname in gw_files:
                    full_path = os.path.join(local_folder_path, fname)
                    if not os.path.exists(full_path): raise ValueError( f"File '{full_path}' does not exist (local_gw_reference_folder)"            )

                    # Key becomes the remote filename (safe, with no dots)
                    remote_key = "CopiedFromLocal_" + fname.replace(".", "_")

                    # Assign SinglefileData into the dynamic namespace
                    inputs.local_files_tocopy_toremote[remote_key] = SinglefileData(file=full_path)

                #Prepare interpolation files - i.e. we need the OUTCAR G0W0 for the QP corrections
                args_interpolation['path_sparse_GW_remote'] = "./"
                args_interpolation['gw_file_string'] = None #Base value, will be redefined in the cycle
                for key,value in inputs.local_files_tocopy_toremote.items():
                    if "OUTCAR" in str(key):
                        args_interpolation['gw_file_string'] = str(key)

                ##Build the copy-map  { remote_filename : SinglefileData }
                ##We assume that all files references by args_interpolation['filename'] are inside a single folder
                ##at local_folder_path 
                #local_copy_dict = {}
                #for fname in [args_interpolation['filename']]:
                #    full_path = os.path.join(local_folder_path, fname)
                #    if not os.path.exists(full_path):raise ValueError(f"File '{full_path}' does not exist (local_gw_reference_folder).")
                #    key_dict = "CopiedFromLocal_"+fname.replace(".","_")
                #    local_copy_dict[key_dict] = SinglefileData(file=full_path)
                ##Provide mapping to Vasp2wInitScriptCalculation
                #inputs.local_files_tocopy_toremote = Dict(dict=local_copy_dict)
                ##In the remote sandbox the files are placed in "./"
                #args_interpolation['path_sparse_GW_remote'] = "./"   
                ##Script receives space-separated file list
                #args_interpolation['gw_file_string'] = " ".join(args_interpolation['filename'])
                    
                        
            #BRANCH B – REMOTE GW FOLDER (classic old behavior)
            #   → user provides a RemoteData pointing to the GW folder
            #   → script can read "OUTCAR.3" etc. directly on the cluster
            elif flag_use_interp and flag_has_remote :
                remote_folder_node = self.inputs.ns_interpolation.remote_gw_reference_folder
                args_interpolation['path_sparse_GW_remote'] = remote_folder_node.get_remote_path()
                # Pass filename (script will read them inside the remote path)
                args_interpolation['gw_file_string'] = " ".join(args_interpolation['filename'])


            #[3]the interpolation script (optional argument)
            #  The calcJob Vasp2wInitScriptCalculation receives a SinglefileData as input and add to the local_copy_list
            #  The SinglefileData is then copied inside the remote sandbox folder (where the calculation will be submitted 
            #   to slurm) and renamed to "script_init.py":
            #       [..]
            #       local_copy_list = []
            #       local_copy_list.append((self.inputs.local_initscript.uuid, self.inputs.local_initscript.filename, "script_init.py"))
            #       calcinfo.local_copy_list = local_copy_list
            #  The workchain _vasp_initscript_workchain is simply a wrapper to Vasp2wInitScriptCalculation
            #  This workchain takes self.inputs.ns_interpolation.local_initscript and passes to the Vasp2wInitScriptCalculation
            #   SinglefileData input. ns_interpolation.local_initscript's default value is the interpolation script
            #  localed in aiida_vasp.workchains.vMBPT.utils_interpolationclasses.v2.py - 
            #
            #  We need to pass the script name that will be run; as mentioned it's ALWAYS script_init.py + the local_pat
            args_interpolation['interpolation_script_remote_filename']= "script_init.py"
            inputs.local_initscript = self.inputs.ns_interpolation.local_initscript
            self.report("Using the provided local interpolation script:"+str(inputs.local_initscript) )

        
            #The command TO RUN THE INTERPOLATION SCRIPT is added to inputs.options.prepend_text = str_prepend_command
            if flag_use_interp :
                str_prepend_command =( str(self.inputs.ns_interpolation.python_sourcing_env_command.value) +"\n"
                                       "python3 "               +str(args_interpolation['interpolation_script_remote_filename'])   +"  "
                                       "--path_sparse_GW "      +str(args_interpolation['path_sparse_GW_remote'])   +"  "
                                       "--sparse_GW_filename "  +str(args_interpolation['gw_file_string'])     +"  "     
                                       "--path_dense_DFT_toInterp "  +str("./")                                 +"  "
                                       "--nbandsgw_dense "  +str(args_interpolation['nbandsgw_to_interpolate']) ) 
#                str_prepend_command =( "source activate aiida-vasp" +"\n"
#                                       "python3 "               +str(args_interpolation['interpolation_script_remote_filename'])   +"  "
#                                       "--path_sparse_GW "      +str(args_interpolation['path_sparse_GW_remote'])   +"  "
#                                       "--sparse_GW_filename "  +str(args_interpolation['gw_file_string'])     +"  "     
#                                       "--path_dense_DFT_toInterp "  +str("./")                                 +"  "
#                                       "--nbandsgw_dense "  +str(args_interpolation['nbandsgw_to_interpolate']) ) 
            else: str_prepend_command = ""
          
                
                


            #[2][Defining INCAR inputs]## ------ # ------ # ------ # ------ # ------ # ------ # ------ # ------ ###       
            incar = {'incar': {'ismear':0 , 'sigma':0.02 , 'prec':'NORMAL' , 'algo':'TDHF' , 'antires':0 , 'lmodelhf':'.TRUE.', 'nbseeig':50}  }
            incar['incar']['nbands'] = np.shape( DFT_lastWorkchain_node.outputs.bands.get_bands() )[1]  #bands array's dimensions are [#spin , #kpoints , #bands]
            if ('encut'                  in self.inputs['ns_parameters']):  incar['incar']['encut'] = self.inputs.ns_parameters.encut.value
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):  incar['incar']['ispin'] = 2
            else:   incar['incar']['ispin'] = 1
            if ('nbseeig'  in self.inputs['ns_parameters']):  incar['incar']['nbseeig']  = self.inputs.ns_parameters.nbseeig.value
            else:  incar['incar']['nbseeig']  = 20
            
            ##[Determine model-BSE flags : starting from the screening parameters]
            incar['incar']['aexx']     = self.inputs.ns_BSE.static_inverse_diel.value
            incar['incar']['hfscreen'] = self.inputs.ns_BSE.screening_parameter.value
            ##[Then parameters for the BSE matrix - if they are explicitly passed we use those, otherwise use an internal estimation]   
            # Determine_BSE_parameters takes the bands DFT and determine how many valence/conduction bands must be included to consider
            # all transitions (at IPA level) below energy_window_goal ; G0W0_gap is also included to approximate G0W0 bands
            input_G0W0_gap       = self.inputs.ns_BSE.G0W0_gap.value if "G0W0_gap" in self.inputs.ns_BSE else None
            input_optical_enwin  = self.inputs.ns_BSE.optical_energy_window.value if "optical_energy_window" in self.inputs.ns_BSE else None
            BSE_params_estimated = determine_BSE_parameters( bandsdata = DFT_lastWorkchain_node.outputs.bands , 
                                                             energy_window_goal = input_optical_enwin         ,
                                                             G0W0_gap=input_G0W0_gap                          )
            self.report( BSE_params_estimated['log'] )
        
            # Parameters are defined from BSE_params_estimated; if thet explicitly passed to input.arguments, override it:
            incar['incar']['ibse'] = self.inputs.ns_parameters.ibse.value
            incar['incar']['nbandso']  = BSE_params_estimated['NBANDSO']
            incar['incar']['nbandsv']  = BSE_params_estimated['NBANDSV']
            # Explicit NBANDSV/O - OMEGAMAX override via workchain ports ns_BSE_NBANDSV / ns_BSE_NBANDSO 
            if ('OMEGAMAX' in self.inputs.get('ns_BSE', {}))  : 
                incar['incar']['omegamax']  = self.inputs.ns_BSE.OMEGAMAX.value
            if ('NBANDSV' in self.inputs.ns_BSE) or ('NBANDSO' in self.inputs.ns_BSE):
                if ('NBANDSV' in self.inputs.ns_BSE) and ('NBANDSO' in self.inputs.ns_BSE):
                    incar['incar']['nbandso']  = self.inputs.ns_BSE.NBANDSO.value
                    incar['incar']['nbandsv']  = self.inputs.ns_BSE.NBANDSV.value
                else: 
                    raise ValueError( "ns_BSE_NBANDSV and ns_BSE_NBANDSO must be both set or both unset." )

            ##[We have not included the QP correction via interpolation in this case, let's use a simple SCISSOR]
            if not flag_use_interp : 
                incar['incar']['scissor'] = BSE_params_estimated['SCISSOR']
          
            ##[Lastly, optimization options]
            #We first need to understand if GPU are present or no, because optimization options change a lot.
            num_GPU_perNode = 0
            opts = self.inputs.options.get_dict() 
            if 'custom_scheduler_commands' in opts:
                tokens = opts['custom_scheduler_commands'].replace("=",":").split(":")
                if 'gpu' in tokens:
                        i = tokens.index('gpu')
                        num_GPU_perNode = int(tokens[i + 1])

            if num_GPU_perNode > 0:
                #This follows the advice on https://vasp.at/wiki/Best_practices_for_Bethe-Salpeter_calculations
                # i.e. KPAR=num of GPUs. 
                #Given that NCCL for VASP imposes #mpiranks = #gpus, this means that all wavefunctions are stored on every MPI rank, 
                #which eliminates the need to send/receive the orbitals during the calculation of the matrix elements.
                num_nodes = inputs.options.get_dict()['resources']['num_machines']
                incar['incar']['kpar'] = num_GPU_perNode * num_nodes

                #https://www.vasp.at/wiki/index.php/Category:Bethe-Salpeter_equations
                #When running BSE calculations on GPUs, we recommend not setting OMEGAMAX or setting it to a larger value so that 
                #all the bands selected in NBANDSV and NBANDSO are included in the kernel. Otherwise, additional data transfers 
                #between CPU and GPU might be required, which leads to a serious performance degradation on GPUs. 
                incar['incar'].pop('omegamax', None)
            else:
                #In our internal test OMEGAMAX had the strongest impact on the reduction of the BSE routine times.
                #We cannot set for GPUs, see above
                incar['incar']['omegamax'] = BSE_params_estimated['OMEGAMAX']





            threshold_cell_volume_for_PRECFOCK = 250
            # Default behaviour follows https://vasp.at/wiki/Best_practices_for_Bethe-Salpeter_calculations
            #In large cells, the FFTs may take up the majority of the time in the calculation of the matrix elements, and 
            #reducing the FFT grid can largely speed up the calculation. For the large cells, even low precision can be found
            #sufficiently accurate, but the convergence with PRECFOCK must be investigated for each system.
            #where threshold for considering a system "large" is threshold_cell_volume_for_PRECFOCK.
            #set_PRECFOCK_to overrides this default: if it is set_PRECFOCK_to=True it's always set to Fast, if it's
            #False it's always left to Normal (and thus not defined)
            #
            #In internal tests PRECFOCK is almost always useful with negligible cost in term of precision, but let's stick
            if ("set_PRECFOCK_to" in self.inputs.ns_BSE) :
                 if self.inputs.ns_BSE.set_PRECFOCK_to.value : incar['incar']['precfock'] = "Fast"    
            elif self.inputs.structure.get_cell_volume() > threshold_cell_volume_for_PRECFOCK :
                incar['incar']['precfock'] = "Fast"


            #[2][Final logging before submission]
            # ------------------------------------------------------------------

            str_log_final =  "\n [BSE Job Configuration Summary] Explicitly defined parameters (before submitting VaspInitScriptWorkChain):"
            str_log_final += "\n Remind: VaspmBSEInitScriptWorkChain -> calls VaspInitScriptWorkChain -> runs Vasp2wInitScriptCalculation\n"    
            kmesh = self.inputs.kpoints.get_kpoints_mesh()[0]
            str_log_final +=     f"   KPOINTS mesh      : {kmesh}\n"            
            if "encut" in self.inputs.ns_parameters:
                str_log_final += f"   ENCUT             : {self.inputs.ns_parameters.encut.value:.1f} eV\n"
            if "kpar" in incar["incar"]:
                str_log_final += f"   KPAR              : {incar['incar']['kpar']}\n"
            if "precfock" in incar["incar"]:
                str_log_final += f"   PRECFOCK          : {incar['incar']['precfock']}\n"
            if "ibse" in incar["incar"]:
                str_log_final += f"   IBSE              : {incar['incar']['ibse']}\n"    

            nvb = incar["incar"].get("nbandsv", None)
            ncb = incar["incar"].get("nbandso", None)
            if nvb is not None and ncb is not None:
                str_log_final += f"   NBANDSV / NBANDSO : {nvb} / {ncb}\n"
            if "scissor" in incar["incar"]:
                str_log_final += f"   SCISSOR           : {incar['incar']['scissor']:.3f} eV\n"
            str_log_final += (   f"   AEXX              : {incar['incar']['aexx']:.3f}\n"
                                 f"   HFSCREEN          : {incar['incar']['hfscreen']:.3f}\n"            )
            if "omegamax" in incar["incar"]:
                str_log_final += f"   OMEGAMAX          : {incar['incar']['omegamax']:.3f} eV\n"
            self.report(str_log_final)
            inputs.parameters = Dict( incar ) #convert to AiiDA format  




            #[4][Define input.options]### ------ # ------ # ------ # ------ # ------ # ------ # ------ # ------ ###       
            dict_entry_options = AttributeDict()
            dict_entry_options.account = self.inputs.options.get_dict()['account']
            dict_entry_options.qos           = self.inputs.options.get_dict()['qos']
            dict_entry_options.resources     = self.inputs.options.get_dict()['resources']
            dict_entry_options.queue_name    = self.inputs.options.get_dict()['queue_name']
            dict_entry_options.max_memory_kb = self.inputs.options.get_dict()['max_memory_kb']
            dict_entry_options.max_wallclock_seconds = self.inputs.options.get_dict()['max_wallclock_seconds']         
            dict_entry_options.prepend_text  = str_prepend_command
            if "custom_scheduler_commands" in self.inputs.options : 
                dict_entry_options.custom_scheduler_commands  = self.inputs.options.get_dict()['custom_scheduler_commands']        
            inputs.options = Dict( dict_entry_options )
            #inputs.metadata = AttributeDict()
            #inputs.metadata.options= Dict( dict_entry_options )

            self.ctx.inputs = prepare_process_inputs(inputs, namespaces=['dynamics','verify','local_files_tocopy_toremote'])
        
            runningProcessNode = self.submit( self._vasp_initscript_workchain , **self.ctx.inputs)
            self.report('launching {}<{}> '.format(self._vasp_initscript_workchain.__name__, runningProcessNode.pk))
            return ToContext(wk_DFT_interpolated_BSE=append_(runningProcessNode))


    def elaborate_results(self):
        self.out("dielectrics"        , self.ctx.wk_DFT_interpolated_BSE[-1].outputs.dielectrics        )
        #We add an if because calculations determined with iterative methods (IBSE=1 and IBSE=3)
        if "opticaltransitions" in self.ctx.wk_DFT_interpolated_BSE[-1].outputs:
            self.out("opticaltransitions" , self.ctx.wk_DFT_interpolated_BSE[-1].outputs.opticaltransitions )

        if self.inputs.copy_result_locally.value == True:
            self_kpt_mesh_concatenated = "".join( [str(kpt) for kpt in self.ctx.inputs.kpoints.get_kpoints_mesh()[0] ] )
            self_pid = str( self.pid )
            foldername = "3.1_mBSE_k"+self_kpt_mesh_concatenated +"_id"+self_pid
            full_foldername = os.path.join(os.getcwd(), foldername)
            os.makedirs( full_foldername , exist_ok=True)
    
            self.ctx.wk_DFT_interpolated_BSE[-1].outputs.retrieved.copy_tree( full_foldername )
    
