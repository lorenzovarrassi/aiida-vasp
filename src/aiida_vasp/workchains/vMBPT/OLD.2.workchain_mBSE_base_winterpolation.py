import numpy as np
from copy import deepcopy
from aiida import orm
import itertools
import scipy
import os.path
from io import StringIO
from aiida.orm import Code, Int, Float, Str, Dict, Bool , List , RemoteData , ArrayData , BandsData , XyData , SinglefileData
from aiida.orm import load_node
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine  import WorkChain, calcfunction , ToContext , append_ , submit, while_ , if_ , submit, run
from aiida.tools.data.array.kpoints import get_kpoints_path
from aiida_vasp.utils.workchains  import prepare_process_inputs
from aiida_vasp.utils.aiida_utils import get_data_class
from aiida.common.extendeddicts   import AttributeDict
from aiida_vasp.utils.workchains  import site_magnetization_to_magmom

from .workchain_wrapper_VaspWorkchain_initscript import VaspInitScriptWorkChain


import warnings

from aiida import load_profile
load_profile()



class VaspmBSEInitScriptWorkChain(WorkChain):
    _vasp_workchain = WorkflowFactory('vasp.vasp')
    _vasp_initscript_workchain = VaspInitScriptWorkChain

    @classmethod
    def define(cls, spec):
            super(VaspmBSEInitScriptWorkChain, cls).define(spec) 

            spec.expose_inputs( cls._vasp_workchain            , exclude=('parameters','settings','options')) 
            spec.expose_inputs(cls._vasp_initscript_workchain  , exclude=('parameters','settings','options')) 



            spec.input('ns_parameters.encut'                  , valid_type=Float      , required=False , help='Cutoff energy for the wavefunction in eV. ENCUT variable in VASP.')  #ns stands for namespace
            spec.input('ns_parameters.nbands'                 , valid_type=Int        , required=False , help='Total number of bands included in the DFT and G0W0 runs. NBANDS variable in VASP.'  )   
            spec.input('ns_parameters.magnetic_moment_onsite' , valid_type=Dict       , required=False , help='Starting collinear on-site magnetic moment ; Syntax is {ElName:value}')
            
            spec.input("options" , valid_type=Dict)

            path_interpolationscript_default = os.path.join( importlib.import_module('aiida_vasp').__path__[0] , "workchains/vMBPT/utils_interpolationclasses.v2.py")
            SFData_default = SingleFileData( file=path_interpolationscript_default ) 
            spec.input("scissor" , valid_type=Int , required=False)
            spec.input("ns_interpolation.G0W0_reference"      , valid_type=RemoteData     , required=True  )
            spec.input("ns_interpolation.remote_initscript"   , valid_type=RemoteData     , required=False )
            spec.input("ns_interpolation.local_initscript"    , valid_type=SinglefileData , required=False , default=lambda:SFData_default )
            spec.input("ns_interpolation.nbandsgw_to_interpolate" , valid_type=Int        , required=False )
            
            spec.input("ns_BSE.static_inverse_diel"  , valid_type=Float , required=True  , help='Required for analytic diagonal screening in mBSE.' )
            spec.input("ns_BSE.screening_parameter"  , valid_type=Float , required=True  , help='Required for analytic diagonal screening in mBSE.' )
            
            spec.input("ns_BSE.energy_window"        , valid_type=Float , required=False , help="Required for the automatic determination of the NBANDSV/NBANDSO given a target energy window")
            spec.input("ns_BSE.G0W0_gap"             , valid_type=Float , required=False , help="Required for the automatic determination of the NBANDSV/NBANDSO given a target energy window")
            spec.input("ns_BSE.OMEGAMAX"             , valid_type=Float , required=False , help='Required for analytic diagonal screening in mBSE.' )
            spec.input("ns_BSE_NBANDSV"              , valid_type=Int   , required=False , help='Alternative to the target energy window.' )
            spec.input("ns_BSE_NBANDSO"              , valid_type=Int   , required=False , help='Alternative to the target energy window.' )



            spec.output("dielectrics"        , valid_type=ArrayData )
            spec.output("opticaltransitions" , valid_type=ArrayData )

            spec.outline(
               cls.prepare_run_DFTground_NSP     ,
               #cls.prepare_run_DFTground_SP      ,
               cls.prepare_run_interpolation_BSE  ,
               cls.elaborate_results    ,

            )
            
            spec.exit_code(402,'ONE_OR_MORE_GW_FAILED',message='One or more GW calculations failed.')
            spec.exit_code(403,'NON_EXISTENT_MODE'    ,message='The inserted mode does not exist - Please choose between custom , final , standard , memory-conserving.')

    
    def prepare_run_DFTground_NSP(self):
            ##[PARTE 1: The Non-Spin-Polarized DFT ground-state]
            self.ctx.inputs_DFTgr_NSP = AttributeDict()
            self.ctx.inputs_DFTgr_NSP.update(self.exposed_inputs(self._vasp_workchain))
            self.ctx.inputs_DFTgr_NSP.clean_workdir=Bool(False)

            ##[Part 3][Defining INCAR]
            input_params = {'incar': {'ediff':1E-7 , 'algo':"Normal" , 'ismear':0 , 'sigma':0.02 , 'prec':'Accurate' , 'nelm':200 , 'lmaxmix':4 , 'loptics':'.TRUE.'}}
            if ('encut'  in self.inputs['ns_parameters']):  input_params['incar']['encut']  = self.inputs.ns_parameters.encut
            if ('nbands' in self.inputs['ns_parameters']):  input_params['incar']['nbands'] = self.inputs.ns_parameters.nbands
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
    
            def _determine_BSEmatrix_dimension( energyWindow_goal , G0W0_gap , bandsData_dense_DFTgr , max_bandsInMatrix=4 ):
                

                b_band = bandsData_dense_DFTgr.get_array('bands')
                b_occ  = bandsData_dense_DFTgr.get_array('occupations')
                
                #This is a workaround for the case where the highest occupied band is not the same for all k-points
                #idx_HO_forDifferentKpts is the index of the highest occupied band for each k-point
                #NOTE: for aiida_vasp 3.1.0  b_occ shape is (#bands , #kpoints)
                #      for aiida_vasp 4.1.0  b_occ shape is (#kpoints , #bands). 
                idx_HO_forDifferentKpts = [np.where(b_occ[idx_k,:-1] - b_occ[idx_k,1:] > 0)[0][0] for idx_k in range(np.shape(b_occ)[0])]
                if min(idx_HO_forDifferentKpts) == max(idx_HO_forDifferentKpts) : idx_HO =  min(idx_HO_forDifferentKpts)
                else:
                    warnings.warn('The highest occupied band is not the same for all k-points. Using the first k-point index as reference.')
                    idx_HO = idx_HO_forDifferentKpts[0]
         
                #bval_ho =max( array of the values of the band w/ index idx_HO for all k-points )
                #bcon_lu =min( array of the values of the band w/ index idx_HO+1 for all k-points )
                #Thus, bval_ho and bcon_lu are the absolute band energies of the valence band maximum and conduction band minimum respectively
                bval_ho = max( b_band[ : , idx_HO   ] )
                bcon_lu = min( b_band[ : , idx_HO+1 ] )  
                
                #For each of the max_bandsInMatrix valence bands below the highest occupied one, we determine the min and max band energies over all k-points
                #Similarly, for each of the max_bandsInMatrix conduction bands above the lowest unoccupied one, we determine the min and max band energies over all k-points 
                bval_eachb_min = [ min(b_band[ : , bVal_idx ])  for bVal_idx in range(idx_HO-max_bandsInMatrix+1 , idx_HO+1) ]
                bval_eachb_max = [ max(b_band[ : , bVal_idx ])  for bVal_idx in range(idx_HO-max_bandsInMatrix+1 , idx_HO+1) ]
                cval_eachb_min = [ min(b_band[ : , bCon_idx ])  for bCon_idx in range(idx_HO+1 , idx_HO+max_bandsInMatrix+1) ]
                cval_eachb_max = [ max(b_band[ : , bCon_idx ])  for bCon_idx in range(idx_HO+1 , idx_HO+max_bandsInMatrix+1) ]
                bval_eachb_min.reverse()
                bval_eachb_max.reverse()

                #Compute the deltas from the reference points
                bval_eachb_min_deltaFromHO = bval_ho - bval_eachb_min
                bval_eachb_max_deltaFromHO = bval_ho - bval_eachb_max
                cval_eachb_min_deltaFromLU = cval_eachb_min - bcon_lu
                cval_eachb_max_deltaFromLU = cval_eachb_max - bcon_lu


                minTransition_forEachCouple = [ G0W0_gap +cval_eachb_min_deltaFromLU[idx] +bval_eachb_max_deltaFromHO[idx] for idx in range (max_bandsInMatrix) ]
                minTransition_forEachCouple = np.array( minTransition_forEachCouple )

                return ( np.argmin( minTransition_forEachCouple < energyWindow_goal ) + 1 )
    
            inputs = AttributeDict()
            #inputs.update(self.exposed_inputs(self._next_workchain))
            inputs.update(self.exposed_inputs( self._vasp_initscript_workchain))
            inputs.clean_workdir=Bool(False)
        

            #[ Define input.settings ]
            inputs.settings = Dict()
            inputs.settings['parser_settings'] = {'include_node': ['kpoints','dielectrics','opticaltransitions'] ,
                                                  'exclude_node': ['bands'] }
            inputs.settings['ADDITIONAL_REMOTE_COPY_LIST'] = ['WAVEDER','CONTCAR'] 
            inputs.settings['ADDITIONAL_RETRIEVE_LIST']    = ['BSEFATBAND','vaspout.h5'] 
            # we also add the .h5 file; if not present, aiida will simply not retrieve it without errors

            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                DFT_lastWorkchain_node = self.ctx.finishedWC_DFTgr_SP[-1]
            else:
                DFT_lastWorkchain_node = self.ctx.finishedWC_DFTgr_NSP[-1]
                
            inputs.restart_folder = DFT_lastWorkchain_node.outputs.remote_folder


            ##[Define Interpolation-related stuff ]
            ## It may read the input.settings['ADDITIONAL_LOCAL_COPY_LIST'] to add the interpolation script to the local copy list
            ## It may read the input.settings['ADDITIONAL_REMOTE_COPY_LIST'] to add other files to the remote copy list
            args_interpolation = AttributeDict()
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                args_interpolation['folder_dense_DFTgr'] = self.ctx.finishedWC_DFTgr_SP[-1].outputs.remote_folder.get_remote_path()
            else:
                args_interpolation['folder_dense_DFTgr']  = self.ctx.finishedWC_DFTgr_NSP[-1].outputs.remote_folder.get_remote_path()
            args_interpolation['folder_sparse_G0W0ref']   = self.inputs.ns_interpolation.G0W0_reference.get_remote_path()   
            args_interpolation['nbandsgw_to_interpolate'] = self.inputs.ns_interpolation.nbandsgw_to_interpolate.value
            
            if ('remote_initscript' in self.inputs['ns_interpolation']):
                args_interpolation['interpolation_script']    = self.inputs.ns_interpolation.remote_initscript.get_remote_path()   
                self.report("Using the provided remote interpolation script:"+str(args_interpolation['interpolation_script']) )
            elif ('local_initscript' in self.inputs['ns_interpolation']):
                #inputs.settings['ADDITIONAL_LOCAL_COPY_LIST'] = List([ self.inputs['ns_interpolation']['local_initscript'] ])
                inputs.local_initscript = self.inputs.ns_interpolation.local_initscript
                args_interpolation['interpolation_script']    = "script_init.py"
            else:
                 args_interpolation['interpolation_script'] = None
                 self.report("WARNING: No interpolation script has been provided.")

            if args_interpolation['interpolation_script'] is not None:
                str_prepend_command =( "source activate aiida-vasp" +"\n"
                                        "python3 " +str(args_interpolation['interpolation_script'])                +"  "
                                        "--path_sparse_GW " +str(args_interpolation['folder_sparse_G0W0ref'])      +"  "
                                        "--path_dense_DFT_toInterp "  +str("./")                                   +"  "
                                        "--nbandsgw_dense " +str(args_interpolation['nbandsgw_to_interpolate'])  )#+"  " 
                                        #"--path_dense_DFT_reference " +str(args_interpolation['folder_dense_DFTgr']) )
)
                #TODO: WOULD BE MORE ROBUST TO USE ABSOLUTE PATH, i.e.
                #from pathlib import Path
                #str_prepend_command =( "source activate aiida-vasp" +"\n"
                #                        "python3 " +str(args_interpolation['interpolation_script'])                +"  "
                #                        "--path_sparse_GW " +str( Path(args_interpolation['folder_sparse_G0W0ref']).absolute()  )      +"  "
                #                        "--path_dense_DFT_toInterp "  +str("./")                                   +"  "
                #                        "--nbandsgw_dense " +str(args_interpolation['nbandsgw_to_interpolate'])  )#+"  " 
                #                        #"--path_dense_DFT_reference " +str(args_interpolation['folder_dense_DFTgr']) )
            else:   
                str_prepend_command = "echo 'WARNING:   No interpolation script has been provided. Skipping interpolation step.'"
            self.report("\nInterpolation script call:\n"+str_prepend_command+"\n\n")
    

            #[ Define input.options ]        
            dict_entry_options = AttributeDict()
            dict_entry_options.account = self.inputs.options.get_dict()['account']
            dict_entry_options.qos     = self.inputs.options.get_dict()['qos']
            dict_entry_options.resources     = self.inputs.options.get_dict()['resources']
            dict_entry_options.queue_name    = self.inputs.options.get_dict()['queue_name']
            dict_entry_options.max_memory_kb = self.inputs.options.get_dict()['max_memory_kb']
            dict_entry_options.max_wallclock_seconds = self.inputs.options.get_dict()['max_wallclock_seconds']         
            dict_entry_options.prepend_text = str_prepend_command
            inputs.options = Dict( dict_entry_options )
            #inputs.metadata = AttributeDict()
            #inputs.metadata.options= Dict( dict_entry_options )

            ##[Determine BSE properties]
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                num_band = _determine_BSEmatrix_dimension( self.inputs.ns_BSE.OMEGAMAX , self.inputs.ns_BSE.G0W0_gap , self.ctx.finishedWC_DFTgr_SP[-1].outputs.bands  )
            else:
                num_band = _determine_BSEmatrix_dimension( self.inputs.ns_BSE.OMEGAMAX , self.inputs.ns_BSE.G0W0_gap , self.ctx.finishedWC_DFTgr_NSP[-1].outputs.bands )
            
            
            ##[Defining INCAR inputs]             
            incar = {'incar': {'ismear':0 , 'sigma':0.02 , 'prec':'NORMAL' , 'algo':'TDHF' , 'antires':0 , 'lmodelhf':'.TRUE.', 'nbseeig':50}  }

            incar['incar']['nbands'] = np.shape( DFT_lastWorkchain_node.outputs.bands.get_bands() )[1]  #bands array's dimensions are [#spin , #kpoints , #bands]

            if ('encut' in self.inputs['ns_parameters']):   incar['incar']['encut']  = self.inputs.ns_parameters.encut.value
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):  incar['incar']['ispin'] = 2
            else:   incar['incar']['ispin'] = 1

            incar['incar']['nbandso']  = num_band 
            incar['incar']['nbandsv']  = num_band 
            incar['incar']['omegamax'] = self.inputs.ns_BSE.OMEGAMAX
            incar['incar']['aexx']     = self.inputs.ns_BSE.static_inverse_diel.value
            incar['incar']['hfscreen'] = self.inputs.ns_BSE.screening_parameter.value

            inputs.parameters = Dict( incar ) #convert to AiiDA format  
           
            self.ctx.inputs = prepare_process_inputs(inputs, namespaces=['dynamics','verify'])
            


            runningProcessNode = self.submit( self._vasp_initscript_workchain , **self.ctx.inputs)
            #self.report('launching {}<{}> '.format(self._next_workchain.__name__, runningProcessNode.pk))
            return ToContext(wk_DFT_interpolated_BSE=append_(runningProcessNode))

    def elaborate_results(self):

    
        self.out("dielectrics"        , self.ctx.wk_DFT_interpolated_BSE[-1].outputs.dielectrics        )
        self.out("opticaltransitions" , self.ctx.wk_DFT_interpolated_BSE[-1].outputs.opticaltransitions )

        self_kpt_mesh_concatenated = "".join( [str(kpt) for kpt in self.ctx.inputs.kpoints.get_kpoints_mesh()[0] ] )
        self_pid = str( self.pid )
        foldername = "3.1_mBSE_k"+self_kpt_mesh_concatenated +"_id"+self_pid
        full_foldername = os.path.join(os.getcwd(), foldername)
        os.makedirs( full_foldername , exist_ok=True)

        self.ctx.wk_DFT_interpolated_BSE[-1].outputs.retrieved.copy_tree( full_foldername )

