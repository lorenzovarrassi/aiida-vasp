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
from .workchain_base import VaspDFTGWWorkChain

import warnings

from aiida import load_profile
load_profile()



class VaspmBSEInterpolatedWorkChain(WorkChain):
    _next_workchain = WorkflowFactory('vasp.vasp')

    @classmethod
    def define(cls, spec):
            super(VaspmBSEInterpolatedWorkChain, cls).define(spec) 

            spec.expose_inputs(cls._next_workchain      , exclude=('parameters','settings','options')) 

            spec.input('ns_parameters.encut'                  , valid_type=Float      , required=False , help='cutoff energy for the wavefunction in eV. ENCUT variable in VASP.')  #ns stands for namespace
            spec.input('ns_parameters.nbands'                 , valid_type=Int        , required=False , help='total number of bands included in the DFT and G0W0 runs. NBANDS variable in VASP.'  )   
            spec.input('ns_parameters.magnetic_moment_onsite' , valid_type=Dict       , required=False , help='Starting collinear on-site magnetic moment ; Syntax is {ElName:value}')
            
            spec.input("options" , valid_type=Dict)

            spec.input("ns_interpolation.G0W0_reference"          , valid_type=RemoteData  , required=True )
            spec.input("ns_interpolation.interpolation_script"    , valid_type=RemoteData  , required=True )
            spec.input("ns_interpolation.nbandsgw_to_interpolate" , valid_type=Int         , required=False)
            
            spec.input("ns_BSE.static_inverse_diel"  , valid_type=Float , required=True )
            spec.input("ns_BSE.screening_parameter"  , valid_type=Float , required=True )   
            spec.input("ns_BSE.G0W0_gap"             , valid_type=Float , required=True )
            spec.input("ns_BSE.OMEGAMAX"             , valid_type=Float , required=True )

            #spec.output("dielectrics"        , valid_type=ArrayData )
            #spec.output("opticaltransitions" , valid_type=ArrayData )

            spec.outline(
               cls.prepare_run_DFTground_NSP     ,
               cls.prepare_run_DFTground_SP      ,
               cls.prepare_run_interpolation_BSE  ,
               cls.elaborate_results    ,

            )
            
            spec.exit_code(402,'ONE_OR_MORE_GW_FAILED',message='One or more GW calculations failed.')
            spec.exit_code(403,'NON_EXISTENT_MODE'    ,message='The inserted mode does not exist - Please choose between custom , final , standard , memory-conserving.')

    
    def prepare_run_DFTground_NSP(self):
            ##[PARTE 1: The Non-Spin-Polarized DFT ground-state]
            inputs_DFTgr_NSP = AttributeDict()
            inputs_DFTgr_NSP.ns_option , inputs_DFTgr_NSP.ns_parameters = AttributeDict() , AttributeDict()
            inputs_DFTgr_NSP.update(self.exposed_inputs(self._next_workchain))
            inputs_DFTgr_NSP.clean_workdir=Bool(False)
            
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                inputs_DFTgr_NSP.ns_option.compute_dipole_transition_mat = Bool(False) 
            else: 
                inputs_DFTgr_NSP.ns_option.compute_dipole_transition_mat = Bool(True) 
            inputs_DFTgr_NSP.ns_option.select_single_iteration = Bool(False) 
            inputs_DFTgr_NSP.ns_option.select_algo_Exact       = Bool(False)             
            inputs_DFTgr_NSP.ns_option.run_G0W0 = Bool(False)

            dict_entry_options = AttributeDict()
            dict_entry_options.account = self.inputs.options.get_dict()['account']
            dict_entry_options.qos     = self.inputs.options.get_dict()['qos']
            dict_entry_options.resources     = self.inputs.options.get_dict()['resources']
            dict_entry_options.queue_name    = self.inputs.options.get_dict()['queue_name']
            dict_entry_options.max_memory_kb = self.inputs.options.get_dict()['max_memory_kb']
            dict_entry_options.max_wallclock_seconds = self.inputs.options.get_dict()['max_wallclock_seconds']
            inputs_DFTgr_NSP.options = Dict( dict_entry_options )



            self.ctx.inputs_DFTgr_NSP = inputs_DFTgr_NSP
            runningWC_DFTgr_NSP = self.submit(VaspDFTGWWorkChain , **self.ctx.inputs_DFTgr_NSP) 
            self.report('\n [Ground-State-1] launching DFT-groundState - NonSpinPolarized vasp.vasp workchain <{}> \n\n'.format(runningWC_DFTgr_NSP.pk))
            return ToContext(finishedWC_DFTgr_NSP=append_(runningWC_DFTgr_NSP))            

    def prepare_run_DFTground_SP(self):
            ##[PARTE 2: The Non-Spin-Polarized DFT ground-state]
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                inputs_DFTgr_SP = AttributeDict()
                inputs_DFTgr_SP.ns_option , inputs_DFTgr_SP.ns_parameters , inputs_DFTgr_SP.ns_reference = AttributeDict() , AttributeDict() , AttributeDict()
                inputs_DFTgr_SP.update(self.exposed_inputs(self._next_workchain))
                inputs_DFTgr_SP.clean_workdir=Bool(False)
            
                inputs_DFTgr_SP.ns_option.compute_dipole_transition_mat = Bool(True)  
                inputs_DFTgr_SP.ns_option.select_single_iteration = Bool(False) 
                inputs_DFTgr_SP.ns_option.select_algo_Exact       = Bool(False)  
                inputs_DFTgr_SP.ns_option.run_G0W0 = Bool(False)

                inputs_DFTgr_SP.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite

                inputs_DFTgr_SP.ns_reference.DFTgr_RemoteData = self.ctx.finishedWC_DFTgr_NSP[-1].outputs.RemoteData_DFT

                dict_entry_options = AttributeDict()
                dict_entry_options.account = self.inputs.options.get_dict()['account']
                dict_entry_options.qos     = self.inputs.options.get_dict()['qos']
                dict_entry_options.resources     = self.inputs.options.get_dict()['resources']
                dict_entry_options.queue_name    = self.inputs.options.get_dict()['queue_name']
                dict_entry_options.max_memory_kb = self.inputs.options.get_dict()['max_memory_kb']
                dict_entry_options.max_wallclock_seconds = self.inputs.options.get_dict()['max_wallclock_seconds']
                inputs_DFTgr_SP.options = Dict( dict_entry_options )
   
                self.ctx.inputs_DFTgr_SP = inputs_DFTgr_SP
                runningWC_DFTgr_SP = self.submit(VaspDFTGWWorkChain , **self.ctx.inputs_DFTgr_SP) 
                self.report('\n [Ground-State-1] launching DFT-groundState - SpinPolarized vasp.vasp workchain <{}> \n\n'.format(runningWC_DFTgr_SP.pk))
                return ToContext(finishedWC_DFTgr_SP=append_(runningWC_DFTgr_SP))                
 



 
    def prepare_run_interpolation_BSE(self):
    
            def _determine_BSEmatrix_dimension( energyWindow_goal , G0W0_gap , bandsData_dense_DFTgr ):
                max_bandsInMatrix = 4

                #b_band = node_dense_DFTgr.outputs.bands_DFT.get_array('bands')
                #b_occ  = node_dense_DFTgr.outputs.bands_DFT.get_array('occupations')
                b_band = bandsData_dense_DFTgr.get_array('bands')
                b_occ  = bandsData_dense_DFTgr.get_array('occupations')
                
                #This is a workaround for the case where the highest occupied band is not the same for all k-points
                #idx_HO_forDifferentKpts is the index of the highest occupied band for each k-point
                idx_HO_forDifferentKpts = [np.where(b_occ[idx_k,:-1] - b_occ[idx_k,1:] > 0)[0][0] for idx_k in range(np.shape(b_occ)[1])]
                if min(idx_HO_forDifferentKpts) == max(idx_HO_forDifferentKpts) : idx_HO =  min(idx_HO_forDifferentKpts)
                else:
                    warnings.warn('The highest occupied band is not the same for all k-points. Using the first k-point index as reference.')
                    idx_HO = idx_HO_forDifferentKpts[0]
         
                #bval_ho =max( array of the values of the band w/ index idx_HO for all k-points )
                bval_ho = max( b_band[ : , idx_HO   ] )
                bcon_lu = min( b_band[ : , idx_HO+1 ] )   
                bval_eachb_min = [ min(b_band[ : , bVal_idx ])  for bVal_idx in range(idx_HO-max_bandsInMatrix+1 , idx_HO+1) ]
                bval_eachb_max = [ max(b_band[ : , bVal_idx ])  for bVal_idx in range(idx_HO-max_bandsInMatrix+1 , idx_HO+1) ]
                cval_eachb_min = [ min(b_band[ : , bCon_idx ])  for bCon_idx in range(idx_HO+1 , idx_HO+max_bandsInMatrix+1) ]
                cval_eachb_max = [ max(b_band[ : , bCon_idx ])  for bCon_idx in range(idx_HO+1 , idx_HO+max_bandsInMatrix+1) ]
                bval_eachb_min.reverse()
                bval_eachb_max.reverse()
                bval_eachb_min_deltaFromHO = bval_ho - bval_eachb_min
                bval_eachb_max_deltaFromHO = bval_ho - bval_eachb_max
                cval_eachb_min_deltaFromLU = cval_eachb_min - bcon_lu
                cval_eachb_max_deltaFromLU = cval_eachb_max - bcon_lu

                minTransition_forEachCouple = [ G0W0_gap +cval_eachb_min_deltaFromLU[idx] +bval_eachb_max_deltaFromHO[idx] for idx in range (max_bandsInMatrix) ]
                minTransition_forEachCouple = np.array( minTransition_forEachCouple )

                return ( np.argmin( minTransition_forEachCouple < energyWindow_goal ) + 1 )
    
            inputs = AttributeDict()
            inputs.update(self.exposed_inputs(self._next_workchain))
            inputs.clean_workdir=Bool(False)
    
            ##[Interpolation stuff]
            args_interpolation = AttributeDict()
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                args_interpolation['folder_dense_DFTgr'] = self.ctx.finishedWC_DFTgr_SP[-1].outputs.RemoteData_DFT.get_remote_path()
            else:
                args_interpolation['folder_dense_DFTgr']  =  self.ctx.finishedWC_DFTgr_NSP[-1].outputs.RemoteData_DFT.get_remote_path()
            args_interpolation['folder_sparse_G0W0ref']   = self.inputs.ns_interpolation.G0W0_reference.get_remote_path()   
            args_interpolation['interpolation_script']    = os.path.join( self.inputs.ns_interpolation.interpolation_script.get_remote_path() , '_module_interpolation_2024-02-18.py' )           
            args_interpolation['nbandsgw_to_interpolate'] = self.inputs.ns_interpolation.nbandsgw_to_interpolate.value
            print("\nInterpolation script call:")
            str_prepend_command =( "source activate aiida-vasp" +"\n"
                                   "python3 " +str(args_interpolation['interpolation_script'])           +"  "
                                   "--path_sparse_GW " +str(args_interpolation['folder_sparse_G0W0ref']) +"  "
                                   "--path_dense_DFT " +str(args_interpolation['folder_dense_DFTgr'])    +"  "
                                   "--nbandsgw_dense " +str(args_interpolation['nbandsgw_to_interpolate'])              )
            print(str_prepend_command,"\n\n")
    

            ##[Interpolation stuff][defining options in order to include interpolation command]
            dict_entry_options = AttributeDict()
            dict_entry_options.account = self.inputs.options.get_dict()['account']
            dict_entry_options.qos     = self.inputs.options.get_dict()['qos']
            dict_entry_options.resources     = self.inputs.options.get_dict()['resources']
            dict_entry_options.queue_name    = self.inputs.options.get_dict()['queue_name']
            dict_entry_options.max_memory_kb = self.inputs.options.get_dict()['max_memory_kb']
            dict_entry_options.max_wallclock_seconds = self.inputs.options.get_dict()['max_wallclock_seconds']         
            dict_entry_options.prepend_text = str_prepend_command
            inputs.options = Dict( dict_entry_options )
            
        
            ##[Determine BSE properties]
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                num_band = _determine_BSEmatrix_dimension( self.inputs.ns_BSE.OMEGAMAX , self.inputs.ns_BSE.G0W0_gap , self.ctx.finishedWC_DFTgr_SP[-1].outputs.bands_DFT  )
            else:
                num_band = _determine_BSEmatrix_dimension( self.inputs.ns_BSE.OMEGAMAX , self.inputs.ns_BSE.G0W0_gap , self.ctx.finishedWC_DFTgr_NSP[-1].outputs.bands_DFT )
            
            
            ##[Defining inputs]          
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                DFT_lastWorkchain_node = self.ctx.finishedWC_DFTgr_SP[-1]
            else:
                DFT_lastWorkchain_node = self.ctx.finishedWC_DFTgr_NSP[-1]
                       
    
            inputs.settings = Dict()
            inputs.settings['parser_settings'] = {'include_node': ['kpoints','dielectrics','opticaltransitions'] ,
                                                  'exclude_node': ['bands'] }
            inputs.settings['ADDITIONAL_REMOTE_COPY_LIST'] = ['WAVEDER','CONTCAR'] 
            inputs.settings['ADDITIONAL_RETRIEVE_LIST']    = ['BSEFATBAND','vaspout.h5'] 
            # we also add the .h5 file; if not present, aiida will simply not retrieve it without errors

            inputs.restart_folder = DFT_lastWorkchain_node.outputs.RemoteData_DFT
           
            incar = {'incar': {'ISMEAR':0 , 'SIGMA':0.02 , 'PREC':'NORMAL' , 'ALGO':'TDHF' , 'ANTIRES':0 , 'LMODELHF':'.TRUE.', 'NBSEEIG':50}  }

            incar['incar']['NBANDS'] = np.shape( DFT_lastWorkchain_node.outputs.bands_DFT.get_bands() )[1]  #bands array's dimensions are [#spin , #kpoints , #bands]
            
            
                
            if ('encut' in self.inputs['ns_parameters']):   incar['incar']['ENCUT']  = self.inputs.ns_parameters.encut.value
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):  incar['incar']['ISPIN'] = 2
            else:   incar['incar']['ISPIN'] = 1

            incar['incar']['NBANDSO']  = num_band 
            incar['incar']['NBANDSV']  = num_band 
            incar['incar']['OMEGAMAX'] = self.inputs.ns_BSE.OMEGAMAX
            incar['incar']['AEXX']     = self.inputs.ns_BSE.static_inverse_diel.value
            incar['incar']['HFSCREEN'] = self.inputs.ns_BSE.screening_parameter.value

            inputs.parameters = DataFactory('dict')(dict=incar) #convert to AiiDA format  
           
            self.ctx.inputs = prepare_process_inputs(inputs, namespaces=['dynamics','verify'])
            

            runningProcessNode = self.submit(self._next_workchain, **self.ctx.inputs)
            self.report('launching {}<{}> '.format(self._next_workchain.__name__, runningProcessNode.pk))
            return ToContext(wk_DFT_interpolated_BSE=append_(runningProcessNode))

    def elaborate_results(self):
        self.ctx.wk_DFT_interpolated_BSE
    
        #self.out("dielectrics"        , self.ctx.wk_DFT_interpolated_BSE[-1].outputs.dielectrics        )
        #self.out("opticaltransitions" , self.ctx.wk_DFT_interpolated_BSE[-1].outputs.opticaltransitions )

