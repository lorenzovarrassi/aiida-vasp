import numpy as np
from copy import deepcopy
from aiida import orm
import itertools
from aiida.orm import Code, Int, Float, Str, Dict, Bool , List , RemoteData , ArrayData , BandsData , XyData
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine  import WorkChain, calcfunction , ToContext , append_ , submit, while_ , if_ , submit, run
from aiida.tools.data.array.kpoints import get_kpoints_path
from aiida_vasp.utils.workchains  import prepare_process_inputs
from aiida_vasp.utils.aiida_utils import get_data_class
from aiida.common.extendeddicts   import AttributeDict
from aiida_vasp.utils.workchains  import site_magnetization_to_magmom
from workchain_BasisExtrapolation import input_magnetic_moment_tomagmom

import warnings

from aiida import load_profile
load_profile()

# the workflow achieves an automatic Wannerization of the calculation node specified by ns_reference.RemoteData.
# The starting projections are automatically determined using the selected columns of the density matrix (SCDM).
# This workchain executes a VASP simulation  with Wannier90 and outputs the wannierized bands.
class wkc_Wannier(WorkChain):
    _next_workchain = WorkflowFactory('vasp.vasp')

    @classmethod
    def define(cls, spec):
            super(wkc_Wannier, cls).define(spec) 

            spec.expose_inputs(cls._next_workchain      , exclude=('parameters','settings')) 
            
            spec.input('ns_parameters.magnetic_moment_onsite' , valid_type=Dict       , required=False , help='Starting collinear on-site magnetic moment ; Syntax is {ElName:value}')
            spec.input('ns_reference.RemoteData'              , valid_type=RemoteData ,                  help='the wavefunction (WAVECAR) to be Wannierized' )

            spec.output("wannier_bands"    , valid_type=ArrayData , help="Wannierized bands.")
            spec.output("bands_label_list" , valid_type=List      , help="bands label list, as printed by Wannier90.")

            spec.outline(
                cls.prepare_run_DFT_Wannierize     ,
                cls.elaborate_results
            )
            
            spec.exit_code(402,'ONE_OR_MORE_GW_FAILED',message='One or more GW calculations failed.')
            spec.exit_code(403,'NON_EXISTENT_MODE'    ,message='The inserted mode does not exist - Please choose between custom , final , standard , memory-conserving.')

 
    def prepare_run_DFT_Wannierize(self):
        input_DFT = AttributeDict()
        input_DFT.update(self.exposed_inputs(self._next_workchain))
        input_DFT.clean_workdir=Bool(False)
    
        #Define parser settings
        settings   = AttributeDict({'parser_settings': {}})
        dict_entry = {'add_bands': True , 'add_maximum_number_pw': True , 'add_ENMAXarray': True , 'add_NGarray': True , 'add_kpoints' : True , 'skip_parameters_validation':True}
        settings.parser_settings.update(dict_entry)
        input_DFT.settings = DataFactory('dict')(dict=settings)

        #make sure that the WAVECAR from the folder we passed as inputs is copied - it's what we want to Wannierize!
        input_DFT.restart_folder = self.inputs.ns_reference.RemoteData
        input_DFT.fileToIncludeFromRestartFolder = List(list=['WAVECAR'])

        #Get_kpoints_path is an internal function of AiiDA and serves for the Automatic computation of k-point paths
        #It's based on the seekpath tool by Giovanni Pizzi
        #The path is first saved in the out variable; then a string formatted in order to respect 
        #the syntax of the VASP incar is saved inside str_segment.
        out = get_kpoints_path( self.inputs.structure )['parameters'].get_dict()
        str_segment=""
        for segment in out['path']:
            str_segment = ( str_segment 
                        + " "+str(segment[0][0]) +" "+ str(out['point_coords'][segment[0]]).replace("[","").replace("]","").replace(",","") 
                        + " "+str(segment[1][0]) +" "+ str(out['point_coords'][segment[1]]).replace("[","").replace("]","").replace(",","") 
                        +"\n")

        #Define the basic INCAR flags; if the flag "magnetic_moment_onsite" is passed, we asked for a magnetic calculation
        if ("magnetic_moment_onsite" in self.inputs["ns_parameters"] ):
            incar = {'incar': { 'LORBIT':11 , 'ISMEAR':0 , 'SIGMA':0.02 , 'PREC':'NORMAL' , 'ISPIN':2 , 'ALGO':'None' , 'NELM':1}}
            _ , incar['incar']['MAGMOM'] = input_magnetic_moment_tomagmom(self.inputs.structure , self.inputs['ns_parameters']['magnetic_moment_onsite'].get_dict())
        else:
            incar = {'incar': { 'LORBIT':11 , 'ISMEAR':0 , 'SIGMA':0.02 , 'PREC':'NORMAL' , 'ISPIN':1 , 'ALGO':'None' , 'NELM':1}}
        #Define the INCAR flags referring to Wannier90
        incar['incar']['LWAVE']  =  '.FALSE.'    
        incar['incar']['LCHARG'] =  '.FALSE.' 
        incar['incar']['LWANNIER90_RUN'] = '.TRUE.'
        incar['incar']['LSCDM'] = '.TRUE.'
        incar['incar']['LWRITE_MMN_AMN'] = '.FALSE.'
        # incar['incar']['LWANNIER90_AUTO_WINDOW'] = '.TRUE.'
        
        #And finally add the INCAR flags referring to the kpoint_path calculated before
        incar['incar']['WANNIER90_WIN']= ('dis_num_iter = 500\ndis_conv_tol = 1e-8\nnum_iter = 1000\nconv_tol = 1e-4\nbands_plot = true'
                                        + '\nbegin kpoint_path'
                                        + '\n' + str_segment
                                        + 'end kpoint_path')

                    
        input_DFT.parameters = DataFactory('dict')(dict=incar) #convert to AiiDA format        
        self.ctx.inputs_DFT = prepare_process_inputs(input_DFT, namespaces=['dynamics','verify'])
           
        runningProcessNode = self.submit(self._next_workchain, **self.ctx.inputs_DFT)
        self.report('launching {}<{}> '.format(self._next_workchain.__name__, runningProcessNode.pk))
        return ToContext(wk_DFT=append_(runningProcessNode))

    
    def elaborate_results(self):
        """
        parse the wannierized bands resulting from the Wannier90 run, store and return them.
        """
        
        def _extract_bands_from_wannier_bands_dat( wannier_band ):
            """
            This function parses the content from wannier90 wannier90_band.dat to a Numpy array.
            """
            bands_idx_lim = np.array( [line_idx for line_idx , line in enumerate(wannier_band.splitlines()) if line.isspace()] )        
            bands = np.zeros( [len(bands_idx_lim) , bands_idx_lim[0] , 2]  )
            for line_idx , line in enumerate(wannier_band.splitlines()):
                idx_section = np.where( ~(line_idx > bands_idx_lim) )[0][0]
                idx_kpt     = line_idx % (bands_idx_lim[0] +1)
                if not line.isspace(): bands[idx_section , idx_kpt , :] = [ line.split()[0] , line.split()[1] ]
            return bands    
        
        def _extract_labels_from_labelinfo( label_str ):
            """
            This function parses the content from wannier90 wannier90_band.labelinfo.dat to suitable format for AiiDA.
            """
            labels = []
            #for line in open( label_file ):
            for line_idx , line in enumerate(label_str.splitlines()) :
                ##DEBUG - print(line.split())
                if len(line.strip()) > 0 :
                    labels.append([ line.split()[0]      ,  int(line.split()[1])-1  , float(line.split()[2]) , 
                                   float(line.split()[3]) , float(line.split()[4])  , float(line.split()[5])])
            return labels
        

        #First extract and parse the actual bands, as saved in the wannier90_band.dat files.
        bands_array = DataFactory('core.array')()        
        if ("magnetic_moment_onsite" in self.inputs["ns_parameters"] ):
            wannier_band_spinDw = self.ctx.wk_DFT[-1].outputs.retrieved.get_object_content('wannier90.2_band.dat')
            wannier_band_spinUp = self.ctx.wk_DFT[-1].outputs.retrieved.get_object_content('wannier90.1_band.dat')
            
            wannier_band_spinDw_extracted = _extract_bands_from_wannier_bands_dat( wannier_band_spinDw )
            wannier_band_spinUp_extracted = _extract_bands_from_wannier_bands_dat( wannier_band_spinUp )

            bands_array.set_array('spinUp', wannier_band_spinUp_extracted )
            bands_array.set_array('spinDw', wannier_band_spinDw_extracted )       
        else:
            wannier_band = self.ctx.wk_DFT[-1].outputs.retrieved.get_object_content('wannier90_band.dat')

            wannier_band_extracted = _extract_bands_from_wannier_bands_dat( wannier_band )
         
            bands_array.set_array('spinUp', wannier_band_extracted )
        bands_array.store()
        self.out("wannier_bands" , bands_array)
          
          
        #The parse and extract the labelinfo as saved into wannier90_band.labelinfo.dat
        labels_list =  []
        if ("magnetic_moment_onsite" in self.inputs["ns_parameters"] ):
            wannier_labelinfo_spinDw = self.ctx.wk_DFT[-1].outputs.retrieved.get_object_content("wannier90.2_band.labelinfo.dat")
            wannier_labelinfo_spinUp = self.ctx.wk_DFT[-1].outputs.retrieved.get_object_content("wannier90.1_band.labelinfo.dat")
            
            wannier_labelinfo_spinDw_extracted = _extract_labels_from_labelinfo( wannier_labelinfo_spinDw )
            wannier_labelinfo_spinUp_extracted = _extract_labels_from_labelinfo( wannier_labelinfo_spinUp )
            
            labels_list = [ wannier_labelinfo_spinUp_extracted , wannier_labelinfo_spinDw_extracted ]
        else:
            wannier_labelinfo = self.ctx.wk_DFT[-1].outputs.retrieved.get_object_content("wannier90_band.labelinfo.dat")
       
            wannier_labelinfo_extracted = _extract_labels_from_labelinfo( wannier_labelinfo )
            
            labels_list = [ wannier_labelinfo_extracted ]
            
        #Save, store & output.
        labels_list_aiida = List(labels_list)
        labels_list_aiida.store()
        self.out("bands_label_list" , labels_list_aiida)
    
    
    
    
        