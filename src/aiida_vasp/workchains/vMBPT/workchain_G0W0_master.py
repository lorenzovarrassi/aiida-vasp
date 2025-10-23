import numpy as np
from copy import deepcopy
from aiida import orm
import itertools
from aiida.orm import Code, Int, Float, Str, Dict, Bool , List , RemoteData , ArrayData , BandsData , XyData , KpointsData
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine  import WorkChain, calcfunction , ToContext , append_ , submit, while_ , if_ , submit, run
from aiida_vasp.utils.workchains import prepare_process_inputs
from aiida_vasp.utils.aiida_utils import get_data_class
from aiida.common.extendeddicts  import AttributeDict
from aiida_vasp.utils.workchains import site_magnetization_to_magmom

import warnings
from .workchain_G0W0_ExtrapolationScheme import VaspG0W0BasisExtrWorkChain , input_magnetic_moment_tomagmom
from .workchain_G0W0_base import VaspDFTGWWorkChain
from .workchain_G0W0_kptsConv import VaspMBPTKptsConvWorkChain
from .utils_calcfunctions import input_magnetic_moment_tomagmom 




# Main entry point for the G0W0 AiiDA-VASP workflow.
# The workflow is based on the basis-set correction scheme(see 10.1103/PhysRevB.90.075125 and https://arxiv.org/pdf/2410.15948 ).
#
# The idea is that a single G0W0 calculation is run on the dense k-point mesh. The basis-set error on the Quasiparticle (QP) energies,
# i.e. the error associated to truncating the basis-set is estimated (through an extrapolation to the infinite basis-set limite on the sparse k-mesh) and then corrected.
# Also the errors on the Quasiparticle energies associated to the use of NON norm-conserving pseudopotential can be estimated and corrected in a similary way:
#      The G0W0 data on the dense k-point mesh is launched using the (computationally cheaper) standard PAWS, the error on the QP energies can be estimated
#      and later corrected.

class VaspG0W0CompleteWorkChain(WorkChain):
    _next_workchain = WorkflowFactory('vasp.vasp')

    @classmethod
    def define(cls, spec):
            super(VaspG0W0CompleteWorkChain, cls).define(spec) 

            spec.expose_inputs(cls._next_workchain         , exclude=('kpoints','potential_family','potential_mapping','parameters','settings')) 
            spec.expose_inputs(VaspG0W0BasisExtrWorkChain  , exclude=('kpoints','potential_family','potential_mapping','ns_reference','ns_option')) 
            spec.expose_inputs(VaspMBPTKptsConvWorkChain   , exclude=('kpoints','potential_family','potential_mapping','ns_opt_converge','ns_option')) 

            
            spec.input('cutoff_startingReferenceDFT' ,  valid_type=Float , required=False , help='Reference energy cut-off used for the extrapolations.'  ) 
            
            spec.input('potential.family_USPAW'  , valid_type=Str  , help='AiiDA-VASP potential family for the US-PAWs.'   )
            spec.input('potential.mapping_USPAW' , valid_type=Dict , help='AiiDA-VASP potential mapping for the US-PAWs.'  )            
            spec.input('potential.family_NCPAW'  , valid_type=Str  , help='AiiDA-VASP potential family for the NC-PAWs.'  , required=False )
            spec.input('potential.mapping_NCPAW' , valid_type=Dict , help='AiiDA-VASP potential mapping for the NC-PAWs.' , required=False ) 
            
            spec.input('ns_option.should_NV_bePerformed'   , valid_type=Bool , required=False , default=lambda: Bool(False) , help='Force the execution of the norm-violation correction.')
            spec.input('ns_option.perform_Wannerization'   , valid_type=Bool , required=False , default=lambda: Bool(False) , help='Wannierize the bands of the G0W0 6x6x6 Dense calculation.' )
            spec.input('ns_option.perform_KptsConvergence' , valid_type=Bool , required=False , default=lambda: Bool(True)  , help='Perform the kpts Convergence - the converged k-mesh will be used for the dense calculation (and not the sparse).' )
            spec.input('ns_option.use_initial_DFTgr_forExtrapolationG0W0s'   , valid_type=Bool , required=False , default=lambda: Bool(False) , help='Use the initial DFT ground state for all G0W0 calculations used in the interpolation, without redoing for each data points.')
            #If self.inputs.ns_option.use_initial_DFTgr_forExtrapolationG0W0s.value is True, then for all G0W0s used as data points for the Extrapolation do NOT run the DFTgr, 
            #and pass as a starting point DIRECTLY to the DFTvo (of the various data points) the WAVECAR of the single DFTgr run at the beginning to get ENMAX/(NGX NGY NGZ).
            #The problem is that the single DFTgr is run using an ENCUT which is usually different to the ENCUTs of all three data-points, and thus it may cause errors.
            #if False, DFTgr is rerun for each data-point; safer, because each DFTgr is therefore run at the same ENCUT of the corresponding G0W0; however three more calcs (at least)
            #should be submitted.          

            kpoints_sparse_defaultvalue = DataFactory('core.array.kpoints')()
            kpoints_sparse_defaultvalue.set_kpoints_mesh([3,3,3])
            kpoints_dense_defaultvalue = DataFactory('core.array.kpoints')()
            kpoints_dense_defaultvalue.set_kpoints_mesh([6,6,6])
            spec.input('kpoints.sparse' , valid_type= DataFactory('core.array.kpoints') , required=False , default=lambda:kpoints_sparse_defaultvalue , help='k-point mesh used for the corrections. By default 3x3x3.') 
            spec.input('kpoints.dense'  , valid_type= DataFactory('core.array.kpoints') , required=False , default=lambda:kpoints_dense_defaultvalue  , help='k-point mesh used for final dense G0W0 run. By default 6x6x6.') 

            spec.output("Correction_BasisSet"      , valid_type=Dict , help="Error on the G0W0 Dense QP gaps due to basis-set incompleteness error.")
            spec.output("Correction_NormViolation" , valid_type=Dict , required=False , help="Error on the G0W0 Dense QP gaps due to USPAW norm violation.")



            spec.outline(
                #DFT ground state calculation with USPAW 
                #a non-spin-polarized calculation is ALWAYS run first; if a magnetic calculation is required, a second
                #spin-polarized calculation is run using the non-spin-polarized as a starting point.
                cls.prepare_run_DFTground_USPAW_NSP     ,
                cls.prepare_run_DFTground_USPAW_SP      ,
                
                #Kpts-Convergence for the dense k-point mesh
                cls.prepare_run_kpts_convergence        ,
                
                #Basis set incompleteness correction
                cls.prepare_run_Extrapolation_BS        ,  
                
                #Norm violation correction - if required
                if_(cls.determine_flag_NC)(
                    #a non-spin-polarized calculation is ALWAYS run first; if a magnetic calculation is required, a second
                    #spin-polarized calculation is run using the non-spin-polarized as a starting point.
                    cls.prepare_run_DFTground_NCPAW     , 
                    #The actual Norm violation correction
                    cls.prepare_run_Extrapolation_NV ,    
                ),
                
                #The basis set and norm-violation corrections are run on a typically sparse k-mesh (by default 3x3x3)
                #Here we run the "final" G0W0 calculation, on the dense k-mesh. 
                cls.prepare_run_DFTground_USPAW_NSP     ,
                cls.prepare_run_DFTground_USPAW_SP      ,
                cls.prepare_run_DFT_G0W0_dense          ,
                cls.elaborate_results                   ,
                
                #Calls the wannerization procedure
                cls.prepare_run_wannerization
            )
            
            spec.exit_code(402,'ONE_OR_MORE_GW_FAILED',message='One or more GW calculations failed.')

 
    def prepare_run_DFTground_USPAW_NSP(self):
            """
            Prepare and run the initial non-spin-polarized (NSP) DFT ground-state (thus with a limited number of conduction bands).
            The wavefunction of this calculation will represent the starting point for all subsequent G0W0s.
            Note: this functions calls the sub-workchain VaspDFTGWWorkChain to actually run the DFT.
            
            The function is called two times: first on the sparse k-point mesh, as a starting point for extrapolation G0W0s.
            Then on the dense k-point mesh.            
            """
            ##[The Non-Spin-Polarized DFT ground-state]
            inputs_DFTgr_NSP = AttributeDict()
            inputs_DFTgr_NSP.ns_option , inputs_DFTgr_NSP.ns_parameters = AttributeDict() , AttributeDict()
            inputs_DFTgr_NSP.update(self.exposed_inputs(self._next_workchain))
            inputs_DFTgr_NSP.clean_workdir=Bool(False)

            #we use the USPAWS
            #inputs_DFTgr_NSP.kpoints = self.inputs.kpoints.sparse
            inputs_DFTgr_NSP.potential_family  = self.inputs.potential.family_USPAW
            inputs_DFTgr_NSP.potential_mapping = self.inputs.potential.mapping_USPAW
            
            #The DFTground state before the extrapolations is used to determine ENMAX and the FFT-mesh used (NGX , NGY , NGZ)
            #And, if self.inputs.ns_option.use_initial_DFTgr_forExtrapolationG0W0s is True, is used also a DFTgr starting point for the G0W0 calcs of the extrapolation 
            #(which skips the DFTgr and start from DFTvo taking the WAVECAR of THIS DFTgr as input)
            inputs_DFTgr_NSP.ns_option.run_1DFTgr       = Bool(True)
            inputs_DFTgr_NSP.ns_option.run_2DFTvo_3G0W0 = Bool(False)

            inputs_DFTgr_NSP.ns_option.verbose = Bool(True)


            #The previous part is common to both calls of the function; from now on we differentiate the cases for the sparse k-point mesh
            #and the dense k-point mesh
            if ("extrBS" in self.ctx):  #if it's present it means we are calling this AFTER running the BS extrapolation - thus for the dense k-point mesh
                inputs_DFTgr_NSP.kpoints = self.inputs.kpoints.dense 
                inputs_DFTgr_NSP.ns_option.calculationLabel = Str("Reference DFT Dense : NSP USPAW")                                                                                                    
                inputs_DFTgr_NSP.ns_parameters.encut = Float(  self.ctx.finishedWC_extrBS[-1].outputs.pairs_nbands_encuts.get_array('x_array')[0]  )
            else:                       
                #If self.ctx.extrBS does not exist, prepare_run_Extrapolation_BS has not been yet called. 
                #Thus we are preparing the inputs for the sparse k-mesh (for the extrapolation data)
                inputs_DFTgr_NSP.kpoints = self.inputs.kpoints.sparse  
                inputs_DFTgr_NSP.ns_option.calculationLabel = Str("Reference DFT : NSP USPAW")
                if ("cutoff_startingReferenceDFT" in self.inputs): inputs_DFTgr_NSP.ns_parameters.encut = self.inputs.cutoff_startingReferenceDFT

            if ("extrBS" in self.ctx):
                self.ctx.inputs_DFTgr_NSP_dense = inputs_DFTgr_NSP
                runningWC_DFTgr_NSP_dense = self.submit(VaspDFTGWWorkChain , **self.ctx.inputs_DFTgr_NSP_dense) 
                self.report('\n [Ground-State-1] launching DFT-groundState - Dense Kmesh - NonSpinPolarized vasp.vasp workchain <{}> \n\n'.format(runningWC_DFTgr_NSP_dense.pk))
                return ToContext(finishedWC_DFTgr_NSP_dense=append_(runningWC_DFTgr_NSP_dense))
            else:  
                self.ctx.inputs_DFTgr_NSP = inputs_DFTgr_NSP
                runningWC_DFTgr_NSP = self.submit(VaspDFTGWWorkChain , **self.ctx.inputs_DFTgr_NSP) 
                self.report('\n [Ground-State-1] launching DFT-groundState - NonSpinPolarized vasp.vasp workchain <{}> \n\n'.format(runningWC_DFTgr_NSP.pk))
                return ToContext(finishedWC_DFTgr_NSP=append_(runningWC_DFTgr_NSP))            

    def prepare_run_DFTground_USPAW_SP(self):
            """
            If we consider the material spin-polarized  (i.e. by passing the magnetic_moment_onsite list variable among the inputs)
            we run an additional DFT ground state calculation, using the DFT ground state from prepare_run_DFTground_USPAW_NSP as a 
            starting point. In this additional calculation, the magnetic moment are iniatilized following magnetic_moment_onsite.
            Note: this functions calls the sub-workchain VaspDFTGWWorkChain to actually run the DFT.
            """
            ##[PARTE 2: The Non-Spin-Polarized DFT ground-state]
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                inputs_DFTgr_SP = AttributeDict()
                inputs_DFTgr_SP.ns_option , inputs_DFTgr_SP.ns_parameters , inputs_DFTgr_SP.ns_reference = AttributeDict() , AttributeDict() , AttributeDict()

                inputs_DFTgr_SP.update(self.exposed_inputs(self._next_workchain))
                inputs_DFTgr_SP.clean_workdir=Bool(False)

                inputs_DFTgr_SP.potential_family  = self.inputs.potential.family_USPAW
                inputs_DFTgr_SP.potential_mapping = self.inputs.potential.mapping_USPAW
            
                inputs_DFTgr_SP.ns_option.run_1DFTgr = Bool(True)
                inputs_DFTgr_SP.ns_option.run_2DFTvo_3G0W0 = Bool(False)
                
                inputs_DFTgr_SP.ns_option.verbose = Bool(True)
            
                inputs_DFTgr_SP.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite


                if ("extrBS" in self.ctx):  #if it's present it means we are calling this AFTER running the BS extrapolation
                    inputs_DFTgr_SP.kpoints = self.inputs.kpoints.dense  
                    inputs_DFTgr_SP.ns_option.calculationLabel = Str("Reference DFT Dense : SP USPAW")
                    inputs_DFTgr_SP.ns_parameters.encut  = Float(  self.ctx.finishedWC_extrBS[-1].outputs.pairs_nbands_encuts.get_array('x_array')[0]  )
                    inputs_DFTgr_SP.ns_reference.DFTgr_RemoteData = self.ctx.finishedWC_DFTgr_NSP_dense[-1].outputs.RemoteData_DFT
                else:                       #if self.ctx.extrBS does not exist, prepare_run_Extrapolation_BS has not been yet called.      
                    inputs_DFTgr_SP.kpoints = self.inputs.kpoints.sparse 
                    inputs_DFTgr_SP.ns_option.calculationLabel = Str("Reference DFT : SP USPAW")
                    inputs_DFTgr_SP.ns_reference.DFTgr_RemoteData = self.ctx.finishedWC_DFTgr_NSP[-1].outputs.RemoteData_DFT
                    if ("cutoff_startingReferenceDFT" in self.inputs): inputs_DFTgr_SP.ns_parameters.encut = self.inputs.cutoff_startingReferenceDFT


                #Now it's time to actually submit the VaspDFTGWWorkChain !
                if ("extrBS" in self.ctx):
                    self.ctx.inputs_DFTgr_SP_dense = inputs_DFTgr_SP
                    runningWC_DFTgr_SP_dense = self.submit(VaspDFTGWWorkChain , **self.ctx.inputs_DFTgr_SP_dense) 
                    self.report('\n [Ground-State-1] launching DFT-groundState - Dense Kmesh - SpinPolarized vasp.vasp workchain <{}> \n\n'.format(runningWC_DFTgr_SP_dense.pk))
                    return ToContext(finishedWC_DFTgr_SP_dense=append_(runningWC_DFTgr_SP_dense))
                else:   
                    self.ctx.inputs_DFTgr_SP = inputs_DFTgr_SP
                    runningWC_DFTgr_SP = self.submit(VaspDFTGWWorkChain , **self.ctx.inputs_DFTgr_SP) 
                    self.report('\n [Ground-State-1] launching DFT-groundState - SpinPolarized vasp.vasp workchain <{}> \n\n'.format(runningWC_DFTgr_SP.pk))
                    return ToContext(finishedWC_DFTgr_SP=append_(runningWC_DFTgr_SP))                
 
    def prepare_run_DFTground_NCPAW(self):
            """
            Analogous to prepare_run_DFTground_USPAW_SP - but for the NCPAW
            """
            self.ctx.inputs_DFTgr_NC = AttributeDict() 
            self.ctx.inputs_DFTgr_NC.ns_option , self.ctx.inputs_DFTgr_NC.ns_parameters , self.ctx.inputs_DFTgr_NC.ns_reference = AttributeDict() , AttributeDict() , AttributeDict()
            self.ctx.inputs_DFTgr_NC.update(self.exposed_inputs(self._next_workchain))
            self.ctx.inputs_DFTgr_NC.update(self.exposed_inputs(self._next_workchain))
            self.ctx.inputs_DFTgr_NC.clean_workdir=Bool(False) 
            
            self.ctx.inputs_DFTgr_NC.kpoints           = self.ctx.inputs_DFTgr_NSP.kpoints
            self.ctx.inputs_DFTgr_NC.potential_family  = self.inputs.potential.family_NCPAW
            self.ctx.inputs_DFTgr_NC.potential_mapping = self.inputs.potential.mapping_NCPAW   
            
            self.ctx.inputs_DFTgr_NC.ns_option.calculationLabel = Str("Reference DFT :NCPAW")
  
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                self.ctx.inputs_DFTgr_NC.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite
                self.ctx.inputs_DFTgr_NC.ns_reference.DFTgr_RemoteData = self.ctx.finishedWC_DFTgr_SP[-1].outputs.RemoteData_DFT
            else:
                self.ctx.inputs_DFTgr_NC.ns_reference.DFTgr_RemoteData = self.ctx.finishedWC_DFTgr_NSP[-1].outputs.RemoteData_DFT
                #self.ctx.inputs_DFTgr_NC.fileToIncludeFromRestartFolder = List(list=['CHGCAR'])  #, 'WAVECAR' ; only CHGCAR following https://www.vasp.at/vasp-workshop/handsonIV.pdf

            runningWC_DFTgr_NC = self.submit(VaspDFTGWWorkChain , **self.ctx.inputs_DFTgr_NC)                 
            self.report('\n [Ground-State-3] launching DFT-groundState - NC-PAW vasp.vasp workchain <{}> \n\n'.format(runningWC_DFTgr_NC.pk))
            return ToContext(finishedWC_DFTgr_SP=append_(runningWC_DFTgr_NC))
    
    
    def prepare_run_kpts_convergence(self):
        # If the user asked to skip, keep the provided dense mesh around for later steps
        if (self.inputs['ns_option']['perform_KptsConvergence'].value is False):
            self.report("\n [KptsConv] Skipped (ns_option.perform_KptsConvergence = False).")            
        
        kconv_inputs = AttributeDict()
        #kconv_inputs.update(self.exposed_inputs(self._next_workchain))         # structure, code, resources, etc.
        kconv_inputs.update(self.exposed_inputs(VaspMBPTKptsConvWorkChain))       


        # Potentials: converge on US-PAW (cheaper), consistent with the dense run
        kconv_inputs.potential_family  = self.inputs.potential.family_USPAW
        kconv_inputs.potential_mapping = self.inputs.potential.mapping_USPAW


        kconv_inputs.ns_parameters = AttributeDict()
        # [param -1] To keep calculations lighter, we use NOMEGA=1 for the kpts-conv
        kconv_inputs.ns_parameters.nomega = Int(1)  # explicit as requested
        # [param -2] Define encut as 0.80 * max( ENMAXarray ) ; 0.80 in order to keep the calculations for the G0W0 more lighter.
        try:
            if ('finishedWC_DFTgr_SP' in self.ctx):     DFTgr_node = self.ctx.finishedWC_DFTgr_SP[-1]
            elif ('finishedWC_DFTgr_NSP' in self.ctx):  DFTgr_node = self.ctx.finishedWC_DFTgr_NSP[-1]
            arr =   DFTgr_node.outputs.ENMAXarray.get_array('ENMAXarray')
            encut_from_enmax = float(np.max(arr))
            kconv_inputs.ns_parameters.encut = Float(encut_from_enmax)
        except Exception: pass
        # [param -3] And the usual magnetic_moments
        if 'magnetic_moment_onsite' in self.inputs.ns_parameters:
            kconv_inputs.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite
       
        #In order to control the convergence via the explicit k-mesh and not the k-density (which is more intuitive)
        #We have to supply both the kmesh.startingValue and kmesh.maxValue of kMesh; only the step is optional (default=1)
        #We use the dense mesh passed as input to this VaspG0W0CompleteWorkChain as startingValue
        dense_mesh, _ = self.inputs.kpoints.dense.get_kpoints_mesh()
        kconv_inputs.ns_kpoints = AttributeDict() ; kconv_inputs.ns_kpoints.kMesh = AttributeDict()
        kconv_inputs.ns_kpoints.kMesh.startingValue = List(list=[int(dense_mesh[0]), int(dense_mesh[1]), int(dense_mesh[2])])
        kconv_inputs.ns_kpoints.kMesh.maxValue      = List([20,20,20])
        
        self.report("\n [KptsConv] Submitting VaspMBPTKptsConvWorkChain…")
        running = self.submit(VaspMBPTKptsConvWorkChain, **kconv_inputs)
        return ToContext(finishedWC_KptsConv=append_(running))



    
          

    def prepare_run_DFT_G0W0_dense(self):
        """
        Prepare the inputs for the Dense G0W0 runs
        """
        self.ctx.input_DFTG0W0 = AttributeDict()
        self.ctx.input_DFTG0W0.ns_option , self.ctx.input_DFTG0W0.ns_parameters , self.ctx.input_DFTG0W0.ns_reference = AttributeDict() , AttributeDict() , AttributeDict()
        self.ctx.input_DFTG0W0.update(self.exposed_inputs(self._next_workchain))
        self.ctx.input_DFTG0W0.clean_workdir = Bool(False)            
        self.ctx.input_DFTG0W0.potential_family  = self.inputs.potential.family_USPAW
        self.ctx.input_DFTG0W0.potential_mapping = self.inputs.potential.mapping_USPAW
        self.ctx.input_DFTG0W0.ns_option.run_1DFTgr = Bool(False)
        self.ctx.input_DFTG0W0.ns_option.run_2DFTvo_3G0W0 = Bool(True)   
        
        
        
        self.ctx.input_DFTG0W0.kpoints = self.inputs.kpoints.dense   
        
        #finishedWC_extrBS contains the AiiDA nodes of the G0W0 calculations used for the extrapolations
        #finishedWC_extrBS[-1].outputs.pairs_nbands_encuts contains the encut/nbands of the first G0W0 node used for the extrapolation
        #which is the one with the lowest cutoffs among the nodes in finishedWC_extrBS.
        self.ctx.input_DFTG0W0.ns_parameters.encut  = Float(  self.ctx.finishedWC_extrBS[-1].outputs.pairs_nbands_encuts.get_array('x_array')[0]    )
        self.ctx.input_DFTG0W0.ns_parameters.nbands = Int(    self.ctx.finishedWC_extrBS[-1].outputs.pairs_nbands_encuts.get_array('y_array_0')[0]  )
        self.ctx.input_DFTG0W0.ns_parameters.nomega = Int( 200 ) 
        
        #Use the DFT dense calculation finishedWC_DFTgr_SP_dense[-1]/finishedWC_DFTgr_NSP_dense[-1] as a starting point 
        #Meaning that the WAVECAR (and the CHGCAR) will be copied from these RemoteData
        #In case of a magnetic calculation, the magnetic_moment_onsite variable is also passed
        if ("magnetic_moment_onsite" in self.inputs["ns_parameters"]):
            self.ctx.input_DFTG0W0.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite 
            self.ctx.input_DFTG0W0.ns_reference.DFTgr_RemoteData =  self.ctx.finishedWC_DFTgr_SP_dense[-1].outputs.RemoteData_DFT
        else:
            self.ctx.input_DFTG0W0.ns_reference.DFTgr_RemoteData =  self.ctx.finishedWC_DFTgr_NSP_dense[-1].outputs.RemoteData_DFT

        if ("encut_chi_low" in self.ctx.finishedWC_extrBS[-1].inputs.ns_extrapolation):
            if self.ctx.finishedWC_extrBS[-1].inputs.ns_extrapolation.encut_chi_low == Bool(True) :
                self.ctx.input_DFTG0W0.ns_parameters.encut_chi = Float( self.ctx.input_DFTG0W0.ns_parameters.encut * 0.50 )
        else:
            self.ctx.input_DFTG0W0.ns_parameters.encut_chi = Float( self.ctx.input_DFTG0W0.ns_parameters.encut * 0.63 )


        self.report("\n [wkc_KptsConv][G0W0 - Launching VaspDFTGWWorkChain on dense k-mesh")
        runningWC_G0W0 = self.submit(VaspDFTGWWorkChain , **self.ctx.input_DFTG0W0) 
        return ToContext(WC_G0W0=append_(runningWC_G0W0))       
    

                
    def prepare_run_Extrapolation_BS(self):
        """
        prepare and submit the VaspG0W0BasisExtrWorkChain on the USPAW (required to estimate the basis set extrapolation error).
        """
         
        ###Extrapolation - BasisSet 
        self.ctx.extrBS = AttributeDict() 
        self.ctx.extrBS.inputs = AttributeDict() ; self.ctx.extrBS.inputs.ns_parameters , self.ctx.extrBS.inputs.ns_reference = AttributeDict() , AttributeDict()
        
        self.ctx.extrBS.inputs.update( self.exposed_inputs(VaspG0W0BasisExtrWorkChain) )    
        self.ctx.extrBS.inputs.potential_family  = self.inputs.potential.family_USPAW
        self.ctx.extrBS.inputs.potential_mapping = self.inputs.potential.mapping_USPAW   
        #self.ctx.finishedWC_DFTgr_NSP[-1].outputs.kpoints does not work for VASP G0W0s: it doesn't use VASP automatic generation but define manually the points inside KPOINTS 
        # - may give error in screened_2e.F -> use inputs self.inputs.kpoints.sparse, which alows get_kpoints_mesh() for VASP automatic generation
        #self.ctx.finishedWC_DFTgr_NSP[-1].outputs.kpoints does not work for determine_completeBasis_encutNband: it requires explicit k-mesh -> use self.ctx.finishedWC_DFTgr_SP[-1].outputs.kpoints 
        #  which allows get_kpoints() used to determine complete basis.
        self.ctx.extrBS.inputs.kpoints = self.inputs.kpoints.sparse   

        #Here source_wcNode is the initial DFTgr in this workflow
        if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
            self.ctx.extrBS.inputs.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite
            source_wcNode =  self.ctx.finishedWC_DFTgr_SP[-1]
        else:
            source_wcNode =  self.ctx.finishedWC_DFTgr_NSP[-1]
        
        #We will use for several things: 
        #1] The extrapolation workchain needs the ENMAX of the POTCAs + NGX,NGY,NGZ of the FFT grid from the DFT starting point
        self.ctx.extrBS.inputs.ns_reference.DFTgr_NGarray    = source_wcNode.outputs.NGarray            
        self.ctx.extrBS.inputs.ns_reference.DFTgr_ENMAXarray = source_wcNode.outputs.ENMAXarray
        
        #2] We defined nbandsgw based on the occupation of the source_wcNode
        #NBANDSGW Definition: We want to define NBANDSGW = #occupied.states + 6; in theory we extrapolate just gap, so we would need #occ.states +1 or +2 ; +6 just for safety:
        occ = source_wcNode.outputs.bands_DFT.get_array("occupations") < 0.45
        c_kptNum = np.shape(occ)[1]
        try: 
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                bndIdx_HOMO_spinUp = max ( [ np.where(occ[0,kptIdx,1:] != occ[0,kptIdx,:-1] )[0][0]    for kptIdx in range(c_kptNum)] )               
                bndIdx_HOMO_spinDw = max ( [ np.where(occ[1,kptIdx,1:] != occ[1,kptIdx,:-1] )[0][0]    for kptIdx in range(c_kptNum)] )               
                self.ctx.extrBS.inputs.ns_parameters.nbandsgw = Int( max(bndIdx_HOMO_spinUp,bndIdx_HOMO_spinDw) + 6 )  
            else:
                bndIdx_HOMO = max ( [ np.where(occ[0,kptIdx,1:] != occ[0,kptIdx,:-1] )[0][0]    for kptIdx in range(c_kptNum)] )               
                self.ctx.extrBS.inputs.ns_parameters.nbandsgw = Int( bndIdx_HOMO + 6 )    
        except: pass
    
        #3] As alternative for starting point of the three G0W0 data points
        #And if use_initial_DFTgr_forExtrapolationG0W0s is True also the WAVECAR (copied by passing the remote data)(CHGCAR is also by default copied, but reused by setting ISTART=1 only for magnetic)
        #If ns_reference.DFTgr_RemoteData is set in the extrapolation workchain, by default it reuse it as DFTgr for the G0W0 data points (thus skipping the step DFTgr in the path DFTgr -> DFTvo -> G0W0 and reusing this)
        #Otherwise it is used
        if self.inputs.ns_option['use_initial_DFTgr_forExtrapolationG0W0s'].value == True:
            self.ctx.extrBS.inputs.ns_reference.DFTgr_RemoteData = source_wcNode.outputs.RemoteData_DFT
    
    
        #4] The kpoints related
        self.ctx.extrBS.inputs.ns_reference.DFTgr_kpoints    = source_wcNode.outputs.kpoints            
        #self.ctx.finishedWC_DFTgr_NSP[-1].outputs.kpoints does not work for VASP G0W0s: it doesn't use VASP automatic generation but define manually the points inside KPOINTS
        # - may give error in screened_2e.F -> use inputs self.inputs.kpoints.sparse, which alows get_kpoints_mesh() for VASP automatic generation
        #self.ctx.finishedWC_DFTgr_NSP[-1].outputs.kpoints does not work for determine_completeBasis_encutNband: it requires explicit k-mesh 
        #-> use self.ctx.finishedWC_DFTgr_SP[-1].outputs.kpoints which allows get_kpoints() used to determine complete basis.

        runningWC_extrBS = self.submit(VaspG0W0BasisExtrWorkChain, **self.ctx.extrBS.inputs)
        return ToContext(finishedWC_extrBS = append_(runningWC_extrBS))            
        
    def prepare_run_Extrapolation_NV(self):
        ###Extrapolation - NormViolation ## -------------- ## -------------- ## -------------- ##
        self.ctx.extrNV = AttributeDict()
        self.ctx.extrNV.inputs = AttributeDict() ; self.ctx.extrNV.inputs.ns_parameters = AttributeDict() ; self.ctx.extrNV.inputs.ns_reference = AttributeDict()
         
        self.ctx.extrNV.inputs.update(self.exposed_inputs(VaspG0W0BasisExtrWorkChain))    
        self.ctx.extrNV.inputs.potential_family  = self.inputs.potential.family_NCPAW
        self.ctx.extrNV.inputs.potential_mapping = self.inputs.potential.mapping_NCPAW   
        #self.ctx.finishedWC_DFTgr_NSP[-1].outputs.kpoints does not work for VASP G0W0s: it doesn't use VASP automatic generation but define manually the points inside KPOINTS - may give error in screened_2e.F -> use inputs self.inputs.kpoints.sparse, which alows get_kpoints_mesh() for VASP automatic generation
        #self.ctx.finishedWC_DFTgr_NSP[-1].outputs.kpoints does not work for determine_completeBasis_encutNband: it requires explicit k-mesh -> use self.ctx.finishedWC_DFTgr_SP[-1].outputs.kpoints which allows get_kpoints() used to determine complete basis.
        self.ctx.extrNV.inputs.kpoints = self.inputs.kpoints.sparse  
        
        self.ctx.extrNV.inputs.ns_parameters = deepcopy( self.ctx.extrBS.inputs.ns_parameters )   

        if self.inputs.ns_option.should_NV_bePerformed:
            self.ctx.extrNV.extr = self.submit(VaspG0W0BasisExtrWorkChain, **self.ctx.extrNV.inputs)
            #key = f'extrapolation-NormViolation'
            #self.to_context(**{key: self.ctx.extrNV.extr})        
            runningWC_extrNV = self.submit(VaspG0W0BasisExtrWorkChain, **self.ctx.extrNV.inputs)
            return ToContext(finishedWC_extrNV = append_(runningWC_extrNV))            

    def determine_flag_NC(self):
        if ( 'should_NV_bePerformed' in self.inputs['ns_option'] ): self.ctx.flag_NV = self.inputs['ns_option']['should_NV_bePerformed']
        else: self.ctx.flag_NV = Bool(True)
        return self.ctx.flag_NV
        
        

    def prepare_run_wannerization(self):
        """
        Prepare and call the workchain_wannerization, which wannierize the bands of the G0W0 dense calculation.
        """
        if self.inputs['ns_option']['perform_Wannerization']:  
            inputs_Wan = AttributeDict()
            inputs_Wan.ns_option , inputs_Wan.ns_parameters = AttributeDict() , AttributeDict()
            inputs_Wan.update(self.exposed_inputs(self._next_workchain))
            inputs_Wan.clean_workdir=Bool(False)
    
            inputs_Wan.kpoints = self.inputs.kpoints.dense
            inputs_Wan.potential_family  = self.inputs.potential.family_USPAW
            inputs_Wan.potential_mapping = self.inputs.potential.mapping_USPAW
    
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                inputs_Wan.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite
    
            #The Wannerization workchain tries to wannierize the WAVECAR contained in the ns_reference.RemoteData node
            #We pass of course the node representing the G0W0 Dense data
            inputs_Wan.ns_reference.RemoteData = self.ctx.WC_G0W0[-1].outputs.RemoteData_DFT
            self.ctx.inputs_Wan = inputs_Wan
            runningWC_Wan = self.submit(VaspG0W0BasisExtrWorkChain, **self.ctx.inputs_Wan)
            return ToContext(finishedWC_WanV = append_(runningWC_Wan))            



    def elaborate_results(self):
        """
        Extract and store the BS and NV corrections, together to the Dense G0W0 data.
        If the Wannieration is performed, extract store and reutrn also that
        """
              
        str_log_BS = "\n [VaspDFTGWWorkChain pk="+str(self.node.pk)+"][elaborate_results - correction_BS ]\n"
        correction_BS = AttributeDict()
        if ('magnetic_moment_onsite' in  self.inputs['ns_parameters']):  
            idx_toIterate = list( itertools.product( ['spinUp','spinDw'] , ['G0W0_dir', 'G0W0_Ind','G0W0_Gam']) )
            correction_BS['spinUp'] , correction_BS['spinDw'] = AttributeDict() , AttributeDict()
        else: 
            idx_toIterate = list( itertools.product( ['spinUp'] , ['G0W0_Dir', 'G0W0_Ind','G0W0_Gam']) )
            correction_BS['spinUp'] = AttributeDict() 

        for idx in idx_toIterate:
            #correction_BS[idx[0]][idx[1]] = self.ctx.finishedWC_extrBS[-1].outputs.extrapolated.get_dict()['gaps'][idx[0]][idx[1]] - self.ctx.WC_G0W0[-1].outputs.ns_gaps.get_dict()[idx[0]][idx[1]]
            correction_BS[idx[0]][idx[1]] = (  self.ctx.finishedWC_extrBS[-1].outputs.extrapolated.get_dict()['gaps'][idx[0]][idx[1]] 
                                             - self.ctx.finishedWC_extrBS[-1].outputs.ns_gaps.get_dict()[idx[0]][idx[1]][0]           )
            str_log_BS = str_log_BS + "\n > correction "+str(idx[0]) +" "+str(idx[1])+" : "+str(correction_BS[idx[0]][idx[1]])
        #for idx in idx_toIterate:
        self.report(str_log_BS)

        correction_BS = Dict(dict=correction_BS)
        correction_BS.store()


        if self.ctx.flag_NV:
            correction_NV = AttributeDict()
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):  
                correction_NV['spinUp'] , correction_NV['spinDw'] = AttributeDict() , AttributeDict()
            else: 
                correction_NV['spinUp'] = AttributeDict() 

            for idx in idx_toIterate:
                correction_NV[idx[0]][idx[1]] = self.ctx.finishedWC_extrNV[-1].outputs.extrapolated.get_dict()['gaps'][idx[0]][idx[1]] - self.ctx.finishedWC_extrNV[-1].outputs.ns_gaps.get_dict()[idx[0]][idx[1]]

            correction_NV = Dict(dict=correction_NV)
            correction_NV.store()
            self.out("Correction_NormViolation" , correction_NV )

        self.out("Correction_BasisSet"      , correction_BS )
        
