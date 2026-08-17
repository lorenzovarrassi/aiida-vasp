import numpy as np
from copy import deepcopy
import itertools
from aiida.orm import Int, Float, Str, Dict, Bool , List , RemoteData ,  KpointsData
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine  import WorkChain, calcfunction , ToContext , append_ , submit, while_ , if_ , submit, run
from aiida_vasp.utils.workchains import prepare_process_inputs
from aiida.common.extendeddicts  import AttributeDict
from aiida.common.exceptions import InputValidationError

from .workchain_G0W0_ExtrapolationScheme import VaspG0W0BasisExtrWorkChain
from .workchain_G0W0_base import VaspDFTGWWorkChain
from .workchain_G0W0_kptsConv import VaspG0W0KptsConvWorkChain
from .utils_helpers_extrapolation import input_magnetic_moment_tomagmom 
from .utils_helpers_extrapolation import get_closest_EncutNband_multiple, get_EncutNbandFitParams_completeBasis_quadratic
from .utils_helpers_G0W0 import get_encut_from_potcar_mapping



# Main entry point for the G0W0 AiiDA-VASP workflow.
# The workflow is based on the basis-set correction scheme(see 10.1103/PhysRevB.90.075125 and https://arxiv.org/pdf/2410.15948 ).
#
# The idea is that a single G0W0 calculation is run on the dense k-point mesh. The basis-set error on the Quasiparticle (QP) energies,
# i.e. the error associated to truncating the basis-set is estimated (through an extrapolation to the infinite basis-set limite on the sparse k-mesh) and then corrected.
# Also the errors on the Quasiparticle energies associated to the use of NON norm-conserving pseudopotential can be estimated and corrected in a similary way:
#      The G0W0 data on the dense k-point mesh is launched using the (computationally cheaper) standard PAWS, the error on the QP energies can be estimated
#      and later corrected.

class VaspG0W0CompleteWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
            super(VaspG0W0CompleteWorkChain, cls).define(spec) 

            spec.expose_inputs(VaspDFTGWWorkChain          , exclude=('options','potential_family','potential_mapping','kpoints',
                                                                      'ns_parameters', 'ns_reference', 'ns_option',' kpoints'   )) 
            #VaspDFTGWWorkChain expose:
            #   from spec.expose_inputs(WorkflowFactory('vasp.vasp') : options - potential_family - potential_mapping - kpoints
            #                                                          (parameters - settings of vasp.vasp are NOT exposed by  VaspDFTGWWorkChain)
            #   ns_parameters - ns_optimization - ns_reference - ns_option - kpoints
            #Of those we expose only ns_optimization; the rest is controlled internally.          
            
            spec.expose_inputs(VaspG0W0BasisExtrWorkChain  , exclude=('kpoints','potential_family','potential_mapping','options','ns_reference'   ,'ns_option')) 
            #VaspG0W0BasisExtrWorkChain expose:
            #   from spec.expose_inputs(VaspDFTGWWorkChain): potential_family - potential_mapping -options -  ns_optimization 
            #                                               (the rest of is VaspDFTGWWorkChain is not exposed and controlled internally by VaspG0W0BasisExtrWorkChain).
            #	ns_extrapolation - ns_parameters - kpoints - ns_reference - ns_options
            spec.expose_inputs(VaspG0W0KptsConvWorkChain   , exclude=('kpoints','potential_family','potential_mapping','options','ns_opt_converge','ns_option','ns_kpoints')) 
            
            spec.input('ns_potential.family_USPAW'  , valid_type=Str  , required=True , help='AiiDA-VASP potential family for the US-PAWs.'   )
            spec.input('ns_potential.mapping_USPAW' , valid_type=Dict , required=True , help='AiiDA-VASP potential mapping for the US-PAWs.'  )            
            spec.input('ns_potential.family_NCPAW'  , valid_type=Str  , required=False, help='AiiDA-VASP potential family for the NC-PAWs.'   )
            spec.input('ns_potential.mapping_NCPAW' , valid_type=Dict , required=False, help='AiiDA-VASP potential mapping for the NC-PAWs.' ) 
           
            spec.input('ns_parameters.magnetic_moment_onsite' , valid_type=Dict  , required=False, help='Starting collinear on-site magnetic moment ; Syntax is {ElName:value}')
            spec.input('ns_parameters.nomega'                 , valid_type=Int   , required=False, default=lambda: Int(200) , help='number of frequency points for the chi and sigma calculation in G0W0 runs. Default is 96.') 
           
            spec.input('ns_option.perform_NV_correction'      , valid_type=Bool  , required=False, help='Force the execution of the norm-violation correction if True, or disable completely.')
            spec.input('ns_option.perform_Wannerization'      , valid_type=Bool  , required=False, default=lambda: Bool(False) , help='Wannierize the bands of the G0W0 6x6x6 Dense calculation.' )
            spec.input('ns_option.perform_KptsConvergence'    , valid_type=Bool  , required=False, default=lambda: Bool(False) , help='Perform the kpts Convergence - the converged k-mesh will be used for the dense calculation (and not the sparse).' )
            spec.input('ns_option.kptsConvergence_threshold'  , valid_type=Float , required=False, default=lambda:Float(0.1)   , help='Threshold for the kpts Convergence in eV.' )
            spec.input('ns_option.use_lower_encut_chi'        , valid_type=Bool  , required=False, default=lambda: Bool(False) , help='Set the value of ENCUTGW (cutoff on the response function) to 1/2 of the wavefunction cutoff instead of the standard of 2/3')
            spec.input('ns_option.use_fraction_enmax_as_starting_cutoff', 
                                                                valid_type=Float , required=False, default=lambda:Float(1.0)   , help=('By default the first G0W0 used for the extrapolation and the G0W0 on the dense kmesh use encut = max(ENMAX of the potcar);' 
                                                                                                                                       'by setting this value we set a specific fraction of that value.'))
            spec.input('ns_option.use_initial_DFTgr_forExtrapolation', 
                                                                valid_type=Bool  , required=False, default=lambda: Bool(False) , help='Use the initial DFT ground state for all G0W0 calculations used in the interpolation, without redoing for each data points.')
            #If self.inputs.ns_option.use_initial_DFTgr_forExtrapolation.value is True, then for all G0W0s used as data points for the Extrapolation do NOT run the DFTgr, 
            #and pass as a starting point DIRECTLY to the DFTvo (of the various data points) the WAVECAR of the single DFTgr run at the beginning to get ENMAX/(NGX NGY NGZ).
            #The problem is that the single DFTgr is run using an ENCUT which is usually different to the ENCUTs of all three data-points, and thus it may cause errors.
            #if False, DFTgr is rerun for each data-point; safer, because each DFTgr is therefore run at the same ENCUT of the corresponding G0W0; however three more calcs (at least)
            #should be submitted.          
            spec.input('ns_option.options_for_extrapolation'     , valid_type=Dict  , required=True,     help='Scheduler options (AiiDA Dict) used only for the initial DFT and the extrapolation workchains (both BS and NV), replacing the standard inputs.option.' )
            spec.input('ns_option.options_for_dense'             , valid_type=Dict  , required=True,     help='Scheduler options (AiiDA Dict) used only for the dense G0W0 workchain.' )


            kpoints_sparse_defaultvalue = DataFactory('core.array.kpoints')()
            kpoints_sparse_defaultvalue.set_kpoints_mesh([1,1,1])
            kpoints_dense_defaultvalue = DataFactory('core.array.kpoints')()
            kpoints_dense_defaultvalue.set_kpoints_mesh([6,6,6])
            spec.input('kpoints.sparse' , valid_type=KpointsData , required=False , default=lambda:kpoints_sparse_defaultvalue , help='k-point mesh used for the corrections. By default 3x3x3.') 
            spec.input('kpoints.dense'  , valid_type=KpointsData , required=False , default=lambda:kpoints_dense_defaultvalue  , help="If perform_KptsConvergence is True, it's used as the starting point for the k-pts convergence. If perform_KptsConvergence is False (defautl), it's used as the k-point mesh used for final dense G0W0 run. By default 6x6x6.") 

            spec.output("Correction_BasisSet"      , valid_type=Dict , required=True ,  help="Error on the G0W0 Dense QP gaps due to basis-set incompleteness error.")
            spec.output("Correction_NormViolation" , valid_type=Dict , required=False, help="Error on the G0W0 Dense QP gaps due to USPAW norm violation.")



            spec.outline(
                cls.initialize , 
                
                #DFT ground state calculation with USPAW 
                #a non-spin-polarized calculation is ALWAYS run first; if a magnetic calculation is required, a second
                #spin-polarized calculation is run using the non-spin-polarized as a starting point.
                cls.prepare_run_DFTground_USPAW      ,
                
                #Kpts-Convergence for the dense k-point mesh
                cls.prepare_run_kpts_convergence     ,
                
                #Basis set incompleteness correction
                cls.prepare_run_Extrapolation_BS     ,  
                
                #Norm violation correction - if required
                if_(cls.determine_flag_NC)(
                    #a non-spin-polarized calculation is ALWAYS run first; if a magnetic calculation is required, a second
                    #spin-polarized calculation is run using the non-spin-polarized as a starting point.
                    cls.prepare_run_DFTground_NCPAW  , 
                    #The actual Norm violation correction
                    cls.prepare_run_Extrapolation_NV ,    
                ),
                
                #The basis set and norm-violation corrections are run on a typically sparse k-mesh (by default 3x3x3)
                #Here we run the "final" G0W0 calculation, on the dense k-mesh. 
                cls.prepare_run_DFT_G0W0_dense       ,
                cls.elaborate_results                ,
                
                #Calls the wannerization procedure
                cls.prepare_run_wannerization
            )
            
            spec.exit_code(402,'ONE_OR_MORE_GW_FAILED',message='One or more GW calculations failed.')


    def initialize(self):
            def _get_total_mpi_tasks(options: Dict) -> int:
                """Return total MPI ranks from an AiiDA options Dict."""
                opts = options.get_dict()
                try:
                    res = opts["resources"]
                    return int(res["num_machines"]) * int(res["num_mpiprocs_per_machine"])
                except KeyError as exc:
                    raise InputValidationError(f"Invalid scheduler options: missing key {exc}")


            #[1] MPI-taks used for the dense should be a multiple of the one used for the extrapolation
            #    This simplifies a lot the setting of nbands_stride
            options_main = self.inputs.ns_option.options_for_dense
            options_extr = self.inputs.ns_option.options_for_extrapolation
            numtasks_main = _get_total_mpi_tasks(options_main)
            numtasks_extr = _get_total_mpi_tasks(options_extr)
            if numtasks_main % numtasks_extr != 0:
                    raise InputValidationError( "Invalid parallelisation setup:\n"
                                                f"  main options:        {numtasks_main} MPI tasks\n"
                                                f"  extrapolation opts:  {numtasks_extr} MPI tasks\n"
                                                "Requirement: main_tasks must be a multiple of extrapolation_tasks." )    
            
            #[²] Implement the encut_chi_value option
            if self.inputs.ns_option.use_lower_encut_chi.value:
                self.ctx.encutgw_fraction_of_encut =  1/2 
            else:
                self.ctx.encutgw_fraction_of_encut =  0.63
                
            #[3] Implement the use_initial_DFTgr_forExtrapolation
            potcar_mapping_US = self.inputs.ns_potential.mapping_USPAW.get_dict()
            self.ctx.US_reference_cutoff = get_encut_from_potcar_mapping(potcar_mapping_US)
            self.ctx_US_cutoff_starting_value = self.ctx.US_reference_cutoff * self.inputs.ns_option.use_fraction_enmax_as_starting_cutoff.value

            #[4] Should be NV performed?
            #TODO : implement logic: if US POTCARs involved have norm violation > threshold, se to True
            if ('family_NCPAW' in self.inputs.ns_potential) and ('mapping_NCPAW' in self.inputs.ns_potential) :
                print("TODO")
                self.ctx.is_NV_correction_feasible = False
                potcar_mapping_NC = self.inputs.ns_potential.mapping_NCPAW.get_dict()
                self.ctx.NC_reference_cutoff = get_encut_from_potcar_mapping(potcar_mapping_NC)
                self.ctx_NC_cutoff_starting_value = self.ctx.NC_reference_cutoff * self.inputs.ns_option.use_fraction_enmax_as_starting_cutoff.value
            else:
                self.ctx.is_NV_correction_feasible = False
                self.ctx_NC_cutoff_starting_value = self.ctx_US_cutoff_starting_value
                
                

    def prepare_run_DFTground_USPAW(self):
            """  Prepare and run the initial non-spin-polarized (NSP) DFT ground-state (thus with a limited number of conduction bands).
            The wavefunction of this calculation will represent the starting point for all subsequent G0W0s.
            Note: this functions calls the sub-workchain VaspDFTGWWorkChain to actually run the DFT.
            
            The function is called two times: first on the sparse k-point mesh, as a starting point for extrapolation G0W0s.
            Then on the dense k-point mesh.            
            """
            ##[The Non-Spin-Polarized DFT ground-state]
            inputs_DFTgr = AttributeDict({'ns_option':AttributeDict(), 'ns_parameters':AttributeDict()})
            inputs_DFTgr.update(self.exposed_inputs(VaspDFTGWWorkChain))
            inputs_DFTgr.clean_workdir=Bool(False)
            # Use dedicated scheduler options
            inputs_DFTgr.options = self.inputs.ns_option.options_for_extrapolation

            #we use the USPAWS
            #inputs_DFTgr.kpoints = self.inputs.kpoints.sparse
            inputs_DFTgr.potential_family  = self.inputs.ns_potential.family_USPAW
            inputs_DFTgr.potential_mapping = self.inputs.ns_potential.mapping_USPAW
            
            #The DFTground state before the extrapolations is used to determine ENMAX and the FFT-mesh used (NGX , NGY , NGZ)
            #And, if self.inputs.ns_option.use_initial_DFTgr_forExtrapolation is True, is used also a DFTgr starting point for the G0W0 calcs of the extrapolation 
            #(which skips the DFTgr and start from DFTvo taking the WAVECAR of THIS DFTgr as input)
            inputs_DFTgr.ns_option.run_1DFTgr       = Bool(True)
            inputs_DFTgr.ns_option.run_2DFTvo_3G0W0 = Bool(False)


            if ('magnetic_moment_onsite' in self.inputs.ns_parameters):
                inputs_DFTgr.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite
            inputs_DFTgr.ns_parameters.encut = self.ctx_US_cutoff_starting_value

            inputs_DFTgr.kpoints = self.inputs.kpoints.sparse  
            inputs_DFTgr.ns_option.calculation_label = Str("Reference DFT : USPAW")
            #This could be overriden using a value supplied by the user, for now let's avoid that.
            
            self.ctx.inputs_DFTgr = inputs_DFTgr
            runningWC_DFTgr = self.submit(VaspDFTGWWorkChain , **self.ctx.inputs_DFTgr) 
            self.report('\n [Ground-State-1] launching DFT-groundState vasp.vasp workchain <{}> \n\n'.format(runningWC_DFTgr.pk))
            return ToContext(finishedWC_DFTgr=append_(runningWC_DFTgr))            

    def prepare_run_DFTground_NCPAW(self):
            """  Analogous to prepare_run_DFTground_USPAW_SP - but for the NCPAW """
            inputs_DFTgr_NC = AttributeDict({ 'ns_option':AttributeDict(), 'ns_parameters':AttributeDict(), 'ns_reference':AttributeDict()})
            inputs_DFTgr_NC.update(self.exposed_inputs(VaspDFTGWWorkChain))
            inputs_DFTgr_NC.clean_workdir=Bool(False) 
            # Use dedicated scheduler options
            inputs_DFTgr_NC.options = self.inputs.ns_option.options_for_extrapolation
                
            inputs_DFTgr_NC.kpoints           = self.ctx.inputs_DFTgr.kpoints
            inputs_DFTgr_NC.potential_family  = self.inputs.ns_potential.family_NCPAW
            inputs_DFTgr_NC.potential_mapping = self.inputs.ns_potential.mapping_NCPAW   
                
            inputs_DFTgr_NC.ns_option.calculation_label = Str("Reference DFT :NCPAW")
    
            inputs_DFTgr_NC.ns_reference.DFTgr_RemoteData = self.ctx.finishedWC_DFTgr[-1].outputs.RemoteData_DFT
            if ('magnetic_moment_onsite' in self.inputs.ns_parameters):
                inputs_DFTgr_NC.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite
            inputs_DFTgr_NC.ns_parameters.encut = self.ctx_NC_cutoff_starting_value
            
            self.ctx.inputs_DFTgr_NC = inputs_DFTgr_NC
            runningWC_DFTgr_NC = self.submit(VaspDFTGWWorkChain , **self.ctx.inputs_DFTgr_NC)                 
            self.report('\n [Ground-State-3] launching DFT-groundState - NC-PAW vasp.vasp workchain <{}> \n\n'.format(runningWC_DFTgr_NC.pk))
            return ToContext(finishedWC_DFTgr_NC=append_(runningWC_DFTgr_NC))
        
    def prepare_run_kpts_convergence(self):
            # If the user asked to skip, keep the provided dense mesh around for later steps
            if (self.inputs['ns_option']['perform_KptsConvergence'].value is False):
                self.report("\n [KptsConv] Skipped (ns_option.perform_KptsConvergence == False).")            
            else:
                self.report("\n [KptsConv] Preparing the KptsConv WorkChain (due to ns_option.perform_KptsConvergence == True).")            
            
                kconv_inputs = AttributeDict({ 'ns_kpoints':AttributeDict(), 'ns_parameters':AttributeDict() })
                kconv_inputs.ns_kpoints.kmesh = AttributeDict()
                kconv_inputs.update(self.exposed_inputs(VaspG0W0KptsConvWorkChain))       
        
                kconv_inputs.options = self.inputs['ns_option']['options_for_extrapolation']


                # Potentials: converge on US-PAW (cheaper), consistent with the dense run
                kconv_inputs.potential_family  = self.inputs.ns_potential.family_USPAW
                kconv_inputs.potential_mapping = self.inputs.ns_potential.mapping_USPAW
        
        
                # [param -1] To keep calculations lighter, we use NOMEGA=1 for the kpts-conv
                kconv_inputs.ns_parameters.nomega = Int(1)  # explicit as requested
                # [param -2] Define encut as 0.80 * max( ENMAXarray ) ; 0.80 in order to keep the calculations for the G0W0 more lighter.
                
                if ('magnetic_moment_onsite' in self.inputs.ns_parameters):
                    kconv_inputs.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite

                
                source_DFTgr_wcnode = self.ctx.finishedWC_DFTgr[-1]
                #We do not want to use encut=enmax but a lower one
                #It's known that the k-points and (encut-nbands) convergences for GWs are not interdependent
                #In the sense that the converged k-point.mesh does not depend from the encut/nbands used for studying the convergence
                #To be more computationally efficient we therefore lower the encut/nbands          
                DFTgr_ENMAXarr = source_DFTgr_wcnode.outputs.ENMAXarray.get_array('ENMAXarray')
                DFTgr_ENMAXmax = float( np.max(DFTgr_ENMAXarr) )
                fraction_enmax_used_as_encut = 0.75
                kconv_inputs.ns_parameters.encut = fraction_enmax_used_as_encut*DFTgr_ENMAXmax
                    
                    
                DFTgr_NGarray  = source_DFTgr_wcnode.outputs.NGarray
                DFTgr_kpts     = source_DFTgr_wcnode.outputs.kpoints
                DFTgr_cell     = source_DFTgr_wcnode.inputs.structure
                #Now Reconstruct NBANDS from this encut value using the complete-basis constraint - Using same logic as in extrapolation:
                #in VASP NBANDS should be a multiple of GW_mpithrd_num - otherwise an error is passed
                
                opts = self.inputs.ns_option.options_for_extrapolation.get_dict()
                total_mpi = int(opts["resources"]["num_machines"]) * int(opts["resources"]["num_mpiprocs_per_machine"])
                kpar = int(kconv_inputs.ns_optimization.kpar.value)
                GW_mpithrd_num = Int(total_mpi // kpar)
                params_fit = get_EncutNbandFitParams_completeBasis_quadratic( DFTgr_kpts, DFTgr_cell, DFTgr_NGarray, Float(DFTgr_ENMAXmax) )
                    
                nbands_entry, log = get_closest_EncutNband_multiple(
                        DFTgr_kpts, DFTgr_cell, DFTgr_NGarray,
                        Float(DFTgr_ENMAXmax), params_fit,
                        GW_mpithrd_num,
                        kconv_inputs.ns_parameters.encut, 
                        Str("encut"),flag_twoSidesRounding=Bool(True)  )
                nbands_value = int( nbands_entry['nbands'] )
                kconv_inputs.ns_parameters.nbands = Int( nbands_value )
                #kconv_inputs.ns_parameters.nbands = Int( 40 )
                
                self.report(f" [KptsConv] Complete-basis nbands reconstructed: {nbands_value} for encut { kconv_inputs.ns_parameters.encut.value}")
                                
        
                # [param -3] And the usual magnetic_moments
                if 'magnetic_moment_onsite' in self.inputs.ns_parameters:
                    kconv_inputs.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite
            
                #In order to control the convergence via the explicit k-mesh and not the k-density (which is more intuitive)
                #We have to supply both the kmesh.starting_mesh and kmesh.max_mesh of kmesh; only the step is optional (default=1)
                #We use the dense mesh passed as input to this VaspG0W0CompleteWorkChain as starting_mesh
                kconv_inputs.ns_kpoints.kmesh.starting_mesh = KpointsData()
                kconv_inputs.ns_kpoints.kmesh.max_mesh      = KpointsData()
                tmp_dense_mesh = self.inputs.kpoints.dense.get_kpoints_mesh()[0]
                kconv_inputs.ns_kpoints.kmesh.starting_mesh.set_kpoints_mesh( tmp_dense_mesh )
                kconv_inputs.ns_kpoints.kmesh.max_mesh.set_kpoints_mesh( [20,20,20] )
            
                self.report("\n [KptsConv] Submitting VaspG0W0KptsConvWorkChain…")
                running = self.submit(VaspG0W0KptsConvWorkChain, **kconv_inputs)
                return ToContext(finishedWC_KptsConv=append_(running))

    def prepare_run_DFT_G0W0_dense(self):
            input_DFTG0W0 = AttributeDict({ 'ns_option':AttributeDict(), 'ns_parameters':AttributeDict() })
            input_DFTG0W0.update(self.exposed_inputs(VaspDFTGWWorkChain))
            input_DFTG0W0.clean_workdir = Bool(False)            
            input_DFTG0W0.potential_family  = self.inputs.ns_potential.family_USPAW
            input_DFTG0W0.potential_mapping = self.inputs.ns_potential.mapping_USPAW
            input_DFTG0W0.options = self.inputs['ns_option']['options_for_dense']
            input_DFTG0W0.ns_option.run_1DFTgr       = Bool(True)
            input_DFTG0W0.ns_option.run_2DFTvo_3G0W0 = Bool(True)   
            
            #If the kpts convergence has not been performed, use the dense k-mesh provided in the inputs
            if (self.inputs['ns_option']['perform_KptsConvergence'].value is False):
                input_DFTG0W0.kpoints = self.inputs.kpoints.dense   
            else:
                input_DFTG0W0.kpoints = self.ctx.kmesh_converged        
                
            #finishedWC_extrBS contains the AiiDA nodes of the G0W0 calculations used for the extrapolations
            #finishedWC_extrBS[-1].outputs.pairs_nbands_encuts contains the encut/nbands of the first G0W0 node used for the extrapolation
            #which is the one with the lowest cutoffs among the nodes in finishedWC_extrBS.
            input_DFTG0W0.ns_parameters.encut  = Float(  self.ctx.finishedWC_extrBS[-1].outputs.pairs_nbands_encuts.get_array('x_array')[0]    )
            input_DFTG0W0.ns_parameters.nbands = Int(    self.ctx.finishedWC_extrBS[-1].outputs.pairs_nbands_encuts.get_array('y_array_0')[0]  )
            input_DFTG0W0.ns_parameters.nomega = self.inputs.ns_parameters.nomega
            input_DFTG0W0.ns_parameters.encut_chi = Float( input_DFTG0W0.ns_parameters.encut.value * self.ctx.encutgw_fraction_of_encut )
            
            #Use the DFT dense calculation finishedWC_DFTgr_SP_dense[-1]/finishedWC_DFTgr_dense[-1] as a starting point 
            #Meaning that the WAVECAR (and the CHGCAR) will be copied from these RemoteData
            #In case of a magnetic calculation, the magnetic_moment_onsite variable is also passed
            if ("magnetic_moment_onsite" in self.inputs["ns_parameters"]):
                input_DFTG0W0.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite 

            self.ctx.input_DFTG0W0 = input_DFTG0W0
            self.report("\n [VaspG0W0CompleteWorkChain][G0W0 - Launching VaspDFTGWWorkChain on dense k-mesh")
            runningWC_G0W0 = self.submit(VaspDFTGWWorkChain , **self.ctx.input_DFTG0W0) 
            return ToContext(WC_G0W0=append_(runningWC_G0W0))       
             
    def __helper_prepare_extrapolation_workchain_input(self , potential_family, potential_mapping , 
                                                      source_DFTgr_wcnode, 
                                                      cutoff_starting_value=None):
            input_extr = AttributeDict({ 'ns_parameters':AttributeDict(),      'ns_reference':AttributeDict(), 
                                         'ns_optimization':AttributeDict(), 'ns_extrapolation':AttributeDict(),
                                         'ns_option':AttributeDict()})
            
            #[1] Default stuff : scheduler options , parser settings  + potcars and kpoints 
            input_extr.update( self.exposed_inputs(VaspG0W0BasisExtrWorkChain) ) 
                        

            input_extr.potential_family  = potential_family
            input_extr.potential_mapping = potential_mapping 
            #self.ctx.finishedWC_DFTgr[-1].outputs.kpoints does not work for VASP G0W0s: it doesn't use VASP automatic generation but define manually the points inside KPOINTS 
            # - may give error in screened_2e.F -> use inputs self.inputs.kpoints.sparse, which alows get_kpoints_mesh() for VASP automatic generation
            #self.ctx.finishedWC_DFTgr[-1].outputs.kpoints does not work for determine_completeBasis_encutNband: it requires explicit k-mesh -> use self.ctx.finishedWC_DFTgr_SP[-1].outputs.kpoints 
            #  which allows get_kpoints() used to determine complete basis.
            input_extr.kpoints = self.inputs.kpoints.sparse   
            
            if ('magnetic_moment_onsite' in self.inputs.ns_parameters):
                input_extr.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite
            
            #[2] define cutoff_starting_value if it's passed
            if cutoff_starting_value !=None:
                input_extr.ns_extrapolation.cutoff_starting_value = cutoff_starting_value
            
            #[3] source_DFTgr_wcnode represents the previous DFT ground state in the workflow - which will be used to extract the NGarray and ENMAXarray 
            #    and may be used as a starting point to take the WAVECAR  for the DFTvos steps
            #[4] Ad hoc inputs required for the extrapolations
            #We will use for several things: 
            #The extrapolation workchain needs the ENMAX of the POTCAs + NGX,NGY,NGZ of the FFT grid from the DFT starting point
            input_extr.ns_reference.DFTgr_NGarray    = source_DFTgr_wcnode.outputs.NGarray            
            input_extr.ns_reference.DFTgr_ENMAXarray = source_DFTgr_wcnode.outputs.ENMAXarray
            input_extr.ns_reference.DFTgr_kpoints    = source_DFTgr_wcnode.outputs.kpoints
            #source_DFTgr_wcnode.outputs.kpoints is not used to start the the DFT/G0W0 calculations, but to determine the cutoff-bands relationship under the full basis constraint.
            #The AiiDA-parser saves in outputs.kpoints the explicit full list of kpoints; this is what we need to determine the cutoff-bands relationship, but it's not fit to
            #pass to the DFT/G0W0 calculation that will run to be used as data points for the extrapolation as it may give error in screened_2e.F


            #[5] We defined nbandsgw based on the occupation of the source_DFTgr_wcnode
            #NBANDSGW Definition: We want to define NBANDSGW = #occupied.states + 6; in theory we extrapolate just gap, so we would need #occ.states +1 or +2 ; +6 just for safety:
            occ = source_DFTgr_wcnode.outputs.bands_DFT.get_array("occupations") < 0.45
            c_kptNum = np.shape(occ)[1]
            try: 
                if 'magnetic_moment_onsite' in self.inputs.ns_parameters:
                    bndIdx_HOMO_spinUp = max ( [ np.where(occ[0,kptIdx,1:] != occ[0,kptIdx,:-1] )[0][0]    for kptIdx in range(c_kptNum)] )               
                    bndIdx_HOMO_spinDw = max ( [ np.where(occ[1,kptIdx,1:] != occ[1,kptIdx,:-1] )[0][0]    for kptIdx in range(c_kptNum)] )               
                    input_extr.ns_parameters.nbandsgw = Int( max(bndIdx_HOMO_spinUp,bndIdx_HOMO_spinDw) + 6 )  
                else:
                    bndIdx_HOMO = max ( [ np.where(occ[kptIdx,1:] != occ[kptIdx,:-1] )[0][0]    for kptIdx in range(c_kptNum)] )               
                    input_extr.ns_parameters.nbandsgw = Int( bndIdx_HOMO + 6 )    
            except Exception as exc: 
                    self.report(f"[WARN] Failed to determine nbandsgw automatically: {exc}")
        
            #[6] Use the WAVECAR from source_DFTgr_wcnode starting point of the G0W0 workflows in the extrapolation
            #    If use_initial_DFTgr_forExtrapolation is True the WAVECAR is copied (by passing the remotefolder of source_DFTgr_wcnode as restart folder)
            #    (CHGCAR is also by default copied, but reused by setting ISTART=1 only for magnetic)
            #    If ns_reference.DFTgr_RemoteData is set in the extrapolation workchain, by default it reuse it as DFTgr for the G0W0 data points (thus skipping the step DFTgr in the path DFTgr -> DFTvo -> G0W0 and reusing this)
            if self.inputs.ns_option.use_initial_DFTgr_forExtrapolation.value:
                input_extr.ns_reference.DFTgr_RemoteData = source_DFTgr_wcnode.outputs.RemoteData_DFT
                
            #[7] Define the scheduler options: options_for_extrapolation and options_for the dense
            input_extr.ns_option.options_for_extrapolation =  self.inputs['ns_option']['options_for_extrapolation']
            input_extr.ns_option.constraint_nbands_divisor = (self.inputs.ns_option.options_for_dense.get_dict()['resources']['num_machines'] *
                                                              self.inputs.ns_option.options_for_dense.get_dict()['resources']['num_mpiprocs_per_machine'] )
            return input_extr
        
    def prepare_run_Extrapolation_BS(self):
            """ Prepare and submit the VaspG0W0BasisExtrWorkChain on the USPAW (required to estimate the basis set extrapolation error). """  
            self.ctx.input_extrBS = self.__helper_prepare_extrapolation_workchain_input(
                                        potential_family      = self.inputs.ns_potential.family_USPAW  ,
                                        potential_mapping     = self.inputs.ns_potential.mapping_USPAW ,
                                        source_DFTgr_wcnode   = self.ctx.finishedWC_DFTgr[-1]          ,
                                        cutoff_starting_value = self.ctx_US_cutoff_starting_value      )
            runningWC_extrBS = self.submit(VaspG0W0BasisExtrWorkChain, **self.ctx.input_extrBS)
            return ToContext(finishedWC_extrBS = append_(runningWC_extrBS))            
            
    def prepare_run_Extrapolation_NV(self):
            """ Prepare and submit the VaspG0W0BasisExtrWorkChain on the NCPAW (required to estimate the norm violation error). """  
            self.ctx.input_extrNV = self.__helper_prepare_extrapolation_workchain_input(
                                        potential_family    = self.inputs.ns_potential.family_NCPAW  ,
                                        potential_mapping   = self.inputs.ns_potential.mapping_NCPAW ,
                                        source_DFTgr_wcnode = self.ctx.finishedWC_DFTgr_NC[-1]       ,
                                        cutoff_starting_value = self.ctx_NC_cutoff_starting_value    )
            runningWC_extrNV = self.submit(VaspG0W0BasisExtrWorkChain, **self.ctx.input_extrNV)
            return ToContext(finishedWC_extrNV = append_(runningWC_extrNV)) 

    def determine_flag_NC(self):
            self.ctx.flag_NV = Bool(False)
            if ( 'perform_NV_correction' in self.inputs.ns_option ) :
                self.ctx.flag_NV = self.inputs['ns_option']['perform_NV_correction']
            else: self.ctx.flag_NV = Bool( self.ctx.is_NV_correction_feasible )
            return self.ctx.flag_NV
        
    def prepare_run_wannerization(self):
            """ Prepare and call the workchain_wannerization, which wannierize the bands of the G0W0 dense calculation. """
            if self.inputs['ns_option']['perform_Wannerization']:  
                inputs_Wan = AttributeDict({ 'ns_option':AttributeDict(), 'ns_parameters':AttributeDict() })
                inputs_Wan.update(self.exposed_inputs(VaspDFTGWWorkChain))
                inputs_Wan.clean_workdir=Bool(False)
        
                inputs_Wan.kpoints = self.inputs.kpoints.dense
                inputs_Wan.potential_family  = self.inputs.ns_potential.family_USPAW
                inputs_Wan.potential_mapping = self.inputs.ns_potential.mapping_USPAW
        
                if ('magnetic_moment_onsite' in self.inputs.ns_parameters):
                    inputs_Wan.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite
        
                #The Wannerization workchain tries to wannierize the WAVECAR contained in the ns_reference.RemoteData node
                #We pass of course the node representing the G0W0 Dense data
                inputs_Wan.ns_reference.RemoteData = self.ctx.WC_G0W0[-1].outputs.RemoteData_DFT
                self.ctx.inputs_Wan = inputs_Wan
                runningWC_Wan = self.submit(VaspG0W0BasisExtrWorkChain, **self.ctx.inputs_Wan)
                return ToContext(finishedWC_WanV = append_(runningWC_Wan))            

    def elaborate_results(self):
            """ Extract and store the BS and NV corrections, together to the Dense G0W0 data.
            If the Wannieration is performed, extract store and reutrn also that.        """
                
            str_log_BS = "\n [VaspDFTGWWorkChain pk="+str(self.node.pk)+"][elaborate_results - correction_BS ]\n"
            is_spinpol = ("magnetic_moment_onsite" in self.inputs.ns_parameters)
            spin_labels = ("spinUp", "spinDw") if is_spinpol else ("spinUp",)

            extrBS = self.ctx.finishedWC_extrBS[-1]
            extrap_gaps_BS = extrBS.outputs.extrapolated.get_dict()["gaps"]      # spin -> Dir/Ind/Gam (scalars)
            ns_gaps_BS     = extrBS.outputs.ns_gaps_G0W0.get_dict()              # spin -> Dir/Ind/Gam (lists)
        
            correction_BS = AttributeDict({sp: AttributeDict() for sp in spin_labels})
            for sp in spin_labels:
                for key in ("Dir", "Ind", "Gam"):
                    correction_BS[sp][key] = float(extrap_gaps_BS[sp][key]) - float(ns_gaps_BS[sp][key][0])
                    str_log_BS += f"\n > correction {sp} {key}: {correction_BS[sp][key]}"

            self.report(str_log_BS)
            node_BS = Dict(dict=correction_BS)
            node_BS.store()
            self.out("Correction_BasisSet", node_BS)

            # -------------------------
            # Norm-violation correction (NV)
            if self.ctx.flag_NV:
                extrNV = self.ctx.finishedWC_extrNV[-1]
                extrap_gaps_NV = extrNV.outputs.extrapolated.get_dict()["gaps"]
                ns_gaps_NV     = extrNV.outputs.ns_gaps_G0W0.get_dict()

                correction_NV = AttributeDict({sp: AttributeDict() for sp in spin_labels})
                for sp in spin_labels:
                    for key in ("Dir", "Ind", "Gam"):
                        correction_NV[sp][key] = float(extrap_gaps_NV[sp][key]) - float(ns_gaps_NV[sp][key][0])
    
                node_NV = Dict(dict=correction_NV)
                node_NV.store()
                self.out("Correction_NormViolation", node_NV) 
          
            def __copy_remote_to_local(self, remote_data, label):
                import os
                from aiida.common.folders import SandboxFolder

                pid = str(self.pid)
                foldername = f"{label}_id{pid}"
                target = os.path.join(os.getcwd(), foldername)
                os.makedirs(target, exist_ok=True)

                with SandboxFolder() as sandbox:
                    # copy *entire* remote folder into sandbox
                    remote_data.getfile('.', sandbox.abspath)
                    sandbox.copytree(target)
            self.__copy_remote_to_local( self.ctx.WC_G0W0[-1].outputs.RemoteData_DFT, "2.1-DFT"    )
            self.__copy_remote_to_local( self.ctx.WC_G0W0[-1].outputs.RemoteData_G0W0, "2.2-GW"    )
                
        
