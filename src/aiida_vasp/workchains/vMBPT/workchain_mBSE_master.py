import numpy as np

from aiida.engine import WorkChain, ToContext, submit
from aiida.orm import Int, Float, KpointsData, ArrayData, KpointsData, Bool, Str 
from aiida.common.extendeddicts import AttributeDict

from .workchain_mBSE_base_winterpolation import VaspmBSEInitScriptWorkChain
from .workchain_mBSE_kptsConv import VaspmBSEKptsConvWorkChain
from .utils_helpers_mBSE import _extract_opticalgap_fromWorkchainNode

class VaspmBSECompleteWorkChain(WorkChain):
    """
    Minimal master mBSE workchain.

   [1][INPUTS/OUTPUTS: OVERVIEW
   Expose inputs from VaspmBSEInitScriptWorkChain - This brings in (from the base mBSE WC), among others:
     code                 : Code          → used by vasp.vasp and init script step
     options              : Dict          → SLURM options, prepend_text, etc.
     structure            : StructureData
     kpoints              : KpointsData   [Exluded here]
     potential_family     : Str
     potential_mapping    : Dict({element: POTCAR})
     
   Namespace ns_parameters.*      
     ns_parameters.encut                  : Float (optional)
     ns_parameters.nbands                 : Int   (optional)
     ns_parameters.magnetic_moment_onsite : Dict  (optional)
     ns_parameters.ibse                   : Int   (optional, default=2)
     ns_parameters.kpar                   : Int   (optional, default=1)
     
   Namespace ns_interpolation.*   
     ns_interpolation.local_initscript
     ns_interpolation.use_interpolation
     ns_interpolation.nbandsgw_to_interpolate
     ns_interpolation.remote_gw_reference_folder
     ns_interpolation.local_gw_reference_folder
     ns_interpolation.gw_reference_filename
     
   Namespace ns_BSE.*             
     ns_BSE.static_inverse_diel    : Float (required)
     ns_BSE.screening_parameter    : Float (required)
     ns_BSE.G0W0_gap               : Float (optional)
     ns_BSE.optical_energy_window  : Float (optional)
     ns_BSE.OMEGAMAX               : Float (optional)
     ns_BSE_NBANDSV                : Int (optional)
     ns_BSE_NBANDSO                : Int (optional)
     
   Expose inputs from VaspmBSEKptsConvWorkChain - for k-points convergence control (ns_kpoints namespace)
   We EXCLUDE 'inputs.kpoints' explicitly because in step 1 (k-convergence) kpoints are generated internally
   -> in step 2 (full mBSE) we plug in the converged k-mesh.
     ns_kpoints.kmesh.starting_mesh : KpointsData    Starting k-mesh for the k-point convergence.
     ns_kpoints.kmesh.max_mesh      : KpointsData    Maximum k-mesh tested in the convergence.
     ns_kpoints.kmesh.step          : KpointsData    Step between k-meshes (default = [1,1,1]).
     ns_kpoints.convergence_threshold : Float        Threshold on |Δ(optical_gap)| between successive meshes.

     ns_converge_BSE.NBANDSO        : Int
     ns_converge_BSE.NBANDSV        : Int
     NOTE: the ns_converge_BSE namespace is used exclusively during the k-point convergence stage, while ns_BSE is reserved for the final mBSE calculation.
           During k-point conv., the goal is to determine a converged k-mesh at minimal computational cost. NBANDSV/O are therefore internally automatically set-up to NBANDSO = NBANDSV = 2, sufficient to track the convergence of the optical onse t,    
           which serves as a proxy for full convergence. ns_converge_BSE.NBANDSO and ns_converge_BSE.NBANDSV serve to override the standard values.
           Once the k-mesh is converged, the final mBSE run uses the full ns_BSE parameters, allowing a dense BSE subspace (via optical_energy_window or explicit NBANDSO/V) to compute dielectric functions/optical transitions on the converged k-point grid. 

    Outputs created by the workchain:
        - kmesh_converged     : KpointsData
                   Final converged k-mesh from the k-points convergence stage.
        - optical_gap_kconv   : Float (optical gap from converged k-mesh, if present)
                   Optical gap at the converged k-mesh.
        - dielectrics         : ArrayData (from final full mBSE)
        - opticaltransitions  : ArrayData (from final full mBSE)
 
    
   [2]STEP-BY-STEP LOGIC
    Step 1: run k-point convergence (VaspmBSEKptsConvWorkChain)
        - uses all standard mBSE inputs
        - but *forces* NBANDSV = NBANDSO = 2
        - and removes ns_BSE.optical_energy_window (so no energy-window logic)

    Step 2: run full mBSE (VaspmBSEInitScriptWorkChain)
        - same inputs as user provided
        - but kpoints = converged k-mesh from step 1
        - uses ns_BSE.optical_energy_window (and/or explicit NBANDSV/O) as given.
 
    Step 3: finalize:
        - wires out kmesh_converged + optical_gap_kconv and the final mBSE outputs. 
    """

    _mbse_base_wc     = VaspmBSEInitScriptWorkChain
    _mbse_kptsconv_wc = VaspmBSEKptsConvWorkChain

    @classmethod
    def define(cls, spec):
        super(VaspmBSECompleteWorkChain, cls).define(spec)
        
        spec.expose_inputs(cls._mbse_base_wc,      exclude=('kpoints'   ,'ns_reference', 'ns_option')  )
        # From VaspBSEInitScriptWorkChain we expose:
        #    ns_parameters.encut                  , valid_type=Float , required=False
        #    ns_parameters.nbands                 , valid_type=Int   , required=False   
        #    ns_parameters.magnetic_moment_onsite , valid_type=Dict  , required=False
        #    ns_parameters.ibse                   , valid_type=Int   , required=False
        #    ns_parameters.kpar                   , valid_type=Int   , required=False          
        #    ns_parameters.nbseeig                , valid_type=Int   , required=False         
        #    
        #    ns_interpolation.local_initscript            , valid_type=SinglefileData , required=False
        #    ns_interpolation.use_interpolation           , valid_type=Bool           , required=True
        #    ns_interpolation.nbandsgw_to_interpolate     , valid_type=Int            , required=False
        #    ns_interpolation.remote_gw_reference_folder  , valid_type=RemoteData     , required=False 
        #    ns_interpolation.local_gw_reference_folder   , valid_type=Str            , required=False 
        #    ns_interpolation.gw_reference_filename       , valid_type=Str            , required=False
        #    ns_interpolation.python_sourcing_env_command , valid_type=Str            , required=True
        #    
        #    ns_BSE.static_inverse_diel   , valid_type=Float , required=True
        #    ns_BSE.screening_parameter   , valid_type=Float , required=True
        #    ns_BSE.G0W0_gap              , valid_type=Float , required=False
        #    ns_BSE.optical_energy_window , valid_type=Float , required=False
        #    ns_BSE.OMEGAMAX              , valid_type=Float , required=False
        #    ns_BSE.NBANDSV               , valid_type=Int   , required=False
        #    ns_BSE.NBANDSO               , valid_type=Int   , required=False 
        #    ns_BSE.set_PRECFOCK_to_Fast  , valid_type=Bool  , required=False 
  
        spec.expose_inputs(cls._mbse_kptsconv_wc,  exclude=('ns_kpoints','ns_reference','ns_converge_BSE')      )
        # From VaspmBSEKptsConvWorkChain we expose:
        #    ns_converge.dielfunction_convergence    , valid_type=Bool
        #    ns_converge.opticalgap_convergence      , valid_type=Bool
        #    ns_converge.dielfunction_distance       , valid_type=Str
        #    ns_converge.convergence_dynamic_control , valid_type=Bool
        #    ns_converge.dielfunction_window         , valid_type=Float  NOTE: the NBANDSV and NBANDSO of all mBSE calculations during kpts-convergence are usually determined automatically based on the dielfunction_window parameters
        spec.input("ns_converge_BSE.NBANDSO"                , valid_type=Int   , required=False , help="number of occupied bands included in the bse matrix for all calculations used for the k-point convergence.")
        spec.input('ns_converge_BSE.NBANDSV'                , valid_type=Int   , required=False , help="number of unoccupied (virtual) bands included in the bse matrix for all calculations used for the k-point  convergence.")

        kpoints_step_defaultvalue = KpointsData(); kpoints_step_defaultvalue.set_kpoints_mesh([1, 1, 1])
        kpoints_step_maxvalue = KpointsData();     kpoints_step_maxvalue.set_kpoints_mesh([20, 20, 20])
        spec.input('ns_kpoints.starting_mesh',   valid_type=KpointsData, required=True,   help="Starting k-mesh for the k-point convergence."         )
        spec.input('ns_kpoints.max_mesh',        valid_type=KpointsData, required=False,  default=lambda: kpoints_step_maxvalue,     help="Maximum k-mesh to be tested in the convergence."         )
        spec.input('ns_kpoints.step',            valid_type=KpointsData, required=False,  default=lambda: kpoints_step_defaultvalue, help="Step size for the k-point mesh."    )
        spec.input('ns_kpoints.convergence_threshold', valid_type=Float, required=False,  default=lambda: Float(0.35), help="Convergence threshold on the optical gap in eV."  )

        spec.output('kmesh_converged',  valid_type=KpointsData)
        spec.output('optical_gap'    ,  valid_type=Float, required=False)
        spec.expose_outputs(cls._mbse_base_wc, include=('dielectrics', 'opticaltransitions')   )

        spec.outline(
            cls.run_kpoints_convergence,
            cls.run_full_mbse,
            cls.finalize,
        )

    # ------------------------------------------------------------------
    #[1] Run k-point convergence with NBANDSV/O fixed to 2/2
    def run_kpoints_convergence(self):
        inputs_kconv = AttributeDict( {'ns_converge_BSE':AttributeDict(), 'ns_kpoints':AttributeDict(), 'ns_converge':AttributeDict(),})

        # Take everything the user gave to this master that is relevant
        # to VaspmBSEInitScriptWorkChain: code, structure, options,
        # potential_family/mapping, ns_parameters, ns_interpolation,
        # ns_BSE, ns_BSE_NBANDSV/O, ...
        inputs_kconv.update(self.exposed_inputs(self._mbse_kptsconv_wc))

        inputs_kconv.ns_converge_BSE.static_inverse_diel = self.inputs.ns_BSE.static_inverse_diel.value
        inputs_kconv.ns_converge_BSE.screening_parameter = self.inputs.ns_BSE.screening_parameter.value
        inputs_kconv.ns_converge_BSE.G0W0_gap            = self.inputs.ns_BSE.G0W0_gap.value

        # BSE overrides ONLY for k-convergence ----
        # Force minimal BSE subspace - the idea is that we want to converge only the onset
        # as use it as proxy for the whole convergence.
        # Setting this overrides the logic based on the "optical_energy_window" input
        if ("NBANDSV" in self.inputs.ns_converge_BSE) and ("NBANDSO" in self.inputs.ns_converge_BSE) :
            inputs_kconv.ns_converge_BSE.NBANDSV = self.inputs.ns_converge_BSE.NBANDSV.value
            inputs_kconv.ns_converge_BSE.NBANDSO = self.inputs.ns_converge_BSE.NBANDSO.value
        else:
            inputs_kconv.ns_converge_BSE.NBANDSV = Int(2)
            inputs_kconv.ns_converge_BSE.NBANDSO = Int(2)

        # Add ns_kpoints namespace as expected by VaspmBSEKptsConvWorkChain
        inputs_kconv.ns_kpoints.kmesh = AttributeDict()
        inputs_kconv.ns_kpoints.kmesh.starting_mesh   = self.inputs.ns_kpoints.starting_mesh
        inputs_kconv.ns_kpoints.kmesh.max_mesh        = self.inputs.ns_kpoints.max_mesh
        inputs_kconv.ns_kpoints.kmesh.step            = self.inputs.ns_kpoints.step
        inputs_kconv.ns_kpoints.convergence_threshold = self.inputs.ns_kpoints.convergence_threshold
        inputs_kconv.ns_converge.convergence_dynamic_control = Bool(True)

        running = self.submit(self._mbse_kptsconv_wc, **inputs_kconv)
        self.report(
            f"[VaspmBSECompleteWorkChain] Launched k-point convergence WC <{running.pk}>"
        )
        return ToContext(wc_kconv=running)

    # ------------------------------------------------------------------
    #[2] Run the full mBSE calculation on the converged k-mesh
    def run_full_mbse(self):
        inputs_full = AttributeDict( {'ns_option':AttributeDict(), 'ns_BSE':AttributeDict(),  'ns_parameters':AttributeDict()  }  )

        # Here we use *exactly* what the user provided to the master WC
        # for VaspmBSEInitScriptWorkChain, including ns_BSE.optical_energy_window
        # and/or explicit NBANDSV/O (if they set them).
        inputs_full.update(self.exposed_inputs(self._mbse_base_wc))
        
        inputs_full.ns_BSE.set_PRECFOCK_to_Fast = Bool(True)
        inputs_full.ns_option.calculation_label = Str("mBSE final")
        inputs_full.ns_parameters.nbseeig = Int(250)
        inputs_full.ns_parameters.ibse = Int(2)

        # Replace only kpoints with the converged k-mesh
        inputs_full.kpoints = self.ctx.wc_kconv.outputs.kmesh_converged

        running = self.submit(self._mbse_base_wc, **inputs_full)
        self.report(
            f"[VaspmBSECompleteWorkChain] Launched final mBSE WC <{running.pk}> "
            f"with k-mesh {inputs_full.kpoints.get_kpoints_mesh()[0]}"
        )
        return ToContext(wc_full_mbse=running)

    # ------------------------------------------------------------------
    #[3] wire outputs
    def finalize(self):
        # k-mesh + optical gap from convergence step
        self.out('kmesh_converged', self.ctx.wc_kconv.outputs.kmesh_converged)

        optgap, _ = _extract_opticalgap_fromWorkchainNode(self.ctx.wc_full_mbse)
        node_optgap = Float(optgap)
        node_optgap.store()
        self.out('optical_gap', node_optgap )

        # Final mBSE outputs
        self.out('dielectrics',        self.ctx.wc_full_mbse.outputs.dielectrics)
        self.out('opticaltransitions', self.ctx.wc_full_mbse.outputs.opticaltransitions)

