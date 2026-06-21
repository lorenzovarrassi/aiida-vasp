import numpy as np

from aiida.engine import WorkChain, ToContext, submit
from aiida.orm import Int, Float, KpointsData, ArrayData, KpointsData, Bool, Str 
from aiida.common.extendeddicts import AttributeDict

from .workchain_mBSE_base_winterpolation import VaspmBSEInitScriptWorkChain
from .workchain_mBSE_convergence import VaspmBSEKptsConvWorkChain, VaspmBSENBandsConvWorkChain
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
     
   Expose inputs from VaspmBSEKptsConvWorkChain - for k-points convergence control (ns_converge.kpoints namespace)
   We EXCLUDE 'inputs.kpoints' explicitly because in step 1 (k-convergence) kpoints are generated internally
   -> in step 2 (full mBSE) we plug in the converged k-mesh.
     ns_converge.kpoints.starting_mesh         : KpointsData    Starting k-mesh for the k-point convergence.
     ns_converge.kpoints.max_mesh              : KpointsData    Maximum k-mesh tested in the convergence.
     ns_converge.kpoints.step                  : KpointsData    Step between k-meshes (default = [1,1,1]).
     ns_converge.kpoints.convergence_threshold : Float          Threshold on |Δ(optical_gap)| between successive meshes
                                                                 for THIS stage only (independent from the
                                                                 nbands-convergence stage's own threshold below).

     ns_converge.kpoints.NBANDSO        : Int
     ns_converge.kpoints.NBANDSV        : Int
     NOTE: ns_converge.kpoints.NBANDSO/NBANDSV are used exclusively during the k-point convergence stage, while ns_BSE is reserved for the final mBSE calculation.
           During k-point conv., the goal is to determine a converged k-mesh at minimal computational cost. NBANDSV/O are therefore internally automatically set-up to NBANDSO = NBANDSV = 2, sufficient to track the convergence of the optical onse t,
           which serves as a proxy for full convergence. ns_converge.kpoints.NBANDSO and ns_converge.kpoints.NBANDSV serve to override the standard values.
           Once the k-mesh is converged, the final mBSE run uses the full ns_BSE parameters, allowing a dense BSE subspace (via optical_energy_window or explicit NBANDSO/V) to compute dielectric functions/optical transitions on the converged k-point grid.

   Expose inputs from VaspmBSENBandsConvWorkChain - for BSE band-subspace convergence (ns_converge.nbandsvo namespace)
   This stage is run on the SAME low/cheap k-mesh as ns_converge.kpoints.starting_mesh (NOT the converged
   dense mesh from step 1), since the BSE band subspace required to cover a given independent-particle
   transition window does not depend strongly on k-mesh density, and the converged mesh is not yet
   available at this point in the workflow anyway.
     ns_converge.nbandsvo.convergence_threshold : Float  Threshold for THIS stage only (independent from the
                                                          k-points-convergence stage's own threshold above).
     ns_converge.nbandsvo.threshold_start        : Float  Starting optical-window threshold (eV).
     ns_converge.nbandsvo.threshold_max          : Float  Maximum threshold (eV); aborts if exceeded.
     ns_converge.nbandsvo.threshold_step         : Float  Threshold increment (eV) per iteration.
     ns_converge.nbandsvo.bandsdata              : BandsData  DFT bands (+occupations) for the IPA transition matrix.
     ns_converge.nbandsvo.num_bands_included     : Int    Safety upper bound on bands scanned (default 20).

     ns_converge.static_inverse_diel / .screening_parameter / .G0W0_gap : shared BSE screening params used
           by BOTH convergence stages - always derived internally from ns_BSE.* (not user-facing here).

    Outputs created by the workchain:
        - kmesh_converged     : KpointsData
                   Final converged k-mesh from the k-points convergence stage.
        - nbandsv_converged   : Int
                   Converged NBANDSV from the BSE band-subspace convergence stage.
        - nbandso_converged   : Int
                   Converged NBANDSO from the BSE band-subspace convergence stage.
        - optical_gap         : Float (optical gap from the final full mBSE, if present)
        - dielectrics         : ArrayData (from final full mBSE)
        - opticaltransitions  : ArrayData (from final full mBSE)


   [2]STEP-BY-STEP LOGIC
    Step 1: run k-point convergence (VaspmBSEKptsConvWorkChain)
        - uses all standard mBSE inputs
        - but *forces* NBANDSV = NBANDSO = 2
        - and removes ns_BSE.optical_energy_window (so no energy-window logic)

    Step 2: run BSE band-subspace convergence (VaspmBSENBandsConvWorkChain)
        - uses the same low/cheap k-mesh as step 1's starting_mesh (fixed, not the converged mesh)
        - determines NBANDSV/NBANDSO by iterating the IPA transition-energy window

    Step 3: run full mBSE (VaspmBSEInitScriptWorkChain)
        - same inputs as user provided
        - but kpoints = converged k-mesh from step 1
        - and ns_BSE.NBANDSV/NBANDSO = converged values from step 2 (always override
          whatever the user passed, since the whole point of this workchain is to
          determine them through convergence rather than guess them upfront)

    Step 4: finalize:
        - wires out kmesh_converged, nbandsv_converged, nbandso_converged, optical_gap
          and the final mBSE outputs.
    """

    _mbse_base_wc        = VaspmBSEInitScriptWorkChain
    _mbse_kptsconv_wc    = VaspmBSEKptsConvWorkChain
    _mbse_nbandsconv_wc  = VaspmBSENBandsConvWorkChain

    @classmethod
    def define(cls, spec):
        super(VaspmBSECompleteWorkChain, cls).define(spec)
        
        spec.expose_inputs(cls._mbse_base_wc,      exclude=('kpoints', 'ns_reference', 'ns_option')  )
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
        #        
        #    ns_optimization.set_PRECFOCK_to_Fast  , valid_type=Bool  , required=False 
        #    ns_optimization.lreal                 , valid_type=Bool  , required=False
        
        # NOTE: both expose_inputs calls below touch the shared ns_converge.* namespace (both
        # children inherit it from the same template base class) - PortNamespace.absorb() rebuilds
        # a namespace's ENTIRE port dict from scratch every time it is re-exposed (it does not do
        # an incremental per-leaf merge), so any manual ns_converge.* sub-fields added BETWEEN these
        # two calls would be silently wiped out by the second one. That's why every manual
        # spec.input('ns_converge....') call below is deferred until AFTER both expose_inputs calls.
        spec.expose_inputs(
            cls._mbse_kptsconv_wc, exclude=(
                'ns_reference', 'ns_converge.kpoints', 'ns_converge.convergence_threshold',
                'ns_converge.static_inverse_diel', 'ns_converge.screening_parameter', 'ns_converge.G0W0_gap',
            )
        )
        # From VaspmBSEKptsConvWorkChain we expose:
        #    ns_converge.dielfunction_convergence    , valid_type=Bool
        #    ns_converge.opticalgap_convergence      , valid_type=Bool
        #    ns_converge.dielfunction_distance       , valid_type=Str
        #    ns_converge.convergence_dynamic_control , valid_type=Bool
        #    ns_converge.dielfunction_window         , valid_type=Float  NOTE: the NBANDSV and NBANDSO of all mBSE calculations during kpts-convergence are usually determined automatically based on the dielfunction_window parameters

        spec.expose_inputs(
            cls._mbse_nbandsconv_wc, exclude=(
                'ns_reference', 'ns_converge.nbandsvo.starting_mesh', 'ns_converge.convergence_threshold',
                'ns_converge.static_inverse_diel', 'ns_converge.screening_parameter', 'ns_converge.G0W0_gap',
            )
        )
        # From VaspmBSENBandsConvWorkChain we additionally expose:
        #    ns_converge.nbandsvo.threshold_start     , valid_type=Float
        #    ns_converge.nbandsvo.threshold_max       , valid_type=Float
        #    ns_converge.nbandsvo.threshold_step      , valid_type=Float
        #    ns_converge.nbandsvo.bandsdata           , valid_type=BandsData
        #    ns_converge.nbandsvo.num_bands_included  , valid_type=Int
        # ns_converge.nbandsvo.starting_mesh is excluded above: this stage always reuses
        # ns_converge.kpoints.starting_mesh internally (see run_nbands_convergence) rather than
        # asking the user to supply the same low-density mesh a second time under a different name.

        # ---- Manual ns_converge.* additions (deferred until after both expose_inputs calls - see NOTE above) ----
        # ns_converge.kpoints.*, ns_converge.{kpoints,nbandsvo}.convergence_threshold and the BSE screening
        # params are excluded from both expose_inputs calls above and redeclared here: each convergence
        # stage needs its OWN convergence_threshold (independent of the other), the kpoints stage needs
        # its OWN NBANDSV/NBANDSO/mesh inputs, and the BSE screening params are derived internally from
        # ns_BSE.* rather than asked of the user a second time here.
        spec.input("ns_converge.kpoints.NBANDSO"           , valid_type=Int   , required=False , default=lambda: Int(2), help="number of occupied bands included in the bse matrix for all calculations used for the k-point convergence.")
        spec.input('ns_converge.kpoints.NBANDSV'           , valid_type=Int   , required=False , default=lambda: Int(2), help="number of unoccupied (virtual) bands included in the bse matrix for all calculations used for the k-point  convergence.")

        kpoints_step_defaultvalue = KpointsData(); kpoints_step_defaultvalue.set_kpoints_mesh([1, 1, 1])
        kpoints_step_maxvalue = KpointsData();     kpoints_step_maxvalue.set_kpoints_mesh([20, 20, 20])
        spec.input('ns_converge.kpoints.starting_mesh',   valid_type=KpointsData, required=True,   help="Starting k-mesh for the k-point convergence."         )
        spec.input('ns_converge.kpoints.max_mesh',        valid_type=KpointsData, required=False,  default=lambda: kpoints_step_maxvalue,     help="Maximum k-mesh to be tested in the convergence."         )
        spec.input('ns_converge.kpoints.step',            valid_type=KpointsData, required=False,  default=lambda: kpoints_step_defaultvalue, help="Step size for the k-point mesh."    )
        spec.input('ns_converge.kpoints.convergence_threshold', valid_type=Float, required=False,  default=lambda: Float(0.35), help="Convergence threshold (eV) for the k-point convergence stage only."  )
        spec.input('ns_converge.nbandsvo.convergence_threshold', valid_type=Float, required=False, default=lambda: Float(0.35), help="Convergence threshold (eV) for the NBANDSV/NBANDSO convergence stage only."  )

        spec.output('kmesh_converged',    valid_type=KpointsData)
        spec.output('nbandsv_converged',  valid_type=Int)
        spec.output('nbandso_converged',  valid_type=Int)
        spec.output('optical_gap'    ,  valid_type=Float, required=False)
        spec.expose_outputs(cls._mbse_base_wc, include=('dielectrics', 'opticaltransitions')   )

        spec.exit_code(400, 'KPOINTS_CONVERGENCE_FAILED',
                       message='The k-point convergence sub-workchain (VaspmBSEKptsConvWorkChain) '
                               'did not finish successfully.')
        spec.exit_code(401, 'NBANDS_CONVERGENCE_FAILED',
                       message='The NBANDSV/NBANDSO convergence sub-workchain (VaspmBSENBandsConvWorkChain) '
                               'did not finish successfully.')
        spec.exit_code(402, 'FULL_MBSE_FAILED',
                       message='The final full mBSE sub-workchain (VaspmBSEInitScriptWorkChain) '
                               'did not finish successfully.')

        spec.outline(
            cls.run_kpoints_convergence,
            cls.run_nbands_convergence,
            cls.run_full_mbse,
            cls.finalize,
        )

    # ------------------------------------------------------------------
    #[1] Run k-point convergence with NBANDSV/O fixed to 2/2
    def run_kpoints_convergence(self):
        # Take everything the user gave to this master that is relevant to
        # VaspmBSEKptsConvWorkChain: ns_parameters, ns_interpolation, plus the
        # shared ns_converge.* criteria fields (dielfunction_window, etc.) it exposes.
        inputs_kconv = AttributeDict(self.exposed_inputs(self._mbse_kptsconv_wc))
        # exposed_inputs() bookkeeps exposure at the top-level port name ('ns_converge' as a whole),
        # not per nested leaf - so it returns the FULL self.inputs.ns_converge, including the OTHER
        # stage's 'nbandsvo' sub-namespace (which lives under the same shared ns_converge parent but
        # isn't a valid port on VaspmBSEKptsConvWorkChain). Strip it before submitting, or AiiDA's
        # port validation rejects it as an "Unexpected port" on a non-dynamic namespace.
        inputs_kconv.ns_converge.pop('nbandsvo', None)

        inputs_kconv.ns_converge.static_inverse_diel = self.inputs.ns_BSE.static_inverse_diel.value
        inputs_kconv.ns_converge.screening_parameter = self.inputs.ns_BSE.screening_parameter.value
        inputs_kconv.ns_converge.G0W0_gap            = self.inputs.ns_BSE.G0W0_gap.value

        # BSE overrides ONLY for k-convergence ----
        # Force minimal BSE subspace - the idea is that we want to converge only the onset
        # as use it as proxy for the whole convergence.
        # Setting this overrides the logic based on the "optical_energy_window" input
        # Always present (default=2 each, see spec.input above) - no need to guess a fallback here.
        inputs_kconv.ns_converge.kpoints = AttributeDict()
        inputs_kconv.ns_converge.kpoints.NBANDSV       = self.inputs.ns_converge.kpoints.NBANDSV.value
        inputs_kconv.ns_converge.kpoints.NBANDSO       = self.inputs.ns_converge.kpoints.NBANDSO.value
        inputs_kconv.ns_converge.kpoints.starting_mesh = self.inputs.ns_converge.kpoints.starting_mesh
        inputs_kconv.ns_converge.kpoints.max_mesh      = self.inputs.ns_converge.kpoints.max_mesh
        inputs_kconv.ns_converge.kpoints.step          = self.inputs.ns_converge.kpoints.step
        # Each convergence stage gets its OWN independent convergence_threshold at the master's
        # public-facing level (ns_converge.kpoints.* vs ns_converge.nbandsvo.*, see run_nbands_convergence) -
        # here we route the kpoints-stage one into the child's shared ns_converge.convergence_threshold port.
        inputs_kconv.ns_converge.convergence_threshold = self.inputs.ns_converge.kpoints.convergence_threshold
        inputs_kconv.ns_converge.convergence_dynamic_control = Bool(True)

        running = self.submit(self._mbse_kptsconv_wc, **inputs_kconv)
        self.report(
            f"[VaspmBSECompleteWorkChain] Launched k-point convergence WC <{running.pk}>"
        )
        return ToContext(wc_kconv=running)

    # ------------------------------------------------------------------
    #[2] Run the BSE band-subspace (NBANDSV/NBANDSO) convergence, on the SAME
    #    low/cheap k-mesh used to start step 1 - not the converged dense mesh.
    def run_nbands_convergence(self):
        if not self.ctx.wc_kconv.is_finished_ok:
            self.report(
                f"[VaspmBSECompleteWorkChain] k-point convergence WC <{self.ctx.wc_kconv.pk}> "
                f"did not finish successfully (exit_status={self.ctx.wc_kconv.exit_status}); aborting."
            )
            return self.exit_codes.KPOINTS_CONVERGENCE_FAILED

        # Take everything the user gave to this master that is relevant to
        # VaspmBSENBandsConvWorkChain: ns_converge.* shared criteria fields,
        # ns_converge.nbandsvo.*, plus whatever VaspmBSEInitScriptWorkChain
        # inputs it also re-exposes.
        inputs_nbconv = AttributeDict(self.exposed_inputs(self._mbse_nbandsconv_wc))
        # Same leakage as in run_kpoints_convergence above, mirrored: strip the OTHER stage's
        # 'kpoints' sub-namespace, which isn't a valid port on VaspmBSENBandsConvWorkChain.
        inputs_nbconv.ns_converge.pop('kpoints', None)

        inputs_nbconv.ns_converge.static_inverse_diel = self.inputs.ns_BSE.static_inverse_diel.value
        inputs_nbconv.ns_converge.screening_parameter = self.inputs.ns_BSE.screening_parameter.value
        inputs_nbconv.ns_converge.G0W0_gap            = self.inputs.ns_BSE.G0W0_gap.value
        # Independent convergence_threshold for this stage - see the matching comment in
        # run_kpoints_convergence above. Read from the LEAKED nbandsvo blob (still on self.inputs,
        # untouched) into the child's BARE ns_converge.convergence_threshold port, then drop the
        # master-only nbandsvo.convergence_threshold copy below - the child's own nbandsvo
        # sub-namespace has no such leaf (see the pop right below).
        inputs_nbconv.ns_converge.convergence_threshold = self.inputs.ns_converge.nbandsvo.convergence_threshold
        inputs_nbconv.ns_converge.nbandsvo.pop('convergence_threshold', None)

        # Fixed, low-density k-mesh: reuse the same starting_mesh given for the
        # k-point convergence study (step 1). The BSE band subspace needed to
        # cover a given IPA transition window is roughly k-mesh independent, so
        # there is no need to wait for kmesh_converged from step 1.
        inputs_nbconv.ns_converge.nbandsvo.starting_mesh = self.inputs.ns_converge.kpoints.starting_mesh

        running = self.submit(self._mbse_nbandsconv_wc, **inputs_nbconv)
        self.report(
            f"[VaspmBSECompleteWorkChain] Launched NBands convergence WC <{running.pk}> "
            f"on fixed k-mesh {self.inputs.ns_converge.kpoints.starting_mesh.get_kpoints_mesh()[0]}"
        )
        return ToContext(wc_nbconv=running)

    # ------------------------------------------------------------------
    #[3] Run the full mBSE calculation on the converged k-mesh and converged NBANDSV/NBANDSO
    def run_full_mbse(self):
        if not self.ctx.wc_nbconv.is_finished_ok:
            self.report(
                f"[VaspmBSECompleteWorkChain] NBands convergence WC <{self.ctx.wc_nbconv.pk}> "
                f"did not finish successfully (exit_status={self.ctx.wc_nbconv.exit_status}); aborting."
            )
            return self.exit_codes.NBANDS_CONVERGENCE_FAILED

        inputs_full = AttributeDict( {'ns_option':AttributeDict(),       'ns_parameters':AttributeDict() ,
                                      'ns_optimization':AttributeDict(), 'ns_BSE':AttributeDict(),       }  )

        # Here we use *exactly* what the user provided to the master WC
        # for VaspmBSEInitScriptWorkChain, including ns_BSE.optical_energy_window
        # and/or explicit NBANDSV/O (if they set them).
        inputs_full.update(self.exposed_inputs(self._mbse_base_wc))
        
        inputs_full.ns_optimization.set_PRECFOCK_to_Fast = Bool(True)
        inputs_full.ns_optimization.lreal = Bool(True)
        inputs_full.ns_option.calculation_label = Str("mBSE final")
        inputs_full.ns_option.calculation_tag   = Str("_final")
        # ns_option is excluded from the base_wc expose above (we rebuild it from scratch so
        # calculation_label/calculation_tag are always ours), but copy_result_locally_path is
        # still a master-level input (exposed via the kptsconv/nbandsconv children instead) -
        # carry it through explicitly or the final stage would silently fall back to os.getcwd().
        if "copy_result_locally_path" in self.inputs.ns_option:
            inputs_full.ns_option.copy_result_locally_path = self.inputs.ns_option.copy_result_locally_path
        inputs_full.ns_parameters.nbseeig = Int(250)
        inputs_full.ns_parameters.ibse = Int(2)

        # Replace kpoints with the converged k-mesh, and NBANDSV/NBANDSO with the
        # converged band-subspace values - these always override whatever the user
        # may have passed in ns_BSE.NBANDSV/NBANDSO/optical_energy_window, since the
        # purpose of this workchain is to determine them through convergence.
        inputs_full.kpoints = self.ctx.wc_kconv.outputs.kmesh_converged
        inputs_full.ns_BSE.NBANDSV = self.ctx.wc_nbconv.outputs.nbandsv_converged
        inputs_full.ns_BSE.NBANDSO = self.ctx.wc_nbconv.outputs.nbandso_converged

        running = self.submit(self._mbse_base_wc, **inputs_full)
        self.report(
            f"[VaspmBSECompleteWorkChain] Launched final mBSE WC <{running.pk}> "
            f"with k-mesh {inputs_full.kpoints.get_kpoints_mesh()[0]}"
        )
        return ToContext(wc_full_mbse=running)

    # ------------------------------------------------------------------
    #[4] wire outputs
    def finalize(self):
        if not self.ctx.wc_full_mbse.is_finished_ok:
            self.report(
                f"[VaspmBSECompleteWorkChain] Final mBSE WC <{self.ctx.wc_full_mbse.pk}> "
                f"did not finish successfully (exit_status={self.ctx.wc_full_mbse.exit_status}); aborting."
            )
            return self.exit_codes.FULL_MBSE_FAILED

        # k-mesh + NBANDSV/NBANDSO from the two convergence stages
        self.out('kmesh_converged',   self.ctx.wc_kconv.outputs.kmesh_converged)
        self.out('nbandsv_converged', self.ctx.wc_nbconv.outputs.nbandsv_converged)
        self.out('nbandso_converged', self.ctx.wc_nbconv.outputs.nbandso_converged)

        optgap, _ = _extract_opticalgap_fromWorkchainNode(self.ctx.wc_full_mbse)
        node_optgap = Float(optgap)
        node_optgap.store()
        self.out('optical_gap', node_optgap )

        # Final mBSE outputs
        self.out('dielectrics',        self.ctx.wc_full_mbse.outputs.dielectrics)
        self.out('opticaltransitions', self.ctx.wc_full_mbse.outputs.opticaltransitions)

