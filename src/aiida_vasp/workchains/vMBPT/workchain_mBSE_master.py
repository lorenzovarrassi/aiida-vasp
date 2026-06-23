import numpy as np

from aiida.engine import WorkChain, ToContext, submit
from aiida.orm import Int, Float, KpointsData, ArrayData, KpointsData, Bool, Str, BandsData
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
     NOTE: ns_BSE.NBANDSV/NBANDSO are excluded entirely at this master level (not redeclared) -
           the final mBSE run's NBANDSV/NBANDSO always come from either the nbandsvo-convergence
           result, or (if that stage is disabled) ns_converge.nbandsvo.superseded_nbandsv/
           superseded_nbandso below. There is no separate ns_BSE.NBANDSV/NBANDSO override path here.

   Expose inputs from VaspmBSEKptsConvWorkChain - for k-points convergence control (ns_converge.kpoints namespace)
   We EXCLUDE 'inputs.kpoints' explicitly because in step 1 (k-convergence) kpoints are generated internally
   -> in step 2 (full mBSE) we plug in the converged k-mesh.
     ns_converge.kpoints.starting_mesh         : KpointsData    Starting k-mesh for the k-point convergence SEARCH
                                                                 (incremented across iterations - this is the search
                                                                 starting point, not a fixed value).
     ns_converge.kpoints.max_mesh              : KpointsData    Maximum k-mesh tested in the convergence.
     ns_converge.kpoints.step                  : KpointsData    Step between k-meshes (default = [1,1,1]).
     ns_converge.kpoints.convergence_threshold : Float          Threshold on |Δ(optical_gap)| between successive meshes
                                                                 for THIS stage only (independent from the
                                                                 nbands-convergence stage's own threshold below).

     ns_converge.kpoints.NBANDSO_fixed_for_convergence  : Int (default=2)
     ns_converge.kpoints.NBANDSV_fixed_for_convergence  : Int (default=2)
     NOTE: these are used exclusively during the k-point convergence stage as a cheap, FIXED proxy
           subspace (they do not vary as the k-mesh search progresses) - they are NOT the final
           production NBANDSV/NBANDSO. The goal of this stage is to determine a converged k-mesh at
           minimal computational cost; NBANDSO=NBANDSV=2 is normally sufficient to track convergence
           of the optical onset, which serves as a proxy for full convergence.
           Once the k-mesh is converged, the final mBSE run instead uses the converged (or
           superseded, if disabled) NBANDSV/NBANDSO from the nbandsvo-convergence stage below.

   Expose inputs from VaspmBSENBandsConvWorkChain - for BSE band-subspace convergence (ns_converge.nbandsvo namespace)
     ns_converge.nbandsvo.kmesh_fixed_for_convergence : KpointsData  Fixed, low-density k-mesh held
           constant throughout this stage (cheap proxy, NOT the final production mesh).
           Self-contained, with its own default ([8,8,8]) - no dependency on
           ns_converge.kpoints.starting_mesh (the BSE band subspace needed to cover a given IPA
           transition window is roughly k-mesh independent, and the converged dense mesh from
           stage 1 is not yet known at this point in the workflow anyway).
     ns_converge.nbandsvo.convergence_threshold : Float  Threshold for THIS stage only (independent from the
                                                          k-points-convergence stage's own threshold above).
     ns_converge.nbandsvo.threshold_start        : Float  Starting optical-window threshold (eV).
     ns_converge.nbandsvo.threshold_max          : Float  Maximum threshold (eV); aborts if exceeded.
     ns_converge.nbandsvo.threshold_step         : Float  Threshold increment (eV) per iteration.
     ns_converge.nbandsvo.bandsdata              : BandsData  DFT bands (+occupations) for the IPA transition matrix.
     ns_converge.nbandsvo.num_bands_included     : Int    Safety upper bound on bands scanned (default 20).

     ns_converge.static_inverse_diel / .screening_parameter / .G0W0_gap : shared BSE screening params used
           by BOTH convergence stages - always derived internally from ns_BSE.* (not user-facing here).

   Per-stage enable/disable switches (both default True), and the explicit "final value when
   disabled" fields each one needs (no default - validated as required at runtime only when the
   matching stage is actually disabled; see run_kpoints_convergence/run_nbands_convergence):
     ns_converge.kpoints.enabled          : Bool  If False, skip k-point convergence; use
                                                   ns_converge.kpoints.superseded_kmesh (required) instead.
     ns_converge.kpoints.superseded_kmesh : KpointsData  Final k-mesh used when kpoints.enabled=False.
     ns_converge.nbandsvo.enabled              : Bool  If False, skip NBANDSV/NBANDSO convergence; use
                                                        superseded_nbandsv/superseded_nbandso (required) instead.
     ns_converge.nbandsvo.superseded_nbandsv   : Int  Final NBANDSV used when nbandsvo.enabled=False.
     ns_converge.nbandsvo.superseded_nbandso   : Int  Final NBANDSO used when nbandsvo.enabled=False.

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
    Step 1: run k-point convergence (VaspmBSEKptsConvWorkChain), unless ns_converge.kpoints.enabled
            is False, in which case this stage is skipped and ns_converge.kpoints.superseded_kmesh
            (required in that case) is used directly as the final k-mesh.
        - uses all standard mBSE inputs
        - but *forces* NBANDSV = NBANDSO = ns_converge.kpoints.NBANDSV/NBANDSO_fixed_for_convergence (default 2)
        - and removes ns_BSE.optical_energy_window (so no energy-window logic)

    Step 2: run BSE band-subspace convergence (VaspmBSENBandsConvWorkChain), unless
            ns_converge.nbandsvo.enabled is False, in which case this stage is skipped and
            ns_converge.nbandsvo.superseded_nbandsv/superseded_nbandso (required in that case) are
            used directly as the final BSE subspace.
        - uses a fixed, low/cheap k-mesh (ns_converge.nbandsvo.kmesh_fixed_for_convergence,
          default [8,8,8]) - never the converged mesh
        - determines NBANDSV/NBANDSO by iterating the IPA transition-energy window

    Step 3: run full mBSE (VaspmBSEInitScriptWorkChain)
        - same inputs as user provided
        - but kpoints = converged (or superseded, if skipped) k-mesh from step 1
        - and ns_BSE.NBANDSV/NBANDSO = converged (or superseded, if skipped) values from step 2

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
        
        # ns_BSE.NBANDSV/NBANDSO are excluded here entirely (not redeclared) - the master no longer
        # has any use for them: the final mBSE run's NBANDSV/NBANDSO come exclusively from either
        # the nbandsvo-convergence result or, when that stage is disabled,
        # ns_converge.nbandsvo.superseded_nbandsv/superseded_nbandso (see below).
        spec.expose_inputs(cls._mbse_base_wc,      exclude=('kpoints', 'ns_reference', 'ns_option', 'ns_BSE.NBANDSV', 'ns_BSE.NBANDSO')  )
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
        #
        #    ns_optimization.set_PRECFOCK_to_Fast  , valid_type=Bool  , required=False
        #    ns_optimization.lreal                 , valid_type=Bool  , required=False
        #
        # NOTE: both expose_inputs calls below touch the shared ns_converge.* namespace (both
        # children inherit it from the same template base class) - PortNamespace.absorb() rebuilds
        # a namespace's ENTIRE port dict from scratch every time it is re-exposed (it does not do
        # an incremental per-leaf merge), so any manual ns_converge.* sub-fields added BETWEEN these
        # two calls would be silently wiped out by the second one. That's why every manual
        # spec.input('ns_converge....') call below is deferred until AFTER both expose_inputs calls.
        spec.expose_inputs(
            cls._mbse_kptsconv_wc, exclude=(
                'ns_reference', 'ns_converge.kpoints', 'ns_converge.convergence_threshold',
                'ns_converge.select_earlier_point_at_convergence',
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
                'ns_reference', 'ns_converge.nbandsvo.kmesh_fixed_for_convergence', 'ns_converge.convergence_threshold',
                'ns_converge.select_earlier_point_at_convergence',
                'ns_converge.static_inverse_diel', 'ns_converge.screening_parameter', 'ns_converge.G0W0_gap',
                'ns_converge.nbandsvo.threshold_start', 'ns_converge.nbandsvo.threshold_max',
                'ns_converge.nbandsvo.bandsdata',
            )
        )
        # From VaspmBSENBandsConvWorkChain we additionally expose:
        #    ns_converge.nbandsvo.threshold_step      , valid_type=Float
        #    ns_converge.nbandsvo.num_bands_included  , valid_type=Int
        # ns_converge.nbandsvo.kmesh_fixed_for_convergence is excluded above and redeclared below
        # with its own independent default ([8,8,8]) - self-contained, no dependency on
        # ns_converge.kpoints.starting_mesh (which belongs to the OTHER stage).
        # ns_converge.nbandsvo.threshold_start/threshold_max/bandsdata are required=True on the
        # child but are ALSO excluded and redeclared below with required=False (no default - none
        # of these have a defensible universal fallback) - they are only actually needed when
        # ns_converge.nbandsvo.enabled=True, validated at runtime in run_nbands_convergence. Without
        # this, disabling the nbandsvo stage would still force the user to supply band data and a
        # threshold window that are never used - the same class of bug as ns_converge.kpoints.starting_mesh.

        # ---- Manual ns_converge.* additions (deferred until after both expose_inputs calls - see NOTE above) ----
        # ns_converge.kpoints.*, ns_converge.{kpoints,nbandsvo}.convergence_threshold and the BSE screening
        # params are excluded from both expose_inputs calls above and redeclared here: each convergence
        # stage needs its OWN convergence_threshold (independent of the other), the kpoints stage needs
        # its OWN NBANDSV/NBANDSO/mesh inputs, and the BSE screening params are derived internally from
        # ns_BSE.* rather than asked of the user a second time here.
        spec.input("ns_converge.kpoints.NBANDSO_fixed_for_convergence" , valid_type=Int , required=False , default=lambda: Int(2), help="number of occupied bands included in the bse matrix for all calculations used for the k-point convergence (cheap proxy, not the final value).")
        spec.input('ns_converge.kpoints.NBANDSV_fixed_for_convergence' , valid_type=Int , required=False , default=lambda: Int(2), help="number of unoccupied (virtual) bands included in the bse matrix for all calculations used for the k-point convergence (cheap proxy, not the final value).")

        kpoints_step_defaultvalue = KpointsData(); kpoints_step_defaultvalue.set_kpoints_mesh([1, 1, 1])
        kpoints_step_maxvalue = KpointsData();     kpoints_step_maxvalue.set_kpoints_mesh([20, 20, 20])
        # NOT required=True: only actually needed if kpoints.enabled=True (the k-conv search
        # itself) - i.e. it can legitimately be omitted when kpoints.enabled=False. Validated at
        # runtime in run_kpoints_convergence below - same no-default, required-only-when-needed
        # pattern as superseded_kmesh. Unrelated to nbandsvo.kmesh_fixed_for_convergence below,
        # which has its own independent default and never falls back to this field.
        spec.input('ns_converge.kpoints.starting_mesh',   valid_type=KpointsData, required=False,   help="Starting k-mesh for the k-point convergence search. Required if "
                                                                                                            "ns_converge.kpoints.enabled=True."  )
        spec.input('ns_converge.kpoints.max_mesh',        valid_type=KpointsData, required=False,  default=lambda: kpoints_step_maxvalue,     help="Maximum k-mesh to be tested in the convergence."         )
        spec.input('ns_converge.kpoints.step',            valid_type=KpointsData, required=False,  default=lambda: kpoints_step_defaultvalue, help="Step size for the k-point mesh."    )
        spec.input('ns_converge.kpoints.convergence_threshold', valid_type=Float, required=False,  default=lambda: Float(0.35), help="Convergence threshold (eV) for the k-point convergence stage only."  )
        spec.input('ns_converge.kpoints.select_earlier_point_at_convergence', valid_type=Bool,
                   required=False, default=lambda: Bool(False),
                   help="Prefer the earlier/cheaper point of the converged k-mesh pair "
                        "over the later/safer one, for the k-point convergence stage only.")

        # Fixed, low-density k-mesh used throughout the NBANDSV/NBANDSO convergence stage. Self-
        # contained - independent default, no cross-dependency on/fallback to
        # ns_converge.kpoints.starting_mesh (which belongs conceptually to the OTHER stage).
        nbandsvo_kmesh_defaultvalue = KpointsData(); nbandsvo_kmesh_defaultvalue.set_kpoints_mesh([8, 8, 8])
        spec.input('ns_converge.nbandsvo.kmesh_fixed_for_convergence', valid_type=KpointsData, required=False,
                   default=lambda: nbandsvo_kmesh_defaultvalue,
                   help="Fixed, low-density k-mesh for the NBANDSV/NBANDSO convergence stage (cheap "
                        "proxy, not the final production mesh). Default: [8,8,8].")
        spec.input('ns_converge.nbandsvo.convergence_threshold', valid_type=Float, required=False, default=lambda: Float(0.35), help="Convergence threshold (eV) for the NBANDSV/NBANDSO convergence stage only."  )
        spec.input('ns_converge.nbandsvo.select_earlier_point_at_convergence', valid_type=Bool,
                   required=False, default=lambda: Bool(False),
                   help="Prefer the earlier/cheaper point of the converged NBANDSV/NBANDSO "
                        "pair over the later/safer one, for the NBANDSV/NBANDSO convergence "
                        "stage only.")

        # Required=True on the child, but NOT here: only actually needed if
        # ns_converge.nbandsvo.enabled=True. Validated at runtime in run_nbands_convergence -
        # same no-default, required-only-when-needed pattern as ns_converge.kpoints.starting_mesh.
        spec.input('ns_converge.nbandsvo.threshold_start', valid_type=Float, required=False,
                   help="Starting optical window threshold (eV) for the first calculation. "
                        "Required if ns_converge.nbandsvo.enabled=True.")
        spec.input('ns_converge.nbandsvo.threshold_max', valid_type=Float, required=False,
                   help="Maximum optical window threshold (eV); convergence aborts if exceeded. "
                        "Required if ns_converge.nbandsvo.enabled=True.")
        spec.input('ns_converge.nbandsvo.bandsdata', valid_type=BandsData, required=False,
                   help="DFT band structure (with occupations) used to build the IPA transition "
                        "matrix. Required if ns_converge.nbandsvo.enabled=True.")

        # ---- Per-stage enable/disable switches ----
        # If a stage is disabled, its sub-workchain is never submitted; the corresponding
        # "superseded_*" value below is used directly for the final mBSE run instead (see
        # run_kpoints_convergence / run_nbands_convergence). These have NO default (unlike the
        # cheap proxy fields above, there's no defensible universal fallback for a final
        # production value) - they are validated as required AT RUNTIME, only when the matching
        # stage is actually disabled.
        spec.input('ns_converge.kpoints.enabled', valid_type=Bool, required=False, default=lambda: Bool(True),
                   help="If False, skip the k-point convergence stage entirely and use "
                        "ns_converge.kpoints.superseded_kmesh (required in that case) as the k-mesh "
                        "for the final mBSE run.")
        spec.input('ns_converge.kpoints.superseded_kmesh', valid_type=KpointsData, required=False,
                   help="Final k-mesh for the full mBSE run when ns_converge.kpoints.enabled=False. "
                        "Required in that case (validated at runtime - see run_kpoints_convergence); "
                        "unused/ignored otherwise.")
        spec.input('ns_converge.nbandsvo.enabled', valid_type=Bool, required=False, default=lambda: Bool(True),
                   help="If False, skip the NBANDSV/NBANDSO convergence stage entirely and use "
                        "ns_converge.nbandsvo.superseded_nbandsv/superseded_nbandso (required in that "
                        "case) for the final mBSE run.")
        spec.input('ns_converge.nbandsvo.superseded_nbandsv', valid_type=Int, required=False,
                   help="Final NBANDSV for the full mBSE run when ns_converge.nbandsvo.enabled=False. "
                        "Required in that case (validated at runtime - see run_nbands_convergence); "
                        "unused/ignored otherwise.")
        spec.input('ns_converge.nbandsvo.superseded_nbandso', valid_type=Int, required=False,
                   help="Final NBANDSO for the full mBSE run when ns_converge.nbandsvo.enabled=False. "
                        "Required in that case (validated at runtime - see run_nbands_convergence); "
                        "unused/ignored otherwise.")

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
        spec.exit_code(403, 'MISSING_SUPERSEDED_KMESH',
                       message='ns_converge.kpoints.enabled=False but ns_converge.kpoints.superseded_kmesh '
                               'was not supplied.')
        spec.exit_code(404, 'MISSING_SUPERSEDED_NBANDS',
                       message='ns_converge.nbandsvo.enabled=False but ns_converge.nbandsvo.superseded_nbandsv '
                               'and/or superseded_nbandso were not supplied.')
        spec.exit_code(405, 'MISSING_KPOINTS_STARTING_MESH',
                       message='ns_converge.kpoints.enabled=True but ns_converge.kpoints.starting_mesh '
                               'was not supplied.')
        spec.exit_code(406, 'MISSING_NBANDSVO_CONVERGENCE_INPUTS',
                       message='ns_converge.nbandsvo.enabled=True but one or more of '
                               'ns_converge.nbandsvo.bandsdata/threshold_start/threshold_max '
                               'were not supplied.')

        spec.outline(
            cls.run_kpoints_convergence,
            cls.run_nbands_convergence,
            cls.run_full_mbse,
            cls.finalize,
        )

    # ------------------------------------------------------------------
    #[1] Run k-point convergence with NBANDSV/O fixed to 2/2
    def run_kpoints_convergence(self):
        if not self.inputs.ns_converge.kpoints.enabled.value:
            if 'superseded_kmesh' not in self.inputs.ns_converge.kpoints:
                self.report(
                    "[VaspmBSECompleteWorkChain] ns_converge.kpoints.enabled=False but "
                    "ns_converge.kpoints.superseded_kmesh was not supplied; aborting."
                )
                return self.exit_codes.MISSING_SUPERSEDED_KMESH
            self.report(
                "[VaspmBSECompleteWorkChain] k-point convergence disabled "
                "(ns_converge.kpoints.enabled=False) -> using ns_converge.kpoints.superseded_kmesh "
                "directly as the k-mesh for the final mBSE run."
            )
            self.ctx.kmesh_converged = self.inputs.ns_converge.kpoints.superseded_kmesh
            return

        if 'starting_mesh' not in self.inputs.ns_converge.kpoints:
            self.report(
                "[VaspmBSECompleteWorkChain] ns_converge.kpoints.enabled=True but "
                "ns_converge.kpoints.starting_mesh was not supplied; aborting."
            )
            return self.exit_codes.MISSING_KPOINTS_STARTING_MESH

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

        inputs_kconv.ns_converge.static_inverse_diel = self.inputs.ns_BSE.static_inverse_diel
        inputs_kconv.ns_converge.screening_parameter = self.inputs.ns_BSE.screening_parameter
        if 'G0W0_gap' in self.inputs.ns_BSE:
            inputs_kconv.ns_converge.G0W0_gap = self.inputs.ns_BSE.G0W0_gap

        # BSE overrides ONLY for k-convergence ----
        # Force minimal BSE subspace - the idea is that we want to converge only the onset
        # as use it as proxy for the whole convergence.
        # Setting this overrides the logic based on the "optical_energy_window" input
        # Always present (default=2 each, see ns_converge.kpoints.NBANDSV/NBANDSO_fixed_for_convergence
        # spec.input above) - no need to guess a fallback here.
        inputs_kconv.ns_converge.kpoints = AttributeDict()
        inputs_kconv.ns_converge.kpoints.NBANDSV_fixed_for_convergence = self.inputs.ns_converge.kpoints.NBANDSV_fixed_for_convergence
        inputs_kconv.ns_converge.kpoints.NBANDSO_fixed_for_convergence = self.inputs.ns_converge.kpoints.NBANDSO_fixed_for_convergence
        inputs_kconv.ns_converge.kpoints.starting_mesh = self.inputs.ns_converge.kpoints.starting_mesh
        inputs_kconv.ns_converge.kpoints.max_mesh      = self.inputs.ns_converge.kpoints.max_mesh
        inputs_kconv.ns_converge.kpoints.step          = self.inputs.ns_converge.kpoints.step
        # Each convergence stage gets its OWN independent convergence_threshold at the master's
        # public-facing level (ns_converge.kpoints.* vs ns_converge.nbandsvo.*, see run_nbands_convergence) -
        # here we route the kpoints-stage one into the child's shared ns_converge.convergence_threshold port.
        inputs_kconv.ns_converge.convergence_threshold = self.inputs.ns_converge.kpoints.convergence_threshold
        # Same per-stage-split treatment as convergence_threshold above.
        inputs_kconv.ns_converge.select_earlier_point_at_convergence = self.inputs.ns_converge.kpoints.select_earlier_point_at_convergence
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
        # 'wc_kconv' only exists in ctx if run_kpoints_convergence actually submitted a child
        # (ns_converge.kpoints.enabled=True) - if it was disabled, ctx.kmesh_converged was already
        # set directly to superseded_kmesh and there is nothing to check here.
        if 'wc_kconv' in self.ctx:
            if not self.ctx.wc_kconv.is_finished_ok:
                self.report(
                    f"[VaspmBSECompleteWorkChain] k-point convergence WC <{self.ctx.wc_kconv.pk}> "
                    f"did not finish successfully (exit_status={self.ctx.wc_kconv.exit_status}); aborting."
                )
                return self.exit_codes.KPOINTS_CONVERGENCE_FAILED
            self.ctx.kmesh_converged = self.ctx.wc_kconv.outputs.kmesh_converged

        if not self.inputs.ns_converge.nbandsvo.enabled.value:
            missing = [name for name in ('superseded_nbandsv', 'superseded_nbandso')
                       if name not in self.inputs.ns_converge.nbandsvo]
            if missing:
                self.report(
                    f"[VaspmBSECompleteWorkChain] ns_converge.nbandsvo.enabled=False but "
                    f"ns_converge.nbandsvo.{'/'.join(missing)} not supplied; aborting."
                )
                return self.exit_codes.MISSING_SUPERSEDED_NBANDS
            self.report(
                "[VaspmBSECompleteWorkChain] NBANDSV/NBANDSO convergence disabled "
                "(ns_converge.nbandsvo.enabled=False) -> using ns_converge.nbandsvo.superseded_nbandsv/"
                "superseded_nbandso directly for the final mBSE run."
            )
            self.ctx.nbandsv_converged = self.inputs.ns_converge.nbandsvo.superseded_nbandsv
            self.ctx.nbandso_converged = self.inputs.ns_converge.nbandsvo.superseded_nbandso
            return

        missing_required = [name for name in ('bandsdata', 'threshold_start', 'threshold_max')
                             if name not in self.inputs.ns_converge.nbandsvo]
        if missing_required:
            self.report(
                f"[VaspmBSECompleteWorkChain] ns_converge.nbandsvo.enabled=True but "
                f"ns_converge.nbandsvo.{'/'.join(missing_required)} not supplied; aborting."
            )
            return self.exit_codes.MISSING_NBANDSVO_CONVERGENCE_INPUTS

        # Take everything the user gave to this master that is relevant to
        # VaspmBSENBandsConvWorkChain: ns_converge.* shared criteria fields,
        # ns_converge.nbandsvo.*, plus whatever VaspmBSEInitScriptWorkChain
        # inputs it also re-exposes.
        inputs_nbconv = AttributeDict(self.exposed_inputs(self._mbse_nbandsconv_wc))
        # Same leakage as in run_kpoints_convergence above, mirrored: strip the OTHER stage's
        # 'kpoints' sub-namespace, which isn't a valid port on VaspmBSENBandsConvWorkChain.
        inputs_nbconv.ns_converge.pop('kpoints', None)

        inputs_nbconv.ns_converge.static_inverse_diel = self.inputs.ns_BSE.static_inverse_diel
        inputs_nbconv.ns_converge.screening_parameter = self.inputs.ns_BSE.screening_parameter
        if 'G0W0_gap' in self.inputs.ns_BSE:
            inputs_nbconv.ns_converge.G0W0_gap = self.inputs.ns_BSE.G0W0_gap
        # Independent convergence_threshold for this stage - see the matching comment in
        # run_kpoints_convergence above. Read from the master's own ns_converge.nbandsvo.* (NOT
        # the leaked exposed_inputs blob) into the child's BARE ns_converge.convergence_threshold port.
        inputs_nbconv.ns_converge.convergence_threshold = self.inputs.ns_converge.nbandsvo.convergence_threshold
        # Same per-stage-split treatment as convergence_threshold above.
        inputs_nbconv.ns_converge.select_earlier_point_at_convergence = self.inputs.ns_converge.nbandsvo.select_earlier_point_at_convergence

        # Rebuild ns_converge.nbandsvo from scratch with an explicit allowlist of only the fields
        # VaspmBSENBandsConvWorkChain actually has - mirrors how run_kpoints_convergence rebuilds
        # ns_converge.kpoints above. This avoids relying on exposed_inputs()'s leaked full blob (which
        # also carries master-only fields like enabled/superseded_nbandsv/superseded_nbandso/
        # convergence_threshold that have no matching port on the child and would otherwise need to
        # be remembered and popped one by one).
        leaked_nbandsvo = self.inputs.ns_converge.nbandsvo
        inputs_nbconv.ns_converge.nbandsvo = AttributeDict()
        for field in ('bandsdata', 'threshold_start', 'threshold_max', 'threshold_step', 'num_bands_included'):
            inputs_nbconv.ns_converge.nbandsvo[field] = leaked_nbandsvo[field]

        # Fixed, low-density k-mesh for this stage: self-contained, independent default
        # (ns_converge.nbandsvo.kmesh_fixed_for_convergence, default [8,8,8] - see spec.input
        # above), no cross-dependency on ns_converge.kpoints.starting_mesh. The BSE band subspace
        # needed to cover a given IPA transition window is roughly k-mesh independent, so there is
        # no need to wait for kmesh_converged from step 1 either way.
        fixed_kmesh = leaked_nbandsvo.kmesh_fixed_for_convergence
        inputs_nbconv.ns_converge.nbandsvo.kmesh_fixed_for_convergence = fixed_kmesh

        running = self.submit(self._mbse_nbandsconv_wc, **inputs_nbconv)
        self.report(
            f"[VaspmBSECompleteWorkChain] Launched NBands convergence WC <{running.pk}> "
            f"on fixed k-mesh {fixed_kmesh.get_kpoints_mesh()[0]}"
        )
        return ToContext(wc_nbconv=running)

    # ------------------------------------------------------------------
    #[3] Run the full mBSE calculation on the converged k-mesh and converged NBANDSV/NBANDSO
    def run_full_mbse(self):
        # 'wc_nbconv' only exists in ctx if run_nbands_convergence actually submitted a child
        # (ns_converge.nbandsvo.enabled=True) - if it was disabled, ctx.nbandsv_converged/
        # nbandso_converged were already set directly from superseded_nbandsv/superseded_nbandso.
        if 'wc_nbconv' in self.ctx:
            if not self.ctx.wc_nbconv.is_finished_ok:
                self.report(
                    f"[VaspmBSECompleteWorkChain] NBands convergence WC <{self.ctx.wc_nbconv.pk}> "
                    f"did not finish successfully (exit_status={self.ctx.wc_nbconv.exit_status}); aborting."
                )
                return self.exit_codes.NBANDS_CONVERGENCE_FAILED
            self.ctx.nbandsv_converged = self.ctx.wc_nbconv.outputs.nbandsv_converged
            self.ctx.nbandso_converged = self.ctx.wc_nbconv.outputs.nbandso_converged

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

        # Replace kpoints with the converged (or fixed, if that stage was disabled) k-mesh, and
        # NBANDSV/NBANDSO with the converged (or fixed) band-subspace values - these always
        # override whatever the user may have passed in ns_BSE.NBANDSV/NBANDSO/optical_energy_window
        # when the corresponding convergence stage is enabled, since the purpose of this workchain
        # is to determine them through convergence rather than guess them upfront.
        inputs_full.kpoints = self.ctx.kmesh_converged
        inputs_full.ns_BSE.NBANDSV = self.ctx.nbandsv_converged
        inputs_full.ns_BSE.NBANDSO = self.ctx.nbandso_converged

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

        # k-mesh + NBANDSV/NBANDSO - either converged (sub-workchain output) or fixed (stage
        # disabled, value taken directly from inputs), set uniformly in ctx by run_nbands_convergence
        # / run_full_mbse above regardless of which path produced them.
        self.out('kmesh_converged',   self.ctx.kmesh_converged)
        self.out('nbandsv_converged', self.ctx.nbandsv_converged)
        self.out('nbandso_converged', self.ctx.nbandso_converged)

        optgap, _ = _extract_opticalgap_fromWorkchainNode(self.ctx.wc_full_mbse)
        node_optgap = Float(optgap)
        node_optgap.store()
        self.out('optical_gap', node_optgap )

        # Final mBSE outputs
        self.out('dielectrics',        self.ctx.wc_full_mbse.outputs.dielectrics)
        self.out('opticaltransitions', self.ctx.wc_full_mbse.outputs.opticaltransitions)

