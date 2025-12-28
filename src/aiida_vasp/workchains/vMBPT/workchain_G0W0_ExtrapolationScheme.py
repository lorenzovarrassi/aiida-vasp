# pylint: disable=too-many-arguments

import math
import numpy as np
from copy import deepcopy
from aiida.common.extendeddicts import AttributeDict
from aiida import orm
from aiida.orm import Code, Int, Float, Str, Dict, Bool , List , RemoteData , ArrayData , BandsData , XyData , KpointsData
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import WorkChain, calcfunction , ToContext , append_ , submit, while_
from aiida_vasp.utils.workchains import prepare_process_inputs
import warnings
from sklearn.linear_model import LinearRegression
from aiida_vasp.utils.workchains import site_magnetization_to_magmom


from .utils_helpers_extrapolation import get_closest_EncutNband_multiple , get_EncutNbandFitParams_completeBasis_quadratic , input_magnetic_moment_tomagmom
from .workchain_G0W0_base import VaspDFTGWWorkChain




class helper_extrapolations:
    @staticmethod
    def _extrapolate_series_dict( ar_nbandsInput, series_dict,
                                  keys_to_iterate=("Dir", "Ind", "Gam"),
                                  use_num_calc_for_extrapolation=3,
                                  series_name="series", ):
        """
        Generic extrapolation of multiple scalar series y_k versus 1/ar_x.

        Parameters
        ----------
        ar_x : Sequence[int|float]
            x-axis values (e.g. NBANDS), length Ncalc. Tipycally:
            List/array of NBANDS values used for the G0W0 datapoints,
            Length = Ncalc (number of GW datapoints).
            NOTE: Only the *last* `use_num_calc_for_extrapolation` points are used in the fit (moving average window).
            If Ncalc < use_num_calc_for_extrapolation, all available points are used.
        series_dict : dict[str, list[float]]
            dict-of-lists, each list length Ncalc.
            Tipycally represents:  dict-of-lists of G0W0 gaps.
            MUST contain the keys {"Dir","Ind","Gam"} , each with lists of length Ncalc.
            NOTE:  it must be ordered consistently with the ar_nbandsInput, meaning that
                   the i° element of each list of ns_gap comes from a calculation associated
                   to the i° ar_nbandsInput value.
        keys_to_iterate : iterable[str]
            Which keys in series_dict to extrapolate.
        use_num_calc_for_extrapolation : int
            Use last N points (moving window).
        """
        """ Fit scalar quantities versus 1/NBANDS and extrapolate to the infinite-basis limit.
        For each scalar y (gap or QP correction component), We perform a linear regression of y versus x = 1/NBANDS 
            y(NBANDS) ≈ y_inf + a * (1 / NBANDS)
        and report  - y_inf  : the intercept (extrapolated value at 1/NBANDS -> 0)
                    - r2     : coefficient of determination of the linear fit
        This function does NOT check physical consistency; it only fits series.

        Returns
        -------
        extra_dict: for each key of the dict the corresponding a fit is done on the corresponidng list 
                    and the extrapolated value is returned
        """

        def fit(ar_x, ar_y, use_num_calc_for_extrapolation):
                num_calc = min(len(ar_x), use_num_calc_for_extrapolation)
                x = (1.0 / np.asarray(ar_x[-num_calc:], dtype=float)).reshape(-1, 1)
                y = np.asarray(ar_y[-num_calc:], dtype=float)
                reg = LinearRegression().fit(x, y)
                return float(reg.intercept_), float(reg.score(x, y))
        
        # --- extrapolate  ---
        extrap_out = {"r2": {}}
        for k in keys_to_iterate:
                v, r2 = fit(ar_x=ar_nbandsInput, ar_y=series_dict[k], 
                            use_num_calc_for_extrapolation=use_num_calc_for_extrapolation)
                extrap_out[k] = v
                extrap_out["r2"][k] = r2
        
        # optional compact log
        str_log_ar_y=""
        for k in keys_to_iterate:
            ar_y = series_dict[k]
            ar_y = np.asarray(ar_y[-use_num_calc_for_extrapolation:], dtype=float)
            str_log_ar_y+=f"  > y =[key={k}]= {ar_y}\n"
        
        
        num_calc = min(len(ar_nbandsInput), use_num_calc_for_extrapolation)
        nb_fit = [int(v) for v in ar_nbandsInput[-num_calc:]]
        inv_nb = [float(1.0 / float(v)) for v in ar_nbandsInput[-num_calc:]]
        def _fmt_list(vals, fmt):
            return "[" + ", ".join(fmt.format(v) for v in vals) + "]"
        str_r2 = "  ".join(f"{k}={extrap_out['r2'][k]:.4f}" for k in keys_to_iterate)
        str_log_spin = ( f"\n"
                         f"  > nbands_fit={nb_fit}\n"
                         f"  > x = inverse(nbands)={_fmt_list(inv_nb, '{:.4e}')}\n" 
                         f"{str_log_ar_y}"
                         f"  > r2: {str_r2}\n" )
        return extrap_out, str_log_spin
        
    @staticmethod
    def _extrapolate_bands(ar_nbandsInput , bands_list , use_num_calc_for_extrapolation=3 ):
        """ Extrapolate *full band arrays* (G0W0 energies and QP corrections) versus 1/NBANDS.
        For each (k-point, band-index) entry, perform a linear regression over datapoints:
            E_G0W0(k,b; NBANDS) ≈ E_inf(k,b) + a(k,b) * (1 / NBANDS)
        It does this for absolute G0W0 band energies (bands_G0W0) and quasiparticle corrections (bands_QPc = bands_G0W0 - bands_DFT)
        It also returns a per-(k,b) R^2 map for both extrapolations.
    
        Parameters
        ----------
        inputs_array : list[AttributeDict]
            The list of inputs prepared for each VaspDFTGWWorkChain call.
            Used only to attempt to determine `tmp_min_nbandsGW`:
                tmp_min_nbandsGW = min([box.ns_parameters.nbandsgw for box in inputs_array])
            If `nbandsgw` is missing (e.g. metals/semimetals), it falls back to:
                tmp_min_nbandsGW = min(ar_nbandsInput)
            This truncation ensures we only extrapolate bands that are present in all datapoints.
            IMPORTANT: This assumes nbandsgw <= nbands for all datapoints and that the
            first `tmp_min_nbandsGW` bands are comparable across datapoints.
            WHY THIS? a G0W0 calculation has a lot (~hundreds ~thousands) of bands and usually
            and order of magnitude less of QPcorrection computed - we want to extrapolate only those.
    
        ar_nbandsInput : Sequence[int|float]
            List/array of NBANDS values used for the G0W0 datapoints,
            Length = Ncalc (number of GW datapoints).
            NOTE: Only the *last* `use_num_calc_for_extrapolation` points are used in the fit.
            If Ncalc < use_num_calc_for_extrapolation, all available points are used.
    
            IMPORTANT: This function currently uses *all points* in ar_nbandsInput for fits.
            (Unlike gaps extrapolation that uses a moving window.)
    
        runningWC_DFT_G0W0 : dict[int, WorkChainNode]
            List of VaspDFTGWWorkChain node.  Each node MUST expose:
                - outputs.bands_G0W0 : BandsData (AiiDA)
                - outputs.bands_DFT  : BandsData (AiiDA)
    
            The ordering of datapoints in the regression is taken from iterating this dict:
                [ runningWC_DFT_G0W0[WC_idx] for WC_idx in runningWC_DFT_G0W0 ]
    
            (If you want strict correctness, sort keys upstream and build arrays in that order.)
    
        Internal data shapes
        --------------------
        Let:  Ncalc = number of datapoints (GW calculations)
              Nkpt  = number of k-points
              Nband = tmp_min_nbandsGW
    
        The code builds:
          bands_G0W0_upToNBANDSGW_differentCalcStacked : shape (Ncalc, Nkpt, Nband)
          bands_DFT_upToNBANDSGW_differentCalcStacked  : shape (Ncalc, Nkpt, Nband)
          bands_QPc_upToNBANDSGW_differentCalcStacked  : shape (Ncalc, Nkpt, Nband)
    
        Returns
        -------
        extrapolated_bands : dict[str, np.ndarray]
            A plain Python dict suitable to wrap into AiiDA Dict:
                { "bands_G0W0":    ndarray (Nkpt, Nband)   # intercepts E_inf
                  "bands_G0W0_r2": ndarray (Nkpt, Nband)   # R^2 per entry
                  "bands_QPc":     ndarray (Nkpt, Nband)   # intercepts ΔE_inf
                  "bands_QPc_r2":  ndarray (Nkpt, Nband)   # R^2 per entry     }
    
        """
        n_tot = len(bands_list)
        n_fit = min(n_tot, int(use_num_calc_for_extrapolation))
        if n_fit <= 0:
            raise ValueError("bands_list is empty; cannot extrapolate.")
        
        #[1] Preparation: select the coresponding rist n_fit NBANDS values and bands
        #    + inverse the NBANDS array    
        ar_nbandsInput_widowed    = np.asarray(ar_nbandsInput[-n_fit:], dtype=float)
        ar_inverseNbands_widowed  = (1.0 / ar_nbandsInput_widowed).reshape(-1, 1)  # shape (n_fit, 1)
        bands_list_widowed = bands_list[-n_fit:]
    
        # Robust common truncation: only extrapolate bands present in all selected datapoints
        # (avoid relying on nbandsgw being present/consistent)
        tmp_min_nbands = min(b.shape[1] for b in bands_list_widowed)
        # Stack and truncate: (n_fit, Nkpt, Nband)
        bands_stack = np.stack( [b[:, :tmp_min_nbands] for b in bands_list_widowed] , axis=0)  
        c_nkpt, c_nbnd = bands_stack.shape[1], bands_stack.shape[2]
    
        # Allocate outputs: (Nkpt, Nband)
        bands_extrapolated    = np.zeros((c_nkpt, c_nbnd), dtype=float)
        bands_extrapolated_r2 = np.zeros((c_nkpt, c_nbnd), dtype=float)
    
        # Fit each (k,b) independently
        for idx_kpt in range(c_nkpt):
            for idx_b in range(c_nbnd):
                y  = bands_stack[:, idx_kpt, idx_b]
                reg = LinearRegression().fit(ar_inverseNbands_widowed, y)
                bands_extrapolated[idx_kpt, idx_b]    = reg.intercept_
                bands_extrapolated_r2[idx_kpt, idx_b] = reg.score(ar_inverseNbands_widowed, y)
        return { "bands":    bands_extrapolated,
                 "bands_r2": bands_extrapolated_r2, 
                 "metatada": {"n_fit": int(n_fit),
                              "num_of_bands_extrapolated": int(tmp_min_nbands),    }
                 }

    @staticmethod
    def _select_spin_block(block, sp_label):
        """ Return the spin-resolved sub-block if present, otherwise return `block` itself.
        This supports both:
        - non-spin: block is already the payload (no spin keys)
        - spin:     block contains "spinUp"/"spinDw" keys
        """
        return block[sp_label] if isinstance(block, dict) and sp_label in block else block
      
    @staticmethod
    def _collect_gap_and_qpc_series_for_spin_component(runningWC_DFT_G0W0 , sp):
        """ Collect the gaps and QPC from a list of-dicts into a dict-of-lists - for a specific spin channel.
        We have a list of dicts, i.e. list of WorkChainNode, each with .outputs.gaps
        In order to simply the extrapolation we want to collect all G0W0 Dir gaps from the different workchain into a single list
        and the same things for the other gap types.
                
        Notes :This function is does NOT recompute QPc from other outputs.
        If QPc keys are missing, it raises KeyError with a diagnostic message.
            
        Parameters
        ----------
        runningWC_DFT_G0W0 : dict[int, WorkChainNode]
            Mapping idx -> VaspDFTGWWorkChain node. Keys must be sortable.
            Each node must expose:
              - outputs.gaps : Dict with structure gaps["G0W0"][spin][Dir|Ind|Gam]
              - outputs.gaps_QPc : Dict with structure gaps_QPc[spin][Dir|Ind|Gam]
        sp : str
            Spin label, typically "spinUp" or "spinDw". For non-spin-polarized cases,
            you still pass "spinUp" and `_select_spin_block` will return the non-spin payload.
        
        Returns
        -------
        ns_gaps_sp : dict[str, list[float]]
                     {"G0W0_Dir":[...], "G0W0_Ind":[...], "G0W0_Gam":[...]}
        ns_qpc_sp : dict[str, list[float]]
                    {"HOMO_Dir":[...], ..., "LUMO_Gam":[...]}
        """
        ns_gaps_G0W0_sp = {"Dir" : [], "Ind": [], "Gam": [], }
        ns_gaps_qpc_sp  = { "Dir": [], "Ind": [], "Gam": [], }

        for i in sorted(runningWC_DFT_G0W0):
            wc = runningWC_DFT_G0W0[i]
        
            # ---- gaps ----
            gaps_dict  = wc.outputs.gaps.get_dict()["G0W0"]
            g0w0_spin  = helper_extrapolations._select_spin_block(gaps_dict, sp)
        
            qpc_dict = wc.outputs.gaps_QPc.get_dict()
            qpc_spin = helper_extrapolations._select_spin_block(qpc_dict, sp)
        
            for key in ["Dir","Ind","Gam"]:
                ns_gaps_G0W0_sp[key].append( float(g0w0_spin[key]) )
                ns_gaps_qpc_sp[key].append(  float(qpc_spin[key])  )

            
        return ns_gaps_G0W0_sp, ns_gaps_qpc_sp
        
    
    
    
    
# The workchain determines the quasiparticle energies (and gaps) extrapolated to infinity basis-set.
# Up to Four different G0W0 calculations are launched (via the VaspDFTGWWorkChain workchain), with different number of bands included.
# The cutoffs and number of bands of these calculations are constrained by the complete basis hypothesis, 
# that is by including all orbitals that the plane-wave basis set allows to calculate.
#
class VaspG0W0BasisExtrWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
            super(VaspG0W0BasisExtrWorkChain, cls).define(spec)        

            spec.expose_inputs(VaspDFTGWWorkChain  , exclude=('ns_parameters', 'ns_reference', 'ns_option', 'kpoints','options')) 
            #VaspDFTGWWorkChain expose:
            #   from spec.expose_inputs(WorkflowFactory('vasp.vasp') : options - potential_family - potential_mapping - kpoints
            #                                                          (parameters - settings of vasp.vasp are NOT exposed by  VaspDFTGWWorkChain)
            #   ns_parameters - ns_parallelization - ns_reference - ns_option - kpoints
            #Of those we expose only ns_parallelization; the rest is controlled internally.  
            
            spec.input('ns_extrapolation.encut_chi_fraction'    , valid_type=Float     , required=True  , default=lambda: Float(0.63)  , help='if not specified, ENCUTGW defined as 0.63 x ENCUT ; - if specified, the fraction provided is used' )
            spec.input('ns_extrapolation.nbands_stride'         , valid_type=Int       , required=False , help='minimum nbands steps used to increase the number of bands in the fit for the final (an other) modes')
            spec.input('ns_extrapolation.cutoff_fractions'      , valid_type=ArrayData , required=False , help='specify the fractions of the cutoff of the first calculation to be used for the extrapolation; it overrides the standard mode')
            spec.input('ns_extrapolation.cutoff_starting_value' , valid_type=Float     , required=False , help='specify the value of cutoff of the first calculation for the extrapolation; it overrides standard value, which is determined from DFTgr'  )   
            spec.input('ns_extrapolation.r2_threshold'          , valid_type=Float     , required=False , default=lambda: Float(0.85) , help='R2 threshold for the extrapolation; if the R2 of the fit is below this value for at least one of the gaps or QP corrections, an additional G0W0 calculation is performed (up to num_calc_touse_for_extrapolation) and the last three calculations are used for the extrapolation (i.e. a moving window of 3 calculations). Default is 0.85')
            
            spec.input('ns_parameters.magnetic_moment_onsite' , valid_type=Dict        , required=False , help='Starting collinear on-site magnetic moment ; Syntax is {ElName:value}')
            spec.input('ns_parameters.nomega'                 , valid_type=Int         , required=False , default=lambda: Int(96) , help='number of frequency points for the chi and sigma calculation in G0W0 runs. Default is 96.') 

            spec.input('kpoints'                              , valid_type=DataFactory('core.array.kpoints') , help='K-mesh used for VASP G0W0 and DFT runs; get_kpoints_mesh() must work.' )     

            spec.input('ns_reference.starting_RemoteData'     , valid_type=RemoteData  , required=False , help='Folder of the starting-point DFT ground-state; the workflows copies the starting point WAVECAR and CHGCAR from it; the CHGCAR is used only for magnetic collinear calcs.')
            spec.input('ns_reference.DFTgr_NGarray'           , valid_type=ArrayData   , required=True  , help='FFT grid used to determine the complete basis compatible with a given cutoff.')
            spec.input('ns_reference.DFTgr_ENMAXarray'        , valid_type=ArrayData   , required=True  , help='Array containing the ENMAX of all employed POTCARs.') 
            spec.input('ns_reference.DFTgr_kpoints'           , valid_type=DataFactory('core.array.kpoints') , help='K-mesh of the calculation exposing NGarray.' )

            spec.input('ns_option.maximum_iterations' , valid_type=Int  , required=False , default=lambda: Int(1) , help='maximum number of times the workchain will restart a crashed G0W0 runs' ) 

            spec.input('ns_option.options_for_extrapolation'  , valid_type=Dict , required=True,     help='Scheduler options (AiiDA Dict) used only for the initial DFT and the extrapolation workchains (both BS and NV), replacing the standard inputs.option.' )
            spec.input('ns_option.constraint_nbands_divisor'  , valid_type=Int  , required=False,    help='Enforce NBANDS to be a multiple of this value (i.e. NBANDS % nbands_divisor == 0).' )

            spec.output('ENMAX_referenceValue'        , valid_type=Float     , help='reference cutoff values used for determining the cutoff fractions in the extrapolation')
            spec.output('ENMAX_array'                 , valid_type=ArrayData , help='array of cutoff fractions values used for the extrapolation')
            spec.output('pairs_nbands_encuts'         , valid_type=XyData    , help='nbands - cutoff (eV) pairs used in the extrapolation')
                        
            spec.output('extrapolated',       valid_type=Dict , help='extrapolated gaps and QuasiParticle corrections.') 
            spec.output('extrapolated_bands', valid_type=Dict , help='extrapolated bands.') 
            spec.output('ns_maximum_num_calcgaps'     , valid_type=Dict , help='Dict with direct - indirect - Gamma gaps used for the extrapolation.')
            spec.output('ns_gaps_QPc'      , valid_type=Dict , help='Dict with the QP corrections (of the states associated to the direct - indirect - gamma gaps) used for the extrapolation.')
            spec.output('ns_gaps_G0W0'     , valid_type=Dict , help='Dict with the (direct - indirect - gamma) gaps used for the extrapolation.')

            spec.exit_code(402,'ONE_OR_MORE_GW_FAILED',message='One or more GW calculations failed.')

      
      #Original and correct
            spec.outline(
                cls.initialize,
                
                #Prepares the inputs of the several G0W0 data points which will be used for the extropolation.
                #the cutoffs - number of bands of these points are chosen in order to satisfy the complete basis constraint.
                cls.determine_completeBasis_encutNband,
                
                #Submit in parallel the DFT+G0W0 workflows.
                cls.prepare_run_wc_DFT_G0W0_firstThreeG0W0s,

                #If the extrapolation from the first 3 G0W0s show a R2 value below the threshold (0.85) 
                #for at least one of the gaps or QP corrections, run additional G0W0 calculations (up to num_calc_touse_for_extrapolation)
                #and use the last three calculations for the extrapolation (i.e. a moving window of 3 calculations) 
                while_(cls.are_r2_under_threshold)(
                    cls.prepare_run_wc_DFT_G0W0_additionalG0W0s,
                ),

                cls.elaborate_extrapolate_results,
                )

    def initialize(self):
        self.report('\n Started Basis-Extrapolation procedure with potentials: '+str(self.inputs.potential_mapping.get_dict())+"\n")
        
        self.ctx.finishedWC_DFTgr     = []
        self.ctx.finishedWC_DFT_G0W0  = []
        
        # [6] Regarding spin polarization
        self.ctx.is_spinpol  = ("magnetic_moment_onsite" in self.inputs.ns_parameters)
        self.ctx.spin_labels = ("spinUp", "spinDw") if self.ctx.is_spinpol else ("spinUp",)
        
      
    def prepare_run_wc_DFT_G0W0_firstThreeG0W0s(self): 
            """
            The workchain launches 3 (the precise value is defined by self.ctx.num_calc_used_for_extrapolation) istances of VaspDFTGWWorkChain (each represent a complete G0W0 data point). 
            This function prepares the correct input for each call of VaspDFTGWWorkChain.
            """
            #Preparing inputs for each workchain
            self.ctx.inputs_array = []
            self.ctx.runningWC_DFT_G0W0 = {}

            #We prepare the inputs for each call of VaspDFTGWWorkChain, which differ only for the encut and nbands values (and the encut_chi value, which is a fraction of encut)
            #  All inputs, as declared in the define of VaspDFTGWWorkChain, are saved inside the list self.ctx.inputs_array
            #  The idea is that the first entry in inputs_array corresponds to the first call of VaspDFTGWWorkChain AND the FIRST G0W0 data point for the extrapolation.
            #We prepare the inputs for all G0W0 data points which could be used, equal to self.ctx.max_num_runnable_G0W0_calcs
            #  However, only the first self.ctx.num_calc_used_for_extrapolation will be actually launched in this function.
            #  The other calculations will be launched only if the R2 of the extrapolation is below the threshold for at least one of the gaps.
            for ecutNbIdx in range( self.ctx.max_num_runnable_G0W0_calcs ):                
                self.ctx.inputs_array.append( AttributeDict({ 'ns_parameters':AttributeDict(), 'ns_option':AttributeDict(), 'ns_reference':AttributeDict() }) )
                self.ctx.inputs_array[ecutNbIdx].update(self.exposed_inputs(VaspDFTGWWorkChain))
                #ns_parallelization of VaspDFTGWWorkChain is not excluded from expose_inputs and thus is set by update
                self.ctx.inputs_array[ecutNbIdx].clean_workdir    = Bool(False)
                #self.ctx.finishedWC_DFTgr_NSP[-1].outputs.kpoints does not work for VASP G0W0s: it doesn't use VASP automatic generation but define manually the points inside KPOINTS - may give error in screened_2e.F -> use inputs values, which employs VASP automatic generation
                #self.ctx.finishedWC_DFTgr_NSP[-1].outputs.kpoints does not work for determine_completeBasis_encutNband: it requires explicit k-mesh -> in 
                self.ctx.inputs_array[ecutNbIdx]['kpoints'] = self.inputs.kpoints
                self.ctx.inputs_array[ecutNbIdx]['options'] = self.inputs.ns_option.options_for_extrapolation


                
                #This part is different among the different entry of inputs_array - because each entry has a different encut - bnads
                self.ctx.inputs_array[ecutNbIdx].ns_parameters.nbands   = Int(   self.ctx.EncutNbands_completeBasis[ecutNbIdx]["nbands"])
                self.ctx.inputs_array[ecutNbIdx].ns_parameters.encut    = Float( self.ctx.EncutNbands_completeBasis[ecutNbIdx]["encut"] )
                
                #These parameters are instead identical between the different calls of VaspDFTGWWorkChain ; same NOMEGA, same MAGMOM
                self.ctx.inputs_array[ecutNbIdx].ns_parameters.nomega   = self.inputs.ns_parameters.nomega
                if ('magnetic_moment_onsite' in self.inputs.ns_parameters ): 
                    self.ctx.inputs_array[ecutNbIdx].ns_parameters.magnetic_moment_onsite   = self.inputs.ns_parameters.magnetic_moment_onsite
                if ('nbandsgw' in self.inputs.ns_parameters ):   self.ctx.inputs_array[ecutNbIdx].ns_parameters.nbandsgw = self.inputs.ns_parameters.nbandsgw
                
                self.ctx.inputs_array[ecutNbIdx].ns_option.maximum_iterations  = self.inputs.ns_option.maximum_iterations
                self.ctx.inputs_array[ecutNbIdx].ns_option.calculation_label    = Str("extrapolation point "+str(ecutNbIdx) )               
                
                #Run_1DFTgr is false for the G0W0 data points because we reuse the single DFTgr done at the beginning (even if the encut is not really identical to the ones used for the extrapolation)
                #Thus that DFTgr is used a)to determine FFT grid (NGX NGY NGZ) and ENMAX - b)as starting point for the workflow_G0W0_base
                #Reusing DFTgr for the G0W0s data points allows to reduce the overall number of calculations; DFTgr is usually very quick compared to G0W0/DFTvo, but we are worried by the time spent in queue in HPC clusters.
                #However internal testing has been shown that launching a DFTvo (thus ALGO=Exact and NELM=1) on a DFTgr with a different encut may cause error in the band energies; the culprit is NELM=1
                if ('starting_RemoteData' in self.inputs.ns_reference ) :
                    self.ctx.inputs_array[ecutNbIdx].ns_reference.starting_RemoteData = self.inputs.ns_reference.starting_RemoteData        #self.ctx.finishedWC_DFTgr[-1].outputs.remote_folder               
                    self.ctx.inputs_array[ecutNbIdx].ns_option.run_1DFTgr  = Bool(False) 
                else:
                    self.ctx.inputs_array[ecutNbIdx].ns_option.run_1DFTgr   = Bool(True) 
                self.ctx.inputs_array[ecutNbIdx].ns_option.run_2DFTvo_3G0W0 = Bool(True) 

                #now let's manage the ENCUTGW terms:
                #By default they are kp at 0.63*ENCUT - otherwise is overriden.
                self.ctx.inputs_array[ecutNbIdx].ns_parameters.encut_chi = Float( self.ctx.EncutNbands_completeBasis[ecutNbIdx]["encut"] * self.inputs.ns_extrapolation.encut_chi_fraction.value )


            for ecutNbIdx in range( self.ctx.num_calc_touse_for_extrapolation ): 
                #Actually launching each workchain. The workchain are launched in parallel
                #See for details https://aiida.readthedocs.io/projects/aiida-core/en/v2.0.1/topics/workflows/usage.html
                #for ecutNbIdx in range(len(self.ctx.EncutNbands_completeBasis)): 
                self.ctx.runningWC_DFT_G0W0[ecutNbIdx] =  self.submit(VaspDFTGWWorkChain   , **self.ctx.inputs_array[ecutNbIdx]) 
                key = f'WC_DFT_G0W0_{ecutNbIdx}'
                self.to_context(**{key: self.ctx.runningWC_DFT_G0W0[ecutNbIdx]})

                self.report("\n  [preparing the VaspDFTGWWorkChain inputs]"
                    +"\n  > encut/nbands pair index ecutNbIdx:"+str(ecutNbIdx)
                    +"\n  > self.ctx.EncutNbands_completeBasis[ecutNbIdx]"+str(self.ctx.EncutNbands_completeBasis[ecutNbIdx])
                    +"\n  > self.ctx.EncutNbands_completeBasis[ecutNbIdx][nbands]"+str(self.ctx.EncutNbands_completeBasis[ecutNbIdx]["nbands"])
                    +"\n  > self.ctx.EncutNbands_completeBasis[ecutNbIdx][encut] "+str(self.ctx.EncutNbands_completeBasis[ecutNbIdx]["encut"] ) 
                    +"\n  > workchain node :"+str(self.ctx.runningWC_DFT_G0W0[ecutNbIdx])+"\n\n"  )

    def determine_completeBasis_encutNband(self): 
        """ The workflow runs a sequence of DFT+G0W0 calculations with increasing basis size.
        This functions Determine the parameters (ENCUT, NBANDS) to be used by these calculation under two constraints:
        	- All parameters couples(ENCUT, NBANDS) must respect the “complete-basis” constraint.
        	- The NBANDS must be a multiple of the value "self.ctx.nbands_stride" 
    
        General strategy ---------------------------------
        	- determine  1° ENCUT as max(ENMAX of the POTCARs involved).
        	- determined 1° NBANDS from the complete basis constraint from that ENCUT.
        	- determine the 2°, 3° NBANDS as
                NBANDS[2° extrapolation point] = NBANDS[1° extrapolation point] + self.ctx.nbands_stride
                NBANDS[3° extrapolation point] = NBANDS[1° extrapolation point] + self.ctx.nbands_stride*2
       	- determine the 2°, 3° ENCUT from the full basis constraint
        
        Additional constraints:---------------------------------
        The choice of the NBANDS values of all extrapolation points are subjected to additional constraints:
            1] Must be a multiple of the total number of [total num of MPI tasks)/(KPAR value)] (otherwise VASP does not read WAVEDER)
               used for the VASP calculations.
            2] self.ctx.nbands_stride must be >= 0.05 * NBANDS[1° extrapolation point]  (to avoid to close points)
        
        The first extrapolation point must be runnable with BOTH:
            (i)  the MPI layout (i.e. the total num of MPI tasks) used for the dense G0W0 calculation
            (ii) the MPI layout (i.e. the total num of MPI tasks) used for the extrapolation calculations
            Since we explicitly require num_mpi_dense % num_mpi_extrapolation == 0
            the minimal value compatible with both layouts is num_mpi_dense -> thus NBANDS of the first extrapolation point
            MUST be multiple of num_mpi_dense
   
        The subsequent extrapolation points are determined are run ONLY with the extrapolation MPI layout
            (i.e. we assume here that only first extrapolation point is run with the dense MPI setup)
            The Nbands of these extrapolations points should again satisfy the constraint of being a multiple of 
    
        Side effects / expected outputs:
          - sets context variables (e.g. self.ctx.extrPoint_1_starting_cutoff, self.ctx.inputs_array, etc.)
        """

        #[Part1] We extract the required data from the previous DFT ground state for the complete basis determination 
        #DFTgr_NGarray contains the fft dimensions along x , y , z.
        #POTCAR_ENMAX_max contains the maximum ENMAX between all POTCARs used
        DFTgr_NGarray     = self.inputs.ns_reference.DFTgr_NGarray 
        POTCAR_ENMAX_max  = Int( max( self.inputs.ns_reference.DFTgr_ENMAXarray.get_array('ENMAXarray') ) )  #maximum ENMAX between all POTCARs used (POTCAR_ENMAX_max)
        DFTgr_kpts        = self.inputs.ns_reference.DFTgr_kpoints
        DFTgr_cell        = self.inputs.structure
              

        #The extrapolation is governed by 4 parameters:
        # 1.1) extrPoint_1_starting_cutoff - the encut energy used for the first point in the extrapolation
        #                                    conventionally we set it to 1.00*POTCAR_ENMAX_max, where POTCAR_ENMAX_max is the maximum ENMAX between all POTCARs used.
        # 1.2) num_calc_touse_for_extrapolation - the maximum number of calculations used for the extrapolation (typically 4, but could be more with a very low nbands_stride for very large volumes)
        # 1.3) nbands_stride      - the minimum number of bands step used to increase the number of bands in the fit.
        # 1.4) extrPoints_cutoff_fractions   - the fractions of POTCAR_ENMAX_max used for the extrapolation; is an additional, older modes which is altarnative to the 3).
        # Let's define them one-by-one.

        #[1] Let's handle the case in which the user wants to override the standard value of num_calc_touse_for_extrapolation and has thus passed a custom value
        self.ctx.r2_threshold = self.inputs.ns_extrapolation.r2_threshold.value
        self.ctx.num_calc_touse_for_extrapolation = 3
        self.ctx.max_num_runnable_G0W0_calcs      = 4
        #The definition of extrPoints_cutoff_fractions overries the standard mode of determination of the encut-nbands pairs.
        #  num_calc_touse_for_extrapolation should still be = 3, the min is to avoid errors if the user passes just 2 fractions, which means that only 2 G0W0 calculations will be performed.
        #  max_num_runnable_G0W0_calcs is instead = len(cutoff_fractions).
        if hasattr(self.inputs.ns_extrapolation, 'cutoff_fractions') :
            self.ctx.num_calc_touse_for_extrapolation = min( self.ctx.num_calc_touse_for_extrapolation , len(self.inputs.ns_extrapolation.cutoff_fractions.get_array('cutoff_fractions'))  )
            self.ctx.max_num_runnable_G0W0_calcs      = len(self.inputs.ns_extrapolation.cutoff_fractions.get_array('cutoff_fractions'))


        #[2][ 1st Extrapolation Point : ENCUT ]
        #Let's handle the case in which the user wants to override the standard value of POTCAR_ENMAX_max and has thus passed a custom value
        #Then let's save the value (original or overridden) in the ctx.
        if ('cutoff_starting_value' in self.inputs.ns_extrapolation):    
            POTCAR_ENMAX_max = self.inputs.ns_extrapolation.cutoff_starting_value
            self.report(f"overriding ENMAX extracted from DFT ground state with one defined in self.inputs={self.inputs.ns_extrapolation.cutoff_starting_value}")
        self.ctx.extrPoint_1_starting_cutoff = Float(POTCAR_ENMAX_max)

        #[3][ 1st Extrapolation Point : NBANDS ]
        #The complete basis constraint allows to consider a single variables between the encut and nbands as indipendent, and is typically encut;
        #nbands=nbands(encut) is determined by counting how many plane waves are included inside a sphere (of radius determined by encut) 
        #centered in the brillouin zone on a given k-point.
        # in VASP GW, NBANDS MUST be a multiple of (total_mpithrd_num/KPAR); otherwise VASP rounds automatically NBANDS to
        # the nearest higher multiple of (total_mpithrd_num/KPAR);This is problematic, because 
        #   - In older VASP versions, this automatic rounding can break WAVEDER usage
        #     (as if the number bands differ between the current calc and the WAVEDER, WAVEDER is not read).
        #   - IT BREAKS the complete basis hypothesis for that G0W0 calculation.
        # Note: (total_mpithrd_num/KPAR) is called "GW_mpithrd_divisor"; NBANDS must be therefore a multiple of this value.
        # Thus the NBANDS must be subjected to several constraints:
        # a] All NBANDS of each extrapolation G0W0 calculation must be a multiple of total_mpithrd_num_for_extrapolation
        # b] The dense G0W0 calculation can be run  using a different number of MPI-threads
        #    thus we define a second GW_mpithrd_divisor (GW_nbands_divisor_for_dense)
        #    NBANDS of the first extrapolation point must be a multiple of also this value for the same reason
        #    Note that the 2nd, 3rd etc extrapolation points are run only with the extrapolation MPI setup (and not the dense)
        #    and thus should bea multiple of ONLY GW_nbands_divisor_for_dense
        # The GW_mpithrd_divisor must be subjected to two constraints:
        # a] total_mpithrd_num_for_extrapolation must be itself a divisor of GW_nbands_divisor_for_dense
        # b] an edge case should be managed:
        #       If kpar > total_mpithrd_num_for_extrapolation/[..]_for_dense, GW_mpithrd_divisor becomes 0.
        #       This could cause a crash in the later call of the function  get_closest_EncutNband_multiple; therefore we do a max(1, ..)
        total_mpithrd_num_for_extrapolation = (self.inputs.ns_option.options_for_extrapolation.get_dict()['resources']['num_machines'] *
                                               self.inputs.ns_option.options_for_extrapolation.get_dict()['resources']['num_mpiprocs_per_machine'] )
        kpar = self.inputs.ns_parallelization.kpar.value if ("kpar" in self.inputs.ns_parallelization) else 1 
        GW_nbands_divisor_for_extrapolation = max(1, (total_mpithrd_num_for_extrapolation // kpar) ) 
        GW_nbands_divisor_for_dense = ( self.inputs.ns_option.constraint_nbands_divisor.value if ('constraint_nbands_divisor' in self.inputs.ns_option) 
                                         else GW_nbands_divisor_for_extrapolation )
        assert ( max(GW_nbands_divisor_for_dense,GW_nbands_divisor_for_extrapolation) %
                 min(GW_nbands_divisor_for_dense,GW_nbands_divisor_for_extrapolation) ) == 0, "GW_nbands_divisor_for_dense GW_nbands_divisor_for_extrapolation are not multiplesof each others:ERROR"
   
        #If the extrapolation is applied in the standard mode, we:
        # 1) invert the relation nbands=nbands(encut) (determined by the complete-basis-hypothesis) to encut=encut(nbands). The solution is not analytical,
        #    we will use the inverted relation to determine the encut corresponding to a given nbands respecting the complete-basis-hypothesis
        # 2) determine nbands corresponding to the POTCAR_ENMAX_max encut value - this value is extrPoint_1_nbands; 
        # Note : get_closest_EncutNband_multiple returns a a list of dict [{'encut'; <Aiida-Float> , 'nbands': <Aiida-Int>}, thus we select nbands with ['nbands']} , <string representing log>] ; the [0] select the {'encut':..,'nbands':..} dict
        # Note : get_closest_EncutNband_multiple has input argument nbands_divisor_constraint which the resulting nbands MUST be multiple of.
        #        here ve pass GW_nbands_divisor_for_dense; because well'run a later G0W0 using extrPoint_1_nbands NBANDS with a total number MPI-threads = 
        # Note: get_closest_EncutNband_multiple returns a List([ output_array , string_log ])
        #       where output_array = AttributeDict() with keys "encut", "nbands"
        params_fit = get_EncutNbandFitParams_completeBasis_quadratic( kpoints=DFTgr_kpts , structure=DFTgr_cell , 
                                                                      FFT_NG_gridpoints=DFTgr_NGarray , POTCAR_ENMAX_max=POTCAR_ENMAX_max)
        extrPoint_1 = get_closest_EncutNband_multiple( kpoints=DFTgr_kpts    , structure=DFTgr_cell , 
                                                       FFT_NG_gridpoints=DFTgr_NGarray  , 
                                                       POTCAR_ENMAX_max=Float(POTCAR_ENMAX_max) , 
                                                       param_quad=params_fit , 
                                                       nbands_divisor_constraint=Int(GW_nbands_divisor_for_dense) , 
                                                       starting_parameter_value=Float(self.ctx.extrPoint_1_starting_cutoff) , 
                                                       input_type=Str("encut")      , 
                                                       flag_twoSidesRounding=False  , 
                                                       label_str_for_log=Str("1 extr.point" )
                                                     )[0]

        #[4][ 2nd,3nd,etc Extrapolation Point : NBANDS ]
        #Let's determine based on two constraint: minimum stride and MPI-multiple:
        #a] Let's ensure that nbands_stride is at least 5% of extrPoint_1['nbands'] - this avoid many calculations with too close nbands values for large volumes and different calculations with very similar nbands,
        #   where numerical noise could be dominant.
        #b] let's ensure that nbands_stride is a multiple of #MPI-threads 
        #   (because VASP rounds NBANDS to the closest multiple of #MPI-threads/KPAR, and that would create problems), while reading WAVECAR/WAVEDER.
        #   and we determine the minimum multiple of  GW_nbands_divisor_for_extrapolation which is closer (from the above, i.e. still >=) to minimum_nbands_stride
        self.ctx.minimum_nbands_stride = int(0.05*extrPoint_1['nbands'])
        self.ctx.nbands_stride = math.ceil( self.ctx.minimum_nbands_stride / GW_nbands_divisor_for_extrapolation) * GW_nbands_divisor_for_extrapolation

        #Let's handle the case in which the user wants to override the standard value of nbands_stride and has thus passed a custom value
        if ('nbands_stride' in self.inputs.ns_extrapolation ): self.ctx.nbands_stride = self.inputs.ns_extrapolation.nbands_stride.value

        extrPoints_nbands_array = [extrPoint_1['nbands'] + i * self.ctx.nbands_stride for i in range(self.ctx.max_num_runnable_G0W0_calcs)]
        #[5][ 2nd,3nd,etc Extrapolation Point : ENCUT and OVERRIDE CASE]
        #Let's call get_closest_EncutNband_multiple for each nbands value in nbansds_array
        #Nbands is in steps of 0.2*extrPoint_1['nbands'] -> and determining the corresponding encut by the inverted relation encut=encut(nbands) 
        #(in order to satisfy the complete basis hypothesis).
        #The number of calculations is determined by num_calc_touse_for_extrapolation.
        #The results are saved in self.ctx.EncutNbands_completeBasis , which is a list of dict [{'encut': <Aiida-Float> , 'nbands': <Aiida-Int>} , ...] while _logger is a string containing a log of the determinations.
        #Let's also handle the case in which the user wants to use the mode defined by cutoff_fractions.
        # If the user has specified the cutoff_fractions, we use them instead of the previous way to determine the encut-nbands pairs.

        self.ctx.EncutNbands_completeBasis = List()
        self.ctx.EncutNbands_completeBasis_logger = str("")
        if not ('cutoff_fractions' in self.inputs.ns_extrapolation):        
            for nb_idx , nb in enumerate( extrPoints_nbands_array ):
                [extrPoints_dict_nbcutoff , extrPoints_logger] = get_closest_EncutNband_multiple(kpoints=DFTgr_kpts, structure=DFTgr_cell , 
                                                                                                 FFT_NG_gridpoints=DFTgr_NGarray  , 
                                                                                                 POTCAR_ENMAX_max=Float(POTCAR_ENMAX_max) , 
                                                                                                 param_quad=params_fit , 
                                                                                                 nbands_divisor_constraint=Int(self.ctx.nbands_stride) , 
                                                                                                 starting_parameter_value=Float(nb) , input_type=Str("nbands"), 
                                                                                                 flag_twoSidesRounding=Bool(True)   ,
                                                                                                 label_str_for_log=Str(str(nb_idx)+" extr.point")  )
                self.ctx.EncutNbands_completeBasis.append( extrPoints_dict_nbcutoff )
                self.ctx.EncutNbands_completeBasis_logger = self.ctx.EncutNbands_completeBasis_logger + extrPoints_logger + "\n"
        else:
            self.ctx.extrPoints_cutoff_fractions = self.inputs.ns_extrapolation.cutoff_fractions.get_array('cutoff_fractions')
            extrPoints_cutoff_array = np.multiply(self.ctx.extrPoints_cutoff_fractions , POTCAR_ENMAX_max)
            for ec_idx , ec in enumerate( extrPoints_cutoff_array ):
                [extrPoints_dict_nbcutoff , extrPoints_logger] = get_closest_EncutNband_multiple(kpoints=DFTgr_kpts, structure=DFTgr_cell , 
                                                                                                 FFT_NG_gridpoints=DFTgr_NGarray  , 
                                                                                                 POTCAR_ENMAX_max=Float(POTCAR_ENMAX_max) , 
                                                                                                 param_quad=params_fit , 
                                                                                                 nbands_divisor_constraint=Int(GW_nbands_divisor_for_extrapolation) , 
                                                                                                 starting_parameter_value=Float(ec) , input_type=Str("encut") , 
                                                                                                 flag_twoSidesRounding= Bool(True)  ,
                                                                                                 label_str_for_log=Str(str(ec_idx)+" extr.point"  )  )
                self.ctx.EncutNbands_completeBasis.append( extrPoints_dict_nbcutoff )
                self.ctx.EncutNbands_completeBasis_logger = self.ctx.EncutNbands_completeBasis_logger + extrPoints_logger + "\n"
        

        #Inputs handling - printing the input data if requested.
        final_str_log = (
             "\n[preparatory-1 - input of DFT ground state]-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---"
            +"\n  Inside routine determine_completeBasis_encutNband - list input data determined from DFTgr"
            +'\n  > FFT grid NG array   : '+str(DFTgr_NGarray)
            +'\n  > max ENMAX           : '+str(POTCAR_ENMAX_max)
            +'\n  > kpoints mesh        : '+str(DFTgr_kpts)
            +'\n  > cell                : '+str(DFTgr_cell)
            +'\n  > nbands_stride       : '+str(self.ctx.nbands_stride)
            +'\n  > GW_nbands_divisor_for_extrapolation : '+str(GW_nbands_divisor_for_extrapolation)
            +'\n  > GW_nbands_divisor_for_dense         : '+str(GW_nbands_divisor_for_dense)
            +"\n[1 - constraint to the NBANDS]-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- "
            +"\n  > constraint to nbands[ of the 1st extr.points] - it must be a multiple of "
            +"\n    GW_nbands_divisor_for_dense         : "+str(GW_nbands_divisor_for_dense)
            +"\n  > constraint to nbands[ of all extr.points] - they must be multiples of "
            +"\n    GW_nbands_divisor_for_extrapolation : "+str(GW_nbands_divisor_for_extrapolation)    
            +"\n  > constraint to extr.point > 1st - nbands_stride must > 5% of nbands of 1st extr.point"
            +"\n    minimum_nbands_stride : "+str(self.ctx.minimum_nbands_stride)
            +"\n  ↪	final nbands_stride   : "+str(self.ctx.nbands_stride)
            +"\n[2 - determining (encut,nbands) for all extr.points]-- - --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- "    
            +"\n  > corrected encut-nbands couples : "
            +"\n    "+"\n    ".join([str(list_point) for list_point in self.ctx.EncutNbands_completeBasis.value])
            +"\n  > full logs :"
            +"\n"+self.ctx.EncutNbands_completeBasis_logger
			+'\n[2 - determining (encut,nbands) -> determining corrected (encut,nbands)]--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ')
        self.report(final_str_log)
    
    def are_r2_under_threshold(self):
        """ Decide whether to run an additional GW datapoint based on the quality of the gap extrapolation.
        Data sources (required per datapoint WorkChainNode)
        ---------------------------------------------------
        self.ctx.runningWC_DFT_G0W0 : dict[int, WorkChainNode]
            Mapping idx -> VaspDFTGWWorkChain node. Keys must be sortable.
            
            For each WorkChainNode `wc` in self.ctx.runningWC_DFT_G0W0:
              wc.outputs.gaps.get_dict() must contain:
                gaps["G0W0"]["spinUp"]["Dir"|"Ind"|"Gam"]
                gaps["G0W0"]["spinDw"]["Dir"|"Ind"|"Gam"]  (if magnetic)
            
              QP-correction scalars are taken from wc.outputs.gaps_QPc if present with keys:
                gaps_QPc["spinUp"][HOMO_Dir/HOMO_Ind/HOMO_Gam/LUMO_*]
                gaps_QPc["spinDw"][...]
              otherwise they are computed from wc.outputs.bnd_extrema, which must contain:
                bnd_extrema["DFT"]["spinUp"]["HOMO"/"LUMO"]   (per-kpoint arrays)
                bnd_extrema["G0W0"]["spinUp"]["HOMO"/"LUMO"]
                and same for "spinDw".
            
        Convergence criterion
        ---------------------
        Only the R² values of the three G0W0 gap fits (Dir/Ind/Gam) are checked against
        self.inputs.ns_extrapolation.r2_threshold.value.
    
        Returns
        -------
        Bool(True)  -> run an additional datapoint (consider the workflow not converged and still below max calcs)
        Bool(False) -> converged OR max number of datapoints reached
        """
        
        #[1] Input handling
        self.ctx.r2_threshold = float(self.inputs.ns_extrapolation.r2_threshold.value)
        str_log = ( f"\n  VaspG0W0BasisExtrWorkChain pk={self.node.pk} checking if an additional VaspDFTGWWorkChain is required"
                    f"\n  convergence test uses only G0W0 gaps (Dir/Ind/Gam); R² threshold = {self.ctx.r2_threshold}"            )
    

    
        #[3] extrapolate per spin channel ----
        extrap_gap_G0W0, extrap_qpc_QPc = {}, {},
        for sp in self.ctx.spin_labels:
            #[3.1] First Collect the gaps and QPC from a list of-dicts into a dict-of-lists - for a specific spin channel.
            #We have a list of dicts, i.e. list of WorkChainNode, each with .outputs.gaps
            #In order to simply the extrapolation we want to collect all G0W0 Dir gaps from the different workchain into a single list
            #and the same things for the other gap types.
            #The outputting dict are 
            #ns_gaps_sp :  {"G0W0_Dir":[...], "G0W0_Ind":[...], "G0W0_Gam":[...]}
            #ns_qpc_sp  :  {"HOMO_Dir":[...], ..., "LUMO_Gam":[...], etc}
            ns_gaps_G0W0_sp, ns_gaps_qpc_sp = helper_extrapolations._collect_gap_and_qpc_series_for_spin_component(self.ctx.runningWC_DFT_G0W0 , sp)

            #[3.2] Collect NBANDS series
            #Iteration order of a dict is insertion order, not necessarily sorted. -> thus sorted on keys
            wc_keys = sorted(self.ctx.runningWC_DFT_G0W0)
            ar_nbandsInput  = [ self.ctx.runningWC_DFT_G0W0[WC_idx].inputs.ns_parameters['nbands'].value for WC_idx in wc_keys ]
        

            #[3.3] Extrapolate
            ns_gaps_G0W0_sp_extr, slog_gap_g0w0 = helper_extrapolations._extrapolate_series_dict(
                  ar_nbandsInput=ar_nbandsInput,  series_dict=ns_gaps_G0W0_sp, 
                  keys_to_iterate=("Dir","Ind","Gam"),
                  use_num_calc_for_extrapolation=self.ctx.num_calc_touse_for_extrapolation,
                  series_name="G0W0 gaps",   )
            ns_gaps_qpc_sp_extr, slog_gap_qpc   = helper_extrapolations._extrapolate_series_dict(
                  ar_nbandsInput=ar_nbandsInput,  series_dict=ns_gaps_qpc_sp, 
                  keys_to_iterate=("Dir","Ind","Gam"), 
                  use_num_calc_for_extrapolation=self.ctx.num_calc_touse_for_extrapolation,
                  series_name="QPc gaps", )
            str_log += (f"\n  [Spin: {sp}][Extrapolation Data for G0W0 gap]"       + slog_gap_g0w0 +
                        f"  [Spin: {sp}][Extrapolation Data for QP correction gap]"+ slog_gap_qpc )
            extrap_gap_G0W0[sp] = ns_gaps_G0W0_sp_extr
            extrap_qpc_QPc[sp]  = ns_gaps_qpc_sp_extr
    
        # ---- convergence check (gaps only) ----
        def _gaps_ok(extrap , threshold):
            return ( (extrap["r2"]["Dir"]     >= threshold)
                     and (extrap["r2"]["Ind"] >= threshold)
                     and (extrap["r2"]["Gam"] >= threshold) )
    
        flag_is_extrapolation_converged = all(_gaps_ok(extrap_gap_G0W0[sp], self.ctx.r2_threshold) for sp in self.ctx.spin_labels)
    
        # ---- decision ----
        if flag_is_extrapolation_converged:
            self.report(str_log + "  --> THUS: NO ADDITIONAL CALC is required, extrapolation is converged.\n")
            return Bool(False)
    
        if len(self.ctx.runningWC_DFT_G0W0) >= self.ctx.max_num_runnable_G0W0_calcs:
            self.report(str_log + "  --> THUS: NO ADDITIONAL CALC. Maximum number of calculations reached.\n")
            return Bool(False)
    
        self.report(str_log + "  --> THUS: ADDITIONAL CALC IS required, extrapolation is NOT converged.\n")
        return Bool(True)
       
    def prepare_run_wc_DFT_G0W0_additionalG0W0s(self):
        self.report("\n\n Launching additional calculation fro extrapolation")
        print(len(self.ctx.runningWC_DFT_G0W0) , self.ctx.max_num_runnable_G0W0_calcs)
        print(len(self.ctx.EncutNbands_completeBasis) , self.ctx.num_calc_touse_for_extrapolation)

        ecutNbIdx = len(self.ctx.runningWC_DFT_G0W0)
        self.ctx.runningWC_DFT_G0W0[ecutNbIdx] =  self.submit(VaspDFTGWWorkChain   , **self.ctx.inputs_array[ecutNbIdx]) 
        key = f'WC_DFT_G0W0_{ecutNbIdx}'
        self.to_context(**{key: self.ctx.runningWC_DFT_G0W0[ecutNbIdx]})

    def elaborate_extrapolate_results(self):
        """
        Extract direct/indirect/Gamma gaps and QP corrections from all G0W0 datapoints,
        extrapolate them to the infinite basis-set limit, and output the results.
        """
    
        # ---------------------------------------------------------
        #[1] Output ENMAX reference values (useful bookkeeping)
        self.ctx.extrPoint_1_starting_cutoff.store()
        self.out("ENMAX_referenceValue", self.ctx.extrPoint_1_starting_cutoff)
        self.out("ENMAX_array", self.inputs.ns_reference.DFTgr_ENMAXarray)
    
        # ---------------------------------------------------------
        #[2] Collect nbands / encut inputs + report/return
        #Iteration order of a dict is insertion order, not necessarily sorted. -> thus sorted on keys
        wc_keys = sorted(self.ctx.runningWC_DFT_G0W0)
    
        ar_isFinishedOk = [self.ctx.runningWC_DFT_G0W0[i].is_finished_ok for i in wc_keys]
        ar_nbandsInput  = [self.ctx.runningWC_DFT_G0W0[i].inputs.ns_parameters["nbands"].value for i in wc_keys]
        ar_encutInput   = [self.ctx.runningWC_DFT_G0W0[i].inputs.ns_parameters["encut"].value  for i in wc_keys]
    
        str_log = (  "\n > [1] ar_isFinishedOk : " + str(ar_isFinishedOk)
                   + "\n   [2] PAW potentials  : " + str(self.inputs.potential_mapping.get_dict())
                   + "\n > [2] ar_nbandsInput  : " + str(ar_nbandsInput)
                   + "\n > [2] ar_encutInput   : " + str(ar_encutInput)         )
    
        if not all(ar_isFinishedOk):
            return self.exit_codes.ONE_OR_MORE_GW_FAILED
    
        #[3] bookkeeping output : Save encut/nbands pairs
        ar_encut_nbands = XyData()
        ar_encut_nbands.set_x(np.array(ar_encutInput), "encut", "eV")
        ar_encut_nbands.set_y(np.array(ar_nbandsInput), "nbands", "")
        ar_encut_nbands.store()
        self.out("pairs_nbands_encuts", ar_encut_nbands)
    
        # ---------------------------------------------------------
        #[4] Extract gaps and QPc series (explicit, normalized)

        ns_gaps_g0w0_dict = {} ;ns_gaps_qpc_dict = {}
        for sp in self.ctx.spin_labels:
            ns_gaps_g0w0_dict[sp], ns_gaps_qpc_dict[sp] = helper_extrapolations._collect_gap_and_qpc_series_for_spin_component(self.ctx.runningWC_DFT_G0W0, sp)
        ns_gaps_g0w0_dict = Dict(dict=ns_gaps_g0w0_dict)
        ns_gaps_qpc_dict  = Dict(dict=ns_gaps_qpc_dict )
        ns_gaps_g0w0_dict.store()
        ns_gaps_qpc_dict.store()
        self.out("ns_gaps_G0W0", ns_gaps_g0w0_dict )
        self.out("ns_gaps_QPc",  ns_gaps_qpc_dict  )
                
        # ---------------------------------------------------------
        #[5] Extrapolate scalars
        extrap_gap_G0W0, extrap_qpc_QPc = {}, {},
        for sp in self.ctx.spin_labels:
            ns_gaps_G0W0_sp_extr, slog_gap_g0w0 = helper_extrapolations._extrapolate_series_dict(
                          ar_nbandsInput=ar_nbandsInput,  series_dict=ns_gaps_g0w0_dict[sp],
                          keys_to_iterate=("Dir","Ind","Gam"),
                          use_num_calc_for_extrapolation=self.ctx.num_calc_touse_for_extrapolation,
                          series_name="G0W0 gaps",   )
            
            ns_gaps_qpc_sp_extr, slog_gap_qpc   = helper_extrapolations._extrapolate_series_dict(
                          ar_nbandsInput=ar_nbandsInput,  series_dict=ns_gaps_qpc_dict[sp],
                          keys_to_iterate=("Dir","Ind","Gam"), 
                          use_num_calc_for_extrapolation=self.ctx.num_calc_touse_for_extrapolation,
                          series_name="QPc gaps", )
            
            slog_gap_g0w0 = slog_gap_g0w0.replace("\n","\n  [Extr]")
            slog_gap_qpc  = slog_gap_qpc.replace("\n","\n  [Extr]")
            str_log += (f"\n  [Spin: {sp}][Extrapolation Data for G0W0 gap]"       + slog_gap_g0w0 +
                        f"  [Spin: {sp}][Extrapolation Data for QP correction gap]"+ slog_gap_qpc )
            extrap_gap_G0W0[sp] = ns_gaps_G0W0_sp_extr
            extrap_qpc_QPc[sp]  = ns_gaps_qpc_sp_extr
        
        
        extrapolated = Dict(dict={"gaps": ns_gaps_G0W0_sp_extr, "gaps_QPc": ns_gaps_qpc_sp_extr, })
        extrapolated.store()
        self.out("extrapolated", extrapolated)
        # [5b] Custom final report
        n_fit = min(int(self.ctx.num_calc_touse_for_extrapolation), len(ar_nbandsInput))
        def _fmt_list(vals):
            return "[" + ", ".join(f"{float(v):.4f}" for v in vals) + "]"
        report_lines = []
        for sp in self.ctx.spin_labels:
            # optional spin header (remove if you want it fully flat)
            report_lines.append(f"\n  [Final gap report][spin={sp}]")
    
            for key in ("Dir", "Ind", "Gam"):
                ar_vals = ns_gaps_g0w0_dict[sp][key]  # full series (all datapoints)
                report_lines.append(f" > [{n_fit}] ar_bandGap_G0W0_{key}={_fmt_list(ar_vals)}")
    
            for key in ("Dir", "Ind", "Gam"):
                val_node = orm.Float(extrap_gap_G0W0[sp][key])          # unstored on purpose
                r2_node  = orm.Float(extrap_gap_G0W0[sp]["r2"][key])    # unstored on purpose
                report_lines.append(
                    f" >  bandGap_G0W0_{key}_extrapolated={val_node} (r^2={r2_node})"
                )
    
        self.report(str_log + "\n" + "\n".join(report_lines))



        
        # ---------------------------------------------------------
        # [6] Extrapolate full bands (G0W0 and QPc), using same window as scalars
        wc_keys = sorted(self.ctx.runningWC_DFT_G0W0)
        
        ar_nbandsInput  = [ self.ctx.runningWC_DFT_G0W0[WC_idx].inputs.ns_parameters['nbands'].value for WC_idx in self.ctx.runningWC_DFT_G0W0 ]
        bands_G0W0_list = [ self.ctx.runningWC_DFT_G0W0[WC_idx].outputs.bands_G0W0.get_bands() for WC_idx in self.ctx.runningWC_DFT_G0W0 ]
        bands_DFT_list  = [ self.ctx.runningWC_DFT_G0W0[WC_idx].outputs.bands_DFT.get_bands() for WC_idx in self.ctx.runningWC_DFT_G0W0 ]
        ex_gw  = helper_extrapolations._extrapolate_bands( ar_nbandsInput=ar_nbandsInput, 
                                                           bands_list=bands_G0W0_list,
                                                           use_num_calc_for_extrapolation=3, )
        extrapolated_bands = { "bands_G0W0":    ex_gw["bands"],   # .tolist(),
                               "bands_G0W0_r2": ex_gw["bands_r2"],# .tolist(),
                               "meta_G0W0":     ex_gw["metatada"], }
        
        try:
            bands_QPc_list = [gw - dft for gw, dft in zip(bands_G0W0_list, bands_DFT_list)]   
            ex_qpc = helper_extrapolations._extrapolate_bands(
                ar_nbandsInput=ar_nbandsInput, bands_list=bands_QPc_list,
                use_num_calc_for_extrapolation=3,    )
            extrapolated_bands["bands_QPc"]    = ex_qpc["bands"].tolist()
            extrapolated_bands["bands_QPc_r2"] = ex_qpc["bands_r2"].tolist()
            extrapolated_bands["meta_QPc"]     = ex_qpc["metatada"]    
        except Exception as exc:
            self.report(f"Warning : calculation of bands_QP_list failed with error:{Exception}")
        
        extrapolated_bands = Dict(dict=extrapolated_bands)
        extrapolated_bands.store()
        self.out("extrapolated_bands", extrapolated_bands)
        
    def clean_remoteFolder_DFT(self):
            from aiida import orm
            self.report(orm.CalcJobNode)

            cleaned_calcs = []
            for WC_DFTgr in self.ctx.finishedWC_DFTgr:
                for called_descendant_DFTgr in WC_DFTgr.called_descendants:
                    if isinstance(called_descendant_DFTgr, orm.CalcJobNode):
                        self.report(called_descendant_DFTgr)
                        self.report(called_descendant_DFTgr.outputs.remote_folder)
                        try:
                            called_descendant_DFTgr.outputs.remote_folder._clean() # pylint: disable=protected-access
                            cleaned_calcs.append(called_descendant_DFTgr.pk)
                        except (IOError, OSError, KeyError):
                            pass

            for WC_DFT in self.ctx.finishedWC_DFT_G0W0:
                for called_descendant_DFT in WC_DFT.called_descendants:
                    if isinstance(called_descendant_DFT, orm.CalcJobNode):
                        self.report(called_descendant_DFT)
                        self.report(called_descendant_DFT.outputs.remote_folder)
                        try:
                            called_descendant_DFT.outputs.remote_folder._clean() # pylint: disable=protected-access
                            cleaned_calcs.append(called_descendant_DFT.pk)
                        except (IOError, OSError, KeyError):
                            pass

