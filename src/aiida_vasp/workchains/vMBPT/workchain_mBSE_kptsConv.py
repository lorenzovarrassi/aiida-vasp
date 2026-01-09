import numpy as np
from copy import deepcopy
from aiida.orm import Int, Float, Dict, Bool , List , RemoteData , KpointsData, Str
from dataclasses import dataclass
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine  import WorkChain, ToContext , append_, submit, while_ 
from aiida.common.extendeddicts  import AttributeDict
from typing import Optional

from .workchain_mBSE_base_winterpolation import VaspmBSEInitScriptWorkChain
from .utils_helpers_mBSE import _extract_opticalgap_fromWorkchainNode


from aiida import load_profile
load_profile()

#Miscellaneous utils functions
class helper_kptsConv_mBSE:
    @staticmethod
    def _get_kmesh_from_kdensity(latVec , KSPACING , flag_roundInsteadCeil=True ):
            """
            Calculate the k-point mesh dimensions for a given lattice and k-point spacing.
            Args:
              - latVec (array-like): A 3x3 array representing the lattice vectors of the unit cell.
              - KSPACING (in Angstrom^{-1}) represents the smallest allowed spacking between k-points in the BZ; SMALLER values produce DENSER meshes;
                Conversely, The output number of divisions Ni is chosen as the maximum integer that satisfies |b_i| / Ni <= KSPACING
            """
    
            latVec = np.array( latVec )
            recLatVec= np.zeros((3,3))
            Vol= np.abs( np.dot(latVec[0,:] , np.cross(latVec[1,:],latVec[2,:])) )
            recLatVec[0,:]= np.cross(latVec[1,:],latVec[2,:])  /Vol
            recLatVec[1,:]= np.cross(latVec[2,:],latVec[0,:])  /Vol
            recLatVec[2,:]= np.cross(latVec[0,:],latVec[1,:])  /Vol
    
            # List of dim=3 with dimensions of the 3 rec.vectors.
            rec_cell_norm = np.array( [np.linalg.norm( recLatVec[x,:]) for x in range(3)] )
            
            #Exact (and thus fractional) K-mesh corresponding to the EXACT KSPACING
            #  For example, for rec_cell_norm = [0.319, 0.319, 0.319] :    
            #  KSPACING = 0.5 -> array([4.0084, 4.0084, 4.0084])
            #  KSPACING = 0.3 -> array([6.6807, 6.6807, 6.6807])
            kmesh_ideal_fractional = np.array(rec_cell_norm) * 2*np.pi / KSPACING
    
            if flag_roundInsteadCeil:   kmesh = [ max(1.0,np.round(k)) for k in kmesh_ideal_fractional ]
            else:                       kmesh = np.ceil( kmesh_ideal_fractional )
            kmesh = np.array( kmesh ).astype(int)
            return kmesh

    @staticmethod    
    def _get_kspacing_from_kmesh(latVec, kmesh): 
            #k-mesh = 2pi * |bi|/ k-density  ->DataFactory('core.array.kpoints')
            kmesh_tmp = deepcopy( kmesh )
            latVec = np.array( latVec )
            recLatVec= np.zeros((3,3))
            Vol= np.abs( np.dot(latVec[0,:] , np.cross(latVec[1,:],latVec[2,:])) )
            recLatVec[0,:]= np.cross(latVec[1,:],latVec[2,:])  /Vol
            recLatVec[1,:]= np.cross(latVec[2,:],latVec[0,:])  /Vol
            recLatVec[2,:]= np.cross(latVec[0,:],latVec[1,:])  /Vol
    
            rec_cell_norm = [np.linalg.norm( recLatVec[x,:]) for x in range(3)]    
            return [2*np.pi * rec_cell_norm[idx] / kmesh_tmp[idx] for idx in range(len(rec_cell_norm)) ]

    @staticmethod
    def _check_dielectric_convergence(idiel_1, 
                                      idiel_2,
                                     energy_grid,
                                     energy_window,
                                     method="L2_distance",
                                     threshold=1e-3,
                                     channels=(0,1,2)  # xx, yy, zz only
                                     ):
        """ Compare two imaginary dielectric functions (N_energy x N_channels)
            using a specified metric, restricted to selected diagonal channels
            and within (E_min, E_max) energy window.
        Returns:   (bool converged, float distance)
        """
        from scipy.stats import wasserstein_distance
    
        #[1] Apply energy window filter
        E_min, E_max = energy_window
        mask = (energy_grid >= E_min) & (energy_grid <= E_max)
    
        egrid = energy_grid[mask]
    
        # filter dielectric arrays & channels
        #id1 = idiel_1[np.ix_(mask, channels)]   # shape (Nwin, 3)
        #id2 = idiel_2[np.ix_(mask, channels)]   # shape (Nwin, 3)
        id1 = idiel_1[mask][:, channels]
        id2 = idiel_2[mask][:, channels]
    
        #[2] Internal metric definitions
        #[2.1] L2 norm over ω and channels
        def __L2(a, b, e):
            diff_sq = (a - b)**2   # shape: (Nwin, 3)
            f = np.mean(diff_sq, axis=1)   # average over xx,yy,zz
            return np.sqrt(np.trapz(f, e))
    
        #[2.2] L1 norm
        def __L1(a, b, e):
            diff = np.abs(a - b)
            f = np.mean(diff, axis=1)
            return np.trapz(f, e)
    
        #[2.3] Wasserstein (Earth-Mover) distance
        def __Wasserstein(a, b, e):
            """
            Uses the mean value across xx,yy,zz channels.
            Both curves must be normalized to form valid PDFs.
            """
            f1 = np.mean(a, axis=1)
            f2 = np.mean(b, axis=1)
    
            # ensure non-negative
            f1 = np.clip(f1, 0, None)
            f2 = np.clip(f2, 0, None)
    
            # normalize to sum to 1 (become distributions)
            if np.sum(f1) > 0:
                f1 = f1 / np.sum(f1)
            if np.sum(f2) > 0:
                f2 = f2 / np.sum(f2)
    
            return wasserstein_distance(e, e, f1, f2)
    
        #[3] Constructing Dict with all methods and dealing with errors
        metric_map = { "L2_distance": __L2,
                       "L1_distance": __L1,
                       "Wasserstein": __Wasserstein,     }
        if method not in metric_map:
            raise ValueError(f"Unknown method '{method}'. "
                             f"Choose from: {list(metric_map.keys())}")
    
        #[4] compute metric AND determine convergence flag
        distance = metric_map[method](id1, id2, egrid)
        converged = distance < threshold
    
        return converged, distance
    
    @staticmethod
    def _get_energy_of_diel_onset( imdiel , egrid , thr_for_considering_offset = 0.1):
        """  Detect the onset of the imaginary dielectric function ε₂(ω).
        Returns the first energy where average of diagonal component (xx,yy,zz)  exceeds 'eps'.
        Parameters: imdie : (N_energy x 6) ndarray - suppsed ordering is (xx,yy,zz,..non-diag.components..)
                    egrid : (N_energy,) energy array
                    eps   : small threshold to avoid picking numerical noise
        Returns:    float : onset energy """                                 
        diag = imdiel[:, :3]  # consider only xx, yy, zz : thus shape (N, 3)
        diag_mean = np.mean(diag, axis=1)
        idx = np.where(diag_mean > thr_for_considering_offset)[0]
        if len(idx) == 0: return egrid[0]  # fallback: no onset detected
        return egrid[idx[0]]
    
    @staticmethod
    def _collect_successful_mbse_nodes_sorted_by_kpts(WC_MBPT , report_function):
            """ Filter and k-mesh-sort successful mBSE WorkChainNodes.
            Success criterion:  - dielectric output present (idiel + ediel arrays)
            Returns:        list[WorkChainNode]     Successful nodes sorted by increasing k-mesh.
            """
            finished_nodes   = []
            finished_kmeshes = []  
            str_report = ""
            
            #WC_MBPT stores the mBSE workchainNodes, but it's not guaranteed it has all successfully finished nodes 
            #for nodes are ordered (AiiDA may return them in whatever order they completed).  
            for idx , wc in enumerate(WC_MBPT):  #Thus we reconstruct the list of WC succesfully finished
                #We filter “successful” WC_MBPT nodes not by AiiDA  by actual availability of outputs required for convergence
                #If the BSE matrix is solved with a iterative method (IBSE=1/3) has_opt will be false; but has_diel should be 
                #always TRUE for a successful calculation
                has_opt = ('opticaltransitions' in wc.outputs)
                has_diel = ('dielectrics' in wc.outputs and wc.outputs.dielectrics is not None
                            and "idiel" in wc.outputs.dielectrics.get_arraynames()
                            and "ediel" in wc.outputs.dielectrics.get_arraynames())

                if not has_diel:
                    report_function(f"WARNING - WC_MBPT[{idx}] (pk={wc.pk}) missing diel output - skipping for convergence.")
                    continue     
                        
                finished_nodes.append(wc)
                finished_kmeshes.append(wc.inputs.kpoints.get_kpoints_mesh()[0] )
                
            if not finished_nodes: return []
            #Sort by actual kmesh (increasing)
            km_tuples  = [tuple(km) for km in finished_kmeshes]
            idx_sorted = sorted(range(len(km_tuples)), key=lambda i: km_tuples[i])
            return [finished_nodes[i] for i in idx_sorted]
       
    @staticmethod
    def _collect_convergence_record_dict(wc_nodes, diel_onset_threshold=0.1):
            """Build convergence records from sorted mBSE WorkChainNodes.
            Parameters  wc_nodes :      list[WorkChainNode]        Successful mBSE nodes sorted by k-mesh.
            Returns     list[records] : list[AttributeDict]        
                        Each record contains:
                              - kmesh       - optgap        - oscstr
                              - imdiel      - energygrid    - imdiel_onset
            """
            records = []
            for wc in wc_nodes:
                rec = AttributeDict()
                rec["kmesh"] = wc.inputs.kpoints.get_kpoints_mesh()[0]

                # Optical gap
                if 'opticaltransitions' in wc.outputs:
                    gap, osc = _extract_opticalgap_fromWorkchainNode(wc)
                    rec["optgap"] = gap ; rec["oscstr"] = osc
                else:
                    rec["optgap"] = None ; rec["oscstr"] = None

                # Dielectric function
                if 'dielectrics' in wc.outputs:
                    rec["imdiel"]     =  wc.outputs.dielectrics.get_array("idiel")
                    rec["energygrid"] =  wc.outputs.dielectrics.get_array("ediel")
                    rec["imdiel_onset"] = helper_kptsConv_mBSE._get_energy_of_diel_onset(
                                                rec["imdiel"], rec["energygrid"] , 
                                                thr_for_considering_offset=diel_onset_threshold, )
                else:
                    rec["imdiel"]     = None
                    rec["energygrid"] = None
                    rec["imdiel_onset"] = None
                records.append(rec)
            return records

    @staticmethod
    def _collect_consecutive_optgap_and_diel_differences(wc_list , window_size , diel_metric, onset_threshold=0.1):
       """ Return a formatted string summarizing:
           idx | kmesh | optical gap | dielectric distance vs previous
           Additionally returns the lists of Δ(optical gap) and Δ(diel) for external use.
           using the records in wc_list = wc_successful_nodes_kptssorted_elaborated.     
   
           wc_list : list of records with fields:
                     kmesh - optgap - imdiel - imdiel_onset - energygrid       """   

       #[1] Compute dielectric distances wrt previous iteration and accumulate in the list
       ogap_diffs = [] 
       diel_diffs = []
       meta = []
       if len(wc_list) == 0: return ogap_diffs, diel_diffs, meta

       for i in range(len(wc_list)):
           if i == 0:
               ogap_diffs.append(None)
               diel_diffs.append(None)
               meta.append({"energy_window": None, "onset": None})
               continue
   
           prev = wc_list[i - 1]
           last = wc_list[i]
   
           #[2.1] Optical gap difference
           if (prev["optgap"] is not None)  and  (last["optgap"] is not None) :
               ogap_diffs.append( abs(last["optgap"] - prev["optgap"]) )
           else:
               ogap_diffs.append( None )
               
           #[2.2] Dielectric difference 
           #      Starty by skipping if imdiel is absent
           if (prev["imdiel"] is None) or (last["imdiel"] is None) :
               diel_diffs.append(None)
               meta.append({"energy_window": None, "onset": None})
               continue
   
           #[2.3] Define energy window
           onset_prev = prev.get("imdiel_onset")
           onset_last = last.get("imdiel_onset")
           if onset_prev is None:
                   onset_prev = helper_kptsConv_mBSE._get_energy_of_diel_onset(
                                   prev["imdiel"], prev["energygrid"], thr_for_considering_offset=onset_threshold  )
           if onset_last is None:
                   onset_last = helper_kptsConv_mBSE._get_energy_of_diel_onset(
                                   last["imdiel"], last["energygrid"], thr_for_considering_offset=onset_threshold )
               
           onset = min(onset_prev, onset_last)
           E_min = max(onset,               float(last["energygrid"][0])   )
           E_max = min(onset + window_size, float(last["energygrid"][-1])  )
           energy_window = (E_min, E_max)
   
   
           #[2.4] Compute dielectric Δ (no threshold, just raw distance)
           _, diel_distance = helper_kptsConv_mBSE._check_dielectric_convergence(
                       prev["imdiel"], last["imdiel"] ,
                       energy_grid      = last["energygrid"],
                       energy_window    = energy_window,
                       method           = diel_metric ,
                       channels         = (0,1,2)     ,
                       threshold        = 100         )   # irrelevant, we only want distance
           diel_diffs.append(diel_distance)
           meta.append({"energy_window": energy_window, "onset": onset})
       return ogap_diffs, diel_diffs, meta   
   
    @staticmethod
    def _prettyprint_kmesh_gap_diel_summary(wc_list, opt_diffs, diel_diffs, prefix="  "):
        """Format a string summary:
          idx | kmesh | optgap | Δopt | Δdiel(prev)
        """
        out = prefix+"[conv-summary]"+"\n"+prefix+"> Completed mBSE nodes:"
        if len(wc_list) == 0:
            out += "    (none yet)"
            return out

        out += "\n"+prefix+"  idx   kmesh          optgap[eV]       Δopt       Δdiel(prev)"
        out += "\n"+prefix+"  ---------------------------------------------------------------"

        for idx, rec in enumerate(wc_list):
            km = rec["kmesh"]
            kmesh_str = f"[{km[0]}, {km[1]}, {km[2]}]"

            gap_str = f"{rec['optgap']:.4f}" if rec.get("optgap") is not None else "--"

            dop = opt_diffs[idx] if idx < len(opt_diffs) else None
            dop_str = "--" if dop is None else f"{dop:.4f}"

            d = diel_diffs[idx] if idx < len(diel_diffs) else None
            d_str = "--" if d is None else f"{d:.4e}"

            out += (f"\n{prefix}  [{idx:2d}]  {kmesh_str:<14}  {gap_str:<14}  {dop_str:<10}  {d_str}")
        return out

    @staticmethod
    def _compute_next_kmesh( ctx , num_finished_successfully, step_kmesh):
        # First run (no successful calcs yet): use starting mesh
        if num_finished_successfully == 0:
            return deepcopy(ctx.control.start_kmesh)

        # Otherwise increment from last successful kmesh
        last_kmesh = np.array(ctx.control['wc_successful_nodes_kptssorted_elaborated'][-1]["kmesh"], dtype=int)
        step_vec   = np.array(step_kmesh, dtype=int)
        return last_kmesh + step_vec
    
    @staticmethod
    def _return_next_kmesh_or_abort( self , next_kmesh ):
        # safety bounds check
        if np.any(np.array(next_kmesh) > np.array( self.ctx.control.max_kmesh) ):
            str_abort = ( f"\n   --> convergence NOT reached and maximum kmesh exceeded:"
                          f"\n       next kmesh would be {next_kmesh}"
                           f"\n   --> aborting" )
            self.report( self.ctx.str_log + str_abort)
            return self.exit_codes.CONVERGENCE_NOT_FOUND
        self.ctx.control.current_kmesh = np.array(next_kmesh, dtype=int)
        str_cont = ( f"\n   --> convergence NOT reached - continuing : next kmesh = {self.ctx.control.current_kmesh}\n" )
        self.report(self.ctx.str_log + str_cont)
        return True
    
@dataclass
class ConvergenceStatus:
    # configuration used
    use_gap:  bool
    use_diel: bool
    gap_threshold: float
    factor_for_dynamic_conv : float
    diel_threshold: float
    diel_metric: str    #which type of distance between the diel.tensors is used
    diel_window: float  #the window considered for checking the convergence

    # last-step distances (None if not available)
    delta_opt:  Optional[float] = None
    delta_diel: Optional[float] = None

    # per-criterion flags (None if cannot be evaluated yet)
    flag_is_optgap_converged:              Optional[bool] = None
    flag_is_diel_converged:                Optional[bool] = None
    flag_is_diel_larger_4_times_threshold: Optional[bool] = None
    # final decision
    flag_is_converged: bool = False

    def finalize(self):
        """Compute final flag from use_* and per-criterion flags."""
        if self.use_gap and self.use_diel:
            self.flag_is_converged = bool(self.flag_is_optgap_converged) and bool(self.flag_is_diel_converged)
        elif self.use_gap and not self.use_diel:
            self.flag_is_converged = bool(self.flag_is_optgap_converged)
        elif self.use_diel and not self.use_gap:
            self.flag_is_converged = bool(self.flag_is_diel_converged)
        else:
            self.flag_is_converged = False
        return self.flag_is_converged









class VaspmBSEKptsConvWorkChain(WorkChain):
    _next_workchain = WorkflowFactory('vasp.vasp')

    @classmethod
    def define(cls, spec):
            super(VaspmBSEKptsConvWorkChain , cls).define(spec)

            #spec.expose_inputs(cls._next_workchain          , exclude=('kpoints','parameters','settings','potential_family','potential_mapping')) 
            spec.expose_inputs(VaspmBSEInitScriptWorkChain  , exclude=('kpoints','ns_reference', 'ns_BSE') ) 
             
            spec.input( 'ns_kpoints.convergence_threshold'       , valid_type=Float , required=False , default=lambda:Float(0.35), help="minimum converge value in eV" ) 
            spec.input( 'ns_kpoints.kmesh.starting_mesh'         , valid_type= KpointsData , required=True , help="Starting k-mesh for the k-point convergence study." )
            spec.input( 'ns_kpoints.kmesh.max_mesh'              , valid_type= KpointsData , required=True , help="Maximum k-mesh for the k-point convergence study." ) 
            kpoints_step_defaultvalue = DataFactory('core.array.kpoints')() ; kpoints_step_defaultvalue.set_kpoints_mesh([1,1,1])
            spec.input( 'ns_kpoints.kmesh.step'                  , valid_type= KpointsData , required=False , default=lambda:kpoints_step_defaultvalue, help="Step size for the k-point mesh." ) 

            spec.input('ns_converge.dielfunction_convergence'   , valid_type=Bool  , required=False , default=lambda:Bool(True)  , help="Enable/disable convergence check based on imaginary dielectric function." )
            spec.input('ns_converge.opticalgap_convergence'     , valid_type=Bool  , required=False , default=lambda:Bool(False) , help="Enable/disable convergence check based on the optical gap."               )
            spec.input("ns_converge.dielfunction_distance"      , valid_type=Str   , required=False , default=lambda:Str("L2_distance") , help="Distance metric for dielectric-function convergence: 'L2_distance' - 'L1_distance' - 'Wasserstein'.")
            spec.input("ns_converge.dielfunction_window"        , valid_type=Float , required=False , default=lambda:Float(3.5)  , help="Energy window (in eV) starting from onset of the imaginary diel.function - Used for dielectric-function convergence evaluation.")
            spec.input('ns_converge.convergence_dynamic_control', valid_type=Bool  , required=False , default=lambda: Bool(False), help="Enable adaptive k-mesh step refinement.")

            spec.input("ns_converge_BSE.NBANDSO"                , valid_type=Int   , required=False , help="number of occupied bands included in the bse matrix for all calculations used for the convergence.")
            spec.input('ns_converge_BSE.NBANDSV'                , valid_type=Int   , required=False , help="number of unoccupied (virtual) bands included in the bse matrix for all calculations used for the convergence.")
            spec.input("ns_converge_BSE.static_inverse_diel"    , valid_type=Float , required=True  , help='Required for analytic diagonal screening in mBSE.' )
            spec.input("ns_converge_BSE.screening_parameter"    , valid_type=Float , required=True  , help='Required for analytic diagonal screening in mBSE.' )
            spec.input("ns_converge_BSE.G0W0_gap"               , valid_type=Float , required=False , help="G0W0 gap; required to determine SCISSOR")

            #Optional pass-through (kept for symmetry with G0W0; currently not consumed explicitly by the base chain)
            spec.input('ns_reference.starting_RemoteData', valid_type=RemoteData, required=False)
       
            spec.output( 'kmesh_converged' , valid_type= KpointsData , required=True)
            spec.output( 'optical_gap'     , valid_type= Float , required=False)
            
            spec.exit_code(300,'MBSE_CALC_FAILURE'              , message='A mBSE calculations failed -> error has not been handled and resolved -> aborting convergence loop.'  )
            spec.exit_code(301,'CONVERGENCE_NOT_FOUND'          , message='Convergence has not been reached; please relax the threshold / increase the range studied / check the calculations.')
            spec.exit_code(302,'NOT_IMPLEMENTED'                , message='The selected feature is currently only planned and not implemented.')
            spec.exit_code(303,'UNSUPPORTED_DIELFUNCTION_METRIC', message=( "The selected dielectric-function convergence metric ns_converge.dielfunction_distance is not supported. "
                                                                            "Allowed values: 'L2_distance', 'L1_distance', 'Wasserstein'."   ))
            spec.exit_code(304,'NO_CONVERGENCE_REQUESTED'       , message='Both the dielectric and optical convergences are disabled. What should I converge then?.')    
            spec.exit_code(305,'INVALID_MBSE_CONVERGE_PARAMETERS' , message='One or more numerical parameters inside ns_converge (either NBANDSO , NBANDSV or dielfunction_window) are invalid (<1?)')
            
            spec.outline(
                cls.initialize,
                while_(cls.monitor_convergence)(
                     cls.prepare_run_mBSE,
                ),
                cls.elaborate_results     
            )
        
    def initialize(self):
        self.ctx.control = AttributeDict() ; 
        self.ctx.control['convergence'] = []
        self.ctx.control['kmesh_converged']      = None 
        self.ctx.control['kdensity_converged']   = None 
        self.ctx.control["dynamic_step_refined"] = False
        self.ctx.control['fail_counter'] = 0
        self.ctx.WC_MBPT = []

        ##[1][ Input checking regarding the ns_converge namespace ]
        allowed_metrics = ["L2_distance", "L1_distance", "Wasserstein"]
        if self.inputs.ns_converge.dielfunction_distance.value not in allowed_metrics: 
            return self.exit_codes.UNSUPPORTED_DIELFUNCTION_METRIC   
        if (not self.inputs.ns_converge.dielfunction_convergence.value) and (not self.inputs.ns_converge.opticalgap_convergence.value):
            return self.exit_codes.NO_CONVERGENCE_REQUESTED
        if (self.inputs.ns_converge.dielfunction_window.value < 1) or (self.inputs.ns_converge_BSE.NBANDSO.value < 1) or (self.inputs.ns_converge_BSE.NBANDSV.value < 1):
            return self.exit_codes.INVALID_MBSE_CONVERGE_PARAMETERS
        
        ##[ The Kpoint part ]
        #There are two possible ways to control the convergence:
        # 1) through the k-mesh : in this case, both the kmesh.starting_mesh and kmesh.max_mesh of kmesh must be specified; only the step is optional (default=1)
        # 2) through the k-point density : this case is activated only if BOTH kmesh.starting_mesh and kmesh.max_mesh are NOT specified; 
        #                                  and is more flexible, as there are no mandatory values, all values have a default values
        #    Two additional notes:
        #    - if kmesh.starting_mesh is specified, and not kmesh.max_mesh that kmesh is used as starting k-mesh, but the increase of the k-mesh is done through the k-point density 
        #      and not through the k-mesh step
        #    - in this second case, the minimum_constraint of kdensity is always considered, i.e. the starting_mesh of kdensity is automatically increased if it is < minimum_constraint
  
        #[1] Determine control mode ---------------------------------------------
        if ('kmesh' in self.inputs.ns_kpoints and
           'starting_mesh' in self.inputs.ns_kpoints.kmesh and
           'max_mesh' in self.inputs.ns_kpoints.kmesh):
    
           self.ctx.control['control_way'] = 'kmesh'
  
           # Extract integer arrays
           kmesh_start = np.array(  self.inputs.ns_kpoints.kmesh.starting_mesh.get_kpoints_mesh()[0], dtype=int   )
           kmesh_max   = np.array(  self.inputs.ns_kpoints.kmesh.max_mesh.get_kpoints_mesh()[0], dtype=int )
           step_vec    = np.array(  self.inputs.ns_kpoints.kmesh.step.get_kpoints_mesh()[0], dtype=int     )
          
           self.ctx.control.iteration_counter = -1
           self.ctx.control.start_kmesh   = kmesh_start
           self.ctx.control.current_kmesh = kmesh_start
           self.ctx.control.max_kmesh  = kmesh_max
           self.ctx.control.step_kmesh = step_vec
           
           str_log = ( "\n [Initializing K-points convergence]"
                        "\n > Controlling convergence via k-mesh."
                       f"\n   Starting k-mesh    : {kmesh_start}"
                       f"\n   Maximum k-mesh     : {kmesh_max}"
                       f"\n   Step (initial)     : {step_vec}" )
        
        else:
            return self.exit_codes.NOT_IMPLEMENTED           
        self.report(str_log)

    def prepare_run_mBSE(self):
        # Build inputs for the base workchain
        self.ctx.inputs_mBSEbase = AttributeDict({ 'ns_parameters' : AttributeDict() , 'ns_BSE' : AttributeDict() })
        self.ctx.inputs_mBSEbase.update(self.exposed_inputs(VaspmBSEInitScriptWorkChain))
        self.ctx.inputs_mBSEbase.clean_workdir = Bool(False)            

        #[1] Kpoint related stuff
        string_kmesh = np.array2string(   self.ctx.control.current_kmesh , separator=" , ").replace('\n', '')
        self.ctx.inputs_mBSEbase.ns_option.calculation_label = Str("mBSE kConv "+string_kmesh)
        self.ctx.inputs_mBSEbase.kpoints = DataFactory('core.array.kpoints')()
        self.ctx.inputs_mBSEbase.kpoints.set_kpoints_mesh( self.ctx.control.current_kmesh )
        self.ctx.log_launching_run = f"\n [wkc_KptsConv][DFT+mBSE calc - NonSpinPolarized - From scratch]\n  > Lauching DFT+mBSE(NonSpinPolarized) using VaspmBSEInitScriptWorkChain on k-mesh {string_kmesh}\n"
        
        #[1] INCAR flags
        self.ctx.inputs_mBSEbase.ns_BSE.static_inverse_diel = self.inputs.ns_converge_BSE.static_inverse_diel
        self.ctx.inputs_mBSEbase.ns_BSE.screening_parameter = self.inputs.ns_converge_BSE.screening_parameter
        self.ctx.inputs_mBSEbase.ns_BSE.G0W0_gap = self.inputs.ns_converge_BSE.G0W0_gap
        
        #Note: we et ns_BSE.optical_energy_window (which it's used to determine automatically the NBANDSV/O values)
        #only if NBANDSV/O are not passed (and in that case override).
        if ("NBANDSV" in self.inputs.ns_converge_BSE) and ("NBANDSO" in self.inputs.ns_converge_BSE) :   
                self.ctx.inputs_mBSEbase.ns_BSE.NBANDSV = self.inputs.ns_converge_BSE.NBANDSV.value
                self.ctx.inputs_mBSEbase.ns_BSE.NBANDSO = self.inputs.ns_converge_BSE.NBANDSO.value
                self.ctx.log_launching_run += f"\n  > Explicitly passed (occupied/virtual) NBANDSO/NBANDSV = {self.ctx.inputs_mBSEbase.ns_BSE.NBANDSO}/{self.ctx.inputs_mBSEbase.ns_BSE.NBANDSV}" 
        else:
                self.ctx.inputs_mBSEbase.ns_BSE.optical_energy_window = self.inputs.ns_converge.dielfunction_window.value
                self.ctx.log_launching_run += f"\n  > BANDSO/NBANDSV not explicitly passed; determined automatically by the child workchain based on the convergence energy window = {self.ctx.inputs_mBSEbase.ns_BSE.optical_energy_window}"
        
        #https://vasp.at/wiki/Best_practices_for_Bethe-Salpeter_calculations
        #In internal tests PRECFOCK reduces computational cost of the setting-up the BSE matrix of almost 40%
        #with negligible cost in term of precision decrease.
        #Thus we always keep on for kpts-convergence
        self.ctx.inputs_mBSEbase.ns_BSE.set_PRECFOCK_to_Fast = Bool(True)
        self.ctx.log_launching_run += f"\n  > For mBSE kpoints convergence (which does not need to be overly accurate), we set PRECFOCK to Fast."


        
        
        #The following spec.inputs of the VaspmBSEInitScriptWorkChain are exposed, and thus updated here
        #and do not need therefore to be handled explicitly:    
        #    ns_parameters.encut - ns_parameters.nbands - ns_parameters.magnetic_moment_onsite
        #    ns_interpolation.G0W0_reference - ns_interpolation.remote_initscript
        #    ns_interpolation.local_initscript - ns_interpolation.nbandsgw_to_interpolate
        #    ns_BSE.static_inverse_diel - ns_BSE.screening_parameter - ns_BSE.energy_window
        #    ns_BSE.G0W0_gap - ns_BSE.OMEGAMAX - ns_BSE_NBANDSV - ns_BSE_NBANDSO - cissor
        #[1.1] INCAR - related optimization flags
        #If we want to convege the optical gap we need the BSE eigenvalue -> thus diagonalizing the matrix
        #  -> thus IBSE=2; however if we want to converge the optical spectra, we can use the faster Lanczos or time evolutions
        #     The diagonalization scales (system-rank)^3 , while Lanczos/time evolutions scale (system-rank)^2
        #     where the system.rank = #kpts * NBANDSV * NBANDSO
        #  Thus we use the more expensive IBSE=2 only if BSE eigenvalues are requested
        #  Note that only time-evolutions as vasp.6.5.1 is ported to GPU: https://vasp.at/wiki/Category:Bethe-Salpeter_equations#Time_evolution
        #  Furthermore, Lanczos requires an additional parameter, i.e.NOMEGA; thus we use time-evolution
        if self.inputs.ns_converge.opticalgap_convergence.value:
            self.ctx.inputs_mBSEbase.ns_parameters.ibse = Int(2)
            self.ctx.log_launching_run = "\n  > Optical gap convergence is required -> full diagonalization of the BSE matrix is required regardless of dielectric tensor convergence in order to get the eigenvalues -> setting IBSE=2. " 
        elif  self.inputs.ns_converge.dielfunction_convergence.value: 
            self.ctx.inputs_mBSEbase.ns_parameters.ibse = Int(1)
            self.ctx.log_launching_run = "\n  > Dielectric tensor convergence is required and optical gap convergence is not required-> setting IBSE=1"
        self.ctx.inputs_mBSEbase.ns_parameters.nbseeig = Int(0)
        
        #[3] POTCAR related stuff
        self.ctx.inputs_mBSEbase.potential_family  = self.inputs.potential_family
        self.ctx.inputs_mBSEbase.potential_mapping = self.inputs.potential_mapping

        #[4] Not currently used - for future   
        # Reference WAVECAR/CHGCAR if user provided (optional)
        # Optional: allow future use of ns_reference (currently unused by base chain)
        if 'ns_reference' in self.inputs and 'starting_RemoteData' in self.inputs.ns_reference:
            pass                         
        
        running_mBSEbase = self.submit(VaspmBSEInitScriptWorkChain , **self.ctx.inputs_mBSEbase) 
        self.ctx.log_launching_run = "\n  -> Submitted VaspmBSEInitScriptWorkChain pk={running_mBSEbase.pk}"
        self.report( self.ctx.log_launching_run )
        return ToContext(WC_MBPT=append_(running_mBSEbase))
         
    def monitor_convergence_TEST(self): 
        self.ctx.control.iteration_counter += 1
        if self.ctx.control.iteration_counter == 1: return False
        else: return True
    
    def monitor_convergence(self):  
        ##Initialize----------------------------------------------------------------------------
        #[Initialize - 1] Define a AttributeDict which will be used internally for this execution
        #of the monitor_convergence. It's used to group in a single dict the relevant flags/values.
        #Given the line self.ctx.monitor = AttributeDict(), it's reinitialized at each run.
        self.ctx.str_log = ""
        self.ctx.monitor = AttributeDict()
        self.ctx.monitor.convergence_status = ConvergenceStatus(
                diel_threshold = float(self.inputs.ns_kpoints.convergence_threshold.value) ,
                gap_threshold  = float(self.inputs.ns_kpoints.convergence_threshold.value) ,
                use_gap     = bool(self.inputs.ns_converge.opticalgap_convergence.value)   ,
                use_diel    = bool(self.inputs.ns_converge.dielfunction_convergence.value) ,
                diel_metric = str(self.inputs.ns_converge.dielfunction_distance.value)  ,
                diel_window = float(self.inputs.ns_converge.dielfunction_window.value)  , 
                factor_for_dynamic_conv = float(2) )
        
        self.ctx.monitor.control_way  = deepcopy( self.ctx.control['control_way'] )  # 'kmesh' or 'kdensity' 
        #Spin related variables
        self.ctx.monitor.has_spin      = ('magnetic_moment_onsite' in self.inputs['ns_parameters'])
        self.ctx.monitor.spin_channels = ['spinUp', 'spinDw'] if self.ctx.monitor.has_spin else ['spinUp']
        #Define minimum number of calculations required before convergence checks
        #The convergence based on k-mesh uses only 2 (the gaps from 2 consecutive k-meshes); the one based on k-density 3 for numerical stability.
        self.ctx.monitor.min_num_calcs_required_for_conv = 2
        
        
        ##--------------------------------------------------------------------------------------     
        #[1] Hard failure check (no retry logic here)
        # If the last submitted child is finished and NOT ok -> abort.
        # NOTE: base workchain should have already handled its own retries.
        if len(self.ctx.WC_MBPT) > 0:
            last_wc = self.ctx.WC_MBPT[-1]
            if last_wc.is_excepted or (not last_wc.is_finished_ok):
                self.report( f"\n  > ERROR: last mBSE child failed (pk={last_wc.pk})."
                             f"\n    Aborting convergence loop (retries should be handled in base workchain)."  )
                return self.exit_codes.MBSE_CALC_FAILURE        
            
        ##[ELABORATION] ------------------------------------------------------------------------
        #[2] Collect successful nodes sorted by kmesh + build elaborated records for each node (optgap, extract diel, etc)
        #     + Compute differences between opt.gap of consecutive nodes (and distanze between dielectric tensors)
        #     i.e.  construct arrays of all successful mBSE children and consecutive difference optgap / diel 
        # each element in self.ctx.control['wc_successful_nodes_kptssorted'] is a workchainNode
        # while each element in self.ctx.control['wc_successful_nodes_kptssorted_elaborated'] is a dict containing:
        #       - kmesh       - optgap        - oscstr     - imdiel      - energygrid    - imdiel_onset
        self.ctx.control['wc_successful_nodes_kptssorted'] =  helper_kptsConv_mBSE._collect_successful_mbse_nodes_sorted_by_kpts( 
                                                                            self.ctx.WC_MBPT , report_function=self.report           )
        self.ctx.control['wc_successful_nodes_kptssorted_elaborated'] =  helper_kptsConv_mBSE._collect_convergence_record_dict( 
                                                                            self.ctx.control['wc_successful_nodes_kptssorted'] )
        num_mBSE_finished_successfully = len(self.ctx.control['wc_successful_nodes_kptssorted_elaborated'])

        # self.ctx.control['consecutive_wc_optgap_difference'] is the list of differences between opt.gap of consecutive gaps
        # self.ctx.control['consecutive_wc_diel_distance'] is the list of distance based on the metric based on consecutive mBSE calculations
        opt_diffs, diel_diffs, meta = helper_kptsConv_mBSE._collect_consecutive_optgap_and_diel_differences(
                                            wc_list     = self.ctx.control['wc_successful_nodes_kptssorted_elaborated'] ,
                                            window_size = float(self.inputs.ns_converge.dielfunction_window.value)      ,
                                            diel_metric = self.inputs.ns_converge.dielfunction_distance.value           , )
        self.ctx.control['consecutive_wc_optgap_difference'] = opt_diffs
        self.ctx.control['consecutive_wc_diel_distance']     = diel_diffs
        self.ctx.control['consecutive_wc_diel_meta'] = meta 
        #Now logging: summary_str is the log of all previous dist and optgap values
        summary_str = "\n"+ helper_kptsConv_mBSE._prettyprint_kmesh_gap_diel_summary(
                                            self.ctx.control['wc_successful_nodes_kptssorted_elaborated'],
                                            opt_diffs, diel_diffs,            )
        self.ctx.str_log = ( "\n [wkc_KptsConv][monitor_convergence]" + summary_str )
        
        ##[DECISION.BLOCK - 1]------------------------------------------------------------------
        ##[3] Early exit if not enough calculations yet ##--------------------------------------
        if num_mBSE_finished_successfully < (self.ctx.monitor.min_num_calcs_required_for_conv ):
            self.ctx.str_log += (  f"\n  > Not enough successful BSE calculations "
                                   f"({num_mBSE_finished_successfully}/{self.ctx.monitor.min_num_calcs_required_for_conv})."
                                   f"\n    → Launch next calculation." )
            next_kmesh = helper_kptsConv_mBSE._compute_next_kmesh(self.ctx, num_mBSE_finished_successfully,
                                                                  self.ctx.control.step_kmesh ) 
            return helper_kptsConv_mBSE._return_next_kmesh_or_abort( self , next_kmesh )

        ##[DECISION.BLOCK - 2]------------------------------------------------------------------
        ##[3] Fill ConvergenceStatus from last diffs + evaluate flags ##------------------------
        cs = self.ctx.monitor.convergence_status
        cs.delta_opt  = ( self.ctx.control['consecutive_wc_optgap_difference'][-1] 
                          if len(self.ctx.control['consecutive_wc_optgap_difference']) > 0 else None )
        cs.delta_diel = ( self.ctx.control['consecutive_wc_diel_distance'][-1]    
                          if len(self.ctx.control['consecutive_wc_diel_distance']) > 0 else None )

        cs.flag_is_optgap_converged = None
        cs.flag_is_diel_converged   = None
        cs.flag_is_diel_larger_4_times_threshold = None      

        # Optical-gap criterion
        if cs.use_gap:
            if cs.delta_opt is not None:
                cs.flag_is_optgap_converged = bool(cs.delta_opt < cs.gap_threshold)
                self.ctx.str_log += ( f"\n  > [optgap-conv] Δopt={cs.delta_opt:.4f} (thr={cs.gap_threshold}) "
                                        f" -> conv[optg]={cs.flag_is_optgap_converged}" )
            else:
                self.ctx.str_log += "\n  > [optgap-conv] Δopt unavailable → conv[optgap]=None"
        else:
            self.ctx.str_log += "\n  > [optgap-conv] disabled"
                
      
        # Dielectric criterion
        if cs.use_diel:
            if cs.delta_diel is not None:
                cs.flag_is_diel_converged = bool(cs.delta_diel < cs.diel_threshold)
                self.ctx.str_log += ( f"\n  > [diel-conv]  metric={cs.diel_metric} (energy window above gap where convergence is studied)={cs.diel_window} ->"
                                        f" Δdiel={cs.delta_diel:.4e} (thr={cs.diel_threshold}) "
                                        f" -> conv[diel]={cs.flag_is_diel_converged}"                )
                cs.flag_is_diel_larger_4_times_threshold = bool(cs.delta_diel > 2*cs.diel_threshold)
            else:
                self.ctx.str_log += "\n  > [diel-conv] Δdiel unavailable → conv=None"
        else:
            self.ctx.str_log += "\n  > [diel-conv] disabled"

      
        # Finalize combined decision
        final_flag = cs.finalize()
        self.ctx.str_log += ( f"\n  > [final-conv] use_gap={cs.use_gap} use_diel={cs.use_diel} "
                                 f"-> conv[optgap]={cs.flag_is_optgap_converged} conv[diel]={cs.flag_is_diel_converged} "
                                 f"-> final decision={final_flag}"     )

        ##[4.1] Stop or continue
        if bool(cs.flag_is_converged):
            # Converged at last successful kmesh (not "current_kmesh", which is next-to-run)
            last_success_kmesh = np.array(self.ctx.control['wc_successful_nodes_kptssorted_elaborated'][-1]["kmesh"], dtype=int)
            self.ctx.str_log +=  f"\n   --> convergence REACHED : Converged k-mesh = {last_success_kmesh} \n\n"
            self.report(self.ctx.str_log)
            self.ctx.control['kmesh_converged'] = DataFactory('core.array.kpoints')()
            self.ctx.control['kmesh_converged'].set_kpoints_mesh(last_success_kmesh)
            return False
        #[4.2] Not converged → compute next kmesh from last successful point
        else:
            step_for_next_kmesh = np.array(self.ctx.control.step_kmesh)
            if self.inputs.ns_converge.convergence_dynamic_control.value and bool(cs.flag_is_diel_larger_4_times_threshold):
                step_for_next_kmesh = np.array(self.ctx.control.step_kmesh)*2
                self.ctx.str_log += ( "\n  > [dynamic-step] Large Δdiel -> using doubled base-step to accellerate convergence")
            next_kmesh = helper_kptsConv_mBSE._compute_next_kmesh(self.ctx, num_mBSE_finished_successfully, step_for_next_kmesh )
            return helper_kptsConv_mBSE._return_next_kmesh_or_abort( self , next_kmesh )

 
    def elaborate_results(self):
        # Determine final k-mesh
        if 'kmesh_converged' in self.ctx.control and self.ctx.control['kmesh_converged'] is not None:
            # already stored as KpointsData
            node_kpoints = self.ctx.control['kmesh_converged']
            node_kpoints.store()
            self.out('kmesh_converged', node_kpoints)
    
        # Output optical gap if available
        if ( self.ctx.control['convergence'] 
             and self.ctx.control['convergence'][-1].get('optgap') is not None  ):
            optgap_value = self.ctx.control['convergence'][-1]['optgap']
            self.out('optical_gap', Float(optgap_value))
    
    
    
    
