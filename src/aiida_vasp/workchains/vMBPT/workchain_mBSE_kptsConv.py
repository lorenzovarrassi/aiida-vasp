import numpy as np
from copy import deepcopy
from aiida.orm import Int, Float, Dict, Bool , List , RemoteData , KpointsData, Str
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine  import WorkChain, ToContext , append_, submit, while_ 
from aiida.common.extendeddicts  import AttributeDict


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
    def _detect_energy_of_diel_onset( imdiel , egrid , thr_for_considering_offset = 0.1):
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
    def _summary_list_kmesh_gap_diel(wc_list , window_size , diel_metric):
        """ Return a formatted string summarizing:
            idx | kmesh | optical gap | dielectric distance vs previous
            Additionally returns the lists of Δ(optical gap) and Δ(diel) for external use.
            using the records in wc_list = wc_MBPT_successful_sorted_elaborated.     
    
            wc_list : list of records with fields:
                      kmesh - optgap - imdiel - imdiel_onset - energygrid       """   

    
        #[1] header
        out = "\n  > [conv-summary] Completed mBSE nodes:"
        if len(wc_list) == 0:
            out += "    (none yet)"
            return out, [], []
        
    
        #[2] Compute dielectric distances wrt previous iteration and accumulate in the list
        opt_diffs = []
        diel_diffs = []
        for i in range(len(wc_list)):
            if i == 0:
                opt_diffs.append(None)
                diel_diffs.append(None)
                continue
    
            prev = wc_list[i - 1]
            last = wc_list[i]
    
            #[2.1] Optical gap difference
            if prev["optgap"] is not None and last["optgap"] is not None:
                opt_diffs.append(abs(last["optgap"] - prev["optgap"]))
            else:
                opt_diffs.append(None)
                
            #[2.2] Dielectric difference 
            #      Starty by skipping if imdiel is absent
            if prev["imdiel"] is None or last["imdiel"] is None:
                diel_diffs.append(None)
                continue
    
            #[2.3] Define energy window
            onset = min(prev["imdiel_onset"], last["imdiel_onset"])
            E_min = max(onset, last["energygrid"][0])
            E_max = min(onset + window_size, last["energygrid"][-1])
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
    
    
        # Pretty print each entry
        out += "\n      idx   kmesh          optgap[eV]         Δopt       Δdiel(prev)"
        out += "\n      ---------------------------------------------------------------"
    
        for idx, rec in enumerate(wc_list):
            km = rec["kmesh"]
            kmesh_str = f"[{km[0]}, {km[1]}, {km[2]}]"
    
            # optical gap
            if rec["optgap"] is not None: gap_str = f"{rec['optgap']:.4f}"
            else:                         gap_str = "--"
            # Δoptgap from previous
            dop = opt_diffs[idx]
            dop_str = "--" if dop is None else f"{dop:.4f}"
            # dielectric Δ
            d = diel_diffs[idx]
            d_str = "--" if d is None else f"{d:.4e}"
            out += ( f"\n      [{idx:2d}]  "
                     f"{kmesh_str:<12}  "
                     f"{gap_str:<12}  "
                     f"{dop_str:<8}  "
                     f"{d_str}"        )
        return out, opt_diffs, diel_diffs




class VaspmBSEKptsConvWorkChain(WorkChain):
    _next_workchain = WorkflowFactory('vasp.vasp')

    @classmethod
    def define(cls, spec):
            super(VaspmBSEKptsConvWorkChain , cls).define(spec)

            spec.expose_inputs(cls._next_workchain          , exclude=('kpoints','parameters','settings','potential_family','potential_mapping')) 
            spec.expose_inputs(VaspmBSEInitScriptWorkChain  , exclude=('kpoints','ns_reference') ) 
             
            spec.input( 'ns_kpoints.convergence_threshold'       , valid_type=Float , required=False , default=lambda:Float(0.1), help="minimum converge value in eV" ) 
            spec.input( 'ns_kpoints.kmesh.starting_mesh'         , valid_type= KpointsData , required=False , help="Starting k-mesh for the k-point density convergence If not specified, a k-mesh based on the density kdensity.starting_mesh will be used" )
            spec.input( 'ns_kpoints.kmesh.max_mesh'              , valid_type= KpointsData , required=False , help="Maximum k-mesh for the k-point density convergence. If not specified, a k-mesh based on the density kdensity.max_mesh will be used" ) 
            kpoints_step_defaultvalue = DataFactory('core.array.kpoints')() ; kpoints_step_defaultvalue.set_kpoints_mesh([1,1,1])
            spec.input( 'ns_kpoints.kmesh.step'                  , valid_type= KpointsData , required=False , default=lambda:kpoints_step_defaultvalue, help="Step size for the k-point mesh." ) 

            spec.input('ns_converge.dielfunction_convergence'   , valid_type=Bool  , required=False , default=lambda:Bool(True)  , help="Enable/disable convergence check based on imaginary dielectric function." )
            spec.input('ns_converge.opticalgap_convergence'     , valid_type=Bool  , required=False , default=lambda:Bool(False) , help="Enable/disable convergence check based on the optical gap." )
            spec.input("ns_converge.dielfunction_distance"      , valid_type=Str   , required=False , default=lambda:Str("L2_distance") , help="Distance metric for dielectric-function convergence: 'L2_distance' - 'L1_distance' - 'Wasserstein'.")
            spec.input("ns_converge.dielfunction_window"        , valid_type=Float , required=False , default=lambda:Float(3.5)  , help="Energy window (in eV) starting from onset of the imaginary diel.function - Used for dielectric-function convergence evaluation.")
            spec.input('ns_converge.convergence_dynamic_control', valid_type=Bool  , required=False , default=lambda: Bool(False), help="Enable adaptive k-mesh step refinement.")

        
            #Optional pass-through (kept for symmetry with G0W0; currently not consumed explicitly by the base chain)
            spec.input('ns_reference.starting_RemoteData', valid_type=RemoteData, required=False)
       
        
            spec.output( 'kmesh_converged' , valid_type= KpointsData , required=True)
            spec.output( 'optical_gap'     , valid_type= Float , required=False)
            
            spec.exit_code(403,'NODE_HAS_NO_MBSE_OUTPUT' , message='The workchain node has no opticaltransition output - please check that the calculation has completed correctly.')
            spec.exit_code(404,'CONVERGENCE_NOT_FOUND'   , message='Convergence has not been reached; please relax the threshold / increase the range studied / check the calculations.')
            spec.exit_code(405,'NOT_IMPLEMENTED'         , message='as the error say')
            spec.exit_code(410,'UNSUPPORTED_DIELFUNCTION_METRIC', 
                                message=( "The selected dielectric-function convergence metric "
                                          "ns_converge.dielfunction_distance is not supported. "
                                          "Allowed: 'L2_distance', 'L1_distance', 'Wasserstein'."                                ))
            spec.exit_code( 411, 'TWO_CONSECUTIVE_MBSE_FAILURES',
                               message='Two consecutive mBSE calculations failed — aborting convergence loop.'  )
            
            
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

        ##[1][ Checking if metric value passed makes sense ]
        allowed_metrics = ["L2_distance", "L1_distance", "Wasserstein"]
        if self.inputs.ns_converge.dielfunction_distance.value not in allowed_metrics: 
            return self.exit_codes.UNSUPPORTED_DIELFUNCTION_METRIC
            
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
           self.ctx.control.step_kmesh_BASE = deepcopy( step_vec )  #used to reference the initial value
                                                                    #because we could modify the .step_mesh one
           
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
        self.ctx.inputs_mBSEbase = AttributeDict()
        self.ctx.inputs_mBSEbase.ns_parameters = AttributeDict()
        self.ctx.inputs_mBSEbase.update(self.exposed_inputs(VaspmBSEInitScriptWorkChain))

        #[1] INCAR flags
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
        else:
            self.ctx.inputs_mBSEbase.ns_parameters.ibse = Int(1)
        self.ctx.inputs_mBSEbase.ns_parameters.nbseeig = Int(0)
        
        #https://vasp.at/wiki/Best_practices_for_Bethe-Salpeter_calculations
        #In internal tests PRECFOCK reduces computational cost of the setting-up the BSE matrix of almost 40%
        #with negligible cost in term of precision decrease.
        #Thus we always keep on for kpts-convergence
        self.ctx.inputs_mBSEbase.ns_BSE.set_PRECFOCK_to_Fast = Bool(True)
        
        #[2] POTCAR and kpoints related stuff
        self.ctx.inputs_mBSEbase.clean_workdir = Bool(False)            
        self.ctx.inputs_mBSEbase.potential_family  = self.inputs.potential_family
        self.ctx.inputs_mBSEbase.potential_mapping = self.inputs.potential_mapping
         
        self.ctx.inputs_mBSEbase.kpoints = DataFactory('core.array.kpoints')()
        self.ctx.inputs_mBSEbase.kpoints.set_kpoints_mesh( self.ctx.control.current_kmesh )


        #[3] Not currently used - for future   
        # Reference WAVECAR/CHGCAR if user provided (optional)
        # Optional: allow future use of ns_reference (currently unused by base chain)
        if 'ns_reference' in self.inputs and 'starting_RemoteData' in self.inputs.ns_reference:
            pass  # kept for symmetry; not altering the base workchain behavior
                     
        self.report("\n [wkc_KptsConv][DFT+mBSE calc - NonSpinPolarized - From scratch]\n  > Lauching DFT+mBSE(NonSpinPolarized) using VaspmBSEInitScriptWorkChain on k-mesh "
                            +np.array2string(   self.ctx.control.current_kmesh , separator=" , ").replace('\n', '')+"\n")  
        running_mBSEbase = self.submit(VaspmBSEInitScriptWorkChain , **self.ctx.inputs_mBSEbase) 
        return ToContext(WC_MBPT=append_(running_mBSEbase))
         
    def monitor_convergence_TEST(self): 
        self.ctx.control.iteration_counter += 1
        if self.ctx.control.iteration_counter == 1: return False
        else: return True
    
    def monitor_convergence(self): 
        # #counter starts at -1 and changes BEFORE a calc is launched
        # #1°control: counter starts=-1  -> [conv. check] -> increased to 0 -> launched 1° G0W0/mBSE
        # #2°control: counter starts= 0  -> [conv. check] -> increased to 1 -> launched 2° G0W0/mBSE
        # #3°control: counter starts= 1  -> [conv. check] -> increased to 2 -> launched 3° G0W0/mBSE

        #Initialize # ----------------------------------------------------------------------------------------------
        #Initialize - Define a AttributeDict which will be used internally for this execution of the monitor_convergence
        #It's used to group in a single dict the relevant flags/values.
        #Given the line self.ctx.monitor = AttributeDict(), it's reinitialized at each run.
        self.ctx.monitor = AttributeDict()
        self.ctx.monitor.flag_is_converged = False
        self.ctx.monitor.flag_is_optgap_converged = None       
        self.ctx.monitor.flag_is_diel_converged   = None 
        self.ctx.monitor.control_way       = deepcopy( self.ctx.control['control_way'] )  # 'kmesh' or 'kdensity' 
        self.ctx.monitor.thr      = self.inputs.ns_kpoints.convergence_threshold.value
        #Spin related variables
        self.ctx.monitor.has_spin      = ('magnetic_moment_onsite' in self.inputs['ns_parameters'])
        self.ctx.monitor.spin_channels = ['spinUp', 'spinDw'] if self.ctx.monitor.has_spin else ['spinUp']
        #Define minimum number of calculations required before convergence checks
        #The convergence based on k-mesh uses only 2 (the gaps from 2 consecutive k-meshes); the one based on k-density 3 for numerical stability.
        self.ctx.monitor.min_num_calcs_required_for_conv = 2
        

        #Initialize - Logging header
        str_log = ( f"\n [wkc_KptsConv][monitor_convergence] iteration_counter={self.ctx.control.iteration_counter}"
                     "\n  > Remember : iteration_counter starts at (i-1)th -> [conv. check] -> increased to i-th -> launched i-th G0W0/mBSE" 
                    f"\n    In monitor_convergence before launching MBPT calculation idx={self.ctx.control.iteration_counter+1}"                     )
        

        #Initialize - construct arrays of all successful mBSE children
        self.ctx.control['wc_MBPT_successful_sorted'] = []
        self.ctx.control['wc_MBPT_successful_sorted_elaborated'] = []
        if len(self.ctx.WC_MBPT) >= 1:
            #WC_MBPT stores the mBSE workchainNodes, but it's not guaranteed it has all successfully finished nodes 
            #or nodes are ordered (AiiDA may return them in whatever order they completed).
            tmp_finished_wc = []
            tmp_finished_wc_kmesh = []    
            for idx , wc in enumerate(self.ctx.WC_MBPT):  #Thus we reconstruct the list of WC succesfully finished
                #We filter “successful” WC_MBPT nodes not by AiiDA  by actual availability of outputs required for convergence
                #If the BSE matrix is solved with a iterative method (IBSE=1/3) has_opt will be false; but has_diel should be 
                #always TRUE for a successful calculation
                has_opt = ('opticaltransitions' in wc.outputs)
                has_diel = ('dielectrics' in wc.outputs and wc.outputs.dielectrics is not None
                            and "idiel" in wc.outputs.dielectrics.get_arraynames()
                            and "ediel" in wc.outputs.dielectrics.get_arraynames())
                if not has_diel:
                    self.report(f"WARNING - WC_MBPT[{idx}] (pk={wc.pk}) missing diel output - skipping for convergence."     )
                    continue     
                
                tmp_finished_wc.append( wc )
                tmp_finished_wc_kmesh.append( wc.inputs.kpoints.get_kpoints_mesh()[0] )
            
            #Sort by actual kmesh (increasing)
            km_tuples  = [tuple(km) for km in tmp_finished_wc_kmesh]
            idx_sorted = sorted(range(len(km_tuples)), key=lambda i: km_tuples[i])
            if len(km_tuples) > 0:
                self.ctx.control['wc_MBPT_successful_sorted'] = [tmp_finished_wc[i] for i in idx_sorted] #sorting
            
            #Build convergence list (fully populated from sorted finished nodes)
            #the convergence list contains the diel.function / gaps already extreacyrf
            for idx , wc in enumerate( self.ctx.control['wc_MBPT_successful_sorted'] ):  
                record = AttributeDict()
                record["kmesh"] =  wc.inputs.kpoints.get_kpoints_mesh()[0]
            
                if 'opticaltransitions' in wc.outputs:
                    gap_last, osc_last = _extract_opticalgap_fromWorkchainNode(wc)
                    record["optgap"] = gap_last ; record["oscstr"] = osc_last
                else:
                    record["optgap"] = None ; record["oscstr"] = None
                if "dielectrics" in wc.outputs:
                    record["imdiel"]     = wc.outputs.dielectrics.get_array("idiel")
                    record["energygrid"] = wc.outputs.dielectrics.get_array("ediel")
                    record["imdiel_onset"] = helper_kptsConv_mBSE._detect_energy_of_diel_onset(record["imdiel"], record["energygrid"] , 
                                                                  thr_for_considering_offset=0.1         )    
                else:
                    record["imdiel"] = None ; record["energygrid"] = None ; record["imdiel_onset"] = None
                self.ctx.control['wc_MBPT_successful_sorted_elaborated'].append(record)
                self.ctx.control['convergence'].append(record) #: also store the record for final output
    
        
        #Initialize - Print optical.gaps of all finished children workchain
        str_log += "\n    [optgap-conv] Optical gaps of completed nodes:"
        if len(self.ctx.control['wc_MBPT_successful_sorted']) == 0:
            str_log += "    (none yet)"
        else:
            for idx, wc in enumerate(self.ctx.control['wc_MBPT_successful_sorted']):
                if 'opticaltransitions' in wc.outputs:
                      gap, osc = _extract_opticalgap_fromWorkchainNode(wc)
                      str_log += (f"\n      [idx={idx}] kmesh={wc.inputs.kpoints.get_kpoints_mesh()[0]}  \t"
                                f"optgap={gap:.4f} eV   osc={osc:.4f}")
                else: str_log += f"\n      [{idx}]\t(no opticaltransitions output)"
        
        
        ##[DECISION.BLOCK - 1] If two mBSE calcs fails in a row, Error ##-------------------------------------------
        # Determine whether last child succeeded
        if len(self.ctx.WC_MBPT) > 0:
            last_wc = self.ctx.WC_MBPT[-1]     # last child node
            if last_wc not in self.ctx.control['wc_MBPT_successful_sorted']:
                self.ctx.control['fail_counter'] += 1   # last child FAILED
            else:
                self.ctx.control['fail_counter'] = 0    # last child succeeded, reset counter
        if self.ctx.control['fail_counter'] >= 2:
           str_log += ( "\n   > ERROR: Two consecutive mBSE calculations failed."
                        "\n     Aborting convergence loop. Life is hard, sorry."     )
           self.report(str_log)
           return self.exit_codes.TWO_CONSECUTIVE_MBSE_FAILURE
        
        ##[DECISION.BLOCK - 2] Early exit if not enough calculations yet ##-----------------------------------------
        #Counter starts at -1 ; it's increased AFTER the convergence check + but BEFORE launching the calculation
        #The counter increase is the LAST thing done before returning; thus:
        #  1°control: counter starts=-1  -> [conv. check] -> increased to 0 -> launched 1° G0W0/mBSE
        #  2°control: counter starts= 0  -> [conv. check] -> increased to 1 -> launched 2° G0W0/mBSE
        #  3°control: counter starts= 1  -> [conv. check] -> increased to 2 -> launched 3° G0W0/mBSE
        #  4°control: counter starts= 2  -> convergence is checked; if not changed to 3 -> launched 3° G0W0/mBSE
        #  [..]
        #We have N k-density to test:
        #  N-1° control: counter starts= N-3 -> changed to N-2 -> launched N-1° G0W0/mBSE
        #  N°   control: counter starts= N-2 -> changed to N-1 -> launched N° G0W0/mBSE
        #  N+1° control  counter starts= N-1 -> exit with error  
        #[k-mesh case] Consider that the counter is increased BEFORE launching the calculation, and starts at -1; 
        #                so when counter=0 here we are at the conv.check of the second iteration, before launching the calculation
        #                and we have therefore to return True in order to continue and perform the second calculation
        #                at the beginning of the third iteration, (before the third calculation) self.ctx.control.iteration_counter will be == 1
        #                and remember that self.ctx.monitor.min_num_calcs_required_for_conv = 2 for the k-mesh
        
        num_finished_successfully = len(self.ctx.control['wc_MBPT_successful_sorted_elaborated'])
        if num_finished_successfully < (self.ctx.monitor.min_num_calcs_required_for_conv ):
            str_log += ( f"\n  > Not enough successful BSE calculations "
                         f"({num_finished_successfully}/{self.ctx.monitor.min_num_calcs_required_for_conv})."
                         "\n    → Launch next calculation.\n"  )
            self.report(str_log)   
            self.ctx.monitor.flag_is_converged = False


        #[ELABORATION BLOCK - 1][Determining if self.ctx.monitor.flag_is_converged FOR THE kmesh control_way]
        if num_finished_successfully >= (self.ctx.monitor.min_num_calcs_required_for_conv ) :
            
            #the function outputs: summary_str is the log of all previous dist and optgap values
            #                      wc_diffs_optgap is the list of differences between opt.gap of consecutive gaps
            #                      wc_diffs_dieldist is the list of distance based on the metric based on 
            #                      consecutive mBSE calculations
            summary_str , wc_diffs_optgap ,  wc_diffs_dieldist = helper_kptsConv_mBSE._summary_list_kmesh_gap_diel(
                    wc_list     = self.ctx.control['wc_MBPT_successful_sorted_elaborated'] , 
                    window_size = float(self.inputs.ns_converge.dielfunction_window.value) , 
                    diel_metric = self.inputs.ns_converge.dielfunction_distance.value      )
            str_log += summary_str
            
            #[First convergence check : on Optical-gap convergence (use the precomputed Δopt) ]
            delta_opt = wc_diffs_optgap[-1]
            if delta_opt is not None:
                thr_on_optgap = self.inputs.ns_kpoints.convergence_threshold.value
                self.ctx.monitor.flag_is_optgap_converged = (delta_opt < thr_on_optgap)
                str_log += ( f"\n   > [optgap-conv] Δopt={delta_opt:.4f}  (thr={thr_on_optgap}) "
                             f"→ conv={self.ctx.monitor.flag_is_optgap_converged}"              )
          
            #[Second convergence check : the dielectric function (using precomputed diel.distance using the metric)
            delta_diel = wc_diffs_dieldist[-1]
            if delta_diel is not None:
                thr_on_dielfun = self.inputs.ns_kpoints.convergence_threshold.value
                self.ctx.monitor.flag_is_diel_converged = (delta_diel < thr_on_dielfun)
                str_log += ( f"\n  > [diel-conv] Δdiel={delta_diel:.4e}  (thr={thr_on_dielfun}) "
                             f"→ conv={self.ctx.monitor.flag_is_diel_converged}"           )
                
            
        #[ELABORATION BLOCK - 2][Dynamic k-mesh step refinement (based on Δdiel only)]
        #If last diel.distance is > 4*thr_on_dielfun, we ASSUME we are far from convergence
        #Thus we change the step to double initial step to speed up and skip superfluos calcs
            if self.inputs.ns_converge.convergence_dynamic_control.value:
                # the threshold used for dielectric convergence
                thr_on_dielfun = self.inputs.ns_kpoints.convergence_threshold.value
              
                # Only act if Δdiel is available
                if delta_diel is not None:
                    # Condition: Δdiel >> thr (and we consider the >> as > 4*)→ we are far from convergence
                    if (delta_diel > 4.0 * thr_on_dielfun):
                        # read the current step
                        self.ctx.control.step_kmesh = 2 * self.ctx.control.step_kmesh_BASE
                        self.ctx.control["dynamic_step_refined"] = True
                        str_log += (  f"\n  > [dynamic-step] Large Δdiel={delta_diel:.3e} → "
                                      f"using doubled base-step: {self.ctx.control.step_kmesh}")  
                    else:  
                        # reset to original step
                        self.ctx.control.step_kmesh = deepcopy( self.ctx.control.step_kmesh_BASE )
                        self.ctx.control["dynamic_step_refined"] = False

 
            #[Updating self.ctx.monitor.flag_is_converged ]
            use_diel  = self.inputs.ns_converge.dielfunction_convergence.value
            use_gap   = self.inputs.ns_converge.opticalgap_convergence.value
            flag_gap  = self.ctx.monitor.flag_is_optgap_converged
            flag_diel = self.ctx.monitor.flag_is_diel_converged
            if use_diel and use_gap: #Case 1: both metrics enabled
                # check flag_gap / flag_diel because if convergence check fails the value is None
                if (flag_gap is True) and (flag_diel is True):
                    final_flag = True
                    str_log += ( f"\n  > [final-conv] Both criteria active → "
                                 f"optgap={flag_gap}, diel conv={flag_diel} → final conv={final_flag}"  )
                else:
                    final_flag = False
                    str_log += ( f"\n  > [final-conv] Both criteria active → "
                                 f"optgap={flag_gap}, diel conv={flag_diel} → final conv={final_flag}"  )
            elif use_gap and not use_diel: #Case 2: ONLY optical gap convergence
                #final_flag = True if (flag_gap == True) else False
                final_flag = flag_gap
                str_log += ( f"\n  > [final-conv] Using optical-gap convergence only → "
                             f"optgap={flag_gap} → final conv={final_flag}"           )
            
            elif use_diel and not use_gap: #Case 3: ONLY dielectric-function convergence
                #final_flag = (flag_diel == True)
                final_flag = flag_diel
                str_log += ( f"\n  > [final-conv] Using dielectric-function convergence only → "
                             f"diel conv={flag_diel} → final conv={final_flag}"            )
            else: #Case 4: user disabled both (should never happen, but safe fallback)
                final_flag = False
                str_log += ( f"\n  > [final-conv] WARNING: Both convergence metrics disabled → "
                             f"forcing NOT converged."          )
            self.ctx.monitor.flag_is_converged = final_flag
        

        ##[DECISION.BLOCK - 3] Final control logic --------------------------------------------------------------
        # Decide whether to continue or stop based on convergence.
        # The max-number-of-calculations check is already handled in DECISION.BLOCK - 1.
        if bool(self.ctx.monitor.flag_is_converged) :
            str_log += (  f"\n   --> convergence REACHED at iteration {self.ctx.control.iteration_counter}"
                          f"\n       Converged k-mesh = {self.ctx.control.current_kmesh}"                 )
            self.report(str_log)
            
            # store converged kmesh
            self.ctx.control['kmesh_converged'] = DataFactory('core.array.kpoints')()
            self.ctx.control['kmesh_converged'].set_kpoints_mesh(   self.ctx.control.current_kmesh )
            return False  # stop workflow
        else:
            # update kmesh
            self.ctx.control.iteration_counter += 1
            self.ctx.control.current_kmesh = ( self.ctx.control.current_kmesh + self.ctx.control.step_kmesh )

            # safety: check bounds
            if np.any(self.ctx.control.current_kmesh > self.ctx.control.max_kmesh):
                str_log += (  f"\n   --> convergence NOT reached and maximum kmesh exceeded:"
                              f"      next kmesh would be {self.ctx.control.current_kmesh}"
                              f"\n   --> aborting"     )
                self.report(str_log)
                return self.exit_codes.CONVERGENCE_NOT_FOUND

            str_log += ( f"\n   --> convergence NOT reached - continuing"
                         f"\n       next kmesh = {self.ctx.control.current_kmesh}" )
            self.report(str_log)

            return True   # continue workflow
 
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
    
    
    
    
