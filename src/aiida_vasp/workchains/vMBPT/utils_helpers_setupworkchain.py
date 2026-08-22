# -*- coding: utf-8 -*-
from xml.etree import cElementTree as ET
import numpy as np
import os
from scipy.optimize import curve_fit
from pymatgen.io.vasp.outputs import Vasprun , BSVasprun
import warnings
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Code, Str, Int, Dict, Float , KpointsData , RemoteData , FolderData



##[Preliminar definitions] Python functions and scripts
class Helpers_setup_Workchain :
    """
    This module provides a collection of helper functions used to construct the
    input dictionary required by the `VaspmBSEInitScriptWorkChain`.
    It does NOT run calculations, does NOT submit processes.
    It ONLY prepares correctly-typed inputs and derived parameters
 
    
    [1] INPUTS REQUIRED BY THE HELPER FUNCTIONS:
    [Str : vasprun_path] path on local disk of vasprun.xml reference G0W0 calculation (on disk) 
            -> used to extract the G0W0 gap for the interpolation 
               + dielectric function information for the model
    [Str : potential_family] a POTCAR family in the AiiDA database
    [pymatgen Structure obj] a Pymatgen structure (POSCAR)
    [dict : setup_cluster] a user-provided configuration for the cluster
    [dict : slurm] a user-provided SLURM configuration for the submission calculation

    [2] ASSUMPTIONS MADE BY THE HELPERS
    - The reference GW folder contains: vasprun.xml.3 - OUTCAR 
    - POTCAR families are correctly imported into AiiDA
    - The cluster configuration dictionary contains valid fields:
          account, qos, partition, mem_kb_per_task
    - pymatgen structure is fully parsed from POSCAR

    [3] AN OVERVIEW OF WHAT VaspmBSEInitScriptWorkChain requires
    The workchain VaspmBSEInitScriptWorkChain requires the following inputs:
      ├── structure                   (StructureData)    ← NOT provided by helper functions
      ├── kpoints                     (KpointsData)      ← NOT provided by helper functions
      |        these two can be construicted in the main calling script by 
      |        pymatgen_structure = pcs.Structure.from_file( os.path.join( arg_inputs['PATH_REFERENCEGW'],'POSCAR') )
      |        inputs.structure   = structure.StructureData( pymatgen_structure=pymatgen_structure )
      |        inputs.kpoints = KpointsData() ; inputs.kpoints.set_kpoints_mesh( [..] )
      ├── code                        (Code)             ← Provided by _build_slurm_options
      ├── options                     (Dict)             ← Provided by _build_slurm_options
      ├── potential_family            (Str)              ← From _build_potential_mapping
      ├── potential_mapping           (Dict)             ← From _build_potential_mapping
      │
      ├── ns_parameters
      │     ├── encut                 (Float)    [optional - NOT provided by helper functions]
      │     ├── nbands                (Int)      [optional - NOT provided by helper functions]
      │     ├── magnetic_moment_onsite (Dict)    [optional - NOT provided by helper functions]
      │     ├── ibse                  (Int)      [optional - NOT provided by helper functions]
      │     └── kpar                  (Int)      [optional - NOT provided by helper functions]
      │
      ├── ns_interpolation  [MOVED 2026-08-22 - VaspmBSEInitScriptWorkChain no
      │     longer accepts this namespace, and the builder that used to
      │     populate it (_build_interpolation_inputs) now lives in
      │     aiida-vasp-qpcorrection/legacy_reference/legacy_build_interpolation_inputs.py]
      ├── ns_BSE
      │     ├── static_inverse_diel  (Float)    ← from _build_bse_inputs
      │     ├── screening_parameter   (Float)   ← from _build_bse_inputs
      │     ├── G0W0_gap              (Float)   ← from _build_bse_inputs
      │     ├── optical_energy_window (Float)   ← from _build_bse_inputs (default value) OR user
      │     ├── OMEGAMAX              (Float)   [optional override]
      │     ├── NBANDSV               (Int)     [optional override]
      │     └── NBANDSO               (Int)     [optional override]
      """

    # -------------------------------------------------------------------------
    #[1] Extract dielectric screening parameters from epsilon_diag in vasprun.xml
    @staticmethod
    def _estimate_parameters_from_epsilon_diag( vasprun_path ):
        # Internal helper to parse <v> entries ------- #
        # Taken straight from aiida-vasp internal parser.
        def _convert_array2D_f(entry, dim):  # pylint: disable=C0103
                data = None
                if entry is not None:
                    data = np.zeros((len(entry), dim), dtype='double')
                for index, element in enumerate(entry):
                    try:
                        data[index] = np.fromstring(element.text, sep=' ')
                    except ValueError as err:
                            print("ERRORE ERRORE ERRORE")
                return data
        
        # Parse XML
        vasprun = ET.parse( vasprun_path )
        vasprun_ep_diag_ET = vasprun.findall('.//varray[@name="epsilon_diag"]/v')
        vasprun_ep_diag    = np.asarray(_convert_array2D_f(vasprun_ep_diag_ET, 2))
       
        # Extract arrays - the x are the |G|, the y are the diagonal epsilon value eps(|G|)
        x = vasprun_ep_diag[:, 0]
        y = vasprun_ep_diag[:, 1]
        invepsilonatlowestg = vasprun_ep_diag[ np.argmin( vasprun_ep_diag[:,0] ) , 1]
        init_vals = [1.26]

        # Fit exponential dielectric model - the model is the one used for the VASP model-BSE
        def modelDiel(x,a1 , eps_lowestG=invepsilonatlowestg):
        	return 1 - (1-eps_lowestG)*np.exp( -x**2 / (4*a1**2)  )   
        
        #best_val , covar = optimize.curve_fit(modelDiel, x, y, p0=init_vals)
        best_val , covar = curve_fit(modelDiel, x, y, p0=init_vals)
        return [ invepsilonatlowestg , best_val[0] ]


    # -------------------------------------------------------------------------
    #[3] Extract G0W0 band gap using pymatgen
    @staticmethod
    def _extract_gap ( vasprun_path ):
        from pymatgen.io.vasp.outputs import BSVasprun
        from pymatgen.electronic_structure.core import Spin

        BSV     = BSVasprun( vasprun_path )
        BSV_bs  = BSV.get_band_structure()
        BSV_gap = BSV_bs.get_band_gap()['energy']
        return BSV_gap  


    # -------------------------------------------------------------------------
    #[4] Build SLURM options + Code node based on cluster type (cpu/gpu)
    @staticmethod
    def _build_slurm_options( setup_cluster: dict , setup_slurmcalc: dict ) -> tuple[Dict, str]:
        """
        Create an AiiDA Dict of scheduler options and return the selected code string.

        Parameters
        ----------
        setup_cluster : dict
            Dict of cluster setups for 'cpu' and 'gpu', e.g.:
            { 'cpu': {'code': 'vasp@localhost', 'account': 'cin_staff',
                      'qos': 'boost_qos_dbg', 'partition': 'boost_usr_prod',
                      'mem_kb_per_task': 130000000},
              'gpu': {...}         }

        setup_slurmcalc : dict
            Dict with SLURM run settings:
            {'num_nodes': 1, 'ntasks-per-node': 4,
             'are_gpu_available': True, 'time_in_h': 10}

        Returns
        -------
        (Dict, str)
            The AiiDA Dict node with scheduler options and the selected code string.
        """

        # Choose cluster type (gpu or cpu)
        if setup_slurmcalc.get('use_gpu', False):
            cluster = setup_cluster['gpu']
        else:
            cluster = setup_cluster['cpu']

        # Extract core info
        code_string = cluster['code']
        
        # Compose base options
        options = AttributeDict()
        options.account    = cluster.get('account', 'cin_staff')
        options.qos        = cluster.get('qos', '')
        options.queue_name = cluster.get('partition', '')
        #From aiida.schedulers.plugins.slurm
        # lines.append(f'#SBATCH --partition={job_tmpl.queue_name}')
        
        
        tasks_per_node  = int(setup_slurmcalc.get('ntasks-per-node', 1))
        mem_GB_per_task = int(setup_slurmcalc.get('mem_GB_per_task', 1024000))
        max_mem_GB_per_node_available = int(cluster.get('max_mem_GB_per_node_available'))
        options.resources = { 'num_machines': int(setup_slurmcalc.get('num_nodes', 1)) ,
                              'num_mpiprocs_per_machine':tasks_per_node                }
        
        
        # Compose Time
        options.max_wallclock_seconds = int( float(setup_slurmcalc.get('time_in_h', 1)) * 3600 )
        
        # Compose Memory
        #Code in aiida.schedulers.plugins.slurm  https://aiida.readthedocs.io/projects/aiida-core/en/stable/_modules/aiida/schedulers/plugins/slurm.html
        #if job_tmpl.max_memory_kb is not None:
        #    physical_memory_kb = int(job_tmpl.max_memory_kb)
        #    lines.append(f'#SBATCH --mem={physical_memory_kb // 1024}')
        #if job_tmpl.custom_scheduler_commands:
        #    lines.append(job_tmpl.custom_scheduler_commands
        # Inside aiida/schedulers/plugin/slurm.html AiiDA calculates memory using multiple of 1024 : mem={physical_memory_kb // 1024}'
        # For slurm, default units are megabytes.
        memory_GB_per_node = int( mem_GB_per_task * tasks_per_node )
        if memory_GB_per_node > max_mem_GB_per_node_available:
            print(f"INFO : [_build_slurm_options] requested memory_GB_per_node={memory_GB_per_node} "
                  f"exceeds max_mem_GB_per_node_available={max_mem_GB_per_node_available} for this cluster/partition "
                  f"- clamping down to {max_mem_GB_per_node_available}.")
        memory_GB_per_node = min( memory_GB_per_node , max_mem_GB_per_node_available )
        memory_MB_per_node = memory_GB_per_node * 1000
        physical_memory_kb = memory_MB_per_node * 1024 
        options.max_memory_kb = int( physical_memory_kb )

        #Compose use_gpu   
        if setup_slurmcalc.get('use_gpu', False):
            options.custom_scheduler_commands = f"#SBATCH --gres=gpu:{tasks_per_node}"

        # Build AiiDA objects
        code_node      = Code.get_from_string(code_string)
        computer_label = code_string.split("@")[1]
        options_node   = Dict(dict=options)

        return code_node, computer_label, options_node

    # -------------------------------------------------------------------------
    #[5] Build POTCAR mapping (element → POTCAR label)
    @staticmethod
    def _build_potential_mapping(potential_family: str, pymatgen_structure,
                                  flag_prefer_GW: bool = True) -> tuple:
        """
        Construct a POTCAR mapping dictionary {element: potcar_name} based on
        the elements in the given structure and the available POTCARs in the family.
    
        Priority rules:
          - If flag_prefer_GW = True:
              prefer _sv_GW > _d_GW > _GW_new > _h_GW > _GW > _sv > _d > plain
          - If flag_prefer_GW = False:
              prefer _sv > _d > plain > _sv_GW > _d_GW > _GW_new > _h_GW > _GW
    
        Parameters
        ----------
        potential_family : str
            Label of the POTCAR family (e.g. "PBE.54").
        pymatgen_structure : pymatgen.Structure
            Structure whose elements are used to build the mapping.
        flag_prefer_GW : bool, optional
            Whether to prioritize _GW potentials (default: True).
    
        Returns
        -------
        tuple
            (Str(potential_family), Dict(potential_mapping))
            where potential_mapping is a dict[str, str] mapping element symbol to POTCAR name.
        """
        from aiida.orm import Str, Dict
        import aiida_vasp.data.potcar
    
        # Load POTCAR family
        potcar_group = aiida_vasp.data.potcar.PotcarGroup.collection.get(label=potential_family)
    
        # element → list of variants
        potcar_dict = {}
        for node in potcar_group.nodes:
            potcar_dict.setdefault(node.element, []).append(node.full_name)
    
        print("[INFO] POTCARs available in family:")
        for el, variants in potcar_dict.items():
            print(f"    {el}: {variants}")
    
        # Priority order
        if flag_prefer_GW:
            priority = ["_sv_GW", "_d_GW", "_GW_new", "_h_GW", "_GW", "_sv", "_d"]
            print("[INFO] Priority (GW-preferred):", priority)
        else:
            priority = ["_sv", "_d", "_sv_GW", "_d_GW", "_GW_new", "_h_GW", "_GW"]
            print("[INFO] Priority (non-GW preferred):", priority)
    
        potential_mapping = {}
    
        print("\n[INFO] Building element → POTCAR mapping")
        for el in pymatgen_structure.elements:
            elname = str(el)
            candidates = potcar_dict.get(elname, [])
    
            print(f"\n  Element {elname}")
            print(f"    Candidates: {candidates}")
    
            if not candidates:
                raise ValueError(f"No POTCAR found for element {elname} in family {potential_family}")
    
            selected = None
    
            # 1) Priority-based matching: "_sv_GW", "_d_GW", ...
            print("    Trying priority suffix matching...")
            for key in priority:
                for cand in candidates:
                    if key in cand:
                        selected = cand
                        print(f"    → Selected by priority '{key}': {selected}")
                        break
                if selected:
                    break
    
            # 2) Exact match for plain element (e.g., "O", "Ti", "Si")
            if selected is None:
                print("    Trying exact match for plain element...")
                for cand in candidates:
                    if cand == elname:
                        selected = cand
                        print(f"    → Exact match selected: {selected}")
                        break
    
            # 3) Match plain prefix: "O", "Ti", "Si" (as fallback)
            if selected is None:
                print("    Trying plain element prefix match...")
                for cand in candidates:
                    if cand.split("_")[0] == elname:
                        selected = cand
                        print(f"    → Prefix match selected: {selected}")
                        break
    
            # 4) Deterministic fallback
            if selected is None:
                selected = sorted(candidates)[0]
                print(f"    [WARN] No preferred match; deterministic fallback: {selected}")
    
            # Clean name is just full_name (AiiDA-VASP format)
            potcar_clean_name = selected
            print(f"    → Final selection: {potcar_clean_name}")
    
            potential_mapping[elname] = potcar_clean_name
    
        print("\n[INFO] Final POTCAR mapping:")
        for el, pot in potential_mapping.items():
            print(f"    {el}: {pot}")
    
        return Str(potential_family), Dict(potential_mapping)

    # -------------------------------------------------------------------------
    # [6] _build_interpolation_inputs (ns_interpolation) moved out 2026-08-22 -
    # see aiida-vasp-qpcorrection/legacy_reference/legacy_build_interpolation_inputs.py.
    # -------------------------------------------------------------------------
    #[7] Build BSE-related input namespace (ns_BSE)
    @staticmethod
    def _build_bse_inputs(local_folder_gw_reference: str , 
                          gw_reference_filename_vasprun:  str = 'vasprun.xml.3',
                          ) -> AttributeDict:
        """
        Prepare ns_BSE:
          - static_inverse_diel (Float)
          - screening_parameter (Float)
          - G0W0_gap (Float)
          - optical_energy_window (Float)

        Notes
        -----
        The optical window is currently hard-coded to 3.5 eV,
        but can be made user-adjustable in the future.

        Requires vasprun.xml.3 to exist in the reference GW directory.
        """
        vasprun_path = os.path.join(local_folder_gw_reference, gw_reference_filename_vasprun)
        AEXX, HFSCREEN = Helpers_setup_Workchain._estimate_parameters_from_epsilon_diag( vasprun_path )
        gap = Helpers_setup_Workchain._extract_gap( vasprun_path )

        ns_bse = AttributeDict()
        ns_bse["static_inverse_diel"] = Float(AEXX)
        ns_bse["screening_parameter"] = Float(HFSCREEN)
        ns_bse["G0W0_gap"] = Float(gap)
        ns_bse["optical_energy_window"] = Float(3.5)

        return ns_bse

    # -------------------------------------------------------------------------
    #[7.1] Build a BandsData node from the reference DFT/GW vasprun (for IPA-transition-matrix helpers)
    @staticmethod
    def _build_bandsdata_from_vasprun(local_folder_gw_reference: str,
                                      gw_reference_filename_vasprun: str = 'vasprun.xml.3',
                                      ) -> orm.BandsData:
        """
        Parse band energies, occupations and k-points from the reference vasprun.xml
        and wrap them into a stored AiiDA BandsData node.

        Used to feed `_determine_BSE_parameters` (utils_helpers_mBSE.py), which needs
        the DFT eigenvalues/occupations to build the independent-particle transition matrix.

        Requires vasprun.xml.3 (or whichever filename is passed) to exist in the
        reference GW directory.
        """
        from pymatgen.electronic_structure.core import Spin

        # BSVasprun (not the full Vasprun) matches the parser already used by
        # _extract_gap for this exact file - lighter, and proven to work on it.
        vasprun_path = os.path.join(local_folder_gw_reference, gw_reference_filename_vasprun)
        vasprun = BSVasprun(vasprun_path)

        spin_key = Spin.up if Spin.up in vasprun.eigenvalues else next(iter(vasprun.eigenvalues))
        eigenvalues_and_occupations = vasprun.eigenvalues[spin_key]  # shape (n_kpts, n_bands, 2)
        bands = eigenvalues_and_occupations[:, :, 0]
        occupations = eigenvalues_and_occupations[:, :, 1]
        kpoints = np.array(vasprun.actual_kpoints)

        bandsdata = orm.BandsData()
        bandsdata.set_kpoints(kpoints)
        bandsdata.set_bands(bands, units='eV', occupations=occupations)
        bandsdata.store()
        return bandsdata

    # -------------------------------------------------------------------------
    #[7.2] Build the sparse-mesh QP-correction BandsData + KpointsData needed by
    #      aiida-vasp-qpcorrection's VaspQPInterpolationWorkChain (ns_qpcorrection.*)
    #      from a raw, externally-run (non-AiiDA-tracked) GW OUTCAR.
    @staticmethod
    def _build_bandsdata_qpcorrection_from_outcar(local_folder_gw_reference: str,
                                                  gw_reference_filename_outcar: str = 'OUTCAR.3',
                                                  ) -> tuple[orm.BandsData, orm.KpointsData]:
        """
        Parse the "QP shifts <psi_nk| G(iteration)W_0 |psi_nk>" section of a raw,
        externally-run (non–spin-polarized) GW OUTCAR and wrap the resulting
        sparse-mesh QP correction (E_G0W0 - E_DFT, VASP's own "QPC" column) into
        a stored BandsData, plus a KpointsData carrying the sparse mesh dims read
        from the same OUTCAR (`generate k-points for:` line).

        Faithful, scoped port of `BandsState_IO.parse_outcar_spinUnpol`
        (originally `utils_interpolationclasses.v2.py:52-200`, now
        aiida-vasp-qpcorrection/legacy_reference/utils_interpolationclasses.v2.py
        after the 2026-08-22 move) - restricted to exactly
        what `QpInterpolationCalculation`'s `bandsdata_g0w0`/`kpoints_mesh_sparse`
        inputs need (the QPC eigenvalue-correction array + IBZ k-points + mesh),
        dropping the BandsState/structure/misc wrapping the original built around
        it. Spin-unpolarized only - the original script never had a spin-polarized
        variant either.

        CAVEAT: not yet verified against a real GW OUTCAR (none exists in either
        repo's test_data - see handoff.md) - treat as a careful-but-unverified
        translation until such a fixture-based check has been done, same caveat
        as `scripts/interpolation/run.py` and `scripts/wavefun_correct/run.py`.
        """
        from itertools import islice

        outcar_path = os.path.join(local_folder_gw_reference, gw_reference_filename_outcar)
        if not os.path.isfile(outcar_path):
            raise ValueError(f"[_build_bandsdata_qpcorrection_from_outcar] OUTCAR not found: {outcar_path}")

        c_nkpts = None
        lineidx_starts_gw, lineidx_kpts_ibz = [], []
        bs_kpts_mesh = None
        with open(outcar_path, "r") as fobj:
            for idx, line in enumerate(fobj):
                if "NKPTS" in line:
                    c_nkpts = int(line.split()[3])
                elif "QP shifts <psi_nk| G(iteration)W_0 |psi_nk>" in line:
                    lineidx_starts_gw.append(idx)
                elif "generate k-points for:" in line:
                    bs_kpts_mesh = list(map(int, line.split()[3:6]))
                elif "Subroutine IBZKPT returns following result" in line:
                    lineidx_kpts_ibz.append(idx)
        if c_nkpts is None:
            raise ValueError(f"Could not find NKPTS in OUTCAR: {outcar_path}")
        if not lineidx_kpts_ibz:
            raise ValueError(f"Could not find IBZKPT section in OUTCAR: {outcar_path}")
        if not lineidx_starts_gw:
            raise ValueError(
                f"No 'QP shifts <psi_nk| G(iteration)W_0 |psi_nk>' section found in OUTCAR: {outcar_path} "
                "- this does not look like a GW OUTCAR."
            )

        bs_kpts_list = []
        with open(outcar_path, "r") as fobj:
            for i, line in enumerate(fobj):
                if lineidx_kpts_ibz[0] + 6 < i < lineidx_kpts_ibz[0] + c_nkpts + 7:
                    bs_kpts_list.append(list(map(float, line.split()[:3])))
        bs_kpts_list = np.array(bs_kpts_list)

        kpts_lines = np.zeros(c_nkpts, dtype=int)
        lineidx_start_after = lineidx_starts_gw[0]
        with open(outcar_path, "r") as fobj:
            for ln, line in enumerate(islice(fobj, lineidx_start_after, None), start=lineidx_start_after):
                s = line.strip()
                if not s.startswith("k-point"):
                    continue
                parts = s.replace("+", "").split()
                if len(parts) < 6 or parts[2] != ":":
                    continue
                ik = int(parts[1])
                if 1 <= ik <= c_nkpts:
                    kpts_lines[ik - 1] = ln
                    if ik == c_nkpts:
                        break

        c_nbands_printed = kpts_lines[1] - kpts_lines[0] - 4

        qpc = np.zeros((c_nkpts, c_nbands_printed))
        for ik in range(c_nkpts):
            with open(outcar_path, "r") as fobj:
                for line in islice(fobj, kpts_lines[ik] + 3, kpts_lines[ik] + c_nbands_printed + 3):
                    ls = line.split()
                    if len(ls) < 8:
                        continue
                    ib = int(ls[0]) - 1
                    qpc[ik, ib] = float(ls[2]) - float(ls[1])  # E_GW - E_DFT

        bandsdata_qpc = orm.BandsData()
        bandsdata_qpc.set_kpoints(bs_kpts_list)
        bandsdata_qpc.set_bands(qpc, units='eV')
        bandsdata_qpc.store()

        kpoints_mesh_sparse = orm.KpointsData()
        kpoints_mesh_sparse.set_kpoints_mesh(bs_kpts_mesh)

        return bandsdata_qpc, kpoints_mesh_sparse


    # -------------------------------------------------------------------------
    #[8] Pretty print
    @staticmethod
    def _unwrap_aiida(val):
        """Return primitive value for AiiDA data types."""
        try:
            return val.value
        except AttributeError:
            return val

    @staticmethod
    def _format_dict(d, indent=0):
        """Recursively format AttributeDict, Dict, or namespaces as text."""
        sp = "  " * indent
        lines = []
        if isinstance(d, (AttributeDict, dict)):
            for key, val in d.items():
                if isinstance(val, (AttributeDict, dict)):
                    lines.append(f"{sp}- {key}:")
                    lines.extend(Helpers_setup_Workchain._format_dict(val, indent + 1))
                elif hasattr(val, "value"):  # AiiDA data type
                    lines.append(f"{sp}- {key}: {Helpers_setup_Workchain._unwrap_aiida(val)!r}")
                else:
                    lines.append(f"{sp}- {key}: {val}")
        else:
            lines.append(f"{sp}{d}")
        return lines
    
    @staticmethod
    def _build_inputs_summary(inputs: AttributeDict) -> str:
    
        lines = []
        lines.append("========================")
        lines.append(" INPUT SUMMARY")
        lines.append("========================")
    
        # ---- STRUCTURE ----
        if "structure" in inputs:
            st = inputs.structure
            lines.append("\n[STRUCTURE]")
            try:
                formula = st.get_formula()
                n_atoms = len(st.sites)
                lines.append(f"- formula: {formula}")
                lines.append(f"- atoms:   {n_atoms}")
            except Exception:
                lines.append("- (could not read structure metadata)")
    
        # ---- KPOINTS ----
        if "kpoints" in inputs:
            try:
                mesh = inputs.kpoints.get_kpoints_mesh()
                lines.append("\n[KPOINTS]")
                lines.append(f"- mesh: {mesh}")
            except Exception:
                pass
    
        # ---- POTENTIALS ----
        if "potential_family" in inputs:
            lines.append("\n[POTENTIALS]")
            lines.append(f"- family: {Helpers_setup_Workchain._unwrap_aiida(inputs.potential_family)}")
    
        if "potential_mapping" in inputs:
            lines.append("- mapping:")
            potmap = inputs.potential_mapping.get_dict()
            for k, v in potmap.items():
                lines.append(f"    - {k}: {v}")
    
        # ---- CODE + OPTIONS ----
        if "code" in inputs:
            lines.append("\n[CODE + SCHEDULER]")
            lines.append(f"- code: {inputs.code}")
    
        if "options" in inputs:
            lines.append("- scheduler options:")
            opts = inputs.options.get_dict()
            lines.extend(Helpers_setup_Workchain._format_dict(opts, indent=1))
    
        # ---- OTHER NAMESPACES ----
        # detect everything that looks like an AiiDA namespace
        lines.append("\n[NAMESPACES]")
    
        for key, val in inputs.items():
            if key in ("structure", "kpoints", "code", "options",
                       "potential_family", "potential_mapping"):
                continue
    
            if isinstance(val, (AttributeDict, dict)) or hasattr(val, "get_dict"):
                lines.append(f"\n[{key}]")
                if hasattr(val, "get_dict"):
                    # AiiDA Dict-like
                    lines.extend(Helpers_setup_Workchain._format_dict(val.get_dict(), indent=0))
                else:
                    # Normal namespace
                    lines.extend(Helpers_setup_Workchain._format_dict(val, indent=0))
    
        lines.append("\n========================")
        lines.append(" END OF INPUT SUMMARY")
        lines.append("========================")
    
        return "\n".join(lines)
    
    @staticmethod
    def _outcar_potcar_map(outcar_path: str) -> dict[str, str]:
        """Return the AiiDA-Ready potcar mapping (es: {'Ag': 'Ag_GW', 'Cl': 'Cl_GW'})
        parsed from SHA256/POTCAR labels present in an OUTCAR or POTCAR files ."""
        import re
        from pathlib import Path

        # Prefer SHA256 lines:  SHA256 = <hash>  Ag_GW/POTCAR
        re_sha = re.compile(r"^\s*SHA256\s*=\s*[0-9a-f]{64}\s+(?P<label>[^/\s]+)/POTCAR\s*$", re.I)
        # Fallback: POTCAR:    PAW_PBE Ag_GW 06Mar2008
        re_hdr = re.compile(r"^\s*POTCAR:\s+\S+\s+(?P<label>\S+)\s+\S+.*$", re.I)

        mapping, seen_labels = {}, set()
        for line in Path(outcar_path).read_text(errors="replace").splitlines():
            m = re_sha.match(line) or re_hdr.match(line)
            if not m:
                continue
            label = m.group("label").strip()           # e.g. Ag_GW
            if label in seen_labels:
                continue
            seen_labels.add(label)
            elem = label.split("_", 1)[0]              # e.g. Ag from Ag_GW / Ag_sv_GW / Ag_pv
            mapping[elem] = label
        return mapping

