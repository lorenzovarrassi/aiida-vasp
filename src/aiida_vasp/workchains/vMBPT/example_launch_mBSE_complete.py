# pylint: disable=too-many-arguments
"""
Reference example for launching VaspmBSECompleteWorkChain end-to-end:
    [1] k-point convergence (cheap NBANDSV=NBANDSO=2 proxy)
    [2] BSE band-subspace (NBANDSV/NBANDSO) convergence, on the SAME low/cheap
        k-mesh as [1] - not the converged dense mesh
    [3] final full mBSE calculation, using the converged k-mesh from [1] and the
        converged NBANDSV/NBANDSO from [2]

This requires a real SLURM cluster + VASP Code already configured in AiiDA, and a
local reference GW directory (POSCAR, OUTCAR.3.gz, vasprun.xml.3.gz) - it is not
runnable in CI and is not collected as an automated test. Everything that needs
to be adapted to your own setup lives in the `arg_inputs` block below.
"""
import os
import gzip
import zipfile
import shutil
from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Code, Bool, Str, Int, Dict, Float, KpointsData, RemoteData
from aiida.orm.nodes.data import structure
from aiida.engine import run
import pymatgen.core.structure as pcs

from aiida_vasp.workchains.vMBPT.workchain_mBSE_master import VaspmBSECompleteWorkChain
from aiida_vasp.workchains.vMBPT.utils_helpers_setupworkchain import Helpers_setup_Workchain

# Suppress AiiDA deprecations and useless warnings
import warnings
from aiida.common.warnings import AiidaDeprecationWarning
warnings.filterwarnings("ignore", category=AiidaDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.vasp.outputs")


#------------------------- ------ ------------ ------------ ------------ ------------ ------------ ------------------------
#[Part 2: main is launched]### -- ------------ ------------ ------------ ------------ ------------ ------------------------
if __name__ == '__main__':
    from aiida import load_profile
    load_profile()

    #[0][Inputs args - interpolation + convergence related args]###------ ------------ ------------ -----------------------
    arg_inputs = {}
    arg_inputs['CLUSTER_CHOICE'] = 'leonardo'
    arg_inputs['PATH_REFERENCEGW'] = os.path.join(os.getcwd(), '../2.2-GW')
    arg_inputs['BASH_PYTHON_VENV'] = "source ~/venv_vasp_interpolation/bin/activate"
    arg_inputs['POTENTIAL_FAMILY']  = 'PBE.54'

    # [0.1] k-point convergence (stage 1) - also the FIXED low/cheap mesh reused
    #       for the NBands convergence study (stage 2), since the BSE band
    #       subspace needed to cover a given transition window is roughly
    #       k-mesh independent.
    arg_inputs['KMESH_STARTING']            = [8, 8, 8]
    arg_inputs['KMESH_MAX']                 = [20, 20, 20]
    arg_inputs['KPTSCONV_THRESHOLD']        = 0.20   # eV, on the optical gap

    # [0.2] BSE band-subspace (NBANDSV/NBANDSO) convergence (stage 2).
    #       The control variable is an IPA transition-energy window (eV); NBANDSV/
    #       NBANDSO are derived from it via _determine_BSE_parameters, never
    #       incremented directly.
    arg_inputs['NBANDSCONV_THRESHOLD_START'] = 2.0   # eV
    arg_inputs['NBANDSCONV_THRESHOLD_MAX']   = 8.0   # eV
    arg_inputs['NBANDSCONV_THRESHOLD_STEP']  = 1.0   # eV
    arg_inputs['NBANDSCONV_NUM_BANDS_INCLUDED'] = 20
    arg_inputs['DIELFUNCTION_WINDOW']        = 3.0   # eV, shared convergence criterion

    #[Prolog-1][Cluster configurations]#------ ------------ ------------ ------------ ------------ ------------------------
    SETUP_CLUSTER = {'leonardo': {}, 'localhost': {}, 'g100interactive': {}}
    SETUP_SLURM   = {'leonardo': {}, 'localhost': {}, 'g100interactive': {}}
    SETUP_CLUSTER['localhost']['gpu'] = None
    SETUP_CLUSTER['localhost']['cpu'] = {'code': 'vasp@localhost',
                                         'max_mem_GB_per_node_available': 24}
    SETUP_SLURM['localhost'] = {'num_nodes':       1,
                                'ntasks-per-node': 2,
                                'use_gpu':   False,
                                'time_in_h': 0.50}

    SETUP_CLUSTER['leonardo']['gpu'] = {'code'      : 'vasp.6.5.1_std_booster@leonardo_login01',
                                        'account'   : 'cin_staff',
                                        'partition' : 'boost_usr_prod',
                                        'max_mem_GB_per_node_available' : 500,
                                        'max_task_per_node' : 32}
    SETUP_CLUSTER['leonardo']['cpu'] = {'code'      : 'vasp_std_dcgp_BSEsingleprec@leonardo_login01',
                                        'account'   : 'cin_staff',
                                        'partition' : 'dcgp_usr_prod',
                                        'max_mem_GB_per_node_available' : 500}
    SETUP_SLURM['leonardo'] = {'num_nodes':         1,
                               'ntasks-per-node' : 48,
                               'mem_GB_per_task' :  7,
                               'time_in_h':  12,
                               'use_gpu':    False}

    SETUP_CLUSTER['g100interactive']['gpu'] = None
    SETUP_CLUSTER['g100interactive']['cpu'] = {'code'  : 'vasp.6.5.1_std_booster@g100',
                                               'account'   : 'cin_staff',
                                               'partition' : 'g100_usr_interactive',
                                               'max_mem_GB_per_node_available' : 24}
    SETUP_SLURM['g100interactive'] = {'num_nodes':         1,
                                      'ntasks-per-node' : 24,
                                      'mem_GB_per_task' : 4,
                                      'time_in_h':  6,
                                      'use_gpu':    False}

    #[Prolog-2][Decompressing relevant files]#------------- ------------ ------------ ------------ ------------------------
    FILES_TO_UNZIP = ["OUTCAR.3.gz", "vasprun.xml.3.gz"]
    print(f"[INFO] Unzipping input files in: {arg_inputs['PATH_REFERENCEGW']}")
    for fname in FILES_TO_UNZIP:
        file_path = os.path.join(arg_inputs['PATH_REFERENCEGW'], fname)
        if fname.endswith('.gz'):
            output_path = file_path[:-3]  # remove .gz
            with gzip.open(file_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            print(f"    Unzipped: {fname} -> {os.path.basename(output_path)}")
        elif fname.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(arg_inputs['PATH_REFERENCEGW'])
            print(f"    Extracted: {fname}")
        else:
            print(f"    Skipped (unsupported format): {fname}")
    print()

    ####- ------------------------ ------------------------ ------------------------- ------------------------ ------------
    #[1][Build AiiDA input dict for VaspmBSECompleteWorkChain]###----- ------------ ------------ ------------ --------------
    inputs = AttributeDict()

    #[1.1] STRUCTURE - POTENTIAL
    pymatgen_structure = pcs.Structure.from_file(os.path.join(arg_inputs['PATH_REFERENCEGW'], 'POSCAR'))
    inputs.structure   = structure.StructureData(pymatgen_structure=pymatgen_structure)

    # POTCARs - QP corrections from a GW run done with a given POTCAR are not transferable to a
    # DFT run done with different ones (e.g. Ag_GW and Ag_sv_GW), due to different NELECT -> different
    # number of occupied bands -> band-by-band mapping becomes wrong. We therefore read the mapping
    # straight from the reference OUTCAR.
    inputs.potential_family  = arg_inputs['POTENTIAL_FAMILY']
    inputs.potential_mapping = Dict(dict=Helpers_setup_Workchain._outcar_potcar_map(
        os.path.join(arg_inputs['PATH_REFERENCEGW'], 'OUTCAR.3')))

    #[1.2] SLURM + Code
    inputs.code, computer_label, inputs.options = Helpers_setup_Workchain._build_slurm_options(
        SETUP_CLUSTER[arg_inputs['CLUSTER_CHOICE']], SETUP_SLURM[arg_inputs['CLUSTER_CHOICE']])

    #[1.3] INTERPOLATION INPUTS
    inputs.ns_interpolation = Helpers_setup_Workchain._build_interpolation_inputs(
        local_folder_gw_reference=arg_inputs['PATH_REFERENCEGW'])
    inputs.ns_interpolation.python_sourcing_env_command = arg_inputs['BASH_PYTHON_VENV']

    #[1.4] BSE INPUTS (static_inverse_diel, screening_parameter, G0W0_gap, optical_energy_window)
    inputs.ns_BSE = Helpers_setup_Workchain._build_bse_inputs(
        local_folder_gw_reference=arg_inputs['PATH_REFERENCEGW'])

    ####- ------------------------ ------------------------ ------------------------- ------------------------ ------------
    #[2][k-point convergence inputs (stage 1)]###------- ------------ ------------ ------------ ----------------------------
    inputs.ns_kpoints = AttributeDict()
    inputs.ns_kpoints.starting_mesh = KpointsData()
    inputs.ns_kpoints.starting_mesh.set_kpoints_mesh(arg_inputs['KMESH_STARTING'])
    inputs.ns_kpoints.max_mesh = KpointsData()
    inputs.ns_kpoints.max_mesh.set_kpoints_mesh(arg_inputs['KMESH_MAX'])
    inputs.ns_kpoints.convergence_threshold = Float(arg_inputs['KPTSCONV_THRESHOLD'])

    inputs.ns_converge = AttributeDict()
    inputs.ns_converge.dielfunction_window = Float(arg_inputs['DIELFUNCTION_WINDOW'])

    ####- ------------------------ ------------------------ ------------------------- ------------------------ ------------
    #[3][BSE band-subspace (NBANDSV/NBANDSO) convergence inputs (stage 2)]###--- ------------ ------------ -----------------
    # bandsdata: parsed straight from the same reference vasprun used for the BSE/interpolation
    # inputs above, so it requires no extra reference data beyond what stage [1] already needs.
    inputs.ns_nbandsconv = AttributeDict()
    inputs.ns_nbandsconv.bandsdata = Helpers_setup_Workchain._build_bandsdata_from_vasprun(
        local_folder_gw_reference=arg_inputs['PATH_REFERENCEGW'])
    inputs.ns_nbandsconv.threshold_start       = Float(arg_inputs['NBANDSCONV_THRESHOLD_START'])
    inputs.ns_nbandsconv.threshold_max         = Float(arg_inputs['NBANDSCONV_THRESHOLD_MAX'])
    inputs.ns_nbandsconv.threshold_step        = Float(arg_inputs['NBANDSCONV_THRESHOLD_STEP'])
    inputs.ns_nbandsconv.num_bands_included    = Int(arg_inputs['NBANDSCONV_NUM_BANDS_INCLUDED'])

    # Capture the launch-time cwd HERE (in the calling process) rather than letting
    # elaborate_results() call os.getcwd() later - that step may run inside a daemon
    # worker, whose cwd need not match this directory at all (results would silently
    # land somewhere unexpected, e.g. the daemon's home directory).
    inputs.ns_option = AttributeDict()
    inputs.ns_option.copy_result_locally_path = Str(os.getcwd())

    #[4][Submit workchain]###----- ------------ ------------ ------------ ------------ ------------ ------------------------
    print("\nSubmitting VaspmBSECompleteWorkChain...\n")
    print(Helpers_setup_Workchain._build_inputs_summary(inputs))
    run(VaspmBSECompleteWorkChain, **inputs)
