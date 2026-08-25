"""Smoke test for VaspG0W0GroundUpWorkChain on localhost (real vasp_std, 1 CPU, minimal settings).

Goal: exercise the real code path (DFTgr -> DFTvo -> G0W0) end to end to catch bugs in the
rewritten groundup workchain, not to obtain physically converged results.
"""
import os
import sys
from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Bool, Int, Str, KpointsData
from aiida.engine import run_get_node
from aiida import load_profile
import pymatgen.core.structure as pcs
from aiida.orm.nodes.data import structure

load_profile()

from aiida.plugins import WorkflowFactory
from aiida_vasp.workchains.vMBPT.utils_helpers_setupworkchain import Helpers_setup_Workchain

VaspG0W0GroundUpWorkChain = WorkflowFactory('vasp.gw.g0w0_groundup')

HERE = os.path.dirname(os.path.abspath(__file__))

SETUP_CLUSTER = {'cpu': {
    'code': 'vasp@localhost',
    'account': 'local',
    'qos': '',
    'partition': '',
    'max_mem_GB_per_node_available': 20,
}}
SETUP_SLURMCALC = {
    'num_nodes': 1,
    'ntasks-per-node': 1,
    'use_gpu': False,
    'mem_GB_per_task': 4,
    'time_in_h': 2,
}

if __name__ == '__main__':
    pymatgen_structure = pcs.Structure.from_file(os.path.join(HERE, 'POSCAR'))

    inputs = AttributeDict()
    inputs.structure = structure.StructureData(pymatgen_structure=pymatgen_structure)

    inputs.potential_family, inputs.potential_mapping = Helpers_setup_Workchain._build_potential_mapping(
        potential_family='PBE.54', pymatgen_structure=pymatgen_structure, flag_prefer_GW=True)

    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([1, 1, 1])
    inputs.kpoints = kpoints

    inputs.code, computer_label, inputs.options = Helpers_setup_Workchain._build_slurm_options(
        SETUP_CLUSTER, SETUP_SLURMCALC)
    print(f'Using code={inputs.code}, computer_label={computer_label}')

    inputs.nbands = Int(12)
    inputs.nomega = Int(24)

    inputs.optimization = AttributeDict()
    inputs.optimization.kpar = Int(1)
    inputs.optimization.npar = Int(1)
    inputs.optimization.lreal = Bool(False)

    inputs.ns_option = AttributeDict()
    inputs.ns_option.run_1DFTgr = Bool(True)
    inputs.ns_option.run_2DFTvo_3G0W0 = Bool(True)
    inputs.ns_option.calculation_label = Str('BP_smoketest')

    print('Submitting VaspG0W0GroundUpWorkChain (blocking run_get_node)...')
    results, node = run_get_node(VaspG0W0GroundUpWorkChain, **inputs)
    print(f'Finished: pk={node.pk} exit_status={node.exit_status} is_finished_ok={node.is_finished_ok}')
    if not node.is_finished_ok:
        print(f'exit_message={node.exit_message}')
        sys.exit(1)
    print('gaps =', results['gaps'].get_dict())
    if 'gaps_QPc' in results:
        print('gaps_QPc =', results['gaps_QPc'].get_dict())
