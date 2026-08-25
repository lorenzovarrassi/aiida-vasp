"""Regression test suite for the vMBPT GW/BSE groundup workchains.

Consolidates the verification done while rewriting `workchain_G0W0_groundup.py`,
`workchain_atomic_BSE.py` and `workchain_atomic_G0W0.py` (dataclass -> AttributeDict
checkpoint-persistence redesign, `optimization`/`npar` namespace fixes, `check_skip`
off-by-one, `VaspAtomicBSEWorkChain` settings-clobber fix) into a pytest-discoverable
suite, so future edits to these files can be checked for regressions instead of
re-verified by hand.

Two tiers:

  * TestOmegatlHandler - fast, mock-node unit tests for VaspAtomicG0W0WorkChain's
    excepted-GW/OMEGATL retry handler (inspect_process()/handle_gw_exception()). No
    real VASP, no AiiDA process instantiation - just the real, unmodified handler
    methods driven against a duck-typed stand-in for `self` (same pattern as this
    codebase's own regression_harness/). Runs anywhere a profile can be loaded.

  * test_g0w0_groundup_regression / test_bse_groundup_regression (marked
    `integration`) - real end-to-end smoke tests on vasp@localhost (BP, 1 CPU,
    minimal settings) exercising the full DFTgr -> DFTvo -> G0W0[-> BSE] pipeline,
    including checkpoint persistence. Baseline gap values below were captured from
    run_G0W0_v5_persist.log (pk=91366) and run_BSE_v2_persist.log (pk=91449), the
    last known-good runs after all fixes in this session. Skipped automatically if
    vasp@localhost / the PBE.54 POTCAR family aren't available in the loaded profile,
    or if no AiiDA profile can be loaded at all.

Run with:
    source ~/venv_AiiDA_202608_refactor/bin/activate
    pytest src/aiida_vasp/workchains/vMBPT/TestRun/test_vMBPT_regression.py -v
    pytest src/aiida_vasp/workchains/vMBPT/TestRun/test_vMBPT_regression.py -v -m "not integration"
"""
import logging
import os
import types

import pytest

pytest.importorskip('aiida')

from aiida import load_profile  # noqa: E402

try:
    load_profile()
except Exception as exc:  # noqa - broad: no loadable AiiDA profile means nothing below can run
    pytest.skip(f'no AiiDA profile available ({exc}); skipping vMBPT regression suite', allow_module_level=True)

from aiida.common.extendeddicts import AttributeDict  # noqa: E402
from aiida.engine import run_get_node  # noqa: E402
from aiida.orm import Bool, Int, KpointsData, Str  # noqa: E402
from aiida.orm.nodes.data import structure  # noqa: E402
from aiida.plugins import WorkflowFactory  # noqa: E402
import pymatgen.core.structure as pcs  # noqa: E402

from aiida_vasp.workchains.vMBPT.utils_helpers_setupworkchain import Helpers_setup_Workchain  # noqa: E402
from aiida_vasp.workchains.vMBPT.workchain_atomic_G0W0 import VaspAtomicG0W0WorkChain  # noqa: E402
from aiida_vasp.workchains.vMBPT.workchain_G0W0_groundup import VaspG0W0GroundUpWorkChain  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================================
# Part 1 - OMEGATL excepted-retry handler (fast: mock-node, no VASP, no process instantiation)
# =============================================================================================


class FakeNode:
    """Stand-in for a finished/excepted child CalcJobNode - only the attributes the handler reads."""

    def __init__(self, pk, is_excepted=True, is_finished=True, is_finished_ok=False):
        self.pk = pk
        self.is_excepted = is_excepted
        self.is_finished = is_finished
        self.is_finished_ok = is_finished_ok


def make_stub():
    """Duck-typed stand-in for `self`, following this codebase's own regression_harness/ convention.

    `handle_gw_exception` is `@process_handler`-decorated; that wrapper's own bookkeeping needs a
    real `instance.node.base.extras` (a genuinely-stored AiiDA process node) to log which handlers
    ran - not available on a duck-typed stub, and explicitly documented as cosmetic in this
    codebase ("kept for introspection/`verdi process report` only" - not part of the retry logic
    itself). So call the real, unmodified, UNDECORATED function directly via `.__wrapped__` (set
    by the `decorator` package) - this exercises the exact same handler body `inspect_process()`
    would run, just without the introspection-only bookkeeping wrapper around it.
    """
    stub = type('Stub', (), {})()
    stub.ctx = AttributeDict()
    stub.ctx.inputs = AttributeDict()
    stub.ctx.inputs.parameters = {'algo': 'EVGW0', 'nelm': 1, 'nomega': 200}
    stub.ctx.iteration = 1
    stub.ctx.children = []
    stub.messages = []
    stub.report = lambda msg: stub.messages.append(msg)
    stub.exit_codes = VaspAtomicG0W0WorkChain.spec().exit_codes
    stub.handle_gw_exception = lambda node: VaspAtomicG0W0WorkChain.handle_gw_exception.__wrapped__(stub, node)
    return stub


@pytest.fixture
def stub():
    return make_stub()


class TestOmegatlHandler:
    """Regression tests for VaspAtomicG0W0WorkChain.inspect_process() / handle_gw_exception()."""

    def test_first_excepted_node_escalates_omegatl_and_retries(self, stub):
        node = FakeNode(pk=1001)
        stub.ctx.children = [node]
        stub.ctx.iteration = 1

        result = VaspAtomicG0W0WorkChain.inspect_process(stub)

        assert stub.ctx.inputs.parameters.get('omegatl') == 16000
        assert stub.ctx.get('gw_excepted_retry_launched', False) is True
        assert stub.ctx.get('gw_iteration') == 1
        assert result is not None and result.status == 0, 'inspect_process must signal restart (status 0)'

    def test_second_excepted_node_aborts_without_retrying(self, stub):
        node1 = FakeNode(pk=1001)
        stub.ctx.children = [node1]
        stub.ctx.iteration = 1
        VaspAtomicG0W0WorkChain.inspect_process(stub)  # first except: consumes the one allowed retry

        node2 = FakeNode(pk=1002)
        stub.ctx.children = [node1, node2]
        stub.ctx.iteration = 2
        stub.messages = []
        parameters_before = dict(stub.ctx.inputs.parameters)

        result = VaspAtomicG0W0WorkChain.inspect_process(stub)

        assert result == stub.exit_codes.ERROR_EXCEPTED_RETRY_FAILED
        assert stub.ctx.inputs.parameters == parameters_before, 'no infinite-retry / double OMEGATL escalation'

    def test_non_excepted_node_never_calls_handler(self, stub):
        def poison_pill(node):
            raise AssertionError('handle_gw_exception must not be called for a non-excepted node')

        stub.handle_gw_exception = poison_pill
        node = FakeNode(pk=1003, is_excepted=False, is_finished=False)
        stub.ctx.children = [node]
        stub.ctx.iteration = 1

        try:
            VaspAtomicG0W0WorkChain.inspect_process(stub)
        except AssertionError:
            pytest.fail('handle_gw_exception was called for a non-excepted node')
        except Exception:
            # super().inspect_process() needs real BaseRestartWorkChain machinery the stub lacks -
            # falling through to that (and failing there) is expected and is exactly what proves
            # the non-excepted path never reached handle_gw_exception in the first place.
            pass


# =============================================================================================
# Part 1b - restart-folder pull (fast: mock-node, no VASP, no process instantiation)
# =============================================================================================


class FakeVaspOutputs:
    """Stand-in for a finished VaspWorkChain node's `.outputs` namespace - just remote_folder/bands."""

    def __init__(self, remote_folder, bands):
        self.remote_folder = remote_folder
        self.bands = bands


class FakeVaspNode:
    def __init__(self, remote_folder='remote_folder_sentinel', bands='bands_sentinel'):
        self.outputs = FakeVaspOutputs(remote_folder, bands)


def _bind_private(stub, cls, *names):
    """Bind cls's name-mangled private methods onto `stub` so the real, unmodified methods (written as
    `self.__foo()` inside cls's own body, which Python mangles to `self._ClassName__foo` at compile time
    regardless of what `self` actually is at runtime) can be exercised against a duck-typed stub - the
    same "call the real code, not a reimplementation" convention as TestOmegatlHandler's `.__wrapped__`
    trick above, generalized to a chain of private helpers calling each other."""
    prefix = f'_{cls.__name__}__'
    for name in names:
        attr = prefix + name
        setattr(stub, attr, types.MethodType(getattr(cls, attr), stub))


class TestRestartFolderPull:
    """Regression tests for VaspG0W0GroundUpWorkChain's __restart_remote_folder/get_restart_folder - the
    predecessor-pull that replaced the old restart_folders/capture_outputs/seed_restart_if_skipped
    mechanism: a phase's get_restart_folder walks self.ctx.state.nodes for its named predecessor's last
    finished node and reads outputs.remote_folder straight off it (via __get_last_node_by_key), falling
    back to self.inputs.ns_reference.starting_RemoteData (read directly - not copied into self.ctx.state,
    see the comment above WorkflowPhase) if there's no predecessor or it was skipped - no intermediate
    captured state at all. Exercises the real, unmodified private methods against a duck-typed stub (see
    _bind_private above) and the real WorkflowPhase entries built by _build_phases(). Covers the one path
    the real-VASP integration tests below don't reach: '1DFTgr' skipped (both integration tests run with
    run_1DFTgr=True), where '2DFTvo' must fall back to starting_RemoteData.
    """

    @pytest.fixture
    def phases(self):
        return VaspG0W0GroundUpWorkChain._build_phases(None)

    @staticmethod
    def make_stub(nodes, starting_remote_data=None):
        stub = type('Stub', (), {})()
        stub.ctx = AttributeDict()
        stub.ctx.state = AttributeDict()
        stub.ctx.state.nodes = nodes
        stub.inputs = AttributeDict()
        stub.inputs.ns_reference = AttributeDict()
        stub.inputs.ns_reference.starting_RemoteData = starting_remote_data
        stub._build_phases = types.MethodType(VaspG0W0GroundUpWorkChain._build_phases, stub)
        _bind_private(stub, VaspG0W0GroundUpWorkChain,
                       'get_all_phases', 'get_last_node_by_key', 'restart_remote_folder')
        return stub

    def test_1DFTgr_get_restart_folder_reads_starting_remotedata(self, phases):
        phase = phases[0]
        assert phase.key == '1DFTgr'
        stub = self.make_stub(nodes=[[], [], []], starting_remote_data='external_sentinel')

        assert phase.get_restart_folder(stub) == 'external_sentinel'

    def test_2DFTvo_get_restart_folder_pulls_predecessors_node_remote_folder(self, phases):
        phase = phases[1]
        assert phase.key == '2DFTvo'
        node = FakeVaspNode(remote_folder='rf1', bands='bands1')
        stub = self.make_stub(nodes=[[node], [], []], starting_remote_data='external_sentinel')

        assert phase.get_restart_folder(stub) == 'rf1'

    def test_2DFTvo_get_restart_folder_falls_back_when_1DFTgr_was_skipped(self, phases):
        """The skip-path this file's real-VASP integration tests can't reach: no '1DFTgr' node at all
        (nodes[0] empty) - '2DFTvo' must fall back to starting_RemoteData."""
        phase = phases[1]
        stub = self.make_stub(nodes=[[], [], []], starting_remote_data='external_sentinel')

        assert phase.get_restart_folder(stub) == 'external_sentinel'

    def test_3G0W0_get_restart_folder_pulls_predecessors_node_remote_folder(self, phases):
        phase = phases[2]
        assert phase.key == '3G0W0'
        node = FakeVaspNode(remote_folder='rf2', bands='bands2')
        stub = self.make_stub(nodes=[[], [node], []], starting_remote_data='external_sentinel')

        assert phase.get_restart_folder(stub) == 'rf2'

    def test_build_inputs_delegates_to_get_restart_folder(self, phases):
        """build_inputs must ask get_restart_folder for the value rather than recomputing it - verified
        by making _prepare_inputs_DFT a spy that just records what it was called with."""
        phase = phases[1]  # '2DFTvo'
        node = FakeVaspNode(remote_folder='rf1', bands='bands1')
        stub = self.make_stub(nodes=[[node], [], []], starting_remote_data=None)
        stub._VaspG0W0GroundUpWorkChain__get_current_phase = types.MethodType(lambda self: phase, stub)
        recorded = {}
        stub._prepare_inputs_DFT = lambda restart_folder, calc_type: recorded.update(
            restart_folder=restart_folder, calc_type=calc_type)

        phase.build_inputs(stub)

        assert recorded == {'restart_folder': 'rf1', 'calc_type': '2DFTvo'}

    def test_validate_remote_has_required_files_reports_and_fails_on_none(self):
        """The "throw error if neither is available" half of the design: get_restart_folder returning
        None (no predecessor node AND no starting_RemoteData) already gets turned into a clean, reported
        failure here - validate_step then returns the phase's documented missing-files exit code, no new
        error-handling code needed anywhere for this."""
        stub = type('Stub', (), {})()
        stub.messages = []
        stub.report = lambda msg: stub.messages.append(msg)

        ok = VaspG0W0GroundUpWorkChain._VaspG0W0GroundUpWorkChain__validate_remote_has_required_files(
            stub, remote=None, required=['WAVECAR'], label='2DFTvo restart')

        assert ok is False
        assert any('restart folder is None' in msg for msg in stub.messages)


# =============================================================================================
# Part 2 - real end-to-end smoke tests on vasp@localhost (slow, marked `integration`)
# =============================================================================================

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

GAP_TOLERANCE_EV = 0.05  # loose sanity bound, not a bit-exact golden-value match


def _build_base_inputs(calculation_label):
    pymatgen_structure = pcs.Structure.from_file(os.path.join(HERE, 'POSCAR'))

    inputs = AttributeDict()
    inputs.structure = structure.StructureData(pymatgen_structure=pymatgen_structure)

    inputs.potential_family, inputs.potential_mapping = Helpers_setup_Workchain._build_potential_mapping(
        potential_family='PBE.54', pymatgen_structure=pymatgen_structure, flag_prefer_GW=True)

    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([1, 1, 1])
    inputs.kpoints = kpoints

    inputs.code, _computer_label, inputs.options = Helpers_setup_Workchain._build_slurm_options(
        SETUP_CLUSTER, SETUP_SLURMCALC)

    inputs.nbands = Int(12)
    inputs.nomega = Int(24)

    inputs.optimization = AttributeDict()
    inputs.optimization.kpar = Int(1)
    inputs.optimization.npar = Int(1)
    inputs.optimization.lreal = Bool(False)

    inputs.ns_option = AttributeDict()
    inputs.ns_option.run_1DFTgr = Bool(True)
    inputs.ns_option.run_2DFTvo_3G0W0 = Bool(True)
    inputs.ns_option.calculation_label = Str(calculation_label)

    return inputs


def _base_inputs_or_skip(calculation_label):
    try:
        return _build_base_inputs(calculation_label)
    except Exception as exc:  # noqa - broad: any failure here means the environment isn't ready
        pytest.skip(f'vasp@localhost / PBE.54 POTCAR family not available in this profile: {exc}')


def _assert_no_checkpoint_failures(caplog):
    assert 'Exception trying to save checkpoint' not in caplog.text, (
        'checkpoint persistence failed during the run - see the dataclass -> AttributeDict '
        'ctx.state redesign in workchain_G0W0_groundup.py')


@pytest.mark.integration
def test_g0w0_groundup_regression(caplog):
    """DFTgr -> DFTvo -> G0W0 on vasp@localhost. Baseline: run_G0W0_v5_persist.log (pk=91366)."""
    VaspG0W0GroundUpWorkChain = WorkflowFactory('vasp.gw.g0w0_groundup')
    inputs = _base_inputs_or_skip('regtest_g0w0')

    with caplog.at_level(logging.WARNING):
        results, node = run_get_node(VaspG0W0GroundUpWorkChain, **inputs)

    assert node.is_finished_ok, f'exit_status={node.exit_status} exit_message={node.exit_message}'
    _assert_no_checkpoint_failures(caplog)

    gaps = results['gaps'].get_dict()
    assert gaps['G0W0']['spinUp']['Dir'] == pytest.approx(3.5378, abs=GAP_TOLERANCE_EV)
    assert gaps['DFT']['spinUp']['Dir'] == pytest.approx(2.9327, abs=GAP_TOLERANCE_EV)
    assert 'gaps_QPc' in results
    assert results['gaps_QPc'].get_dict()['spinUp']['Dir'] == pytest.approx(0.6051, abs=GAP_TOLERANCE_EV)


@pytest.mark.integration
def test_bse_groundup_regression(caplog):
    """DFTgr -> DFTvo -> G0W0 -> BSE on vasp@localhost. Baseline: run_BSE_v2_persist.log (pk=91449)."""
    VaspBSEGroundUpWorkChain = WorkflowFactory('vasp.gw.bse_groundup')
    inputs = _base_inputs_or_skip('regtest_bse')
    inputs.bse = AttributeDict()
    inputs.bse.optical = AttributeDict()
    inputs.bse.optical.algo = Str('BSE')
    inputs.bse.optical.nbandso = Int(2)
    inputs.bse.optical.nbandsv = Int(2)
    inputs.bse.optical.nbseeig = Int(4)
    inputs.bse.optimization = AttributeDict()
    inputs.bse.optimization.kpar = Int(1)

    with caplog.at_level(logging.WARNING):
        results, node = run_get_node(VaspBSEGroundUpWorkChain, **inputs)

    assert node.is_finished_ok, f'exit_status={node.exit_status} exit_message={node.exit_message}'
    _assert_no_checkpoint_failures(caplog)

    gaps = results['gaps'].get_dict()
    assert gaps['G0W0']['spinUp']['Dir'] == pytest.approx(3.5378, abs=GAP_TOLERANCE_EV)
    assert gaps['DFT']['spinUp']['Dir'] == pytest.approx(2.9327, abs=GAP_TOLERANCE_EV)

    bse_outputs = sorted(k for k in results.keys() if k.startswith('bse'))
    assert bse_outputs, 'expected at least one bse.* output namespace'
