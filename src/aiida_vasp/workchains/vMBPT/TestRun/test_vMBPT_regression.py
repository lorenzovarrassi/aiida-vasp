"""Regression test suite for the vMBPT GW/BSE groundup workchains.

Consolidates the verification done while rewriting `workchain_G0W0_groundup.py`,
`workchain_atomic_BSE.py` and `workchain_atomic_G0W0.py` (dataclass -> AttributeDict
checkpoint-persistence redesign, `optimization`/`npar` namespace fixes, `check_skip`
off-by-one, `VaspAtomicBSEWorkChain` settings-clobber fix) into a pytest-discoverable
suite, so future edits to these files can be checked for regressions instead of
re-verified by hand.

Two tiers:

  * TestOmegatlHandler / TestRestartFolderResolution - fast, mock-node unit tests for
    VaspAtomicG0W0WorkChain's excepted-GW/OMEGATL retry handler (inspect_process()/
    handle_gw_exception()) and for VaspG0W0GroundUpWorkChain's per-phase restart-folder
    resolution. No real VASP, no AiiDA process instantiation - just the real, unmodified
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
from aiida_vasp.workchains.vMBPT.workchain_G0W0_groundup import PhaseStatus, VaspG0W0GroundUpWorkChain  # noqa: E402

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
# Part 1b - restart-folder resolution (fast: mock-node, no VASP, no process instantiation)
# =============================================================================================


class FakeVaspOutputs:
    """Stand-in for a child node's `.outputs` namespace (a real AiiDA NodeLinksManager).

    Supports `in` as well as attribute access, because __get_restart_remote_folder_for_current_phase
    membership-tests `'remote_folder' in node.outputs` before reading it - a node that ran is not
    guaranteed to have attached that output.
    """

    def __init__(self, **outputs):
        self.__dict__.update(outputs)

    def __contains__(self, key):
        return key in self.__dict__


class FakeVaspNode:
    """Stand-in for a submitted child node. Pass outputs explicitly; `FakeVaspNode()` models a node that
    ran but attached none (excepted / killed before the calcjob attached its outputs)."""

    def __init__(self, **outputs):
        self.outputs = FakeVaspOutputs(**outputs)


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


class TestRestartFolderResolution:
    """Regression tests for VaspG0W0GroundUpWorkChain.__get_restart_remote_folder_for_current_phase() - the
    single point resolving "where does the active phase restart from", and the two declarative WorkflowPhase
    fields driving it (predecessor_key / fallback_to_starting_RemoteData), which replaced the old
    per-phase get_restart_folder callable (itself the replacement for restart_folders/capture_outputs/
    seed_restart_if_skipped).

    Resolution walks self.ctx.state.nodes for the named predecessor's last node (via __get_last_node_by_key)
    and reads outputs.remote_folder off it, falling back to self.inputs.ns_reference.starting_RemoteData
    (read directly - not copied into self.ctx.state, see the comment above WorkflowPhase) only if the phase
    permits it. No intermediate captured state at all.

    Exercises the real, unmodified private methods against a duck-typed stub (see _bind_private above) and
    the real WorkflowPhase entries built by _build_phases(). Covers the paths the real-VASP integration
    tests below cannot reach: '1DFTgr' skipped (both integration tests run with run_1DFTgr=True), a
    predecessor node that ran without attaching a remote_folder, and '3G0W0' refusing the fallback.
    """

    @pytest.fixture
    def phases(self):
        return VaspG0W0GroundUpWorkChain._build_phases(None)

    @staticmethod
    def make_stub(nodes, starting_remote_data=None, phase_idx=0):
        """`starting_remote_data=None` models the port not being supplied at all (the AttributeDict key is
        left unset, so reading it raises - exactly like an unset optional AiiDA input port), which is what
        the production try/except is there to absorb."""
        stub = type('Stub', (), {})()
        stub.ctx = AttributeDict()
        stub.ctx.state = AttributeDict()
        stub.ctx.state.nodes = nodes
        stub.ctx.state.phase_idx = phase_idx
        stub.inputs = AttributeDict()
        stub.inputs.ns_reference = AttributeDict()
        if starting_remote_data is not None:
            stub.inputs.ns_reference.starting_RemoteData = starting_remote_data
        stub._build_phases = types.MethodType(VaspG0W0GroundUpWorkChain._build_phases, stub)
        _bind_private(stub, VaspG0W0GroundUpWorkChain, 'get_all_phases', 'get_current_phase',
                       'get_last_node_by_key', 'get_restart_remote_folder_for_current_phase')
        return stub

    @staticmethod
    def resolve(stub):
        return stub._VaspG0W0GroundUpWorkChain__get_restart_remote_folder_for_current_phase()

    def test_phase_dependencies_are_declared_as_data(self, phases):
        """The whole cross-phase dependency graph must be readable off the WorkflowPhase entries alone."""
        assert [(p.key, p.predecessor_key, p.fallback_to_starting_RemoteData) for p in phases] == [
            ('1DFTgr', None, True),
            ('2DFTvo', '1DFTgr', True),
            ('3G0W0', '2DFTvo', False),
        ]

    def test_1DFTgr_resolves_to_starting_remotedata(self):
        stub = self.make_stub(nodes=[[], [], []], starting_remote_data='external_sentinel', phase_idx=0)

        assert self.resolve(stub) == 'external_sentinel'

    def test_1DFTgr_resolves_to_none_when_no_starting_remotedata_supplied(self):
        """The entry-point-from-scratch case both integration tests run: restart_folder ends up None, which
        is legal on the child's non-required port, and '1DFTgr' declares no required_files so validate_step
        never objects."""
        stub = self.make_stub(nodes=[[], [], []], phase_idx=0)

        assert self.resolve(stub) is None

    def test_2DFTvo_pulls_predecessors_remote_folder(self):
        node = FakeVaspNode(remote_folder='rf1', bands='bands1')
        stub = self.make_stub(nodes=[[node], [], []], starting_remote_data='external_sentinel', phase_idx=1)

        assert self.resolve(stub) == 'rf1'

    def test_2DFTvo_falls_back_when_1DFTgr_was_skipped(self):
        """The skip-path this file's real-VASP integration tests can't reach: no '1DFTgr' node at all
        (nodes[0] empty) - '2DFTvo' must fall back to starting_RemoteData."""
        stub = self.make_stub(nodes=[[], [], []], starting_remote_data='external_sentinel', phase_idx=1)

        assert self.resolve(stub) == 'external_sentinel'

    def test_2DFTvo_falls_back_when_predecessor_attached_no_remote_folder(self):
        """A node that ran is not proof of a remote_folder output (excepted / killed mid-flight). That must
        degrade to the fallback, not raise on an unguarded attribute access."""
        stub = self.make_stub(nodes=[[FakeVaspNode()], [], []], starting_remote_data='external_sentinel',
                              phase_idx=1)

        assert self.resolve(stub) == 'external_sentinel'

    def test_3G0W0_pulls_predecessors_remote_folder(self):
        node = FakeVaspNode(remote_folder='rf2', bands='bands2')
        stub = self.make_stub(nodes=[[], [node], []], starting_remote_data='external_sentinel', phase_idx=2)

        assert self.resolve(stub) == 'rf2'

    def test_3G0W0_never_falls_back_to_starting_remotedata(self):
        """fallback_to_starting_RemoteData=False: a DFT ground-state folder has no WAVEDER, so substituting
        it would turn a clear "2DFTvo produced nothing" into a confusing downstream VASP failure. Resolving
        to None instead is what makes validate_step report NO_STARTING_WAVECAR_WAVEDER_forG0W0."""
        for nodes in ([[], [], []], [[], [FakeVaspNode()], []]):
            stub = self.make_stub(nodes=nodes, starting_remote_data='external_sentinel', phase_idx=2)

            assert self.resolve(stub) is None

    def test_prepare_step_hands_the_resolved_folder_to_build_inputs(self):
        """The engine wiring: build_inputs no longer looks its own restart folder up - prepare_step resolves
        it once and passes it in. Verified by making _prepare_inputs_DFT a spy over the real prepare_step."""
        node = FakeVaspNode(remote_folder='rf1', bands='bands1')
        stub = self.make_stub(nodes=[[node], [], []], phase_idx=1)
        stub.ctx.state.terminal = None
        stub.ctx.state.inputs_finalized = None
        stub.ctx.state.phase_statuses = [PhaseStatus.COMPLETED, PhaseStatus.PENDING, PhaseStatus.NOT_STARTED]
        _bind_private(stub, VaspG0W0GroundUpWorkChain,
                       'get_status_current_phase', 'set_status_current_phase')
        recorded = {}

        def spy_prepare_inputs_DFT(restart_folder, calc_type):
            recorded.update(restart_folder=restart_folder, calc_type=calc_type)
            return 'INPUTS_SENTINEL'

        stub._prepare_inputs_DFT = spy_prepare_inputs_DFT

        VaspG0W0GroundUpWorkChain.prepare_step(stub)

        assert recorded == {'restart_folder': 'rf1', 'calc_type': '2DFTvo'}
        assert stub.ctx.state.inputs_finalized == 'INPUTS_SENTINEL'
        assert stub.ctx.state.phase_statuses[1] is PhaseStatus.PREPAREDINPUTS

    def test_validate_remote_has_required_files_reports_and_fails_on_none(self):
        """The "throw error if nothing is available" half of the design: resolution returning None (no
        predecessor node AND no permitted/available fallback) already gets turned into a clean, reported
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
