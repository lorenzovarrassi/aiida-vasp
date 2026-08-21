"""
Golden-file regression harness for VaspDFTGWWorkChain's FSM *control flow*
(workchain_G0W0_base.py) - `initialize`/`update_state`/`validate_step`/
`prepare_step`/`execute_step`.

Purpose
-------
Unlike `harness_vMBPT_inputs.py` (pure input-dict-building functions), this
harness exercises actual *control flow*: which calc_type gets submitted in
which order, how ctx state evolves, and how restart_folders chain from one
phase's output into the next phase's input. This is the safety net for the
FSM genericization of `VaspDFTGWWorkChain` (see handoff.md's "Phase 3
detailed design" section).

Note on what "identical" means across the refactor: the genericization
deliberately replaces the internal state representation (the old flat
`state_execution_enum` with one member per (phase, status) pair, e.g.
`DFTGR_PENDING`/`DFTGR_RUNNING`/`DFTGR_DONE`, becomes `ctx.phase_idx` +
`ctx.phase_status` (generic `PENDING`/`RUNNING`) + `ctx.terminal`
(`COMPLETE`/`FAILED`/`RECOVERY`)). So the raw internal state names are
*expected* to differ before/after and are not byte-diffed; the invariant
contract this harness checks is the **observable** behaviour: the ordered
list of `submissions` (calc_type, class submitted, restart_folder passed
in) and the final outcome (`final_state`). Those must match exactly.

Mirrors `harness_vMBPT_inputs.py`'s approach: duck-typed stand-in for `self`,
real (unmangled) FSM methods called directly, no daemon/no submission/no
database writes. `self.submit(...)` is stubbed to record the call and return
an already-"finished" fake node (submission is synchronous/instantaneous in
this harness, unlike the real daemon) - this sidesteps needing mock-vasp, a
test AiiDA profile, or real `exposed_inputs()` resolution entirely, since we
only care about calc_type ordering and restart_folder chaining, not exact
scientific input values (that would be `harness_vMBPT_inputs.py`'s job for
its own class, if this class ever gets a pure-function input harness too).

Usage
-----
    source ~/venv_AiiDA_202608_refactor/bin/activate
    python regression_harness/harness_G0W0_base_fsm.py \\
        regression_harness/golden/<label>.json

Requires the real `lvarras_aiida` AiiDA profile to be loadable (module-level
`load_profile()`, needed because the target module imports real `aiida.orm`
node classes) but performs no `.store()` calls anywhere - nothing is written
to the database, and nothing is ever actually submitted to a scheduler.
"""
import json
import os.path
import sys

from aiida import load_profile
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import ToContext

load_profile()

from aiida_vasp.workchains.vMBPT.workchain_G0W0_base import (  # noqa: E402
    VaspDFTGWWorkChain,
)

CLS = VaspDFTGWWorkChain
MANGLE = "_VaspDFTGWWorkChain__"


class _ExitCodes:
    """Duck stand-in for `self.exit_codes` - returns a sentinel string per name."""

    def __getattr__(self, name):
        return f"<exit_code {name}>"


class _FakeRemoteFolder:
    """Stand-in for a RemoteData restart folder - supports .listdir() for validate_step."""

    def __init__(self, path, files):
        self._path = path
        self._files = set(files)

    def listdir(self):
        return self._files

    def __repr__(self):
        return f"<FakeRemoteFolder {self._path}>"


class _FakeNode:
    """Stand-in for a finished child WorkChainNode."""

    _pk_counter = [0]

    def __init__(self, calc_type, is_ok=True):
        _FakeNode._pk_counter[0] += 1
        self.pk = _FakeNode._pk_counter[0]
        self.is_finished = True
        self.is_finished_ok = is_ok

        class _Outputs:
            pass

        self.outputs = _Outputs()
        self.outputs.remote_folder = _FakeRemoteFolder(
            f"/remote/scratch/{calc_type}_pk{self.pk}", files=("WAVECAR", "WAVEDER", "CHGCAR")
        )


class _Harness:
    """Duck-typed stand-in for `self` - NOT a real WorkChain instance.

    Carries the real, unmangled FSM methods (`initialize`, `should_wc_continue`,
    `update_state`, `validate_step`, `prepare_step`, `correct_previous_errors`,
    `execute_step`), bound the same way `harness_vMBPT_inputs.py` binds its
    target methods - plus stand-ins for everything they touch that would
    otherwise require a real Process/submission/profile round-trip.
    """

    _PHASES = CLS._PHASES
    initialize = CLS.initialize
    should_wc_continue = CLS.should_wc_continue
    _advance_to_first_runnable_phase = CLS._advance_to_first_runnable_phase
    update_state = CLS.update_state
    validate_step = CLS.validate_step
    prepare_step = CLS.prepare_step
    correct_previous_errors = CLS.correct_previous_errors
    execute_step = CLS.execute_step
    _prepare_inputs_DFT = CLS._prepare_inputs_DFT
    _prepare_inputs_G0W0 = CLS._prepare_inputs_G0W0

    def __init__(self, fail_plan=None):
        self.ctx = AttributeDict()
        self.exit_codes = _ExitCodes()
        self.submissions = []  # captured (attempt_idx, calc_type, class_name, restart_folder_repr)
        # {calc_type: number_of_leading_attempts_that_should_fail}, e.g. {'2DFTvo': 1}
        # fails the FIRST submission of 2DFTvo, succeeds on the retry.
        self.fail_plan = fail_plan or {}

    def exposed_inputs(self, _proc_cls):
        return AttributeDict()

    def report(self, _msg):
        pass

    def submit(self, process_class, **inputs):
        # Determine which calc_type this is from ctx.phase_idx (set just
        # before this call, inside the real execute_step) - recover it the
        # same way execute_step itself does, so this stand-in stays decoupled
        # from execute_step's internals rather than requiring calc_type as an
        # explicit arg.
        calc_type = self._PHASES[self.ctx.phase_idx].key
        attempt_num_before_this_one = self.ctx.state_WC.retries[calc_type]
        is_ok = attempt_num_before_this_one >= self.fail_plan.get(calc_type, 0)
        self.submissions.append(
            {
                "attempt_idx": len(self.submissions),
                "calc_type": calc_type,
                "class_name": getattr(process_class, "__name__", repr(process_class)),
                "restart_folder": repr(inputs.get("restart_folder")),
                "is_ok": is_ok,
            }
        )
        return _FakeNode(calc_type, is_ok=is_ok)


setattr(_Harness, MANGLE + "last_wc_node", staticmethod(CLS._VaspDFTGWWorkChain__last_wc_node))
setattr(_Harness, MANGLE + "handle_failure_with_retry", CLS._VaspDFTGWWorkChain__handle_failure_with_retry)
setattr(_Harness, MANGLE + "validate_remote_has_required_files", CLS._VaspDFTGWWorkChain__validate_remote_has_required_files)
setattr(_Harness, MANGLE + "report_compact_submission", lambda self, *a, **k: None)


def build_fixture(run_1DFTgr: bool, starting_remote=None, maximum_iterations=1, fail_plan=None):
    """One fixed, representative set of synthetic inputs."""
    structure = orm.StructureData(cell=[[4.5, 0, 0], [0, 4.5, 0], [0, 0, 4.5]])
    structure.append_atom(position=(0, 0, 0), symbols="Ga")
    structure.append_atom(position=(2.25, 2.25, 2.25), symbols="N")

    fake_self = _Harness(fail_plan=fail_plan)
    ns_reference = AttributeDict()
    if starting_remote is not None:
        ns_reference.starting_RemoteData = starting_remote

    fake_self.inputs = AttributeDict(
        {
            "structure": structure,
            "ns_parameters": AttributeDict(
                {
                    "encut": orm.Float(400.0),
                    "nbands": orm.Int(64),
                    "nomega": orm.Int(200),
                }
            ),
            "ns_optimization": AttributeDict(
                {
                    "kpar": orm.Int(4),
                    "npar": orm.Int(1),
                    "lreal": orm.Bool(False),
                }
            ),
            "ns_reference": ns_reference,
            "ns_option": AttributeDict(
                {
                    "maximum_iterations": orm.Int(maximum_iterations),
                    "run_1DFTgr": orm.Bool(run_1DFTgr),
                    "run_2DFTvo_3G0W0": orm.Bool(True),
                    "calculation_label": orm.Str("golden-fsm-fixture"),
                }
            ),
        }
    )
    return fake_self


def _state_snapshot(fake_self):
    """Normalized (phase_key, status) snapshot - not a raw enum name, since the
    genericization deliberately changed the internal representation. Only
    `submissions` and `final_state` are treated as the cross-refactor
    invariant contract (see module docstring); this snapshot is kept in the
    output for human debugging only."""
    if fake_self.ctx.terminal is not None:
        return fake_self.ctx.terminal.name
    if fake_self.ctx.phase_idx == -1:
        return "INIT"
    phase = fake_self._PHASES[fake_self.ctx.phase_idx]
    return f"{phase.key}_{fake_self.ctx.phase_status.name}"


def run_one_fsm(run_1DFTgr: bool, starting_remote=None, safety_cap=20, maximum_iterations=1, fail_plan=None):
    fake_self = build_fixture(
        run_1DFTgr=run_1DFTgr,
        starting_remote=starting_remote,
        maximum_iterations=maximum_iterations,
        fail_plan=fail_plan,
    )
    fake_self.initialize()

    state_trace = [_state_snapshot(fake_self)]
    exit_code_returned = None
    attempts = 0
    while fake_self.should_wc_continue() and attempts < safety_cap:
        # Mirror real AiiDA: a step returning a non-None exit code halts the
        # workchain immediately, without running the remaining steps in this
        # outline iteration (matters for the FAILED-path scenarios below,
        # where validate_step/update_state return an exit code and
        # prepare_step/execute_step must NOT then also run).
        for step_fn in (fake_self.update_state, fake_self.validate_step, fake_self.prepare_step,
                        fake_self.correct_previous_errors, fake_self.execute_step):
            result = step_fn()
            if result is not None and not isinstance(result, ToContext):
                exit_code_returned = repr(result)
                break
        state_trace.append(_state_snapshot(fake_self))
        attempts += 1
        if exit_code_returned is not None:
            break

    return {
        "final_state": fake_self.ctx.terminal.name if fake_self.ctx.terminal is not None else None,
        "exit_code_returned": exit_code_returned,
        "state_trace": state_trace,
        "submissions": fake_self.submissions,
    }


def run_harness():
    return {
        "scenario_full_chain_from_scratch": run_one_fsm(run_1DFTgr=True),
        "scenario_skip_dftgr_with_starting_remote": run_one_fsm(
            run_1DFTgr=False,
            starting_remote=_FakeRemoteFolder("/remote/external/starting", files=("WAVECAR", "CHGCAR")),
        ),
        "scenario_no_starting_data_fails_immediately": run_one_fsm(
            run_1DFTgr=False,
            starting_remote=None,
        ),
        "scenario_retry_then_succeed": run_one_fsm(
            run_1DFTgr=True,
            maximum_iterations=2,
            fail_plan={"2DFTvo": 1},  # first 2DFTvo attempt fails, retry succeeds
        ),
        "scenario_retries_exhausted": run_one_fsm(
            run_1DFTgr=True,
            maximum_iterations=1,
            fail_plan={"2DFTvo": 5},  # always fails, only 1 attempt allowed -> FAILED
        ),
    }


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_json_path>", file=sys.stderr)
        sys.exit(1)
    out_path = sys.argv[1]
    result = run_harness()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fhandle:
        json.dump(result, fhandle, indent=2, sort_keys=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
