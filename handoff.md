# Handoff: splitting vMBPT out of aiida-vasp

Branch: `refactor/split-vmbpt` (worktree at `../aiida-vasp-dev-refactor`, off
`support_GWBSE_rebased`). Plan file (for full step-by-step detail):
`~/.claude/plans/this-aiida-vasp-plugin-has-abstract-platypus.md`.

## Goal

`aiida-vasp-dev/src/aiida_vasp/workchains/vMBPT/` currently mixes three
unrelated concerns with no AiiDA entry points enforcing any boundary between
them:
- **(A) base G0W0/BSE workchains** - stays in `aiida-vasp`.
- **(B) convergence & basis-set extrapolation workchains** - moves to a new
  plugin, `aiida-vasp-gwconv`.
- **(C) QP-correction of a WAVECAR via interpolation** - moves to a new
  plugin, `aiida-vasp-qpcorrection`, and is modernized from a shell-script
  `prepend_text` hack (zero AiiDA provenance) into a real
  `WavecarQPModificationCalculation` CalcJob, with a parallel path to swap in
  a GP/ML prediction of QP corrections instead of deterministic
  interpolation (`GPModelData` + `predict_qp_corrections` calcfunction).

This is a large, multi-session change. The overriding constraint is
minimizing regression risk over speed - see "Regression-safety mechanism"
below, used throughout every phase.

## Architecture end-state

- **aiida-vasp** (this repo, branch `refactor/split-vmbpt`): bucket-A
  workchains (`VaspDFTGWWorkChain`, `VaspGWWorkChain`,
  `VaspInitScriptWorkChain`) plus the "common" helpers used by all three
  buckets (`utils_helpers_mBSE.py`, `utils_helpers_setupworkchain.py`,
  `input_magnetic_moment_tomagmom` from `utils_helpers_extrapolation.py`).
  `VaspmBSEInitScriptWorkChain` is slimmed to only consume a `restart_folder`
  RemoteData - it no longer builds interpolation `prepend_text` itself.
  **Planned (Phase 3, not yet built)**: `VaspDFTGWWorkChain`'s internal FSM
  gets genericized (phase-list-driven instead of hardcoded per named phase)
  so `aiida-vasp-qpcorrection`'s workchains can subclass it additively - see
  "Phase 3 detailed design" below for the full rationale and design.
- **aiida-vasp-gwconv** *(done, Phase 2)*: convergence templates + master
  orchestrators (`VaspmBSEConvergenceTemplateWorkChain` + 2 children,
  `VaspmBSECompleteWorkChain`, `VaspG0W0KptsConvWorkChain`,
  `VaspG0W0BasisExtrWorkChain`, `VaspG0W0CompleteWorkChain`) + the
  extrapolation calcfunctions + `wkc_Wannier` (Wannierization, fixed+kept).
- **aiida-vasp-qpcorrection** *(design settled 2026-08-21, not yet built -
  supersedes the plan file's original sketch, see "Phase 3 detailed design"
  below for the full detail)*: `GPModelData` (orm.Data, mirrors
  `ArchiveData`'s store()-override pattern); a single backend-agnostic
  `WavefunEigenCorrectCalculation` (CalcJob, renamed from
  `WavecarQPModificationCalculation`, simplified to always take
  `corrections: BandsData` - the old mutually-exclusive
  `reference_gw_folder`/`corrections` design was dropped); two independent
  corrections-provider CalcJobs (`QpInterpolationCalculation`,
  `QpGPPredictionCalculation`), each producing `qp_corrections: BandsData`;
  two independent WorkChains (`VaspQPInterpolationWorkChain`,
  `VaspQPGPCorrectionWorkChain`, no shared dispatcher) that subclass
  `VaspDFTGWWorkChain` and layer correction+patch+BSE phases on top.

## Decisions log

- Full modernization of interpolation into a real CalcJob now, not deferred.
- Fix all pre-existing bugs found (see below) as part of this work, not
  left as-is.
- New branch `refactor/split-vmbpt` in the existing `aiida-vasp-dev` repo,
  created as a **separate `git worktree`** (`../aiida-vasp-dev-refactor`),
  not an in-place checkout - because the repo has only one working
  directory and the real dev venv's editable install
  (`~/venv_AiiDA_20251209`) is bound to the original `aiida-vasp-dev`
  directory. This keeps real job submissions on `support_GWBSE_rebased`
  completely unaffected while this branch is worked on.
- New sibling plugin repos `aiida-vasp-gwconv/` and `aiida-vasp-qpcorrection/`
  under `01_AiiDA/AiiDA_Develop/` (git-initialized; scaffolding committed in
  Phase 0 as `e5ea4bc`/`db82390`).
- No historical AiiDA database provenance to preserve (confirmed: nothing
  stored in the `lvarras_aiida` profile needs to stay loadable) - no
  backward-compat import shims required.
- Gate each phase: implementation stops after Phase 0/1/2/3 land, for
  explicit review, rather than running everything back-to-back.
- 4 untracked `.bak` files pre-existed in `vMBPT/` in the *original*
  `aiida-vasp-dev` directory (manual backups from a 2026-08-17 fix session,
  never committed). Left untouched there - since they're untracked and
  per-directory, they never appeared in this worktree at all, so no action
  was needed.
- **Golden-file harness environment**: this sandbox has no local AiiDA
  install at all (checked: no `aiida` module, no conda/mamba anywhere). The
  user's real dev/production environment is
  `source ~/venv_AiiDA_20251209/bin/activate` (aiida-core 2.7.2, editable-
  installed against the *original* `aiida-vasp-dev` dir, used for actual job
  submissions) - **do not use or modify this venv for refactor work**, the
  user asked for it to be left completely untouched. Initially (Phase 0/
  early Phase 1) this branch's code was run against it via a
  `PYTHONPATH=<this worktree>/src` override without reinstalling anything
  into it; partway through Phase 1 the user asked for a dedicated venv
  instead, so a fresh one was created:
  `source ~/venv_AiiDA_202608_refactor/bin/activate`, with this worktree's
  package `pip install -e .`'d into it directly (no `PYTHONPATH` override
  needed anymore) and dependency versions matched from the real venv's
  `pip freeze` (`aiida-core==2.7.2` - yanked on PyPI, still installable
  pinned; `numpy`; `scikit-learn==1.8.0`; `pymatgen==2025.10.7`;
  `scipy==1.17.1` - several of these are undeclared deps of vMBPT modules,
  not currently listed in `pyproject.toml`). Both venvs share the same
  `~/.aiida/config.json`/`lvarras_aiida` profile (profiles are per-machine,
  not per-venv). **Use `~/venv_AiiDA_202608_refactor` for all verification
  from here on.** The harness loads the real `lvarras_aiida` profile
  (needed because `workchain_mBSE_base_winterpolation.py` calls
  `load_profile()` at import time) but never calls `.store()` - no writes
  happen.
- **New bug found while building the harness fixture** (see below,
  bug 5) - decided: fix now, in Phase 1, as its own isolated commit; golden
  baseline captured *before* the fix (this is `phase0_pre_bugfix_baseline_*`
  below) so the Phase-1 diff shows exactly the intended fix and nothing
  else. Confirmed not currently used in real submissions
  (`local_gw_reference_folder` branch), but planned for future use, which is
  why it's being fixed rather than left alone.
- **Phase 1 step 3 (the straddling-file split) ripples further than the
  plan text originally called out.** `VaspmBSECompleteWorkChain`
  (`workchain_mBSE_master.py`) and `VaspmBSEConvergenceTemplateWorkChain`
  (`workchain_mBSE_convergence.py`) - both bucket B - both do
  `spec.expose_inputs(VaspmBSEInitScriptWorkChain, ...)`, so they derive
  their exposed inputs *dynamically* from whatever the base class currently
  declares. No code changes were needed in either file for structural
  correctness (confirmed: both still import and build `.spec()` cleanly
  after `ns_interpolation.*` was removed) - but any caller that previously
  supplied `ns_interpolation.*` values to them will now fail at input-
  validation time. Separately, `utils_helpers_setupworkchain.py`'s
  `Helpers_setup_Workchain._build_interpolation_inputs` (a bucket-A shared
  helper) becomes orphaned - it still builds an `ns_interpolation`-shaped
  dict, but nothing accepts that namespace anymore anywhere in this repo.
  Left as inert dead code for now (not deleted) since Phase 3's new
  orchestrating workchain/launch script will likely want to reference or
  adapt its local/remote GW-reference-folder branch logic when rebuilding
  the interpolation call site. This means `example_launch_mBSE_complete.py`
  and all `submit_workchain_mBSEinterpolation*.py` scripts are stale from
  this commit onward until Phase 3 delivers `VaspQPCorrectedWorkChain` as
  their replacement - already anticipated by the plan for
  `submit_workchain_mBSEinterpolation.py` specifically, but not for the
  other two `_master(_vsc5)` scripts (which go through
  `VaspmBSECompleteWorkChain`) or for the shared helper.
- Replaced the old `not ns_interpolation.use_interpolation` scissor toggle
  in `__add_inputs_mBSE_incar` with an explicit `ns_BSE.use_scissor` input
  (default `False`, behaviorally identical to the old default since
  `use_interpolation` defaulted to `True`). This is a new, independent input
  - not a straight rename - since the interpolation decision no longer
  lives on this class at all.
- **Phase 3 architecture, settled 2026-08-21 through discussion before any
  code was written** - see "Phase 3 detailed design" section below for full
  detail. Headline decisions, each with its rationale recorded there:
  `WavecarQPModificationCalculation` renamed to `WavefunEigenCorrectCalculation`
  and simplified to a single `corrections: BandsData` input (dropped the
  mutually-exclusive `reference_gw_folder` branch entirely); PROCAR-like data
  reuses `orm.ProjectionData` (no new Data type); no formal template/ABC
  across the two corrections-provider CalcJobs, and no Code-extras tagging
  for dispatch either - the contract is output-shape-only
  (`qp_corrections: BandsData`); `PortableCode` (not `InstalledCode`) for
  the bundled scripts, registered via 4 small single-purpose functions, with
  no dedup-by-label and no auto-versioning (both explicitly accepted
  trade-offs, not oversights); the two corrections engines are separate,
  non-interchangeable WorkChains, not a runtime-dispatched pair; and
  `VaspDFTGWWorkChain`'s FSM will be genericized (phase-list-driven) so
  these new WorkChains can subclass it additively instead of duplicating
  its retry/state machinery.

## Pre-existing bugs to fix (Phase 1, each an isolated commit)

1. **[Fixed, `eee74e3f`]** `calcs/vasp.py` - `vdw_kernel` input declared
   (`define()`) but its copy-to-remote logic in `write_additional()` was
   deleted in fork commit `9b9d069f`. Restored the copy logic verbatim from
   that commit's pre-deletion state.
2. **[Fixed, `9d8684e4`]** `workchain_wrapper_VaspWorkchain_G0W0.py` imported
   `VaspCalculation` via `from aiida_vasp.calcs.vasp2wInitScript import
   VaspCalculation` (transitive re-export) instead of `from
   aiida_vasp.calcs.vasp import VaspCalculation`. Fixed the import.
3. **[Fixed, moved to gwconv `2d9933f`]** `workchain_G0W0_Wannierization.py`
   had a broken import (`from workchain_BasisExtrapolation import
   input_magnetic_moment_tomagmom` - no such module; real one is
   `utils_helpers_extrapolation.py`, and post-split lives in aiida-vasp).
   User's explicit call: fix and move to gwconv rather than delete (it's a
   real, standalone DFT+Wannier90 workchain, just unwired). Fixed the import
   to `aiida_vasp.workchains.vMBPT.utils_helpers_extrapolation` while
   moving; registered as `vasp.gw.wannierization`.
4. **[Fixed, `728befbe`]** Deleted `utils_calcfunctions.py` (dead duplicate
   of `utils_helpers_extrapolation.py`, unreferenced anywhere).
5. **[Fixed, `bab6cd50`]** **[Found during Phase-0 harness build]**
   `workchain_mBSE_base_winterpolation.py`'s
   `__prepare_inputs_G0W0interpolation` (~line 690-697) passes
   `--path_sparse_GW`/`--sparse_GW_filename` **swapped** relative to what
   `utils_interpolationclasses.v2.py`'s argparse expects (it does
   `os.path.join(args.path_sparse_GW, args.sparse_GW_filename)`, i.e.
   `path_sparse_GW` must be the directory, `sparse_GW_filename` the
   filename - the workchain passes them the other way around). Confirmed via
   the harness's `local` variant: pre-fix, the constructed command is
   `--path_sparse_GW CopiedFromLocal_dummy_OUTCAR_3  --sparse_GW_filename ./`
   which resolves to the broken path `CopiedFromLocal_dummy_OUTCAR_3/.` on
   the interpolation-script side. The `remote_gw_reference_folder` branch
   likely masks this in practice (an absolute 2nd `os.path.join` arg
   discards the 1st), which is presumably why it's gone unnoticed - but the
   `local_gw_reference_folder` branch is genuinely broken today. Fix: swap
   the two `str(args_interpolation[...])` values in the f-string/concat that
   builds `str_launch_command`. Verified via the harness: the only diff vs.
   the pre-fix baseline (both variants) was the two values trading places.
   Note: this whole method was removed from the class two commits later
   (Phase 1 step 3's split), so this fix and its dedicated golden baselines
   (`phase1_post_bug5fix_baseline_*.json`) are now historical record only -
   the corresponding logic will need to be rebuilt (correctly, per this
   fix) wherever Phase 3 reconstructs the interpolation call.

**Minor/cosmetic, optional**: `utils_helpers_setupworkchain.py:526` has
`"\[POTENTIALS]"` - an invalid escape sequence (`SyntaxWarning` under Python
3.12, will become a hard error in a future Python). Harmless today; fix
opportunistically if touching that file for other reasons, not urgent.

## Regression-safety mechanism: golden-file harness

`regression_harness/harness_vMBPT_inputs.py` (this worktree) calls
`VaspmBSEInitScriptWorkChain`'s pure input-building methods directly
against a fixed synthetic fixture - no daemon, no submission, nothing
stored.

Run it:
```
source ~/venv_AiiDA_202608_refactor/bin/activate
cd <this worktree>
python regression_harness/harness_vMBPT_inputs.py \
    regression_harness/golden/<label>.json
```
(no `PYTHONPATH` override needed - this worktree's package is `pip install
-e .`'d directly into this dedicated venv; see the Decisions log entry
above for why this venv exists instead of the real dev venv.)

Baselines, in order (each documents one step in this history):
- `phase0_pre_bugfix_baseline_{remote,local}.json` - pre any Phase-1 change,
  two variants exercising the (now-removed)
  `remote_gw_reference_folder`/`local_gw_reference_folder` branches of
  `__prepare_inputs_G0W0interpolation`.
- `phase1_post_bug5fix_baseline_{remote,local}.json` - immediately after
  bug 5's fix, same two variants; only diff vs. the baselines above is the
  swapped-args fix.
- `phase1_post_split_baseline.json` - after Phase 1 step 3 removed
  `ns_interpolation.*`/`__prepare_inputs_G0W0interpolation` entirely. No
  more remote/local variant (that branch point no longer exists on this
  class). Confirmed byte-identical to the two baselines above once the
  now-absent interpolation-only keys (`init_script_call_command`,
  `local_init_script`, `local_files_to_copy_to_remote_submission_folder`,
  `local_SinglefileData_tocopy_toremote`) are excluded from the comparison.
  **This is the current baseline going forward** - the harness itself no
  longer accepts a `[remote|local]` CLI argument.

**How to use in later phases**: after any change that could plausibly touch
this logic, rerun the harness into a new file and diff against
`phase1_post_split_baseline.json`. Any diff should be *exactly* the
intended change, nothing else. Once confirmed correct-and-intended,
re-capture it as the new baseline for subsequent steps to diff against.

For Phase 3's deepest verification (interpolation numerics moving into
`WavecarQPModificationCalculation`), this extends to a real OUTCAR+WAVECAR
fixture and comparing patched eigenvalues numerically - not started yet,
see Open items.

## Phase 3 detailed design (settled 2026-08-21, not yet implemented)

This section is the settled design worked out through discussion before any
Phase-3 code was written. It supersedes the original plan file's Phase-3
sketch wherever the two disagree - the plan file is left as historical
record, this section is authoritative going forward.

### Naming/contract changes vs. the original plan text

- `WavecarQPModificationCalculation` renamed to **`WavefunEigenCorrectCalculation`**.
- Its input contract is simplified from the plan's original two-mutually-
  exclusive-branches design (`reference_gw_folder` OR `corrections`) down to
  **always `corrections: BandsData`**, nothing else. It becomes a single,
  dumb, backend-agnostic WAVECAR patcher:
  `original_folder: RemoteData` (uncorrected WAVECAR) + `corrections:
  BandsData` -> `remote_folder: RemoteData` (corrected WAVECAR). It has zero
  knowledge of *how* the corrections were computed - interpolation and GP
  prediction both just need to produce a `BandsData` beforehand.
- This pushes all "how are corrections computed" logic into two new,
  independent CalcJobs (below), each producing the same
  `qp_corrections: BandsData` output shape - that shared output shape is the
  *only* contract between them, deliberately not formalized as a shared base
  class / ABC (see "No formal template" below).

### Data types

- **PROCAR data**: reuse `orm.ProjectionData` - confirmed present in the
  installed aiida-core (`~/venv_AiiDA_202608_refactor`); it has
  `set_reference_bandsdata()`/`set_projectiondata()`/`set_orbitals()`, and is
  what `aiida-quantumespresso` already uses for PDOS/orbital-projection
  data. No new Data type needed for this piece.
- **`GPModelData(orm.Data)`** (new): mirrors `ArchiveData`'s `store()`-
  override pattern (private state built in `__init__`, materialized into
  `self.base.repository` just before storing). Holds architecture/config,
  kernel/likelihood hyperparameters, inducing points/variational state,
  normalization info, descriptor definition, serialized `state_dict`.

### CalcJobs

```
WavefunEigenCorrectCalculation(CalcJob)     [shared by both chains]
    in:  code (AbstractCode), original_folder (RemoteData), corrections (BandsData)
    out: remote_folder (RemoteData)

QpInterpolationCalculation(CalcJob)
    in:  code (PortableCode), bandsdata_g0w0 (BandsData), settings (Dict, optional)
    out: qp_corrections (BandsData)

QpGPPredictionCalculation(CalcJob)
    in:  code (PortableCode), gp_model (GPModelData), structure (StructureData),
         bandsdata_dft (BandsData), projection_data (ProjectionData), settings (Dict, optional)
    out: qp_corrections (BandsData), uncertainty (BandsData, optional)
```

### No formal template/ABC for the corrections-providers

Explicit decision: do NOT force a shared `spec.input()` signature across
`QpInterpolationCalculation`/`QpGPPredictionCalculation` - their real inputs
genuinely differ (BandsData-only vs. Structure+BandsData+ProjectionData),
and padding one CalcJob's spec with unused inputs just to look symmetric was
judged an abstraction-for-its-own-sake trap. The only enforced contract is
the shared *output* shape (`qp_corrections: BandsData`).

Also explicitly rejected: tagging a `PortableCode` node with an engine-kind
extra (`code.base.extras['qp_engine_kind'] = 'interpolation' | 'gp_ml'`) as
a runtime dispatch mechanism - unnecessary, since (see below) the two
engines are never dynamically interchangeable at runtime; each lives in its
own hardcoded workchain, so there is nothing to dispatch between.

### Code registration (PortableCode, not InstalledCode)

`PortableCode` chosen over `InstalledCode` specifically because it is **not
bound to any one Computer** - confirmed via
`inspect.signature(orm.PortableCode.__init__)` against the installed
aiida-core: `PortableCode.__init__(self, filepath_executable, filepath_files,
**kwargs)` takes no `computer` argument (unlike
`InstalledCode.__init__(self, computer, filepath_executable, **kwargs)`,
which requires one). AiiDA re-uploads the stored file tree fresh into
whichever computer's work directory the CalcJob actually runs on - so ONE
`PortableCode` node works whether the CalcJob is submitted to `localhost`
(direct scheduler, no queue - cheapest case, e.g. lightweight GP inference)
or a remote SLURM cluster (e.g. co-located with a large WAVECAR
`RemoteData`), with no per-computer duplication. Venv activation (e.g.
today's `source ~/venv_vasp_interpolation/bin/activate`) stays a per-
Computer concern handled via `metadata.options.prepend_text`, unrelated to
which Code is used - `PortableCode` only ships the script files themselves,
it has no notion of a remote Python environment.

Rejected: registering the Code as a `pip install` post-install hook -
flit/modern build backends don't support post-install hooks at all, and
even if they did, there's no guarantee a profile is loaded/writable at
install time (anti-pattern relative to how the rest of the AiiDA ecosystem
handles this - nothing in aiida-core auto-creates computers/codes on
install).

Adopted: lazy get-or-create, split into 4 single-purpose functions (final
naming, to live in `aiida_vasp_qpcorrection/utils/code_registration.py`):

```python
def register_QPinterpolation_portableCode() -> orm.PortableCode:
    """Always builds+stores a brand-new PortableCode. No existence check, no branching."""

def register_GPML_portableCode() -> orm.PortableCode:
    """Same, for the GP/ML script."""

def get_QPinterpolation_portableCode(code: orm.PortableCode | None = None) -> orm.PortableCode:
    return code if code is not None else register_QPinterpolation_portableCode()

def get_GPML_portableCode(code: orm.PortableCode | None = None) -> orm.PortableCode:
    return code if code is not None else register_GPML_portableCode()
```

Explicit, accepted consequences of this exact shape (both raised and
confirmed acceptable during design discussion, not oversights):
- `get_*()` called with no argument creates a **new** Code node every call -
  no label-based dedup/lookup. Two separate calls with no argument produce
  two distinct `PortableCode` nodes with identical content, not one reused
  node. Dedup, if wanted, is the *caller's* job (call `register_*()` once,
  keep the returned node, pass it explicitly from then on) - not the
  helper's.
- No auto-versioning/hashing of the label to detect a stale stored script
  after editing it. If the bundled script changes: old runs correctly keep
  pointing at their original, immutable Code node (this is real provenance
  working as intended, not a bug) - but *new* runs, if they omit the `code`
  argument, would also need a fresh `register_*()` call to pick up the
  change; without one they'd get whichever Code the `get_*` no-arg path
  happens to produce, which does not automatically track script edits. This
  is accepted as a manual/deliberate step, not automatic drift-detection.

Where these get called from: NOT inside the CalcJob classes themselves (a
CalcJob just declares a normal required `code` input, same as any CalcJob).
The calling WorkChain owns the "did I get a Code, or do I need a default"
policy in its `prepare_step`/inputs-building step, e.g.:
```python
code = self.inputs.ns_qpcorrection.get('interpolation_code', None)
builder.code = get_QPinterpolation_portableCode(code)
```

### Two independent WorkChains, not a shared dispatcher

Because interpolation and GP prediction are **not runtime-interchangeable**
- they are two different, separately-launched chains, never swapped
dynamically by a single caller - there is no factory/entry-point dispatch
mechanism here (the `aiida-common-workflows` "generator" pattern was
considered and explicitly rejected as unneeded for exactly this reason:
that pattern earns its cost when callers need to pick an engine generically
at runtime, which is not the case here). Instead: `VaspQPInterpolationWorkChain`
and `VaspQPGPCorrectionWorkChain` are separate concrete classes, each
hardcoded to call its own corrections-provider CalcJob.

### DFT+GW reuse: FSM genericization in VaspDFTGWWorkChain (bucket A) - the big remaining design decision

User's ask: the two QP-correction workchains above should **inherit** from
`VaspDFTGWWorkChain` (bucket A's existing DFTgr -> DFTvo -> G0W0 chain) and
layer their own correction+patch+BSE phases on top of it, reusing its
FSM/retry machinery - rather than composing it via a plain
`self.submit(VaspDFTGWWorkChain, ...)` call from a separate, simpler outer
workchain (which was the original, simpler sketch discussed earlier in this
same design conversation, before this inheritance requirement came up).

**Checked the actual code (`workchain_G0W0_base.py`) before committing to
this, rather than assuming it would just work - found it does NOT support
clean subclass-based extension today:**
- `self.ctx._next_workchain = {'1DFTgr': WorkflowFactory('vasp.vasp'),
  '2DFTvo': WorkflowFactory('vasp.vasp'), '3G0W0': VaspGWWorkChain}` (built
  in `initialize()`, ~line 235) - this dict-dispatch-by-calc_type part is
  already reusable and already proves the "swap the class used for a given
  step" pattern works today (`'3G0W0'` is already overridden to a wrapper
  workchain, `VaspGWWorkChain`, not the raw `vasp.vasp` process).
- BUT `state_execution_enum` (line 108) is a genuine `enum.Enum` with one
  flat member per (phase, status) pair: `INIT`, `DFTGR_PENDING`,
  `DFTGR_RUNNING`, `DFTGR_DONE`, `DFTVO_PENDING`, `DFTVO_RUNNING`,
  `DFTVO_DONE`, `G0W0_PENDING`, `G0W0_RUNNING`, `COMPLETE`, `FAILED`,
  `RECOVERY` (confirmed via direct read of the class body - no
  `G0W0_DONE` member, `G0W0_RUNNING` success goes straight to `COMPLETE`,
  an asymmetry with the other two phases). **Standard Python `Enum` classes
  cannot be extended with new members via subclassing** - this is a hard
  language restriction, not a style choice - so no subclass can add
  `CORRECTION_PENDING`/`BSE_RUNNING`/etc. members to this enum. Any
  genericization has to replace this representation, not extend it.
- `update_state()` (~line 249) is one large sequential if-chain hardcoded to
  these exact named states - e.g. the `G0W0_RUNNING` branch (~line 336-340)
  hardwires `state_execution = COMPLETE` directly on success, with no hook
  for "then move to a correction phase instead."
- `execute_step()` (~line 378) builds two dict literals fresh inside the
  method body (`mapping_enum_to_calc_type_torun`, `mapping_calctype_to_state`),
  both hardcoded to `'1DFTgr'/'2DFTvo'/'3G0W0'` - not class attributes, so
  there is nothing to extend without redeclaring the whole method.
- `validate_step()` similarly branches on named states for its per-phase
  required-file checks (e.g. `DFTVO_PENDING` needs `WAVECAR`, `G0W0_PENDING`
  needs `WAVECAR`+`WAVEDER`).
- `prepare_step()` (~line 408 onward) was not yet read in full detail at
  design time - almost certainly branches per calc_type too, for building
  each phase's inputs; needs inspection before implementation starts.
- **Net effect as the code is written today: subclassing now would mean
  fully overriding `update_state`/`execute_step`/`validate_step` (and
  probably `prepare_step`), duplicating the base's DFT/GW logic inline
  alongside the new phases' logic - fragile, since any future change to the
  base class's FSM would have to be manually re-mirrored in every
  subclass.**

**Decision (agreed with the user, not yet implemented): genericize the FSM
first**, so real additive subclassing becomes possible with zero method-
overriding needed in the QP-correction subclasses. Design:

1. Replace the flat per-phase enum with a composite representation: a
   small, fixed, never-extended `PhaseStatus(Enum)` with just `PENDING`/
   `RUNNING`, covering *any* phase generically, plus `self.ctx.phase_idx: int`
   (an index into an ordered phase list; `-1`/`len(list)` serve as the
   INIT/COMPLETE sentinels) and a separate terminal-state tracker for
   `FAILED`/`RECOVERY` (these stay global, not per-phase).
2. Introduce a `WorkflowPhase` descriptor (frozen dataclass) carrying
   everything that is currently hardcoded per named phase:
   ```python
   @dataclass(frozen=True)
   class WorkflowPhase:
       key: str
       process_class: type                                       # or a resolver, like today's _next_workchain
       build_inputs: Callable[[WorkChain], dict]                  # replaces prepare_step's per-phase branch
       capture_outputs: Callable[[WorkChain, ProcessNode], None]  # replaces update_state's "stash restart_folder for next phase" branch
       required_files: tuple[str, ...] = ()                       # replaces validate_step's per-phase branch
       skip_if: Callable[[WorkChain], bool] | None = None         # replaces run_1DFTgr / run_2DFTvo_3G0W0 skip logic
   ```
3. `VaspDFTGWWorkChain` gets a class-level
   `_PHASES: ClassVar[list[WorkflowPhase]] = [<1DFTgr>, <2DFTvo>, <3G0W0>]`.
   `update_state`/`execute_step`/`validate_step`/`prepare_step` are each
   rewritten ONCE, generically, as loops over
   `self._PHASES[self.ctx.phase_idx]` - inherited unchanged by every
   subclass forever; no per-phase named branches remain anywhere in the
   base class after this.
4. `VaspQPInterpolationWorkChain`/`VaspQPGPCorrectionWorkChain` become
   purely additive subclasses:
   `_PHASES = VaspDFTGWWorkChain._PHASES + [<correction-phase>, <patch-phase>, <BSE-phase>]`,
   with no method overrides needed at all.
5. **The real (non-trivial) work this actually requires, not just
   reshuffling**: `prepare_step`'s current per-phase input-building logic
   and `update_state`'s per-phase output-capture logic (today: stash
   `node.outputs.remote_folder` for the next phase's restart) must be
   extracted into these small callables - and the generic loop can't assume
   one fixed output shape across all phases, since e.g. the correction phase
   produces `qp_corrections: BandsData` while the DFT/GW/patch phases
   produce `remote_folder: RemoteData`. `capture_outputs` is exactly what
   absorbs that per-phase difference.

**New verification requirement this introduces** - distinct in kind from
the input-building golden harness already in place, and does not exist yet:
this refactor changes actual FSM *control flow*, not just data-building, so
the existing harness (which only exercises pure input-dict-building
functions) cannot catch a regression here. Needed before/after this
refactor: run the *current* `VaspDFTGWWorkChain` through a dry-run
mechanism, capture the exact ordered sequence of submitted calc-types plus
their inputs end-to-end, then confirm the genericized version reproduces
that sequence bit-for-bit. Whether `mock-vasp`/`dryrun-vasp` (previously
ruled out for the input-building harness, since they need a real `vasp_std`
and drive an actual short VASP run - see Open items below) are reusable for
*this* different, control-flow-level check is an open question, not yet
re-investigated - it is a different requirement (exercising submission
sequence/order, not comparing raw dict output), so the earlier "not reused"
conclusion for the input-harness does not automatically carry over to it.

## Phase checklist

- [x] **Phase 0**: worktree created; `aiida-vasp-gwconv`/`aiida-vasp-qpcorrection`
      scaffolded (pyproject.toml, package skeleton, README, .gitignore;
      both verified `pip install -e . --no-deps` cleanly); golden-file
      harness built and validated end-to-end; pre-bugfix golden baselines
      captured (both variants); this handoff.md written. All committed
      (`4a69af05` this repo, `e5ea4bc` gwconv, `db82390` qpcorrection).
- [x] **Phase 1** (aiida-vasp): fixed bugs 1 (`eee74e3f`), 2 (`9d8684e4`),
      4 (`728befbe`), 5 (`bab6cd50`, verified via harness - diff was exactly
      the arg swap) as isolated commits; bug 3 (Wannierization) deferred to
      Phase 2 per plan. Registered `aiida.workflows` entry points for the 3
      bucket-A classes (`dc180faf`) plus the slimmed mBSE class
      (`vasp.gw.mbse`, added alongside the split commit). Split
      `workchain_mBSE_base_winterpolation.py` (`d470bd0d`) - stripped
      `ns_interpolation.*`, verified byte-identical harness output on
      everything except the now-removed interpolation-only keys; see
      Decisions log for the wider ripple this uncovered
      (`VaspmBSECompleteWorkChain`/`VaspmBSEConvergenceTemplateWorkChain`'s
      dynamic `expose_inputs`, the now-orphaned
      `Helpers_setup_Workchain._build_interpolation_inputs`, the new
      `ns_BSE.use_scissor` input). Rewrote `workchains/__init__.py`
      (`5c32cd14`) to export only bucket-A classes (also fixed two latent
      `__all__`/star-import bugs found along the way). None of the 7 real
      launch scripts needed import updates (none go through
      `workchains/__init__.py` - confirmed by inspection).
- [x] **Phase 2** (aiida-vasp-gwconv): moved bucket-B files + extrapolation
      calcfunctions (aiida-vasp side: `919c4b57`; gwconv side: `2d9933f`).
      Bug 3 resolved (fixed + moved, see above). Registered 8
      `aiida.workflows` entry points (`vasp.gw.mbse_convergence_template`,
      `vasp.gw.mbse_kpts_conv`, `vasp.gw.mbse_nbands_conv`,
      `vasp.gw.mbse_complete`, `vasp.gw.g0w0_kpts_conv`,
      `vasp.gw.g0w0_basis_extr`, `vasp.gw.g0w0_complete`,
      `vasp.gw.wannierization`) - all verified via `verdi plugin list` and
      `.spec()` build. Updated all 6 affected launch scripts under
      `AiiDALAB_Container/AiiDA_SetupScripts/` (one more than the plan's
      count of 5 - `submit_workchain_mBSEinterpolation_kptsConv.py` was
      also affected) - **these live in a separate, much larger outer git
      repo (root `09-Project-DBMBPT_ML`, not this worktree) and were left
      as uncommitted edits there pending the user's explicit sign-off**,
      per git safety practice of not committing in a repo outside what was
      asked without confirmation. Verified: both packages import cleanly,
      `from ... import *` succeeds on both, all 8 new entry points load
      and build `.spec()`, all 6 launch scripts' new import targets
      resolve, golden harness output unchanged (this phase touches nothing
      it exercises).
- [ ] **Phase 3** (aiida-vasp-qpcorrection + FSM genericization): see "Phase 3
      detailed design" above for the full plan. Progress so far:
      - [x] Read `VaspDFTGWWorkChain` in full (previously only partially
        inspected) - confirmed `prepare_step` branches per calc_type the same
        way the other FSM methods do, as assumed in the design section.
      - [x] Built `regression_harness/harness_G0W0_base_fsm.py` - a
        control-flow golden harness for the FSM (distinct from
        `harness_vMBPT_inputs.py`, which only covers pure input-dict-building).
        Considered and rejected a real mock-vasp/pytest-test-profile based
        integration test (would require first building a correct full
        `exposed_inputs` builder for `vasp.vasp`, a substantial "first
        integration test ever written for this workchain" effort on its own)
        in favor of reusing `harness_vMBPT_inputs.py`'s proven duck-typed
        `self`, real-methods-called-directly pattern - `self.submit(...)` is
        stubbed to record the call and return an already-"finished" fake
        node, so the whole FSM loop runs synchronously in-process with zero
        AiiDA submissions/database writes/new computers-codes. Captures, per
        scenario: the full ordered `state_execution` trace and the ordered
        submission sequence (calc_type, class name, restart_folder repr).
      - [x] Captured `regression_harness/golden/pre_fsm_genericization_baseline.json`
        - two scenarios (full chain from scratch; skip-DFTgr with an
        external `starting_RemoteData`) - both verified by inspection to
        match the FSM's own documented behavior (header comment block in
        `workchain_G0W0_base.py`) exactly: correct calc_type ordering,
        correct per-phase class dispatch (`VaspWorkChain` for 1DFTgr/2DFTvo,
        `VaspGWWorkChain` for 3G0W0), correct restart_folder chaining
        phase-to-phase, correct skip-logic. **This is the safety net the FSM
        genericization rewrite must reproduce bit-for-bit before it's trusted.**
      - [ ] Not yet started: the actual FSM genericization rewrite
        (`PhaseStatus` enum, `WorkflowPhase` dataclass, rewriting
        `update_state`/`execute_step`/`validate_step`/`prepare_step` to loop
        over `_PHASES`), re-running this harness against the genericized code
        and diffing against this baseline, `GPModelData`, the 3 new CalcJobs
        (`WavefunEigenCorrectCalculation`, `QpInterpolationCalculation`,
        `QpGPPredictionCalculation`), `code_registration.py`'s 4 functions,
        the OUTCAR+WAVECAR numerical golden fixture, the two new WorkChains,
        entry point registration, updating the last launch script.

## Known risks & mitigations

- **Straddling-file split (Phase 1 step 3)** was the highest-risk step in
  aiida-vasp itself - done (`d470bd0d`), verified via the golden harness's
  zero-diff check on everything except the interpolation-specific fields.
- **Launch scripts now stale until Phase 3**: `example_launch_mBSE_complete.py`
  and all `submit_workchain_mBSEinterpolation*.py` scripts (3 of them) will
  fail at input-validation time if run today, since they build an
  `ns_interpolation` namespace that no longer exists on
  `VaspmBSEInitScriptWorkChain` (or, transitively, on
  `VaspmBSECompleteWorkChain`). This is expected and deferred to Phase 3's
  `VaspQPCorrectedWorkChain` - do not "fix" these scripts before then.
- **Interpolation-to-CalcJob migration (Phase 3)** is the highest-risk step
  overall - mitigated by capturing a real numerical (patched-eigenvalues)
  golden fixture *before* writing the new CalcJob, per the harness section
  above.
- **`VaspDFTGWWorkChain` FSM genericization (Phase 3, bucket A)** - a new,
  separately-risky piece added by the "Phase 3 detailed design" section
  above: rewriting `update_state`/`execute_step`/`validate_step`/
  `prepare_step` to be phase-list-driven touches bucket-A's core control
  flow, not just aiida-vasp-qpcorrection's new code, so a regression here
  could silently break the existing DFTgr->DFTvo->G0W0 chain for every
  caller, not just the new QP-correction subclasses. Mitigation: a new
  control-flow-level golden check (capture the exact submitted calc-
  type/input sequence before the refactor via a dry-run mechanism, diff
  against the same after) is required *before* trusting this refactor -
  see that section for what this needs and why the existing golden harness
  doesn't cover it. Do not build the QP-correction subclasses on top of the
  genericized FSM until this control-flow golden check passes cleanly.
- **GP model API not finalized** - the separate `Models_Base`
  `GPBackboneHead` refactor (different project) is still in progress; don't
  hardcode `predict_qp_corrections`/`GPModelData` internals against it yet.
- **Bug 5 could have affected real results** if `local_gw_reference_folder`
  had been used historically - confirmed it has not been, so no past-results
  audit is needed, only the forward fix.

## Open items deferred

- **[Resolved in Phase 2]** Wannierization kept, fixed, and moved to gwconv
  - see bug 3 above.
- **New from Phase 2**: get the user's sign-off on the 6 launch-script edits
  under `AiiDALAB_Container/AiiDA_SetupScripts/` (separate outer repo) and
  commit them there if approved - left uncommitted intentionally.
- GP model adapter API - blocked on `Models_Base` project stabilizing.
- OUTCAR+WAVECAR reference fixture for Phase 3's numerical golden check -
  check `test_data/` for something reusable before creating one from
  scratch.
- Whether `mock-vasp`/`dryrun-vasp` (`aiida_vasp/commands/`) are worth
  reusing for any part of the harness - inspected during Phase-0 prep
  (`dryrun_vasp.py`): they require an actual `vasp_std` executable and
  drive a real (short) VASP run, so they're not a fit for the no-cluster,
  pure-Python golden-harness use case here - not reused.
- **New from the Phase 3 design discussion**: whether `mock-vasp`/
  `dryrun-vasp` (or something else) can serve the *different*,
  control-flow-level golden check that `VaspDFTGWWorkChain`'s FSM
  genericization needs (capturing the ordered sequence of submitted calc-
  types/inputs, not raw dict output) - not yet re-investigated; the "not
  reused" conclusion just above was reached for a different purpose and
  doesn't automatically apply here. Needs answering before the FSM
  genericization work starts. See "Phase 3 detailed design" above.
- `VaspDFTGWWorkChain.prepare_step()`'s full body was not yet read/inspected
  during the Phase 3 design discussion (only `initialize`, `should_wc_continue`,
  `update_state`, `validate_step`, `correct_previous_errors`, `execute_step`
  were). Needs reading before implementing the `WorkflowPhase.build_inputs`
  extraction, to confirm it's per-calc_type-branched the same way the other
  methods are (assumed, not yet confirmed).
- Exact `WorkflowPhase` entries for the new correction/patch/BSE phases
  (their `build_inputs`/`capture_outputs` callables, and how
  `VaspQPInterpolationWorkChain`/`VaspQPGPCorrectionWorkChain` expose their
  own inputs - e.g. whether DFT+GW happens inside these workchains via the
  inherited phases, exposing `VaspDFTGWWorkChain`'s inputs directly, which
  is now the resolved answer to the earlier open question of whether GW
  happens inside or upstream of these chains) are designed at the concept
  level above but not yet written as code.
