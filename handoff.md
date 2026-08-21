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
- **aiida-vasp-gwconv**: convergence templates + master orchestrators
  (`VaspmBSEConvergenceTemplateWorkChain` + 2 children,
  `VaspmBSECompleteWorkChain`, `VaspG0W0KptsConvWorkChain`,
  `VaspG0W0BasisExtrWorkChain`, `VaspG0W0CompleteWorkChain`) + the
  extrapolation calcfunctions.
- **aiida-vasp-qpcorrection**: `GPModelData` (orm.Data, mirrors
  `ArchiveData`'s store()-override pattern), `WavecarQPModificationCalculation`
  (CalcJob, mirrors `VaspCalcBase`/`remote_copy_restart_folder`, mutually
  exclusive `reference_gw_folder` vs `corrections` inputs),
  `predict_qp_corrections` calcfunction, and the orchestrating
  `VaspQPCorrectedWorkChain` (DFT -> correction branch ->
  `WavecarQPModificationCalculation` -> aiida-vasp's slimmed
  `VaspmBSEInitScriptWorkChain`).

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

## Pre-existing bugs to fix (Phase 1, each an isolated commit)

1. **[Fixed, `eee74e3f`]** `calcs/vasp.py` - `vdw_kernel` input declared
   (`define()`) but its copy-to-remote logic in `write_additional()` was
   deleted in fork commit `9b9d069f`. Restored the copy logic verbatim from
   that commit's pre-deletion state.
2. **[Fixed, `9d8684e4`]** `workchain_wrapper_VaspWorkchain_G0W0.py` imported
   `VaspCalculation` via `from aiida_vasp.calcs.vasp2wInitScript import
   VaspCalculation` (transitive re-export) instead of `from
   aiida_vasp.calcs.vasp import VaspCalculation`. Fixed the import.
3. `workchain_G0W0_Wannierization.py` has a broken import (`from
   workchain_BasisExtrapolation import input_magnetic_moment_tomagmom` - no
   such module; real one is `utils_helpers_extrapolation.py`) and is
   otherwise unreferenced anywhere. Fix-and-move-to-gwconv or delete - your
   call at Phase 2 execution time (intent not recoverable from code).
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
- [ ] **Phase 2** (aiida-vasp-gwconv): move bucket-B files + extrapolation
      calcfunctions; resolve bug 3 (Wannierization); register entry points;
      update the 5 affected launch scripts; verify installs + imports.
- [ ] **Phase 3** (aiida-vasp-qpcorrection): extract interpolation numerics
      (no behavior change); build/capture the OUTCAR+WAVECAR golden fixture;
      `GPModelData`; `WavecarQPModificationCalculation` (verify against
      fixture); `predict_qp_corrections`; `VaspQPCorrectedWorkChain`
      (end-to-end verify); register entry points; update the last launch
      script.

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
- **GP model API not finalized** - the separate `Models_Base`
  `GPBackboneHead` refactor (different project) is still in progress; don't
  hardcode `predict_qp_corrections`/`GPModelData` internals against it yet.
- **Bug 5 could have affected real results** if `local_gw_reference_folder`
  had been used historically - confirmed it has not been, so no past-results
  audit is needed, only the forward fix.

## Open items deferred

- Wannierization (`workchain_G0W0_Wannierization.py`) keep-and-fix vs delete
  - decide at Phase 2 execution time.
- GP model adapter API - blocked on `Models_Base` project stabilizing.
- OUTCAR+WAVECAR reference fixture for Phase 3's numerical golden check -
  check `test_data/` for something reusable before creating one from
  scratch.
- Whether `mock-vasp`/`dryrun-vasp` (`aiida_vasp/commands/`) are worth
  reusing for any part of the harness - inspected during Phase-0 prep
  (`dryrun_vasp.py`): they require an actual `vasp_std` executable and
  drive a real (short) VASP run, so they're not a fit for the no-cluster,
  pure-Python golden-harness use case here - not reused.
