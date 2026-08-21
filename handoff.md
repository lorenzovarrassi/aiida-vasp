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
  under `01_AiiDA/AiiDA_Develop/` (git-initialized, not yet committed as of
  this writing - see Phase checklist).
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
  real dev environment is `source ~/venv_AiiDA_20251209/bin/activate`
  (aiida-core 2.7.2, editable-installed against the *original*
  `aiida-vasp-dev` dir). The harness runs against this branch's code via
  `PYTHONPATH=<this worktree>/src` override, never touching the venv's own
  editable-install target. The harness loads the real `lvarras_aiida`
  profile (needed because `workchain_mBSE_base_winterpolation.py` calls
  `load_profile()` at import time) but never calls `.store()` - no writes
  happen.
- **New bug found while building the harness fixture** (see below,
  bug 5) - decided: fix now, in Phase 1, as its own isolated commit; golden
  baseline captured *before* the fix (this is `phase0_pre_bugfix_baseline_*`
  below) so the Phase-1 diff shows exactly the intended fix and nothing
  else. Confirmed not currently used in real submissions
  (`local_gw_reference_folder` branch), but planned for future use, which is
  why it's being fixed rather than left alone.

## Pre-existing bugs to fix (Phase 1, each an isolated commit)

1. `calcs/vasp.py` - `vdw_kernel` input declared (`define()`) but its
   copy-to-remote logic in `write_additional()` was deleted in fork commit
   `9b9d069f`. Restore the copy logic.
2. `workchain_wrapper_VaspWorkchain_G0W0.py` imports `VaspCalculation` via
   `from aiida_vasp.calcs.vasp2wInitScript import VaspCalculation`
   (transitive re-export) instead of `from aiida_vasp.calcs.vasp import
   VaspCalculation`. Fix the import.
3. `workchain_G0W0_Wannierization.py` has a broken import (`from
   workchain_BasisExtrapolation import input_magnetic_moment_tomagmom` - no
   such module; real one is `utils_helpers_extrapolation.py`) and is
   otherwise unreferenced anywhere. Fix-and-move-to-gwconv or delete - your
   call at Phase 2 execution time (intent not recoverable from code).
4. Delete `utils_calcfunctions.py` (dead duplicate of
   `utils_helpers_extrapolation.py`, unreferenced anywhere).
5. **[Found during Phase-0 harness build]**
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
   builds `str_launch_command`.

**Minor/cosmetic, optional**: `utils_helpers_setupworkchain.py:526` has
`"\[POTENTIALS]"` - an invalid escape sequence (`SyntaxWarning` under Python
3.12, will become a hard error in a future Python). Harmless today; fix
opportunistically if touching that file for other reasons, not urgent.

## Regression-safety mechanism: golden-file harness

`regression_harness/harness_vMBPT_inputs.py` (this worktree) calls
`VaspmBSEInitScriptWorkChain`'s pure input-building methods
(`__prepare_inputs_DFT`, `__prepare_inputs_mBSE_base`,
`__prepare_inputs_G0W0interpolation`, `__add_inputs_mBSE_incar`) directly
against a fixed synthetic fixture - no daemon, no submission, nothing
stored. Two variants exercise the two mutually-exclusive GW-reference
branches (`remote_gw_reference_folder` / `local_gw_reference_folder`).

Run it:
```
source ~/venv_AiiDA_20251209/bin/activate
cd <this worktree>
PYTHONPATH=$(pwd)/src python regression_harness/harness_vMBPT_inputs.py \
    regression_harness/golden/<label>.json [remote|local]
```

Baselines captured so far (pre any Phase-1 change, on the state this branch
was created from):
- `regression_harness/golden/phase0_pre_bugfix_baseline_remote.json`
- `regression_harness/golden/phase0_pre_bugfix_baseline_local.json`

**How to use in later phases**: after any change that could plausibly touch
this logic, rerun the harness into a new file and diff against the most
recent golden baseline. Bug fixes should show up as *exactly* the intended
diff (e.g. bug 5's fix should flip the two args in
`init_script_call_command` and nothing else); pure refactors (e.g. the
Phase-1 step-3 split of `workchain_mBSE_base_winterpolation.py`) should show
*zero* diff in `dft_inputs`/the non-interpolation parts of `mbse_inputs`.
Once a diff is confirmed correct-and-intended, re-capture it as the new
golden baseline for subsequent steps to diff against.

For Phase 3's deepest verification (interpolation numerics moving into
`WavecarQPModificationCalculation`), this extends to a real OUTCAR+WAVECAR
fixture and comparing patched eigenvalues numerically - not started yet,
see Open items.

## Phase checklist

- [x] **Phase 0**: worktree created; `aiida-vasp-gwconv`/`aiida-vasp-qpcorrection`
      scaffolded (pyproject.toml, package skeleton, README, .gitignore;
      both verified `pip install -e . --no-deps` cleanly); golden-file
      harness built and validated end-to-end; pre-bugfix golden baselines
      captured (both variants); this handoff.md written.
      **Not yet done in Phase 0**: committing this branch's new files, and
      committing the two new repos' initial scaffolding (pending your
      review of this phase before I commit).
- [ ] **Phase 1** (aiida-vasp): fix bugs 1/2/5 as isolated commits (verify
      via harness after each - bug 5's diff should be exactly the arg swap);
      decide bug 3/4 disposition; register `aiida.workflows` entry points
      for bucket-A classes; split `workchain_mBSE_base_winterpolation.py`
      (strip `ns_interpolation.*`, verify zero diff on the non-interpolation
      harness output); update `workchains/__init__.py`; update launch-script
      imports as needed.
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

- **Straddling-file split (Phase 1 step 3)** is the highest-risk step in
  aiida-vasp itself - mitigated by the golden harness's zero-diff check on
  everything except the interpolation-specific fields.
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
