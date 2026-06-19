# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
from aiida.orm import Int, Float, Dict, Bool, List, RemoteData, KpointsData, Str, BandsData
from dataclasses import dataclass
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import WorkChain, ToContext, append_, submit, while_
from aiida.common.extendeddicts import AttributeDict
from typing import Optional

from .workchain_mBSE_base_winterpolation import VaspmBSEInitScriptWorkChain
from .utils_helpers_mBSE import _extract_opticalgap_fromWorkchainNode, _determine_BSE_parameters

from aiida import load_profile
load_profile()


# ==============================================================================
# Shared static helpers  (convergence-parameter-agnostic)
# ==============================================================================

class helper_BSEConv_shared:
    """
    Static helper methods shared between all BSE convergence workchains.
    Nothing here knows whether the control variable is kmesh, optical_window_threshold,
    or anything else.
    """

    @staticmethod
    def _get_kmesh_from_kdensity(latVec, KSPACING, flag_roundInsteadCeil=True):
        """
        Calculate the k-point mesh from a target k-spacing.
        KSPACING (Angstrom^{-1}): smaller -> denser mesh.
        Output Ni satisfies |b_i| / Ni <= KSPACING.
        """
        latVec = np.array(latVec)
        recLatVec = np.zeros((3, 3))
        Vol = np.abs(np.dot(latVec[0, :], np.cross(latVec[1, :], latVec[2, :])))
        recLatVec[0, :] = np.cross(latVec[1, :], latVec[2, :]) / Vol
        recLatVec[1, :] = np.cross(latVec[2, :], latVec[0, :]) / Vol
        recLatVec[2, :] = np.cross(latVec[0, :], latVec[1, :]) / Vol
        rec_cell_norm = np.array([np.linalg.norm(recLatVec[x, :]) for x in range(3)])
        kmesh_ideal_fractional = rec_cell_norm * 2 * np.pi / KSPACING
        if flag_roundInsteadCeil:
            kmesh = [max(1.0, np.round(k)) for k in kmesh_ideal_fractional]
        else:
            kmesh = np.ceil(kmesh_ideal_fractional)
        return np.array(kmesh).astype(int)

    @staticmethod
    def _get_kspacing_from_kmesh(latVec, kmesh):
        latVec = np.array(latVec)
        recLatVec = np.zeros((3, 3))
        Vol = np.abs(np.dot(latVec[0, :], np.cross(latVec[1, :], latVec[2, :])))
        recLatVec[0, :] = np.cross(latVec[1, :], latVec[2, :]) / Vol
        recLatVec[1, :] = np.cross(latVec[2, :], latVec[0, :]) / Vol
        recLatVec[2, :] = np.cross(latVec[0, :], latVec[1, :]) / Vol
        rec_cell_norm = [np.linalg.norm(recLatVec[x, :]) for x in range(3)]
        return [2 * np.pi * rec_cell_norm[i] / kmesh[i] for i in range(3)]

    @staticmethod
    def _check_dielectric_convergence(idiel_1, idiel_2, energy_grid, energy_window,
                                      method="L2_distance", threshold=1e-3, channels=(0, 1, 2)):
        """
        Compare two imaginary dielectric functions within an energy window.
        Returns: (bool converged, float distance)
        """
        from scipy.stats import wasserstein_distance
        E_min, E_max = energy_window
        mask = (energy_grid >= E_min) & (energy_grid <= E_max)
        egrid = energy_grid[mask]
        id1 = idiel_1[mask][:, channels]
        id2 = idiel_2[mask][:, channels]

        def __L2(a, b, e):
            return np.sqrt(np.trapz(np.mean((a - b) ** 2, axis=1), e))

        def __L1(a, b, e):
            return np.trapz(np.mean(np.abs(a - b), axis=1), e)

        def __Wasserstein(a, b, e):
            f1 = np.clip(np.mean(a, axis=1), 0, None)
            f2 = np.clip(np.mean(b, axis=1), 0, None)
            if np.sum(f1) > 0: f1 /= np.sum(f1)
            if np.sum(f2) > 0: f2 /= np.sum(f2)
            return wasserstein_distance(e, e, f1, f2)

        metric_map = {"L2_distance": __L2, "L1_distance": __L1, "Wasserstein": __Wasserstein}
        if method not in metric_map:
            raise ValueError(f"Unknown method '{method}'. Choose from: {list(metric_map.keys())}")
        distance = metric_map[method](id1, id2, egrid)
        return distance < threshold, distance

    @staticmethod
    def _get_energy_of_diel_onset(imdiel, egrid, thr_for_considering_offset=0.1):
        """First energy where mean of diagonal eps_2 components exceeds threshold."""
        diag_mean = np.mean(imdiel[:, :3], axis=1)
        idx = np.where(diag_mean > thr_for_considering_offset)[0]
        return egrid[idx[0]] if len(idx) > 0 else egrid[0]

    @staticmethod
    def _collect_consecutive_optgap_and_diel_differences(records, window_size, diel_metric,
                                                          onset_threshold=0.1):
        """
        Compute delta(optical gap) and delta(dielectric distance) between consecutive records.
        Records are already sorted by the child's sort key.
        Returns: (ogap_diffs, diel_diffs, meta)  - all length == len(records).
        """
        ogap_diffs, diel_diffs, meta = [], [], []
        if not records:
            return ogap_diffs, diel_diffs, meta

        for i, rec in enumerate(records):
            if i == 0:
                ogap_diffs.append(None)
                diel_diffs.append(None)
                meta.append({"energy_window": None, "onset": None})
                continue

            prev = records[i - 1]

            # optical gap difference
            if prev["optgap"] is not None and rec["optgap"] is not None:
                ogap_diffs.append(abs(rec["optgap"] - prev["optgap"]))
            else:
                ogap_diffs.append(None)

            # dielectric distance
            if prev["imdiel"] is None or rec["imdiel"] is None:
                diel_diffs.append(None)
                meta.append({"energy_window": None, "onset": None})
                continue

            onset_prev = prev.get("imdiel_onset") or helper_BSEConv_shared._get_energy_of_diel_onset(
                prev["imdiel"], prev["energygrid"], onset_threshold)
            onset_rec  = rec.get("imdiel_onset")  or helper_BSEConv_shared._get_energy_of_diel_onset(
                rec["imdiel"],  rec["energygrid"],  onset_threshold)
            onset  = min(onset_prev, onset_rec)
            E_min  = max(onset, float(rec["energygrid"][0]))
            E_max  = min(onset + window_size, float(rec["energygrid"][-1]))

            _, diel_distance = helper_BSEConv_shared._check_dielectric_convergence(
                prev["imdiel"], rec["imdiel"],
                energy_grid=rec["energygrid"], energy_window=(E_min, E_max),
                method=diel_metric, channels=(0, 1, 2), threshold=100)
            diel_diffs.append(diel_distance)
            meta.append({"energy_window": (E_min, E_max), "onset": onset})
        return ogap_diffs, diel_diffs, meta

    @staticmethod
    def _return_next_or_abort(workchain, abort_condition, next_description, set_next_fn):
        """
        Generic helper: abort with CONVERGENCE_NOT_FOUND if abort_condition is True,
        otherwise call set_next_fn() to advance the control variable and return True.
        """
        if abort_condition:
            workchain.report(workchain.ctx.str_log +
                             f"\n    --> convergence NOT reached and limit exceeded:"
                             f"\n       next would be: {next_description}"
                             f"\n    --> aborting")
            workchain.ctx.control['last_exit_code_thrown'] = workchain.exit_codes.CONVERGENCE_NOT_FOUND
            return False
        set_next_fn()
        workchain.report(workchain.ctx.str_log +
                         f"\n    --> convergence NOT reached - continuing: next = {next_description}\n")
        return True


# ==============================================================================
# Convergence status dataclass  (shared, unchanged)
# ==============================================================================

@dataclass
class ConvergenceStatus:
    use_gap:  bool
    use_diel: bool
    gap_threshold:  float
    diel_threshold: float
    diel_metric: str
    diel_window: float
    factor_for_dynamic_conv: float

    delta_opt:  Optional[float] = None
    delta_diel: Optional[float] = None

    flag_is_optgap_converged:              Optional[bool] = None
    flag_is_diel_converged:                Optional[bool] = None
    flag_is_diel_larger_2_times_threshold: Optional[bool] = None
    flag_is_converged: bool = False

    def finalize(self):
        if self.use_gap and self.use_diel:
            self.flag_is_converged = bool(self.flag_is_optgap_converged) and bool(self.flag_is_diel_converged)
        elif self.use_gap:
            self.flag_is_converged = bool(self.flag_is_optgap_converged)
        elif self.use_diel:
            self.flag_is_converged = bool(self.flag_is_diel_converged)
        else:
            self.flag_is_converged = False
        return self.flag_is_converged


# ==============================================================================
# TEMPLATE BASE CLASS
# ==============================================================================

class VaspmBSEConvergenceTemplateWorkChain(WorkChain):
    """
    Template workchain for BSE convergence studies.

    Handles all convergence monitoring logic identically for all child classes:
        - threshold checking (optical gap + dielectric distance)
        - iteration control (single ctx.control['current_value'] slot, advanced
          exactly once per loop iteration, before it is ever consumed)
        - result elaboration and output

    Child classes implement the following abstract methods, all expressed in terms
    of a single scalar/array "control value" (k-mesh array for the Kpts child,
    energy threshold for the NBands child):

        _increment_value(last_record)
            Given the last successful record, return the next control value.
            Kpts child  : last_record["kmesh"] + step.
            NBands child : last_record["optical_window_threshold"] + step.

        _is_value_over_max(value)
            Return True if `value` exceeds the allowed maximum.

        _value_to_kpoints(value)
            Return the KpointsData to use for the next calculation, given the
            current control value.
            Kpts child  : wraps `value` directly.
            NBands child : ignores `value`, returns the fixed starting mesh.

        _value_to_bse_overrides(value)
            Return a dict of ns_BSE fields to set for the next calculation, given
            the current control value.
            Kpts child  : ignores `value`, returns fixed {NBANDSV, NBANDSO}.
            NBands child : derives {NBANDSV, NBANDSO} from `value` (the threshold)
                          via the band-pair function.

        _extract_record_from_finished_wc(wc)
            Extracts a convergence record (AttributeDict) from a finished child node.
            Must always populate: kmesh, NBANDSV, NBANDSO, optgap, oscstr,
                                  imdiel, energygrid, imdiel_onset.
            Child adds the control variable it sorts by (e.g. optical_window_threshold).

        _record_sort_key(record)
            Returns a comparable key used to sort successful records in ascending order.
            Kpts child  : tuple(record["kmesh"])
            NBands child : record["optical_window_threshold"]
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(VaspmBSEInitScriptWorkChain,
                           exclude=('kpoints', 'ns_reference', 'ns_BSE', 'ns_optimization'))

        # ---- Convergence criteria (identical for all child classes, never overridden) ----
        spec.input('ns_converge.convergence_threshold',       valid_type=Float, required=False,
                   default=lambda: Float(0.35),
                   help="Convergence threshold (eV) applied to both optical gap and dielectric distance.")
        spec.input('ns_converge.dielfunction_convergence',    valid_type=Bool,  required=False,
                   default=lambda: Bool(True),
                   help="Enable convergence check based on imaginary dielectric function.")
        spec.input('ns_converge.opticalgap_convergence',      valid_type=Bool,  required=False,
                   default=lambda: Bool(False),
                   help="Enable convergence check based on optical gap.")
        spec.input('ns_converge.dielfunction_distance',       valid_type=Str,   required=False,
                   default=lambda: Str("L2_distance"),
                   help="Metric for dielectric convergence: 'L2_distance', 'L1_distance', 'Wasserstein'.")
        spec.input('ns_converge.dielfunction_window',         valid_type=Float, required=False,
                   default=lambda: Float(3.5),
                   help="Energy window (eV) above dielectric onset used for convergence evaluation.")
        spec.input('ns_converge.convergence_dynamic_control', valid_type=Bool,  required=False,
                   default=lambda: Bool(False),
                   help="Double step size when delta_diel > 2 * threshold (accelerates slow convergence).")

        # ---- BSE screening parameters (passed through to every child calculation) ----
        spec.input('ns_converge_BSE.static_inverse_diel', valid_type=Float, required=True,
                   help="Required for analytic diagonal screening in mBSE.")
        spec.input('ns_converge_BSE.screening_parameter', valid_type=Float, required=True,
                   help="Required for analytic diagonal screening in mBSE.")
        spec.input('ns_converge_BSE.G0W0_gap',            valid_type=Float, required=False,
                   help="G0W0 gap used to determine the SCISSOR correction.")

        spec.input('ns_reference.starting_RemoteData', valid_type=RemoteData, required=False)

        # ---- Outputs ----
        spec.output('optical_gap', valid_type=Float, required=False)

        # ---- Exit codes ----
        spec.exit_code(300, 'MBSE_CALC_FAILURE',
                       message='A mBSE child calculation failed; aborting convergence loop.')
        spec.exit_code(301, 'CONVERGENCE_NOT_FOUND',
                       message='Convergence not reached; relax threshold or increase the studied range.')
        spec.exit_code(302, 'NOT_IMPLEMENTED',
                       message='The selected feature is not yet implemented.')
        spec.exit_code(303, 'UNSUPPORTED_DIELFUNCTION_METRIC',
                       message="Unsupported metric in ns_converge.dielfunction_distance. "
                               "Allowed: 'L2_distance', 'L1_distance', 'Wasserstein'.")
        spec.exit_code(304, 'NO_CONVERGENCE_REQUESTED',
                       message='Both dielfunction_convergence and opticalgap_convergence are disabled.')
        spec.exit_code(305, 'INVALID_MBSE_CONVERGE_PARAMETERS',
                       message='One or more parameters in ns_converge are invalid (e.g. window < 1 eV).')

        spec.outline(
            cls.initialize,
            while_(cls.monitor_convergence)(
                cls.prepare_run_mBSE,
            ),
            cls.elaborate_results,
        )

    # --------------------------------------------------------------------------
    # initialize
    # --------------------------------------------------------------------------

    def initialize(self):
        """
        Common initialization. Validates shared inputs, then delegates
        control-variable setup to the child via _initialize_convergence_parameter().
        """
        self.ctx.control = AttributeDict()
        self.ctx.control['successful_records']       = []   # elaborated records, sorted
        self.ctx.control['successful_nodes_sorted']  = []   # raw WorkChainNodes, sorted
        self.ctx.control['last_exit_code_thrown']    = None
        self.ctx.WC_MBPT = []

        # ---- Validate shared ns_converge inputs ----
        allowed_metrics = ["L2_distance", "L1_distance", "Wasserstein"]
        if self.inputs.ns_converge.dielfunction_distance.value not in allowed_metrics:
            self.ctx.control['last_exit_code_thrown'] = self.exit_codes.UNSUPPORTED_DIELFUNCTION_METRIC
            return self.ctx.control['last_exit_code_thrown']
        if (not self.inputs.ns_converge.dielfunction_convergence.value
                and not self.inputs.ns_converge.opticalgap_convergence.value):
            self.ctx.control['last_exit_code_thrown'] = self.exit_codes.NO_CONVERGENCE_REQUESTED
            return self.ctx.control['last_exit_code_thrown']
        if self.inputs.ns_converge.dielfunction_window.value < 1:
            self.ctx.control['last_exit_code_thrown'] = self.exit_codes.INVALID_MBSE_CONVERGE_PARAMETERS
            return self.ctx.control['last_exit_code_thrown']

        # ---- Delegate control-variable initialization to child ----
        # Must set, at minimum: ctx.control['initial_value'], ['current_value'], ['step']
        return self._initialize_convergence_parameter()

    # --------------------------------------------------------------------------
    # prepare_run_mBSE
    # --------------------------------------------------------------------------

    def prepare_run_mBSE(self):
        """
        Build and submit the next mBSE child workchain.
        k-mesh and ns_BSE overrides come entirely from ctx.control['current_value'],
        which has already been advanced by _advance_to_next_or_abort() before this
        method runs; this body is therefore identical for all convergence types.
        """
        value = self.ctx.control['current_value']

        self.ctx.inputs_mBSEbase = AttributeDict({
            'ns_parameters':   AttributeDict(),
            'ns_BSE':          AttributeDict(),
            'ns_optimization': AttributeDict(),
        })
        self.ctx.inputs_mBSEbase.update(self.exposed_inputs(VaspmBSEInitScriptWorkChain))
        self.ctx.inputs_mBSEbase.clean_workdir = Bool(False)

        # ---- [A] k-mesh ----
        next_kpoints = self._value_to_kpoints(value)
        self.ctx.inputs_mBSEbase.kpoints = next_kpoints

        # ---- [B] ns_BSE overrides (NBANDSV, NBANDSO, and/or anything else) ----
        bse_overrides = self._value_to_bse_overrides(value)
        for key, val in bse_overrides.items():
            setattr(self.ctx.inputs_mBSEbase.ns_BSE, key, val)

        # ---- [B.1] BSE screening parameters (always passed through) ----
        self.ctx.inputs_mBSEbase.ns_BSE.static_inverse_diel = self.inputs.ns_converge_BSE.static_inverse_diel
        self.ctx.inputs_mBSEbase.ns_BSE.screening_parameter = self.inputs.ns_converge_BSE.screening_parameter
        if 'G0W0_gap' in self.inputs.ns_converge_BSE:
            self.ctx.inputs_mBSEbase.ns_BSE.G0W0_gap = self.inputs.ns_converge_BSE.G0W0_gap

        # ---- [C] IBSE flag: diagonalize only if eigenvalues are needed ----
        if self.inputs.ns_converge.opticalgap_convergence.value:
            self.ctx.inputs_mBSEbase.ns_parameters.ibse = Int(2)
        elif self.inputs.ns_converge.dielfunction_convergence.value:
            self.ctx.inputs_mBSEbase.ns_parameters.ibse = Int(1)
        self.ctx.inputs_mBSEbase.ns_parameters.nbseeig = Int(0)

        # ---- [D] Optimization flags ----
        self.ctx.inputs_mBSEbase.ns_optimization.set_PRECFOCK_to_Fast = Bool(True)
        self.ctx.inputs_mBSEbase.ns_optimization.lreal = Bool(True)

        # ---- [E] POTCAR ----
        self.ctx.inputs_mBSEbase.potential_family  = self.inputs.potential_family
        self.ctx.inputs_mBSEbase.potential_mapping = self.inputs.potential_mapping

        # ---- [F] Label ----
        label = self._get_calculation_label(next_kpoints, bse_overrides)
        self.ctx.inputs_mBSEbase.ns_option.calculation_label = Str(label)

        self.report(f"\n [prepare_run_mBSE] Submitting -> {label}")
        running = self.submit(VaspmBSEInitScriptWorkChain, **self.ctx.inputs_mBSEbase)

        # ---- [G] Provenance: stash the raw control value as a node extra ----
        # so _extract_record_from_finished_wc never has to guess it back from inputs.
        running.base.extras.set('control_value', self._value_to_jsonable(value))

        return ToContext(WC_MBPT=append_(running))

    @staticmethod
    def _value_to_jsonable(value):
        """Convert a control value (numpy array or scalar) into a JSON-serializable extra."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        return float(value)

    # --------------------------------------------------------------------------
    # monitor_convergence  (identical for all child classes)
    # --------------------------------------------------------------------------

    def monitor_convergence(self):
        """
        Evaluate convergence. Uses the same optical-gap and dielectric-distance
        criteria regardless of which parameter is being varied.
        Returns True (continue iterating) or False (stop).
        """
        self.ctx.str_log = ""
        cs = ConvergenceStatus(
            diel_threshold          = float(self.inputs.ns_converge.convergence_threshold.value),
            gap_threshold           = float(self.inputs.ns_converge.convergence_threshold.value),
            use_gap                 = bool(self.inputs.ns_converge.opticalgap_convergence.value),
            use_diel                = bool(self.inputs.ns_converge.dielfunction_convergence.value),
            diel_metric             = str(self.inputs.ns_converge.dielfunction_distance.value),
            diel_window             = float(self.inputs.ns_converge.dielfunction_window.value),
            factor_for_dynamic_conv = 2.0,
        )
        min_calcs_required = 2

        # ---- [1] Hard failure: last child did not finish ok ----
        # NOTE: this method is used as the predicate of `while_(cls.monitor_convergence)`.
        # `ExitCode` is a plain NamedTuple with no custom `__bool__`, so it is always
        # truthy as a 3-tuple - returning it directly here would make the engine treat
        # a hard failure as "keep looping" instead of stopping. Only `elaborate_results`
        # (a normal outline step, not a predicate) may return the ExitCode itself; here
        # we must return a real `False` and let `ctx.control['last_exit_code_thrown']`
        # carry the exit code through to that step.
        if self.ctx.WC_MBPT:
            last_wc = self.ctx.WC_MBPT[-1]
            if last_wc.is_excepted or not last_wc.is_finished_ok:
                self.report(f"\n  > ERROR: last mBSE child failed (pk={last_wc.pk}). Aborting.")
                self.ctx.control['last_exit_code_thrown'] = self.exit_codes.MBSE_CALC_FAILURE
                return False

        # ---- [2] Collect, extract records, sort ----
        successful_nodes = self._collect_and_sort_successful_nodes(self.ctx.WC_MBPT)
        successful_records = [self._extract_record_from_finished_wc(wc) for wc in successful_nodes]
        self.ctx.control['successful_nodes_sorted'] = successful_nodes
        self.ctx.control['successful_records']      = successful_records
        num_ok = len(successful_records)

        # ---- [3] Consecutive differences ----
        opt_diffs, diel_diffs, meta = helper_BSEConv_shared._collect_consecutive_optgap_and_diel_differences(
            records     = successful_records,
            window_size = float(self.inputs.ns_converge.dielfunction_window.value),
            diel_metric = self.inputs.ns_converge.dielfunction_distance.value,
        )
        self.ctx.control['consecutive_optgap_differences'] = opt_diffs
        self.ctx.control['consecutive_diel_distances']     = diel_diffs
        self.ctx.control['consecutive_diel_meta']          = meta

        self.ctx.str_log = ("\n [monitor_convergence]\n"
                            + self._prettyprint_summary(successful_records, opt_diffs, diel_diffs))

        # ---- [4] Early exit: not enough calculations yet ----
        if num_ok < min_calcs_required:
            self.ctx.str_log += (f"\n  > Not enough successful calculations "
                                 f"({num_ok}/{min_calcs_required}) -> launch next.")
            return self._advance_to_next_or_abort(num_ok)

        # ---- [5] Fill convergence status from last consecutive pair ----
        cs.delta_opt  = opt_diffs[-1]
        cs.delta_diel = diel_diffs[-1]

        if cs.use_gap:
            if cs.delta_opt is not None:
                cs.flag_is_optgap_converged = bool(cs.delta_opt < cs.gap_threshold)
                self.ctx.str_log += (f"\n  > [optgap] delta_opt={cs.delta_opt:.4f} eV"
                                     f" (thr={cs.gap_threshold}) -> conv={cs.flag_is_optgap_converged}")
            else:
                self.ctx.str_log += "\n  > [optgap] delta_opt unavailable -> conv=None"
        else:
            self.ctx.str_log += "\n  > [optgap] disabled"

        if cs.use_diel:
            if cs.delta_diel is not None:
                cs.flag_is_diel_converged = bool(cs.delta_diel < cs.diel_threshold)
                cs.flag_is_diel_larger_2_times_threshold = bool(cs.delta_diel > 2 * cs.diel_threshold)
                self.ctx.str_log += (f"\n  > [diel] metric={cs.diel_metric} window={cs.diel_window} eV"
                                     f" delta_diel={cs.delta_diel:.4e} (thr={cs.diel_threshold})"
                                     f" -> conv={cs.flag_is_diel_converged}")
            else:
                self.ctx.str_log += "\n  > [diel] delta_diel unavailable -> conv=None"
        else:
            self.ctx.str_log += "\n  > [diel] disabled"

        final_flag = cs.finalize()
        self.ctx.str_log += (f"\n  > [final] use_gap={cs.use_gap} use_diel={cs.use_diel}"
                             f" -> conv[gap]={cs.flag_is_optgap_converged}"
                             f" conv[diel]={cs.flag_is_diel_converged}"
                             f" -> converged={final_flag}")

        # ---- [6] Converged ----
        if cs.flag_is_converged:
            self.report(self.ctx.str_log + "\n    --> CONVERGED")
            self._store_converged_result()
            return False

        # ---- [7] Not converged: optionally double step, then advance ----
        if (self.inputs.ns_converge.convergence_dynamic_control.value
                and cs.flag_is_diel_larger_2_times_threshold):
            self.ctx.str_log += "\n  > [dynamic-step] Large delta_diel -> doubling step"
            self._apply_dynamic_step()

        return self._advance_to_next_or_abort(num_ok)

    # --------------------------------------------------------------------------
    # elaborate_results
    # --------------------------------------------------------------------------

    def elaborate_results(self):
        if self.ctx.control['last_exit_code_thrown'] is not None:
            self.report(f"Aborting: {self.ctx.control['last_exit_code_thrown']}")
            return self.ctx.control['last_exit_code_thrown']

        # _output_converged_result() returns an ExitCode (CONVERGENCE_NOT_FOUND) on
        # the defensive failure path - this must be returned, not discarded, or the
        # workchain would silently finish "successfully" without the converged output.
        exit_code = self._output_converged_result()
        if exit_code is not None:
            return exit_code

        records = self.ctx.control.get('successful_records', [])
        if records and records[-1].get('optgap') is not None:
            optgap_node = Float(records[-1]['optgap'])
            optgap_node.store()
            self.out('optical_gap', optgap_node)

    # --------------------------------------------------------------------------
    # Internal helper: advance ctx.control['current_value'] (concrete, shared)
    # --------------------------------------------------------------------------

    def _advance_to_next_or_abort(self, num_ok):
        """
        Compute the next control value (from the last successful record, or the
        initial value if there isn't one yet). If it exceeds the child's maximum,
        abort with CONVERGENCE_NOT_FOUND; otherwise commit it to
        ctx.control['current_value'] so prepare_run_mBSE picks it up next.

        Concrete and identical for every child: the only per-child pieces are
        _increment_value() and _is_value_over_max().
        """
        records = self.ctx.control.get('successful_records', [])
        if not records:
            next_value = deepcopy(self.ctx.control['initial_value'])
        else:
            next_value = self._increment_value(records[-1])

        over_max = self._is_value_over_max(next_value)

        def _set():
            self.ctx.control['current_value'] = next_value

        return helper_BSEConv_shared._return_next_or_abort(
            self, over_max, str(next_value), _set)

    # --------------------------------------------------------------------------
    # Internal helper: collect + sort successful nodes (uses child's sort key)
    # --------------------------------------------------------------------------

    def _collect_and_sort_successful_nodes(self, WC_MBPT):
        """
        Filter nodes that have valid dielectric output, extract a record from each
        to get the sort key, then return them sorted ascending by _record_sort_key.
        """
        valid = []
        for idx, wc in enumerate(WC_MBPT):
            has_diel = ('dielectrics' in wc.outputs
                        and wc.outputs.dielectrics is not None
                        and "idiel" in wc.outputs.dielectrics.get_arraynames()
                        and "ediel" in wc.outputs.dielectrics.get_arraynames())
            if not has_diel:
                self.report(f"WARNING: WC_MBPT[{idx}] (pk={wc.pk}) missing dielectric output -> skipping.")
                continue
            valid.append(wc)

        if not valid:
            return []

        # Sort using the child's key, extracted from a lightweight record per node
        valid.sort(key=lambda wc: self._record_sort_key(self._extract_record_from_finished_wc(wc)))
        return valid

    # --------------------------------------------------------------------------
    # Default pretty-print (child may override for a cleaner column layout)
    # --------------------------------------------------------------------------

    def _prettyprint_summary(self, records, opt_diffs, diel_diffs, prefix="  "):
        out = prefix + "[conv-summary]\n" + prefix + "> Completed mBSE nodes:"
        if not records:
            return out + "    (none yet)"
        out += ("\n" + prefix +
                "  idx   kmesh          NBANDSV  NBANDSO  optgap[eV]   delta_opt   delta_diel(prev)")
        out += "\n" + prefix + "  " + "-" * 75
        for i, rec in enumerate(records):
            km  = rec.get("kmesh", [0, 0, 0])
            km_str  = f"[{km[0]},{km[1]},{km[2]}]"
            nbv_str = str(rec.get("NBANDSV", "--"))
            nbo_str = str(rec.get("NBANDSO", "--"))
            gap_str = f"{rec['optgap']:.4f}" if rec.get("optgap") is not None else "--"
            dop     = opt_diffs[i]  if i < len(opt_diffs)  else None
            dd      = diel_diffs[i] if i < len(diel_diffs) else None
            dop_str = "--" if dop is None else f"{dop:.4f}"
            dd_str  = "--" if dd  is None else f"{dd:.4e}"
            out += (f"\n{prefix}  [{i:2d}]  {km_str:<14}  {nbv_str:<8} {nbo_str:<8}"
                    f" {gap_str:<12}  {dop_str:<11}  {dd_str}")
        return out

    # --------------------------------------------------------------------------
    # ABSTRACT METHODS  - must be implemented by every child class
    # --------------------------------------------------------------------------

    def _initialize_convergence_parameter(self):
        """
        Initialize the child's control variable in self.ctx.control.
        Called at the end of initialize(). Must store at minimum:
            ctx.control['initial_value']    - starting control value (array or scalar)
            ctx.control['current_value']    - same as initial_value at this point
            ctx.control['step']             - increment applied by _increment_value
            ctx.control['initial_NBANDSV']  - used for logging/labels
            ctx.control['initial_NBANDSO']  - used for logging/labels
        Returns None on success or an exit code on failure.
        """
        raise NotImplementedError("Subclass must implement _initialize_convergence_parameter")

    def _increment_value(self, last_record):
        """Return the next control value, given the last successful record."""
        raise NotImplementedError("Subclass must implement _increment_value")

    def _is_value_over_max(self, value):
        """Return True if `value` exceeds the child's allowed maximum."""
        raise NotImplementedError("Subclass must implement _is_value_over_max")

    def _value_to_kpoints(self, value):
        """
        Return a KpointsData for the next child calculation, given the current
        control value.

        KptsConv child  : wraps `value` directly as the next k-mesh.
        NBands child    : always returns the fixed initial mesh (ignores `value`).
        """
        raise NotImplementedError("Subclass must implement _value_to_kpoints")

    def _value_to_bse_overrides(self, value):
        """
        Return a dict of ns_BSE fields to set for the next child calculation, given
        the current control value. Must include at minimum 'NBANDSV' and 'NBANDSO'.

        KptsConv child  : ignores `value`, returns fixed {NBANDSV, NBANDSO}.
        NBands child    : derives {NBANDSV, NBANDSO} from `value` (the threshold)
                          via the band-pair function.
        """
        raise NotImplementedError("Subclass must implement _value_to_bse_overrides")

    def _extract_record_from_finished_wc(self, wc):
        """
        Extract a convergence record (AttributeDict) from a finished child WorkChainNode.
        Must always populate:
            rec["kmesh"]        - k-mesh used (list/array of 3 ints)
            rec["NBANDSV"]      - NBANDSV used (int)
            rec["NBANDSO"]      - NBANDSO used (int)
            rec["optgap"]       - optical gap in eV, or None
            rec["oscstr"]       - oscillator strength, or None
            rec["imdiel"]       - imaginary dielectric array (N_energy x 6), or None
            rec["energygrid"]   - energy grid array, or None
            rec["imdiel_onset"] - onset energy, or None
        Child classes add the field(s) used by their _record_sort_key.
        """
        raise NotImplementedError("Subclass must implement _extract_record_from_finished_wc")

    def _record_sort_key(self, record):
        """
        Return a comparable key for sorting successful records in ascending order.

        KptsConv child  : return tuple(record["kmesh"])
        NBands child    : return record["optical_window_threshold"]
        """
        raise NotImplementedError("Subclass must implement _record_sort_key")

    def _store_converged_result(self):
        """
        Store the converged value in self.ctx.control for later output.
        Called when monitor_convergence detects convergence.
        """
        raise NotImplementedError("Subclass must implement _store_converged_result")

    def _output_converged_result(self):
        """
        Emit the converged value as a named workchain output.
        Called in elaborate_results.
        """
        raise NotImplementedError("Subclass must implement _output_converged_result")

    def _apply_dynamic_step(self):
        """
        Double the step size (called when dynamic control is enabled and delta_diel is large).
        Default no-op; subclass overrides if meaningful.
        """
        pass

    def _get_calculation_label(self, kpoints, bse_overrides):
        """Human-readable label for the child calculation. Subclass may override."""
        km = kpoints.get_kpoints_mesh()[0]
        return (f"mBSE km=[{km[0]},{km[1]},{km[2]}]"
                f" NBANDSV={bse_overrides.get('NBANDSV','?')}"
                f" NBANDSO={bse_overrides.get('NBANDSO','?')}")


# ==============================================================================
# CHILD CLASS 1: K-points convergence
#   varies  : k-mesh
#   fixed   : NBANDSV=initial_NBANDSV, NBANDSO=initial_NBANDSO  (both from input, default 2)
#   sort key: tuple(rec["kmesh"])
#   output  : kmesh_converged (KpointsData)
# ==============================================================================

class VaspmBSEKptsConvWorkChain(VaspmBSEConvergenceTemplateWorkChain):
    """
    BSE convergence over k-mesh density.
    NBANDSV and NBANDSO are kept fixed (default 2) throughout.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('ns_kpoints.kmesh.starting_mesh', valid_type=KpointsData, required=True,
                   help="Starting k-mesh for the convergence study.")
        spec.input('ns_kpoints.kmesh.max_mesh',      valid_type=KpointsData, required=True,
                   help="Maximum k-mesh; convergence aborts if exceeded.")
        kpoints_step_default = DataFactory('core.array.kpoints')()
        kpoints_step_default.set_kpoints_mesh([1, 1, 1])
        spec.input('ns_kpoints.kmesh.step', valid_type=KpointsData, required=False,
                   default=lambda: kpoints_step_default,
                   help="k-mesh increment per iteration.")

        # Fixed BSE band counts during k-convergence (cheap proxy subspace)
        spec.input('ns_converge_BSE.NBANDSV', valid_type=Int, required=False,
                   default=lambda: Int(2),
                   help="Fixed NBANDSV used throughout k-point convergence.")
        spec.input('ns_converge_BSE.NBANDSO', valid_type=Int, required=False,
                   default=lambda: Int(2),
                   help="Fixed NBANDSO used throughout k-point convergence.")

        spec.output('kmesh_converged', valid_type=KpointsData, required=True)

    # ---- Abstract method implementations ----

    def _initialize_convergence_parameter(self):
        kmesh_start = np.array(self.inputs.ns_kpoints.kmesh.starting_mesh.get_kpoints_mesh()[0], dtype=int)
        kmesh_max   = np.array(self.inputs.ns_kpoints.kmesh.max_mesh.get_kpoints_mesh()[0],      dtype=int)
        step_vec    = np.array(self.inputs.ns_kpoints.kmesh.step.get_kpoints_mesh()[0],           dtype=int)

        self.ctx.control['initial_value']   = kmesh_start
        self.ctx.control['current_value']   = kmesh_start
        self.ctx.control['step']            = step_vec
        self.ctx.control['max_value']       = kmesh_max
        self.ctx.control['initial_NBANDSV'] = int(self.inputs.ns_converge_BSE.NBANDSV.value)
        self.ctx.control['initial_NBANDSO'] = int(self.inputs.ns_converge_BSE.NBANDSO.value)
        self.ctx.control['kmesh_converged'] = None

        self.report(
            f"\n [VaspmBSEKptsConvWorkChain] Initializing k-mesh convergence"
            f"\n   Starting k-mesh : {kmesh_start}"
            f"\n   Maximum k-mesh  : {kmesh_max}"
            f"\n   Step            : {step_vec}"
            f"\n   Fixed NBANDSV   : {self.ctx.control['initial_NBANDSV']}"
            f"\n   Fixed NBANDSO   : {self.ctx.control['initial_NBANDSO']}"
        )

    def _increment_value(self, last_record):
        return np.array(last_record["kmesh"], dtype=int) + np.array(self.ctx.control['step'], dtype=int)

    def _is_value_over_max(self, value):
        return bool(np.any(np.array(value) > np.array(self.ctx.control['max_value'])))

    def _value_to_kpoints(self, value):
        kpoints = DataFactory('core.array.kpoints')()
        kpoints.set_kpoints_mesh(np.array(value, dtype=int))
        return kpoints

    def _value_to_bse_overrides(self, value):
        """NBANDSV and NBANDSO stay fixed at their initial values."""
        return {
            'NBANDSV': self.ctx.control['initial_NBANDSV'],
            'NBANDSO': self.ctx.control['initial_NBANDSO'],
        }

    def _extract_record_from_finished_wc(self, wc):
        rec = AttributeDict()
        rec["kmesh"]   = wc.inputs.kpoints.get_kpoints_mesh()[0]
        rec["NBANDSV"] = int(wc.inputs.ns_BSE.NBANDSV) if hasattr(wc.inputs.ns_BSE, 'NBANDSV') else None
        rec["NBANDSO"] = int(wc.inputs.ns_BSE.NBANDSO) if hasattr(wc.inputs.ns_BSE, 'NBANDSO') else None

        if 'opticaltransitions' in wc.outputs:
            rec["optgap"], rec["oscstr"] = _extract_opticalgap_fromWorkchainNode(wc)
        else:
            rec["optgap"] = rec["oscstr"] = None

        if 'dielectrics' in wc.outputs:
            rec["imdiel"]     = wc.outputs.dielectrics.get_array("idiel")
            rec["energygrid"] = wc.outputs.dielectrics.get_array("ediel")
            rec["imdiel_onset"] = helper_BSEConv_shared._get_energy_of_diel_onset(
                rec["imdiel"], rec["energygrid"])
        else:
            rec["imdiel"] = rec["energygrid"] = rec["imdiel_onset"] = None
        return rec

    def _record_sort_key(self, record):
        return tuple(record["kmesh"])

    def _store_converged_result(self):
        last_kmesh = np.array(self.ctx.control['successful_records'][-1]["kmesh"], dtype=int)
        node = DataFactory('core.array.kpoints')()
        node.set_kpoints_mesh(last_kmesh)
        self.ctx.control['kmesh_converged'] = node
        self.ctx.str_log += f"\n    --> Converged k-mesh = {last_kmesh}"

    def _output_converged_result(self):
        node = self.ctx.control.get('kmesh_converged')
        if node is not None:
            node.store()
            self.out('kmesh_converged', node)
        else:
            self.report("kmesh_converged not found in ctx!")
            return self.exit_codes.CONVERGENCE_NOT_FOUND

    def _apply_dynamic_step(self):
        self.ctx.control['step'] = np.array(self.ctx.control['step']) * 2

    def _get_calculation_label(self, kpoints, bse_overrides):
        km = kpoints.get_kpoints_mesh()[0]
        return f"mBSE kConv [{km[0]},{km[1]},{km[2]}]"


# ==============================================================================
# CHILD CLASS 2: NBands convergence  (NBANDSV + NBANDSO together)
#   varies  : optical_window_threshold (eV) -> (NBANDSV, NBANDSO) via band-pair function
#   fixed   : k-mesh at initial_kmesh  (should be a cheap/sparse mesh - this study
#             is run BEFORE the converged dense k-mesh from VaspmBSEKptsConvWorkChain
#             is known, and the BSE band subspace is roughly k-mesh-independent)
#   sort key: rec["optical_window_threshold"]
#   output  : nbandsv_converged (Int), nbandso_converged (Int)
# ==============================================================================

class VaspmBSENBandsConvWorkChain(VaspmBSEConvergenceTemplateWorkChain):
    """
    BSE convergence over the BSE band subspace (NBANDSV + NBANDSO together).

    The control variable is optical_window_threshold (eV): at each iteration
    _determine_BSE_parameters() (utils_helpers_mBSE.py) maps the threshold to a
    physically consistent (NBANDSO, NBANDSV) pair that covers all independent-
    particle (IPA) transitions up to that energy. NBANDSV/NBANDSO are therefore
    NEVER incremented directly (e.g. by +1/+1) - they are always re-derived from
    the threshold via that function. Iterating over the threshold is equivalent
    to systematically expanding the BSE subspace in a physically motivated way.

    k-mesh is kept fixed at the value provided via ns_kpoints.kmesh.starting_mesh.
    This should be a cheap/sparse mesh: the converged k-mesh from
    VaspmBSEKptsConvWorkChain is not required (and is normally not yet available
    when this convergence is run), since the BSE band subspace required to cover
    a given IPA transition window does not depend strongly on the k-mesh density.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # Fixed k-mesh for the whole convergence - intentionally a cheap/sparse mesh.
        spec.input('ns_kpoints.kmesh.starting_mesh', valid_type=KpointsData, required=True,
                   help="Fixed, low-density k-mesh used throughout the NBands convergence.")

        # Control variable: optical window threshold iterated over
        spec.input('ns_nbandsconv.threshold_start', valid_type=Float, required=True,
                   help="Starting optical window threshold (eV) for the first calculation.")
        spec.input('ns_nbandsconv.threshold_max',   valid_type=Float, required=True,
                   help="Maximum optical window threshold (eV); convergence aborts if exceeded.")
        spec.input('ns_nbandsconv.threshold_step',  valid_type=Float, required=False,
                   default=lambda: Float(1.0),
                   help="Step size (eV) by which the threshold is incremented each iteration.")

        # Band-pair determination (wraps _determine_BSE_parameters)
        spec.input('ns_nbandsconv.bandsdata', valid_type=BandsData, required=True,
                   help="DFT band structure (with occupations) used to build the IPA transition "
                        "matrix; see utils_helpers_mBSE._determine_BSE_parameters.")
        spec.input('ns_nbandsconv.num_bands_included', valid_type=Int, required=False,
                   default=lambda: Int(20),
                   help="Safety upper bound on valence/conduction bands scanned when "
                        "determining NBANDSV/NBANDSO from the threshold.")

        spec.output('nbandsv_converged', valid_type=Int, required=True)
        spec.output('nbandso_converged', valid_type=Int, required=True)

    # ---- Band-pair determination ----

    def _nbands_pair_from_threshold(self, threshold_eV):
        """
        Return (NBANDSO, NBANDSV) covering all IPA transitions up to threshold_eV,
        using the physically-motivated helper already used elsewhere in the plugin.
        """
        G0W0_gap = (float(self.inputs.ns_converge_BSE.G0W0_gap.value)
                    if 'G0W0_gap' in self.inputs.ns_converge_BSE else None)
        result = _determine_BSE_parameters(
            bandsdata=self.ctx.bandsdata,
            G0W0_gap=G0W0_gap,
            energy_window_goal=float(threshold_eV),
            num_bands_included=self.ctx.control['num_bands_included'],
        )
        self.report(result['log'])
        return int(result['NBANDSO']), int(result['NBANDSV'])

    # ---- Abstract method implementations ----

    def _initialize_convergence_parameter(self):
        kmesh_fixed     = np.array(self.inputs.ns_kpoints.kmesh.starting_mesh.get_kpoints_mesh()[0], dtype=int)
        threshold_start = float(self.inputs.ns_nbandsconv.threshold_start.value)
        threshold_max   = float(self.inputs.ns_nbandsconv.threshold_max.value)
        threshold_step  = float(self.inputs.ns_nbandsconv.threshold_step.value)

        self.ctx.bandsdata = self.inputs.ns_nbandsconv.bandsdata
        self.ctx.control['num_bands_included'] = int(self.inputs.ns_nbandsconv.num_bands_included.value)

        # Derive initial (NBANDSV, NBANDSO) from starting threshold
        nbandso_start, nbandsv_start = self._nbands_pair_from_threshold(threshold_start)

        self.ctx.control['initial_value']   = threshold_start
        self.ctx.control['current_value']   = threshold_start
        self.ctx.control['step']            = threshold_step
        self.ctx.control['max_value']       = threshold_max
        self.ctx.control['initial_kmesh']   = kmesh_fixed
        self.ctx.control['initial_NBANDSV'] = nbandsv_start
        self.ctx.control['initial_NBANDSO'] = nbandso_start
        self.ctx.control['nbandsv_converged'] = None
        self.ctx.control['nbandso_converged'] = None

        self.report(
            f"\n [VaspmBSENBandsConvWorkChain] Initializing NBands convergence"
            f"\n   Fixed k-mesh (low density)  : {kmesh_fixed}"
            f"\n   Threshold start             : {threshold_start} eV"
            f"\n   Threshold max               : {threshold_max} eV"
            f"\n   Threshold step              : {threshold_step} eV"
            f"\n   Initial (NBANDSO, NBANDSV)  : ({nbandso_start}, {nbandsv_start})"
        )

    def _increment_value(self, last_record):
        return float(last_record["optical_window_threshold"]) + float(self.ctx.control['step'])

    def _is_value_over_max(self, value):
        return float(value) > float(self.ctx.control['max_value'])

    def _value_to_kpoints(self, value):
        """k-mesh is always fixed at the initial (low-density) value."""
        kpoints = DataFactory('core.array.kpoints')()
        kpoints.set_kpoints_mesh(deepcopy(self.ctx.control['initial_kmesh']))
        return kpoints

    def _value_to_bse_overrides(self, value):
        """Derive (NBANDSV, NBANDSO) from the current threshold via the band-pair function."""
        nbandso, nbandsv = self._nbands_pair_from_threshold(value)
        return {'NBANDSV': nbandsv, 'NBANDSO': nbandso}

    def _extract_record_from_finished_wc(self, wc):
        rec = AttributeDict()
        rec["kmesh"]   = wc.inputs.kpoints.get_kpoints_mesh()[0]
        rec["NBANDSV"] = int(wc.inputs.ns_BSE.NBANDSV) if hasattr(wc.inputs.ns_BSE, 'NBANDSV') else None
        rec["NBANDSO"] = int(wc.inputs.ns_BSE.NBANDSO) if hasattr(wc.inputs.ns_BSE, 'NBANDSO') else None

        # The threshold used for this child is stashed as a node extra at submit time
        # (see VaspmBSEConvergenceTemplateWorkChain.prepare_run_mBSE) - read it back
        # directly instead of guessing it from inputs.
        rec["optical_window_threshold"] = wc.base.extras.get('control_value', None)

        if 'opticaltransitions' in wc.outputs:
            rec["optgap"], rec["oscstr"] = _extract_opticalgap_fromWorkchainNode(wc)
        else:
            rec["optgap"] = rec["oscstr"] = None

        if 'dielectrics' in wc.outputs:
            rec["imdiel"]     = wc.outputs.dielectrics.get_array("idiel")
            rec["energygrid"] = wc.outputs.dielectrics.get_array("ediel")
            rec["imdiel_onset"] = helper_BSEConv_shared._get_energy_of_diel_onset(
                rec["imdiel"], rec["energygrid"])
        else:
            rec["imdiel"] = rec["energygrid"] = rec["imdiel_onset"] = None
        return rec

    def _record_sort_key(self, record):
        return record["optical_window_threshold"]

    def _store_converged_result(self):
        last = self.ctx.control['successful_records'][-1]
        self.ctx.control['nbandsv_converged'] = last["NBANDSV"]
        self.ctx.control['nbandso_converged'] = last["NBANDSO"]
        self.ctx.str_log += (f"\n    --> Converged NBANDSV={last['NBANDSV']}"
                             f"  NBANDSO={last['NBANDSO']}"
                             f"  (threshold={last['optical_window_threshold']} eV)")

    def _output_converged_result(self):
        nbv = self.ctx.control.get('nbandsv_converged')
        nbo = self.ctx.control.get('nbandso_converged')
        if nbv is not None and nbo is not None:
            nbv_node = Int(nbv)
            nbo_node = Int(nbo)
            nbv_node.store()
            nbo_node.store()
            self.out('nbandsv_converged', nbv_node)
            self.out('nbandso_converged', nbo_node)
        else:
            self.report("nbandsv/nbandso_converged not found in ctx!")
            return self.exit_codes.CONVERGENCE_NOT_FOUND

    def _apply_dynamic_step(self):
        self.ctx.control['step'] = float(self.ctx.control['step']) * 2

    def _get_calculation_label(self, kpoints, bse_overrides):
        thr = self.ctx.control['current_value']
        return (f"mBSE NBandsConv thr={thr:.2f}eV"
                f" NBANDSV={bse_overrides.get('NBANDSV','?')}"
                f" NBANDSO={bse_overrides.get('NBANDSO','?')}")
