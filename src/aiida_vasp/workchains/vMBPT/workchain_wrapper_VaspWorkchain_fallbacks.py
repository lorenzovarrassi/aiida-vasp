# pylint: disable=too-many-arguments

from aiida.orm import Dict
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.engine import process_handler, ProcessHandlerReport

from aiida_vasp.workchains.v2.vasp import VaspWorkChain


class VaspWorkChainWithFallbacks(VaspWorkChain):
    """Plain VaspWorkChain (standard VaspCalculation) carrying two independent,
    self-gated failure-recovery fallbacks:

      [1] Excepted-GW retry: MPI_ABORT on a GW calc -> retry once with
          OMEGATL=16000.
      [2] Exit-700 resource fallback: ERROR_DID_NOT_FINISH (OOM/walltime) on
          any calc -> retry once with extraresources_fallback_options, if
          supplied.

    Each fallback fires only on its own trigger condition; both are inert
    no-ops otherwise. Safe as the submission class for any VASP calculation,
    not just GW.

    See the DESIGN RATIONALE block above `inspect_process()` for why both
    live in that one method instead of two separate mechanisms.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.exit_code(404, 'ERROR_EXCEPTED_RETRY_FAILED', message='Handler handle_gw_exception did not solve the exception.')
        spec.input("extraresources_fallback_options", valid_type=Dict, required=False, serializer=to_aiida_type,
                   help=("Alternative scheduler options Dict (same shape as 'options': account/qos/"
                         "resources/queue_name/max_memory_kb/max_wallclock_seconds/custom_scheduler_commands), "
                         "used ONLY for the single retry that handler_unfinished_calc_generic grants after an "
                         "ERROR_DID_NOT_FINISH (exit 700) failure - e.g. an OOM kill. If not supplied, the retry "
                         "resubmits with the original options unchanged, exactly as before this input existed."))

    # =====================================================================
    # DESIGN RATIONALE
    # =====================================================================
    #
    # Claim: [1] and [2] are ONE mechanism (inspect_process()), not two.
    #
    # BaseRestartWorkChain.inspect_process() dispatch order, per node:
    #   (a) node.is_excepted  -> abort ERROR_SUB_PROCESS_EXCEPTED, no
    #       handler ever runs.
    #       node.is_killed    -> abort ERROR_SUB_PROCESS_KILLED, same.
    #   (b) otherwise (finished, good or bad exit code) -> walk every
    #       @process_handler on this class by descending `priority`,
    #       stop at the first ProcessHandlerReport(do_break=True).
    #
    #   => @process_handler methods ARE the second half of
    #      inspect_process()'s own body, reached via step (b). They are not
    #      a competing mechanism.
    #
    # Fallback [1] needs an inspect_process() override because step (a)
    # aborts BEFORE any handler runs - an excepted node is structurally
    # invisible to @process_handler. This is a framework constraint, not a
    # style choice: no @process_handler could ever implement [1].
    #
    # Given [1] forces an inspect_process() override, [2] could in
    # principle be folded into it by hand instead of staying a
    # @process_handler. Not done, because:
    #   - super().inspect_process() already IS step (b)'s dispatch loop.
    #     Reimplementing it duplicates that loop for zero benefit.
    #   - VaspWorkChain (v2/vasp.py) registers TWO exit-700 handlers:
    #     handler_unfinished_calc_generic (priority 900, active) and
    #     handler_unfinished_calc_generic_alt (priority 798, disabled).
    #     Hand-rolling risks diverging from that existing pair.
    #   - Any future handler added elsewhere on this class/its parents
    #     keeps working untouched, since step (b) is never bypassed.
    #
    # Composition (why [1] -> [2] chains with no extra code):
    #   attempt 1: excepted, algo=G0W0
    #     -> inspect_process() takes the is_excepted branch
    #     -> OMEGATL=16000 patched in, retry
    #   attempt 2: NOT excepted, exit 700 (OOM)
    #     -> inspect_process() takes the not-excepted branch -> super()
    #     -> super() dispatches to handler_unfinished_calc_generic
    #     -> extraresources_fallback_options swapped in (if supplied), retry
    #   attempt 3: runs with bigger resources.
    #
    #   [1] only ever claims the excepted branch; everything else goes to
    #   the framework's own dispatch, where [2] already lives - no manual
    #   sequencing needed for the chain above to work.
    # =====================================================================

    def inspect_process(self):
        """Override inspect_process to handle excepted calculations, which can happen in G0W0 VASP:
           In particular regarding cases where a too low OMEGATL causes MPI_ABORT without any error message.

           See DESIGN RATIONALE above: mandatory override for the excepted
           case (handlers can't see excepted nodes); exit-700 fallback stays
           a @process_handler, reached via super() below."""

        node = self.ctx.children[self.ctx.iteration - 1]

        # Not excepted (the common case: finished normally, whether with a
        # good or bad exit code) -> hand off entirely to the base class,
        # which runs the normal @process_handler dispatch loop. This is the
        # line that makes handler_unfinished_calc_generic (fallback [2],
        # and anything else registered on this class or its parents) still
        # work exactly as if we'd never touched inspect_process() at all.
        if not node.is_excepted:
            return super().inspect_process()

        # From here on: node.is_excepted is True. No @process_handler could
        # ever have seen this node (see DESIGN RATIONALE) - this is the one
        # case that genuinely requires our own logic instead of the base
        # dispatch loop.
        try:  # Determine if this is a GW run
            parameters = node.inputs.parameters.get_dict()
            algo = str(parameters.get('algo', '')).upper()
            is_gw = algo in ('GW0', 'EVGW0', 'G0W0')
        except Exception as e:
            self.report(f"Error inside overriden inspect_process:{e}")
            is_gw = False

        # Not GW? don't change behavior; abort like base workchain would
        # (this is exactly what BaseRestartWorkChain.inspect_process() would
        # have returned for an excepted node, had we not overridden it).
        if not is_gw:
            return self.exit_codes.ERROR_SUB_PROCESS_EXCEPTED

        # GW + excepted: try once
        self.report(f'Detected GW excepted calculation {node.pk}')

        if self.ctx.get('gw_excepted_retry_launched', False):
            self.report('[VaspWorkChainWithFallbacks] Recovery already attempted once; aborting')
            return self.exit_codes.ERROR_EXCEPTED_RETRY_FAILED

        self.ctx.gw_excepted_retry_launched = True

        report = self.handle_gw_exception(node)
        if report:
            return report.exit_code  # usually 0 -> restart

        # even if handler returned None, retry once
        return None

    @process_handler
    def handle_gw_exception(self, node):
        """Escalate GW parameters if the previous GW calculation failed.

        Called directly as a plain method from inspect_process() above, not
        via the @process_handler dispatch loop (excepted nodes never reach
        it - see DESIGN RATIONALE). Decorator kept for
        introspection/`verdi process report` only.
        """
        if not node.is_excepted:
            return None

        self.report(
            f'GW VaspCalculation<{node.pk}> excepted; '
            'attempting recovery with OMEGATL=16000'
        )

        # count restarts
        self.ctx.gw_iteration = self.ctx.get('gw_iteration', 0) + 1

        # Start from current inputs (either from ctx or from the node)
        # inside the try assume self.ctx.input.parameters is a Dict; if it's not, in the except
        # is read as a python dict/AttributeDict
        try:
            incar_dict = self.ctx.inputs.parameters.get_dict()
            incar = incar_dict.get('incar', {})
        except:
            incar = self.ctx.inputs.parameters
        incar['omegatl'] = 16000
        self.ctx.inputs.parameters = incar

        self.report(f"[G0W0 restart] iter {self.ctx.gw_iteration}: set OMEGATL={incar['omegatl']}")

        return ProcessHandlerReport()  # restart

    @process_handler(priority=900)
    def handler_unfinished_calc_generic(self, node):
        """
        Fallback [2] - exit-700 (OOM/walltime) resource fallback. Reached
        via super().inspect_process() for any non-excepted failure -
        including a retry of a GW calc that previously excepted (see
        DESIGN RATIONALE, "Composition").

        Defers to VaspWorkChain's generic handler for ERROR_DID_NOT_FINISH (exit 700) for all
        of its existing logic (single retry, then abort on a second consecutive failure). The
        only addition: when that handler grants the retry (no exit_code set) and
        `extraresources_fallback_options` was supplied, swap it in for ctx.inputs.metadata['options']
        before the retry - so the one retry attempt runs with larger resources instead of repeating
        the same failure.
        """
        report = super().handler_unfinished_calc_generic(node)
        if report is not None and report.exit_code.status == 0 and 'extraresources_fallback_options' in self.inputs:
            self.report("Calculation did not finish (exit 700) - retrying with extraresources_fallback_options "
                         "(larger scheduler resources) instead of the original options.")
            self.ctx.inputs.metadata['options'] = self.inputs.extraresources_fallback_options.get_dict()
        return report
