# pylint: disable=too-many-arguments

from aiida.orm import Dict
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.engine.processes.workchains.restart import process_handler, ProcessHandlerReport

from aiida_vasp.workchains.v2.vasp import VaspWorkChain


class VaspWorkChainWithResourceFallback(VaspWorkChain):
    """Plain VaspWorkChain (standard VaspCalculation, no script injection) with
    one extra feature: an optional alternate scheduler-options Dict used for
    the single retry `VaspWorkChain.handler_unfinished_calc_generic` already
    grants after an ERROR_DID_NOT_FINISH (exit 700, e.g. OOM/walltime)
    failure - so that one retry can run with larger resources instead of
    repeating the same failure.

    Extracted 2026-08-22 from the old VaspInitScriptWorkChain when the
    script-injection machinery that class also carried (local_init_script /
    local_files_to_copy_to_remote_submission_folder / init_script_call_command,
    backed by the now-deleted Vasp2wInitScriptCalculation) was retired as
    unused dead weight for the mBSE BSE step - this handler was the one part
    of it still needed. See handoff.md for the full story.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("extraresources_fallback_options", valid_type=Dict, required=False, serializer=to_aiida_type,
                   help=("Alternative scheduler options Dict (same shape as 'options': account/qos/"
                         "resources/queue_name/max_memory_kb/max_wallclock_seconds/custom_scheduler_commands), "
                         "used ONLY for the single retry that handler_unfinished_calc_generic grants after an "
                         "ERROR_DID_NOT_FINISH (exit 700) failure - e.g. an OOM kill. If not supplied, the retry "
                         "resubmits with the original options unchanged, exactly as before this input existed."))

    @process_handler(priority=900)
    def handler_unfinished_calc_generic(self, node):
        """
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
