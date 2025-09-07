"""

"""

from typing import Any

from aiida import orm

# pylint: disable=abstract-method, unreachable, undefined-variable
# explanation: pylint wrongly complains about (aiida) Node not implementing query
from aiida.plugins import DataFactory
from aiida.orm import SinglefileData

from aiida_vasp.calcs.vasp import VaspCalculation




class Vasp2wInitScriptCalculation(VaspCalculation):
    """TEST"""

    _default_parser = 'vasp.vasp'

    @classmethod
    def define(cls, spec: Any) -> None:
        super(Vasp2wInitScriptCalculation, cls).define(spec)

        spec.expose_inputs( VaspCalculation ) 
        spec.input("local_initscript" , valid_type=SinglefileData , required=False )

    def prepare_for_submission(self, folder: Any) -> None:
        """Override the method such that we can add the flag that executes Wannier90 in library mode."""
       
        calcinfo = super().prepare_for_submission(folder)

        local_copy_list = []
        local_copy_list.append((self.inputs.local_initscript.uuid, self.inputs.local_initscript.filename, "script_init.py"))
        calcinfo.local_copy_list = local_copy_list

        return calcinfo



