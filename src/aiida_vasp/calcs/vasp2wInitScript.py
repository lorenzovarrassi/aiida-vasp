from typing import Any

from aiida import orm
from aiida.common import InputValidationError

# pylint: disable=abstract-method, unreachable, undefined-variable
# explanation: pylint wrongly complains about (aiida) Node not implementing query
from aiida.plugins import DataFactory
from aiida.orm import SinglefileData , Dict

from aiida_vasp.calcs.vasp import VaspCalculation




class Vasp2wInitScriptCalculation(VaspCalculation):
    """General purpose Calculation for using vasp + running an python script in the folder BEFORE launching the actual VaspCalculation.
       While the Calculation supports a generic python script, it has been developed for the interpolation script."""
    """
    Calculation class that extends the standard AiiDA-VASP calculation (VaspCalculation) by allowing the
    user to copy *local* files (stored as SinglefileData nodes) into the *remote* calculation sandbox folder **before** launching VASP.

    AiiDA normally creates the VASP inputs (INCAR, KPOINTS, POTCAR, etc.) .
    - Transfer additional files from a the restart_folder (of RemoteFolder type) is possible via settings['ADDITIONAL_REMOTE_COPY_LIST']
    - Here we want to transfer additional LOCAL file (local meaning on the computer where the AiiDA demon runs) to the remote sandbox.
      In order to do so we use calcinfo.local_copy_list
    - The Additional data transferred is:
        1) a main initialization script       →   copied inside the remote sandbox folder with the filename 'script_init.py'
           (In the mBSE case this is the interpolation script; but it supports a generic python script)
        2) an arbitrary dictionary of files   →   copied under user-defined filenames
                                                  the filename are the dict's keys - while the dict values are the 
                                                  SinglefileData to be copied over
    """

    _default_parser = 'vasp.vasp'

    @classmethod
    def define(cls, spec: Any) -> None:
        super(Vasp2wInitScriptCalculation, cls).define(spec)

        spec.expose_inputs( VaspCalculation ) 
        spec.input("local_initscript"            , valid_type=SinglefileData , required=False , help="Script copied as 'script_init.py' into the remote sandbox folder." )
        #spec.input("local_files_tocopy_toremote" , valid_type=Dict ,           required=False,  help="Dictionary mapping {remote_filename: SinglefileData}. Each file will be copied into the remote sandbox under the given name.")
        # Dynamic namespace: each key is the REMOTE filename, each value is a SinglefileData
        spec.input_namespace(
            "local_files_tocopy_toremote",
            valid_type=SinglefileData,
            dynamic=True,
            help="Mapping from remote filename -> SinglefileData. Each file will be copied into the remote sandbox under the given name."
        )    



    def prepare_for_submission(self, folder: Any) -> Any:
        """Override the method such that we can add the flag that executes Wannier90 in library mode."""
       
        calcinfo = super().prepare_for_submission(folder)
        local_copy_list = []

        #Definition: local computer is the computer where the workchain is submitted and the aiida demon resides
        #            remote sandbox folder is folder that AiiDA-VASP creates on the remote Cluster to run the calculations
        #[1] copy local_initscript from local computer → script_init.py in remote sandbox folder
        if "local_initscript" in self.inputs:
            local_copy_list.append( (self.inputs.local_initscript.uuid, self.inputs.local_initscript.filename, "script_init.py") )

        #[2]
        if "local_files_tocopy_toremote" in self.inputs:
            #mapping = self.inputs.local_files_tocopy_toremote.get_dict()
            #mapping as a structure has:
            # self.inputs.local_files_tocopy_toremote is a namespace-like mapping: 
            #    {'filename_inremote_1': <SinglefileData: uuid: 7a4b967c-fd34-4ac2-8e22-5aeb83295ebd (unstored)>,
            #     'filename_inremote_2': <SinglefileData: uuid: c2b4df06-1936-4473-bdc4-a0feb0d08ba7 (unstored)>}

            #for remote_filename, node_singlefileData in mapping.items():
            #    node_singlefileData = mapping[remote_filename]
            #    if not isinstance(node_singlefileData, SinglefileData):
            #        raise InputValidationError( f"local_files['{remote_filename}'] must be a SinglefileData" )
            #    local_copy_list.append( (node_singlefileData.uuid, node_singlefileData.filename, remote_filename) )
            for remote_filename, node_singlefile in self.inputs.local_files_tocopy_toremote.items():
                if not isinstance(node_singlefile, SinglefileData):
                    raise InputValidationError(f"Input 'local_files_tocopy_toremote.{remote_filename}' must be a SinglefileData, got {type(node_singlefile)}"                     )

                local_copy_list.append( (node_singlefile.uuid, node_singlefile.filename, remote_filename)          )

        calcinfo.local_copy_list = local_copy_list
        return calcinfo



