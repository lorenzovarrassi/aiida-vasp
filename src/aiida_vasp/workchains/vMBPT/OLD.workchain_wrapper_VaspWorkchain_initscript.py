# pylint: disable=too-many-arguments

from aiida.orm import SinglefileData

from aiida_vasp.workchains.v2.vasp     import VaspWorkChain
from aiida_vasp.calcs.vasp2wInitScript import Vasp2wInitScriptCalculation



class VaspInitScriptWorkChain(VaspWorkChain):
    """ Simple (and as short as possible ) workchain wrapper of the Vasp2wInitScriptCalculation Calcjob;
        It has on job: forwarding the inputs to the calcjob node
    """
    
    _process_class = Vasp2wInitScriptCalculation  # entry point to the CalcJob above

    @classmethod
    def define(cls, spec):
            super(VaspInitScriptWorkChain, cls).define(spec)        

            spec.expose_inputs( VaspWorkChain ) #parameters contains the INCAR, see 
            spec.input("local_initscript"            , valid_type=SinglefileData , required=False , help="Script copied as 'script_init.py' into the remote sandbox folder." )
            #spec.input("local_files_tocopy_toremote" , valid_type=Dict ,           required=False,  help="Dictionary mapping {remote_filename: SinglefileData}. Each file will be copied into the remote sandbox under the given name.")
        
            spec.input_namespace(
                'local_files_tocopy_toremote',
                valid_type=SinglefileData, dynamic=True,
                help="Files to be copied to the remote sandbox." )
            

    def init_inputs(self):
        # call the parent to build ctx.inputs
        exit_code = super().init_inputs()
        if exit_code is not None:
            return exit_code
        # forward our extra input so the CalcJob sees it
        if 'local_initscript' in self.inputs:
            self.ctx.inputs.local_initscript = self.inputs.local_initscript
        if 'local_files_tocopy_toremote' in self.inputs:
                self.ctx.inputs.local_files_tocopy_toremote = self.inputs.local_files_tocopy_toremote
               
        return None
