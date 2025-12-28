# pylint: disable=too-many-arguments

from aiida.orm import SinglefileData, Str, Dict

from aiida_vasp.workchains.v2.vasp     import VaspWorkChain
from aiida_vasp.calcs.vasp2wInitScript import Vasp2wInitScriptCalculation



class VaspInitScriptWorkChain(VaspWorkChain):    
    """VaspInitScriptWorkChain (wrapper) — why this exists and how to use it
    ---------------------------------------------------------------------
    Simple (and as short as possible ) workchain wrapper of the Vasp2wInitScriptCalculation Calcjob;
    it reuses the standard aiida-vasp VaspWorkChain machinery (resource handling, submission, parsing, error handling), 
    but swaps the underlying CalcJob to `Vasp2wInitScriptCalculation`, which supports staging *extra local files* into 
    the remote sandbox before VASP starts:
      (A) VaspWorkChain
          - standard aiida-vasp workflow driver that ultimately launches a CalcJob (_process_class)
          - prepares ctx.inputs (code/structure/kpoints/options/settings/parameters/...)
      (B) Vasp2wInitScriptCalculation  (our CalcJob)
          - extends VaspCalculation by populating `calcinfo.local_copy_list`
          - this enables copying from the *daemon machine* (local) -> *remote sandbox folder*
            before the scheduler job runs
    
    This workchain therefore extends the standard VaspWorkChain by two way:
      (A) : Allow to copy an arbitrary number of file the *daemon machine* (local) -> *remote sandbox folder*
            where the daemon will submit the job
      (B) : Add a generic prolog script before the VASP execution
          - `options.prepend_text` is executed on the remote compute node *before* the VASP executable
          - we use it to run a Python initialization script (typically: GW interpolation) that must happen
            in the same working directory as VASP BEFORE the VASP execution
    
    This wrapper provides three extra inputs:
      1) local_init_script : SinglefileData (optional)
         - staged into the remote sandbox as a fixed filename: 'script_init.py'
         - convention: downstream call uses 'python3 script_init.py ...'
      2) local_files_to_copy_to_remote_submission_folder : dynamic namespace (optional)
         - mapping: remote_filename -> SinglefileData
         - each file is copied into the remote sandbox under its key name
         - typical use: stage reference GW outputs (e.g. OUTCAR.3) so the init script can read them
      3) init_script_call_command : Str (optional)
         - a shell command block that should run *before VASP starts*
         - we do NOT hardcode it into the base workchain anymore; instead, this wrapper merges it into
           `options.prepend_text` in init_inputs().
         - benefit: centralizes “how do we inject pre-run commands” in one place, so higher-level workflows
           only specify *what* to run, not *how to edit prepend_text*.
    
    Implementation notes:
      - init_inputs() first calls super().init_inputs() to build the default ctx.inputs.
      - then it forwards our extra inputs into ctx.inputs using the names expected by the CalcJob
        (`local_initscript`, `local_files_tocopy_toremote`).
      - finally, if init_script_call_command is provided, it appends it to any existing options.prepend_text
        (preserving user-provided prepend_text such as modules/environment setup).
    """  
    
    
    _process_class = Vasp2wInitScriptCalculation  # entry point to the CalcJob above

    @classmethod
    def define(cls, spec):
            super(VaspInitScriptWorkChain, cls).define(spec)        

            spec.expose_inputs( VaspWorkChain ) #parameters contains the INCAR, see 
            spec.input("local_init_script"       , valid_type=SinglefileData , required=False , help="Script copied as 'script_init.py' into the remote sandbox folder." )
        
            spec.input_namespace('local_files_to_copy_to_remote_submission_folder',
                                 valid_type=SinglefileData, dynamic=True,
                                 help="Files to be copied to the remote sandbox." )
            spec.input("init_script_call_command", valid_type=Str, required=False,help=("Shell command(s) appended to options.prepend_text in the job script. "
                                                                                        "Use this to run 'script_init.py' (or other init actions) BEFORE VASP starts. "
                                                                                        "Example: 'source activate aiida-vasp\\npython3 script_init.py --help'" ), )


    def init_inputs(self):
        #[1]All the parent to build ctx.inputs
        exit_code = super().init_inputs()
        if exit_code is not None:
            return exit_code

        #[2]Forward our extra input so the CalcJob sees it
        if 'local_init_script' in self.inputs:
            self.ctx.inputs.local_initscript = self.inputs.local_init_script
        if 'local_files_to_copy_to_remote_submission_folder' in self.inputs:
                self.ctx.inputs.local_files_tocopy_toremote = self.inputs.local_files_to_copy_to_remote_submission_folder
               
        
        #[3]Merge init_script_call_command into options.prepend_text ---
        cmd = ""
        if "init_script_call_command" in self.inputs:
            cmd = (self.inputs.init_script_call_command.value or "").strip()

        if cmd:
            #[3.1] Extract existing prepend_text
            # options is expected to be an AiiDA Dict
            if "options" in self.ctx.inputs.metadata :
                options_dict = self.ctx.inputs.metadata['options']
            else:
                options_dict = {}
            existing_prepend_text = (options_dict.get("prepend_text", "") or "").strip()

            #[3.2] Join safely with exactly one newline between blocks
            merged_prepend_text = "\n".join([x for x in [existing_prepend_text, cmd] if x])

            #[3.3]
            self.ctx.inputs.metadata['options']["prepend_text"] = merged_prepend_text
        return None        
