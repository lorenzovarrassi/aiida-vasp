# pylint: disable=too-many-arguments

from aiida.orm import Dict

from aiida_vasp.workchains.v2.vasp     import VaspWorkChain
from aiida_vasp.calcs.vasp import VaspCalculation
from aiida.engine import process_handler, ProcessHandlerReport

class VaspGWWorkChain(VaspWorkChain):
    """ Wrapper dedicated to G0W0 calculations. 
    This class is introduced with the SINGLE PURPOSE of adding @process_handler specific to GW calcs.  
    """

    _process_class = VaspCalculation

    @classmethod
    def define(cls, spec):
            super(VaspGWWorkChain, cls).define(spec) 
            
            #spec.expose_inputs( VaspWorkChain  ) 
            #spec.expose_outputs( VaspWorkChain ) 
            
            spec.exit_code(404,'ERROR_EXCEPTED_RETRY_FAILED' , message='Handler handle_gw_exection di not s.')



#VaspCalculation fails with "application called MPI_Abort(MPI_COMM_WORLD, 1)"
#which causes	VaspGWWorkChain   Finished [301]
#               VaspCalculation   Excepted
#The 301 corresponds to ERROR_SUB_PROCESS_EXCEPTED in BaseRestartWorkChain
#From [..]/site-packages/aiida/engine/processes/workchain/restart.py
#class BaseRestartWorkChain(WorkChain):
#   def define(cls, spec: 'ProcessSpec') -> None:  # type: ignore[override]
#        super().define(spec)
#        spec.input('max_iterations', valid_type=orm.Int, [..] )
#        spec.input('clean_workdir', valid_type=orm.Bool, [..] )
#        [..]
#        spec.exit_code(301, 'ERROR_SUB_PROCESS_EXCEPTED', message='The sub process excepted.')
#        spec.exit_code(302, 'ERROR_SUB_PROCESS_KILLED', message='The sub process was killed.')        
#  def inspect_process(self) -> Optional['ExitCode']:
#        node = self.ctx.children[self.ctx.iteration - 1]
#        if node.is_excepted:
#            return self.exit_codes.ERROR_SUB_PROCESS_EXCEPTED
#        if node.is_killed:
#            return self.exit_codes.ERROR_SUB_PROCESS_KILLED

    def inspect_process(self):
        """Override inspect_process to handle excepted calculations, which can happen in G0W0 VASP : 
           In particular regarding cases where a too low OMEGATL causes MPI_ABORT without any error message."""
           
           
        node = self.ctx.children[self.ctx.iteration - 1]
        # Normal cases: use standard mechanism (handlers, restarts, etc.)
        if (not node.is_excepted):
            return super().inspect_process()

        else:
            try: # Determine if this is a GW run
                parameters = node.inputs.parameters.get_dict()
                algo  = str(parameters.get('algo', '')).upper()
                is_gw = algo in ('GW0', 'EVGW0', 'G0W0')
            except Exception as e:
                self.report(f"Error inside overriden inspect_process:{e}")
                is_gw=False
                
            # Not GW? don't change behavior; abort like base workchain would
            if not is_gw:
                return self.exit_codes.ERROR_SUB_PROCESS_EXCEPTED
        
            # GW + excepted: try once
            self.report(f'Detected GW excepted calculation {node.pk}')
        
            if self.ctx.get('gw_excepted_retry_launched', False):
                self.report('[VaspGWWorkChain] Recovery already attempted once; aborting')
                return self.exit_codes.ERROR_EXCEPTED_RETRY_FAILED
        
            self.ctx.gw_excepted_retry_launched = True
        
            report = self.handle_gw_exception(node)
            if report:
                return report.exit_code  # usually 0 -> restart
        
            # even if handler returned None, retry once (your requirement)
            return None
        


    @process_handler
    def handle_gw_exception(self, node):
        """  Escalate GW parameters if the previous GW calculation failed.
        """
        if not node.is_excepted:
            return None

        self.report(
            f'GW VaspCalculation<{node.pk}> excepted; '
            'attempting recovery with OMEGATL=16000'
        )
    
        # count restarts
        self.ctx.gw_iteration = self.ctx.get('gw_iteration', 0) + 1
        #parameters = node.inputs.parameters.get_dict()
        #parameters['omegatl'] = 16000
        #incar = {'incar':parameters}
        #self.ctx.inputs.parameters = Dict(dict=incar)
        
        # Start from current inputs (either from ctx or from the node)
        # inside the try assume self.ctx.input.paramets is a Dict; if it's not, in the except
        # is read as a python dict/AttributeDict
        try:
            incar_dict = self.ctx.inputs.parameters.get_dict()
            incar = incar_dict.get('incar', {})
        except:
            incar = self.ctx.inputs.parameters
        incar['omegatl'] = 16000
        self.ctx.inputs.parameters = incar
            
        self.report(f"[G0W0 restart] iter {self.ctx.gw_iteration}: set OMEGATL={incar['omegatl']}")

        return ProcessHandlerReport() #restart
