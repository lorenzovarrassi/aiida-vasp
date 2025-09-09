# pylint: disable=too-many-arguments

import numpy as np
from copy import deepcopy
from aiida.common.extendeddicts import AttributeDict
from aiida import orm
from aiida.orm import Code, Int, Float, Str, Dict, Bool , List , RemoteData , ArrayData , BandsData , XyData , KpointsData
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import WorkChain, calcfunction , ToContext , append_ , submit, while_
from aiida_vasp.utils.workchains import prepare_process_inputs
import warnings
from sklearn.linear_model import LinearRegression
from aiida_vasp.utils.workchains import site_magnetization_to_magmom


from .utils_calcfunctions import  input_magnetic_moment_tomagmom






# The VaspDFTGWWorkChain can run
# - Single DFT calculations : in this case ns_option.run_G0W0 must be set to False (default).
# - DFT AND G0W0 calculations on top of the DFT, using the same number of bands and encut ; 
#   in this case the G0W0 calculation will use the DFT wavefunctions and energies as starting point.
#   The option is activated with ns_option.run_G0W0=True.
#
# For what regards the inputs:
# - If DFTgr_RemoteData is passed, the DFT calculation will use the WAVEVAR in DFTgr_RemoteData as starting point.
# - A spin-polarized calculations can be selected by using magnetic_moment_onsite.
# - The inputs are categorized in different namespaces:
# 	ns_parameters contains the flag for the INCAR and the KpointsData.
# 	ns_parallelization contains the INCAR flags pertaining the parallelization.
# 	ns_option contains miscellaneous controls.
#
# The main workchain outputs the bands and gaps of the DFT and G0W0 calculations, togetherwith the QuasiParticle corrections.
# The remoteData of the DFT and G0W0 are also returned.
class VaspDFTGWWorkChain(WorkChain):
    _next_workchain = WorkflowFactory( 'vasp.vasp' )

    @classmethod
    def define(cls, spec):
            super(VaspDFTGWWorkChain, cls).define(spec)        

            spec.expose_inputs(cls._next_workchain  , exclude=('kpoints','parameters','settings')) #parameters contains the INCAR, see 

            spec.input('ns_parameters.encut'                  , valid_type=Float      , required=False , help='cutoff energy for the wavefunction in eV. encut variable in VASP.')  #ns stands for namespace
            spec.input('ns_parameters.nbands'                 , valid_type=Int        , required=False , help='total number of bands included in the DFT and G0W0 runs. nbands variable in VASP.'  )   
            #Note magnetic_moment_onsite assumes that the calculation is spin-polarized; Spin-Orbit calculations are currently not supported. 
            #If magnetic_moment_onsite  is not passed, the calculation is instead assumed spin non-polarized.
            spec.input('ns_parameters.magnetic_moment_onsite' , valid_type=Dict       , required=False , help='Starting collinear on-site magnetic moment ; Syntax is {ElName:value}')

            spec.input('ns_parameters.nomega'                 , valid_type=Int        , required=False , default=lambda: Int(200) , help='number of frequency points for the chi and sigma calculation in G0W0 runs. Default is 1 (COHSEX).') 
            spec.input('ns_parameters.nbandsgw'               , valid_type=Int        , required=False , help='number of bands for which QP energies are calculated - nbandsGW variable in VASP')  
            spec.input('ns_parameters.encut_chi'              , valid_type=Float      , required=False , help='cutoff energy for the response function in eV - encutGW variable in VASP') 
            spec.input('kpoints'                              , valid_type=DataFactory('core.array.kpoints') , help='K-mesh used for VASP G0W0 and DFT runs; get_kpoints_mesh() must work.' )     

            spec.input('ns_parallelization.kpar'              , valid_type=Int        , required=False , default=lambda: Int(1)      , help='kpar value to be used in G0W0 calculations')
            spec.input('ns_parallelization.npar'              , valid_type=Int        , required=False , default=lambda: Int(1)      , help='NPAR value to be used in G0W0 calculations')
            spec.input('ns_parallelization.lreal'             , valid_type=Bool       , required=False , default=lambda: Bool(False) , help='lreal value to be used in all calculations. If True sets to Auto, otherwise False') 

            spec.input('ns_reference.DFTgr_RemoteData'        , valid_type=RemoteData , required=False , help='the DFT ground state wavefunction (WAVECAR) and CHGCAR will be copied from this RemoteData folder as a starting point' )
            
            spec.input('ns_option.maximum_iterations'              , valid_type=Int  , required=False , default=lambda: Int(2)      , help='maximum number of times the workchain will restart a crashed G0W0 runs.')
            spec.input('ns_option.verbose'                         , valid_type=Bool , required=False , default=lambda: Bool(True)  )
            spec.input('ns_option.run_G0W0'                        , valid_type=Bool , required=False , default=lambda: Bool(True)  , help='If False, run a single G0W0 calculations; if True, run a DFT and G0W0 ON TOP on it, using same encut and number of bands and the DFT wavefunctions and energies as starting point')
#            spec.input('ns_option.compute_dipole_transition_mat'   , valid_type=Bool , required=False , default=lambda: Bool(False) , help='Compute the DFT dipole matrix elements (LOPTICS flag).')
#            spec.input('ns_option.select_algo_Exact'               , valid_type=Bool , required=False , default=lambda: Bool(False) , help='The DFT step will be run with ALGO=Exact.' )
#            spec.input('ns_option.select_single_iteration'         , valid_type=Bool , required=False , default=lambda: Bool(False) , help='If activated, the DFT step will run a single self-consistency step (nelm=1).' )
            spec.input('ns_option.calculationLabel'                , valid_type=Str  , required=False , default=lambda: Str("")     , help='The summary printed at the end will be labeled with this string.')


            spec.output('RemoteData_G0W0' , valid_type=RemoteData , required=False , help='RemoteData for the DFT calculation node.' )
            spec.output('RemoteData_DFT'  , valid_type=RemoteData , required=False , help='RemoteData for the G0W0 calculation node.')
            spec.output('bands_G0W0'      , valid_type=BandsData  , required=False , help='BandsData for the DFT calculation node.' )
            spec.output('bands_DFT'       , valid_type=BandsData  , required=False , help='BandsData for the G0W0 calculation node.')
           
            spec.output('gaps'            , valid_type=Dict       , required=False , help='Direct and indirect gaps for the DFT and G0W0 nodes.' )
            spec.output('gaps_QPc'        , valid_type=Dict       , required=False , help='QP HOMO correction at direct gap kpt')
            spec.output('QP_corrections'  , valid_type=BandsData  , required=False  ) 
            
            spec.output('NGarray'    , valid_type=ArrayData  , required=False , help='FFT grid used.')
            spec.output('ENMAXarray' , valid_type=ArrayData  , required=False , help='Array containing the ENMAX of all employed POTCARs.')           
            spec.output('kpoints'    , valid_type=DataFactory('core.array.kpoints') , help='The actual k-mesh used for VASP G0W0 and DFT runs' )
            
            #spec.expose_outputs(cls._next_workchain) 

            spec.exit_code(401,'REACHED_MAXIMUM_TRY_NUMBER'  ,message='The workflow reached the maximum number of tries.')
            spec.exit_code(100,'GENERIC_EXIT_CODE'           ,message='generic exit code.')


            spec.outline(
                cls.initialize,
                while_(cls.monitor_WCprogress)(     # Check if the previous iteration of the cycle has run correctly; if not, it tries to change the INCAR to correc the errors.
                    cls.prepare_calc_DFT,           # Prepare the DFT calculation; this calculation must have the same number of bands of the G0W0 one.
                    cls.run_calc,                   # Run it.
                    cls.prepare_calc_G0W0,          # Prepare the G0W0 calculation
                    cls.run_calc,                     
                    ),
                cls.elaborate_results,
                #cls.clean_remoteFolder_DFT,
                )

    def run_calc(self):
            if self.inputs.ns_option.verbose:
                str_log = ('\n [VaspDFTGWWorkChain pk='+str(self.node.pk)+" <"+self.inputs.ns_option.calculationLabel.value 
                + "> iteration="+str(self.ctx.control.iteration_counter)+"][run_calc]"
                + '\n  launching a calc? '+str(self.ctx.WCtoRun) )
                #DEprecATED  #if self.inputs.ns_option.compute_dipole_transition_mat: str_log = str_log + '\n                     The DFT run is the preparatory step to G0W0 (LOPTICS=T , nelm=1='
                str_log = str_log + ('\n  at this step we have already done:'
                + '\n  >> WCrecord_DFT='+str(self.ctx.WCrecord_DFT)
                + '\n  >> WCrecord_G0W0='+str(self.ctx.WCrecord_G0W0)+'\n\n')
            

            # The DFT calculation nodes are appended to WCrecord_DFT ; the G0W0s ones to WCrecord_G0W0;                 #DEprecATED 
            # at each iteration of the cycle while_(cls.monitor_WCprogress) a single DFT calculation node is appended   #DEprecATED
            # and eventually (if no error in the DFT are encountered) a single G0W0 one.                                #DEprecATED
            # The flags WCtoRun['DFT'] and WCtoRun['G0W0'] which determine where to append the node, are set in prepare_calc_DFT and prepare_calc_G0W0  #DEprecATED
            if self.ctx.WCtoRun['DFT']==True and self.ctx.WCtoRun['G0W0']==False  :
                runningWC_DFT = self.submit(self._next_workchain , **self.ctx.inputs_DFT_finalized) 
                self.report('launching DFT workchain{}<{}> '.format(self._next_workchain.__name__, runningWC_DFT.pk))
                return ToContext(WCrecord_DFT=append_(runningWC_DFT))

            if self.ctx.WCtoRun['DFT']==False and self.ctx.WCtoRun['G0W0']==True :
                runningWC_G0W0 = self.submit(self._next_workchain   , **self.ctx.inputs_GW_finalized) 
                self.report('launching G0W0 workchain{}<{}> '.format(self._next_workchain.__name__, runningWC_G0W0.pk))
                return ToContext(WCrecord_G0W0=append_(runningWC_G0W0))

    def initialize(self):
            self.ctx.WCrecord_DFT   = []
            self.ctx.WCrecord_G0W0  = []

            #All self.ctx.control variables are used in the monitor_WCprogress functions
            self.ctx.control = AttributeDict()          
            self.ctx.control.iteration_counter = 0
            self.ctx.control.FINISHED_SUCCESSFULLY      = False            
            self.ctx.control.REACHED_MAXIMUM_TRY_NUMBER = False
            

    def prepare_calc_DFT(self):
            ##[Part 1] Defining self.ctx.inputs for DFT calculations <--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/>
            self.ctx.inputs_DFT = AttributeDict() 
            self.ctx.inputs_DFT.update(self.exposed_inputs(self._next_workchain))

            ##[Part 1][Step 1.0] Folder are not cancelled automatically because we may want to keep WAVECARs, WAVEDERs, WFULLs for following calculation. Folder can be cancelled later
            self.ctx.inputs_DFT.clean_workdir=Bool(False)

            ##[Part 2] Defining kpoints - restart data - settings 
            self.ctx.inputs_DFT.kpoints = self.inputs.kpoints

            if ('DFTgr_RemoteData' in self.inputs['ns_reference']):
                self.ctx.inputs_DFT.restart_folder = self.inputs.ns_reference.DFTgr_RemoteData


            self.ctx.inputs_DFT.settings = AttributeDict({'parser_settings': {'include_node': ['bands','kpoints','structure','NGarray','maximum_number_pw']}})
            
            ##[Part 3][Defining INCAR]
            incar = {'incar': {'ediff':1E-7 , 'algo':"Normal" , 'ismear':0 , 'sigma':0.02 , 'prec':'Accurate' , 'nelm':200 , 'lmaxmix':4}}
            if ('encut'  in self.inputs['ns_parameters']):  incar['incar']['encut']  = self.inputs.ns_parameters.encut
            if ('nbands' in self.inputs['ns_parameters']):  incar['incar']['nbands'] = self.inputs.ns_parameters.nbands
            
            if ('kpar'  in self.inputs['ns_parallelization']): incar['incar']['kpar'] = self.inputs.ns_parallelization.kpar.value
            if ('npar'  in self.inputs['ns_parallelization']): incar['incar']['npar'] = self.inputs.ns_parallelization.npar.value
            if self.inputs.ns_parallelization.lreal == True: incar['incar']['lreal'] = 'Auto'
            else:                                            incar['incar']['lreal'] = '.FALSE.'
    
            # If 'magnetic_moment_onsite' is set, we consider the calculation spin-polarized.
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):          #For some reason hasattr(self.inputs['ns_parameters'], 'magnetic_moment_onsite') does not work, always false.
                _ , incar['incar']['magmom'] = input_magnetic_moment_tomagmom(self.inputs.structure , self.inputs['ns_parameters']['magnetic_moment_onsite'].get_dict())
                incar['incar']['ispin']  = 2
                incar['incar']['icharg'] = 1
                incar['incar']['lorbit'] = 11
                incar['incar']['amix_mag'] = 0.8
                incar['incar']['bmix_mag'] = 0.00001
                incar['incar']['amix'] = 0.2
                incar['incar']['bmix'] = 0.00001
                
            #if self.inputs.ns_option.compute_dipole_transition_mat or self.inputs.ns_option.run_G0W0 :
            if self.inputs.ns_option.run_G0W0 :  
                incar['incar']['loptics'] = '.TRUE.'  
                incar['incar']['algo']    = "Exact"
                incar['incar']['nelm']    = 1      
            #if self.inputs.ns_option.select_algo_Exact == True:               incar['incar']['algo'] = "Exact"
            #if (self.inputs['ns_option']['select_single_iteration'] == True): incar['incar']['nelm'] = 1      
            self.ctx.inputs_DFT.parameters = Dict( incar) #convert to AiiDA format


            # Finalized!
            self.ctx.inputs_DFT_finalized = prepare_process_inputs(self.ctx.inputs_DFT , namespaces=['dynamics','verify'])

            

            # If WCrecord_DFT is empty, we have to run a DFT calc.
            if len(self.ctx.WCrecord_DFT) == 0:
                self.ctx.WCtoRun = {'DFT':True , 'G0W0':False}
            else: 
                #This else means if the DFT run is already done - This case happens where the G0W0 at the previous iteration failed;
                #Thus we avoid to redo also the DFT, use the DFT of the previous iteration as a starting point and do not redo it. 
                WC_PreviousIdentical = None
                for previousWC_idx, previousWC in enumerate(self.ctx.WCrecord_DFT) :
                    try:    previousWC_loptics = previousWC.inputs.parameters.get_dict()['incar']['loptics']
                    except: previousWC_loptics = None
                    try:    presentWC_loptics = incar['incar']['loptics']
                    except: presentWC_loptics = None                     
                    try: # if previous DFT exists, extract final encut / nbands / kpoints / loptics and check if identical; if yes, do not required to relaunch         
                        previousWC_encut   = previousWC.inputs.parameters.get_dict()['incar']['encut']
                        previousWC_kpoints = previousWC.inputs.kpoints.get_kpoints_mesh()            
                        previousWC_nbands  = np.shape(previousWC.outputs.bands.get_bands())[1] # BandsData indexs: [0]=spin components; [1]=represents kpts index, [2]=bands
                        
                        if (previousWC.is_finished_ok              and previousWC_encut  == incar['incar']['encut']  and 
                            previousWC_nbands  == incar['nbands']  and previousWC_loptics == presentWC_loptics          ):
                            WC_PreviousIdentical = self.ctx.WCrecord_DFT[previousWC_idx]  
                    except: pass
                
                    if WC_PreviousIdentical != None:
                        self.ctx.WCtoRun = {'DFT':False , 'G0W0':False}        # No need to redo the DFT calc.
                        self.ctx.WCrecord_DFT.append(WC_PreviousIdentical)     # GW routines will copy WAVECAR and WAVEDER from self.ctx.WCrecord_DFT[-1].outputs.remote_folder                
                        self.report("For this iteration we reuse the DFT of" , WC_PreviousIdentical)
                    else:
                        self.ctx.WCtoRun = {'DFT':True  , 'G0W0':False }       # No luck, we should redo the DFT calculation.
    
    def prepare_calc_G0W0(self):
            if  not self.ctx.WCrecord_DFT[-1].is_finished_ok :        
                self.report("\n DFT node <"+str(self.ctx.WCrecord_DFT[-1])+"> has not finished correctly - not even trying G0W0 for this iteration!")
            else:
                self.ctx.inputs_GW = AttributeDict() 
                self.ctx.inputs_GW.update(self.exposed_inputs(self._next_workchain))

                #[Step 0] Folder are not cancelled automatically because we may want to keep WAVECARs, WAVEDERs, WFULLs for following calculations.
                # Folder can be cancelled later by the clean_ method
                self.ctx.inputs_GW.clean_workdir=Bool(False)
                
                #[Step 1] Define the INCAR
                incar = {'incar': {'nelm':1 , 'algo':'GW0' , 'ismear':0 , 'sigma':0.01 , ##'ispin':1 , 
                                   'nomega':self.inputs.ns_parameters.nomega       , 'kpar':self.inputs.ns_parallelization.kpar  ,
                                   'nmaxfockae':2 , 'prec':'Accurate'}} #nmaxfockae is set to 2 Following Klimes et al, 2014.
                if ('encut'  in self.inputs['ns_parameters']):  incar['incar']['encut']  = self.inputs.ns_parameters.encut
                if ('nbands' in self.inputs['ns_parameters']):  incar['incar']['nbands'] = self.inputs.ns_parameters.nbands            
                else: incar['incar']['nbands'] =  np.shape(self.ctx.WCrecord_DFT[-1].outputs.bands.get_bands())[1]   #In altenrnativa : self.ctx.WCrecord_DFT[-1].outputs.get_dict()['run_status']['nbands']



                if ('encut_chi' in self.inputs['ns_parameters']):  
                    incar['incar']['encutgw']      = self.inputs.ns_parameters.encut_chi                                                            
                    incar['incar']['encutgwsoft']  = self.inputs.ns_parameters.encut_chi                                                            
                
                if ('nbandsgw' in self.inputs['ns_parameters']):       incar['incar']['nbandsgw'] = self.inputs.ns_parameters.nbandsgw               #THIS IS TO-TEST           
                if self.inputs.ns_parallelization.lreal: incar['incar']['lreal'] = 'Auto'                                                   
                else: incar['incar']['lreal'] = '.FALSE.'                                                                                      

                if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):     #if hasattr(self.inputs, 'ns_parameters.magnetic_moment_onsite'):
                    incar['incar']['ispin']  = 2
                    _ , incar['incar']['magmom'] = input_magnetic_moment_tomagmom(self.inputs.structure , self.inputs['ns_parameters']['magnetic_moment_onsite'].get_dict())
                    incar['incar']['lorbit'] = 11
                    incar['incar']['istart'] = 1     
                    incar['incar']['icharg'] = 1     
       
                else: 
                    incar['incar']['ispin'] = 1

                #[Step 1.1] correct the INCAR in following run (if the first has failed)
                #At the first iteration  self.ctx.WCrecord_DFT is empty and thus  self.ctx.WCrecord_DFT[-1] does not possess the attribute is_finished_ok; thus the try-except
                try:
                    if (self.ctx.control.iteration_counter == 1) and (not self.ctx.WCrecord_G0W0[-1].is_finished_ok):
                        incar['incar']['lreal']   = 'Auto'   #in order to reduce Memory constraint
                        incar['incar']['omegatl'] = 8000     #in order to improve stability of the frequency integration
                        #incar['incar']['nmaxfockae'] = 1
                        #incar['incar']['kpar']       = 1   
                    elif (self.ctx.control.iteration_counter >= 2) and (not self.ctx.WCrecord_G0W0[-1].is_finished_ok):
                        incar['incar']['lreal']   = 'Auto'   #in order to reduce Memory constraint
                        incar['incar']['omegatl'] = 16000    #in order to improve stability of the frequency integration
                except:
                    pass
     
                self.ctx.inputs_GW.parameters = Dict( incar ) #convert to AiiDA format

                self.ctx.inputs_GW.kpoints = self.inputs.kpoints

                #[Step 2] Define parser settings: add_maximum_number_pw is not really needed here, we add for completenes.
                self.ctx.inputs_GW.settings = Dict()
                self.ctx.inputs_GW.settings['parser_settings'] = {'include_node': ['bands','kpoints','structure']}
                self.ctx.inputs_GW.settings['ADDITIONAL_REMOTE_COPY_LIST'] = ['WAVEDER'] 

                #[Step 3] G0W0 should use DFT's WAVECAR and WAVEDER as a starting point.
                # restart_folder and fileToIncludeFromRestartFolder are VaspCalculation's inputs (passed through expose_inputs).
                self.ctx.inputs_GW.restart_folder = self.ctx.WCrecord_DFT[-1].outputs.remote_folder

                self.ctx.inputs_GW_finalized = prepare_process_inputs(self.ctx.inputs_GW, namespaces=['dynamics','verify'])
            
            if self.inputs.ns_option.run_G0W0 and self.ctx.WCrecord_DFT[-1].is_finished_ok: 
                  self.ctx.WCtoRun = {'DFT':False  , 'G0W0':True}
            else: self.ctx.WCtoRun = {'DFT':False  , 'G0W0':False}

    def monitor_WCprogress(self):
            self.ctx.control.iteration_counter = self.ctx.control.iteration_counter + 1
           

            if self.inputs.ns_option.verbose:
                str_log=('\n [VaspDFTGWWorkChain pk='+str(self.node.pk)+" <"+self.inputs.ns_option.calculationLabel.value+"> ][monitor_WCprogress at the start of iteration" +str(self.ctx.control.iteration_counter)+"]"
                +'\n  >> monitor_WCprogress: WCrecord_DFT='+str(self.ctx.WCrecord_DFT))
                if (len(self.ctx.WCrecord_DFT) >0) : str_log = str_log + '\n                       '+str(' '.join(["  called by wkc "+str(node.pk)+" :"+str(node.called) for node in self.ctx.WCrecord_DFT]))
                str_log = str_log + '\n  >> monitor_WCprogress: WCrecord_G0W0=' +str(self.ctx.WCrecord_G0W0)
                if (len(self.ctx.WCrecord_G0W0)>0) : str_log = str_log + '\n                       '+str(' '.join(["  called by wkc "+str(node.pk)+" :"+str(node.called) for node in self.ctx.WCrecord_G0W0]))
                str_log = str_log + '\n  >> monitor_WCprogress: evaluating start of cycle iteration no:{}'.format(self.ctx.control.iteration_counter)
               

            try:    lastGW_exitCode  = self.ctx.WCrecord_G0W0[-1].exit_status
            except: lastGW_exitCode  = None
            try:    lastDFT_exitCode = self.ctx.WCrecord_DFT[-1].exit_status
            except: lastDFT_exitCode = None
            
            #Brief synthesis of t
            #[case 1]: first iteration, not calculation is yet done: CONTINUE
            #[case 2]: DFT finished ok and G0W0 is not required:     EXIT CYCLE WITH SUCCESS
            #[case 3]: G0W0 finished ok:                             EXIT CYCLE WITH SUCCESS
            #[case 4]: num of tries exceed maximum number; exit workchain with error.
            #[case 5]: retry.
            if len(self.ctx.WCrecord_DFT) == 0:   
                self.report(str_log+'\n  -> monitor_WCprogress: VaspDFTGWWorkChain started first iteration!'+"\n")
                return True
            elif (not self.inputs.ns_option.run_G0W0) and (lastDFT_exitCode == 0) : 
                self.report(str_log+'\n  -> monitor_WCprogress:VaspDFTGWWorkChain cycle exit - DFT calculation at iteration {} finished successfully, G0W0 is not required!'.format(self.ctx.control.iteration_counter-1)+"\n")
                self.ctx.control.FINISHED_SUCCESSFULLY      = True
                return False
            elif lastGW_exitCode == 0:
                self.report(str_log+'\n  >> monitor_WCprogress: VaspDFTGWWorkChain cycle exit - G0W0 calculations at iteration {} finished successfully!'.format(self.ctx.control.iteration_counter-1)+"\n")
                self.ctx.control.FINISHED_SUCCESSFULLY      = True
            elif self.ctx.control.iteration_counter > self.inputs.ns_option.maximum_iterations +1:
                self.report(str_log+'\n  -> monitor_WCprogress: VaspDFTGWWorkChain EXCEEDED maximum number of iterations!'+"\n")
                self.ctx.control.REACHED_MAXIMUM_TRY_NUMBER = True
                return False
            else:
                self.report(str_log+'\n -> something got wrong in last iteration, continuing!'+"\n")
                return True

    def elaborate_results(self):
        """
        From the output of the DFT and G0W0 nodes the function determines the gaps - QP corrections and reports a brief summary.

        """
        
        def elaborate_single_spin_component(bnd_DFT, bnd_G0W0 , bnd_DFT_occ , bnd_G0W0_occ , str_log):
            """
            The function takes the array representing the DFT and G0W0 bands (and occupations) and computes the various gaps.
            The bands / occupations are taken from the outputs of the calculations nodes.
            The functions elaborates bands associated a single-spin; for spin-polarized calculations, elaborate_single_spin_component
            will be called two times, one for each spin component.
            """
            gap_G0W0_Dir =Float(-1) ; gap_G0W0_Ind = Float(-1) ; gap_G0W0_Gam = Float(-1) ;
            gap_DFT_Dir = Float(-1)  ; gap_DFT_Ind = Float(-1)  ; gap_DFT_Gam = Float(-1) ;   
            bndIdx_HOMOar = [] ; bndIdx_LUMOar = []
        
            occ = bnd_DFT_occ < 0.45    #We consider a band occupied if occupancy is > 0.45
            c_kptNum = np.shape(occ)[0] #c_kptNum represents the total number of k-points in the Irreducible Brillouin Zone.
            for kptIdx in range(c_kptNum): 
                #bndIdx_HOMOar and bndIdx_LUMOar are arrays (dimension = c_kptNum) containing the band indexes of the highest occupied / lowest unoccupied bands at each k-point.
                bndIdx_HOMOar.append( np.where(occ[kptIdx,1:] != occ[kptIdx,:-1] )[0][0]    )
                bndIdx_LUMOar.append( np.where(occ[kptIdx,1:] != occ[kptIdx,:-1] )[0][0] +1 )
            

            bnd = AttributeDict()
            bnd['G0W0_HOMO'] = np.array( [ np.array(bnd_G0W0)[:,:][HOIdx]  for HOIdx in zip(range(c_kptNum ), bndIdx_HOMOar) ] )
            bnd['G0W0_LUMO'] = np.array( [ np.array(bnd_G0W0)[:,:][LUIdx]  for LUIdx in zip(range(c_kptNum ), bndIdx_LUMOar) ] )
            bnd['DFT_HOMO']  = np.array( [ np.array(bnd_DFT)[:,:][HOIdx]   for HOIdx in zip(range(c_kptNum ), bndIdx_HOMOar) ] )
            bnd['DFT_LUMO']  = np.array( [ np.array(bnd_DFT)[:,:][LUIdx]   for LUIdx in zip(range(c_kptNum ), bndIdx_LUMOar) ] )
            bnd['QPc_HOMO'] = bnd['G0W0_HOMO'] - bnd['DFT_HOMO']
            bnd['QPc_LUMO'] = bnd['G0W0_LUMO'] - bnd['DFT_LUMO']                
        
            #Prepare the symmary in the str_log string; this string is not initialized from scratch in this function but took as an argument; the function concatenates (and does not overwrite it) its results.
            #This is useful for spin-polarized calculations: in this case each call of elaborate_single_spin_component will add the results of one of the two spin components.
            try: str_log=str_log+("\n >> [2] input encut , nbands  :"
                                +str(lastNode_G0W0.inputs.parameters.get_dict()['incar']['encut'])+" , "
                                +str(lastNode_G0W0.inputs.parameters.get_dict()['incar']['nbands'])     )
            except:pass
            str_log=str_log+("\n >> [2] output maximum num pw at DFT    : "+str(lastNode_DFT.outputs.maximum_number_pw.get_array()[0] )
                            +"\n >> [2] output nbands (effectively used): "+str(lastNode_G0W0.outputs.misc.get_dict()['run_status']['nbands'])
                            +"\n >> [2] k-mesh used and shift: " +str(self.inputs.kpoints.get_kpoints_mesh() )
                            +"\n >> [3] HOMO GW eigenvalues : "+str(bnd['G0W0_HOMO'])
                            +"\n >> [3] LUMO GW eigenvalues : "+str(bnd['G0W0_LUMO'])
                            +"\n >> [3] HOMO DFT eigenvalues: "+str(bnd['DFT_HOMO'] )
                            +"\n >> [3] LUMO DFT eigenvalues: "+str(bnd['DFT_LUMO'] ))
            
            gap = AttributeDict()
            gap['G0W0_Dir'] =  Float( min(bnd['G0W0_LUMO']  - bnd['G0W0_HOMO'])      )
            gap['G0W0_Ind'] =  Float( min(bnd['G0W0_LUMO']) - max(bnd['G0W0_HOMO'])  )
            gap['G0W0_Gam'] =  Float( (bnd['G0W0_LUMO']     - bnd['G0W0_HOMO'])[0]   )
            gap['DFT_Dir']  =  Float( min(bnd['DFT_LUMO']   - bnd['DFT_HOMO'])       )
            gap['DFT_Ind']  =  Float( min(bnd['DFT_LUMO'])  - max(bnd['DFT_HOMO'])   )
            gap['DFT_Gam']  =  Float( (bnd['DFT_LUMO']     - bnd['DFT_HOMO'])[0]     )
             
            #QPc stands for Quasi-Particle corrections, i.e. QP.energies - DFT.energies
            gap_QPc = AttributeDict()
            gap_QPc['HOMO_Dir'] = Float( bnd['QPc_HOMO'][np.argmin(bnd['G0W0_LUMO'] - bnd['G0W0_HOMO'])] )
            gap_QPc['LUMO_Dir'] = Float( bnd['QPc_LUMO'][np.argmin(bnd['G0W0_LUMO'] - bnd['G0W0_HOMO'])] )
            gap_QPc['HOMO_Gam'] = Float( bnd['QPc_HOMO'][0] )
            gap_QPc['LUMO_Gam'] = Float( bnd['QPc_LUMO'][0] )
            gap_QPc['HOMO_Ind'] = Float( bnd['QPc_HOMO'][np.argmax(bnd['G0W0_HOMO'])] )
            gap_QPc['LUMO_Ind'] = Float( bnd['QPc_LUMO'][np.argmin(bnd['G0W0_LUMO'])] )
            
            str_log=str_log+("\n >> [3] gap_G0W0_Dir"+str(gap['G0W0_Dir'])
                            +"\n >> [3] gap_G0W0_Ind"+str(gap['G0W0_Ind'])
                            +"\n >> [3] gap_G0W0_Gam"+str(gap['G0W0_Gam'])
                            +"\n >> [3] gap_DFT_Dir"+str(gap['DFT_Dir'])
                            +"\n >> [3] gap_DFT_Ind"+str(gap['DFT_Ind'])
                            +"\n >> [3] gap_DFT_Gam"+str(gap['DFT_Gam']) 
                            +"\n >> [4] QPc_HOMO_atDir"+str(gap_QPc['HOMO_Dir'])
                            +"\n >> [4] QPc_HOMO_atInd"+str(gap_QPc['HOMO_Ind'])
                            +"\n >> [4] QPc_HOMO_atGam"+str(gap_QPc['HOMO_Gam'])
                            +"\n >> [4] QPc_LUMO_atDir"+str(gap_QPc['LUMO_Dir'])
                            +"\n >> [4] QPc_LUMO_atInd"+str(gap_QPc['LUMO_Ind'])
                            +"\n >> [4] QPc_LUMO_atGam"+str(gap_QPc['LUMO_Gam'])+"\n" )
 
            ##[Output3] Aiida can append -1 to the bands arrays - We want to retrun the DFT and G0W0 bands without these -1 values in the last band indexes.
            #In order to do this we first find the last band index without -1 entries (of indexes firstBnd_toTrim -1 ) and then we keep only those bands.
            #bnd_DFTvo_toTrim and bnd_G0W0_toTrim contain the -1 (not already trimmed).
            bnd_DFTvo_toTrim = np.array( lastNode_DFT.outputs.bands.get_bands(also_occupations=True)   )
            bnd_G0W0_toTrim  = np.array( lastNode_G0W0.outputs.bands.get_bands(also_occupations=True)  )
            c_bnd_num = np.shape( bnd_G0W0_toTrim[:,:,:] )[2]
            c_kpt_num = np.shape( bnd_G0W0_toTrim[:,:,:] )[1]
            
            #Let's determine which is the lowest band index containing -1
            try:
                firstBnd_toTrim = min( [  np.nonzero(np.in1d(bnd_G0W0_toTrim[0,i,:],[-1,-1]))[0][0]   for i in range(c_kpt_num)] )
            except:
                firstBnd_toTrim = c_bnd_num
             
            #Let's keep the bands up to that indexes.
            bnd_DFTvo_trimmed  = np.zeros( [c_kpt_num , firstBnd_toTrim])
            bnd_G0W0_trimmed   = np.zeros( [c_kpt_num , firstBnd_toTrim])
            bnd_DFTvo_trimmed  = bnd_DFTvo_toTrim[0,:, 0:firstBnd_toTrim]
            occ_DFTvo_trimmed  = bnd_DFTvo_toTrim[1,:, 0:firstBnd_toTrim]        
            bnd_G0W0_trimmed   = bnd_G0W0_toTrim[0,:, 0:firstBnd_toTrim]
            occ_G0W0_trimmed   = bnd_G0W0_toTrim[1,:, 0:firstBnd_toTrim]
            #Now let's save the Numpy arrays to AiiDA variables.
            BandsData = DataFactory('core.array.bands')
            bd_G0W0_trimmed = DataFactory('core.array.bands')()
            bd_G0W0_trimmed.set_kpoints(lastNode_G0W0.outputs.bands.get_kpoints())
            bd_G0W0_trimmed.set_bands(bnd_G0W0_trimmed , occupations=occ_G0W0_trimmed) 

            bd_DFT_trimmed  = DataFactory('core.array.bands')()
            bd_DFT_trimmed.set_kpoints(lastNode_DFT.outputs.bands.get_kpoints())
            bd_DFT_trimmed.set_bands(bnd_DFTvo_trimmed , occupations=occ_DFTvo_trimmed)
        
            bd_difference = bd_G0W0_trimmed.get_bands() - bd_DFT_trimmed.get_bands()
            #bd_difference[abs(bd_difference) < 0.005] = 0   #difference on the 3 digit are probably due to rounding, set to 0.
            bd_QPc_fromTrimmed =  DataFactory('core.array.bands')()
            bd_QPc_fromTrimmed.set_kpoints(lastNode_DFT.outputs.bands.get_kpoints())
            bd_QPc_fromTrimmed.set_bands(bd_difference , occupations=occ_DFTvo_trimmed) 

            #list_finalResults = [bd_QPc_fromTrimmed , bnd , gap , gap_QPc , str_log]
            return [bd_QPc_fromTrimmed , bnd , gap , gap_QPc , str_log]
        
        
        #Check for errors - in case return the associated error code.
        if self.ctx.control.REACHED_MAXIMUM_TRY_NUMBER : return self.exit_codes.REACHED_MAXIMUM_TRY_NUMBER
        
        #And then complete the log in the string. 
        if self.inputs.ns_option.verbose: 
            str_log=('\n [VaspDFTGWWorkChain pk='+str(self.node.pk)+" <"+self.inputs.ns_option.calculationLabel.value+"> ][elaborate_results]"
            +"\n >> [1] self.ctx.control.FINISHED_SUCCESSFULLY       ="+str(self.ctx.control.FINISHED_SUCCESSFULLY     )
            +"\n >> [1] self.ctx.control.REACHED_MAXIMUM_TRY_NUMBER  ="+str(self.ctx.control.REACHED_MAXIMUM_TRY_NUMBER)
            +"\n >> [2] PAW potentials used = "+str(self.inputs.potential_mapping.get_dict())                          )
        
        
        #WCrecord_DFT may contain more than one calculations nodes; this means that one (or more) DFT calculations failed.
        #(if the DFT calculations finishes successfully and only the G0W0 fails, the DFT is not re-run).
        #In this case, use the last calculation in the WCrecord_DFT array.
        if  (len(self.ctx.WCrecord_DFT)>0) and  (self.ctx.WCrecord_DFT[-1].is_finished_ok):        
            lastNode_DFT  = self.ctx.WCrecord_DFT[-1]
            self.out('RemoteData_DFT' , self.ctx.WCrecord_DFT[-1].outputs.remote_folder ) 
            self.out('bands_DFT'      , self.ctx.WCrecord_DFT[-1].outputs.bands         )
            self.out('NGarray'        , self.ctx.WCrecord_DFT[-1].outputs.NGarray       ) 
            self.out('ENMAXarray'     , self.ctx.WCrecord_DFT[-1].outputs.ENMAXarray    )  
            self.out('kpoints'        , self.ctx.WCrecord_DFT[-1].outputs.kpoints       )  
        else: lastNode_DFT = None
        
        
        
        #WCrecord_G0W0 may contain more than one calculations nodes; this means that one (or more) G0W0 calculations failed.
        #WCrecord_G0W0[-1] may contain 0 entry if it was just a DFT run -> in this case lastNode_G0W0=None
        #if it contains more than entry, there are two possible cases:
        #a) either all failed and thus WCrecord_G0W0[-1].is_finished_ok -> also in this case lastNode_G0W0=None
        #or b) the cycle controlled by monitor_WCprogress exited because the last one finished successfully.
        if  (len(self.ctx.WCrecord_G0W0)>0) and  (self.ctx.WCrecord_G0W0[-1].is_finished_ok):        
            lastNode_G0W0 = self.ctx.WCrecord_G0W0[-1]
            self.out('RemoteData_G0W0' , lastNode_G0W0.outputs.remote_folder)
            self.out('bands_G0W0'      , lastNode_G0W0.outputs.bands)
        else: lastNode_G0W0 = None 
  
        if not(lastNode_G0W0 is None):
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):#This means that the calculation is spin-polarized.
                #elaborate_single_spin_component is called once for each spin-component
                bd_QPcorr_spUp , bnd_spUp , gap_spUp , gap_QPc_spUp , str_log_spUp =  elaborate_single_spin_component(
                                                                                    np.expand_dims( lastNode_DFT.outputs.bands.get_array("bands")[0,:,:] , 0) ,
                                                                                    np.expand_dims( lastNode_G0W0.outputs.bands.get_array("bands")[0,:,:] , 0) ,
                                                                                    np.expand_dims( lastNode_DFT.outputs.bands.get_array("occupations")[0,:,:]  , 0) , 
                                                                                    np.expand_dims( lastNode_G0W0.outputs.bands.get_array("occupations")[0,:,:] , 0) , str_log) 
                bd_QPcorr_spDw , bnd_spDw , gap_spDw , gap_QPc_spDw , str_log_spDw =  elaborate_single_spin_component(
                                                                                    np.expand_dims( lastNode_DFT.outputs.bands.get_array("bands")[1,:,:] , 0) ,
                                                                                    np.expand_dims( lastNode_G0W0.outputs.bands.get_array("bands")[1,:,:] , 0) ,
                                                                                    np.expand_dims( lastNode_DFT.outputs.bands.get_array("occupations")[1,:,:]  , 0) , 
                                                                                    np.expand_dims( lastNode_G0W0.outputs.bands.get_array("occupations")[1,:,:] , 0) , str_log) 
                #Let's create an addiation BandsData output variable containing the QuasiParticle Corrections.
                bd_QPcorr = DataFactory('core.array.bands')()
                bd_QPcorr.set_kpoints(lastNode_DFT.outputs.bands.get_kpoints())
                bd_QPcorr.set_bands( np.stack([ bd_QPcorr_spUp.get_array("bands") , bd_QPcorr_spDw.get_array("bands") ]) ,
                                      occupations= np.stack([ bd_QPcorr_spUp.get_array("occupations") , bd_QPcorr_spDw.get_array("occupations") ]) )
                gap     = Dict(dict = {"spinUp":gap_spUp     , "spinDw":gap_spDw })
                gap_QPc = Dict(dict = {"spinUp":gap_QPc_spUp , "spinDw":gap_QPc_spDw }) 
                str_log="\n [1 Spin Component]"+str_log_spUp+"\n [2 Spin Component]"+str_log_spDw     
           
            else:                                                                            
                bd_QPcorr , bnd , gap , gap_QPc , str_log = elaborate_single_spin_component(lastNode_DFT.outputs.bands.get_array("bands")  ,
                                                                                    lastNode_G0W0.outputs.bands.get_array("bands")         ,
                                                                                    lastNode_DFT.outputs.bands.get_array("occupations")    , 
                                                                                    lastNode_G0W0.outputs.bands.get_array("occupations")   , str_log)
                gap = Dict(dict = {"spinUp":gap})
                gap_QPc = Dict(dict = {"spinUp":gap_QPc}) 
            
            self.report(str_log)
                    
            bd_QPcorr.store() ; gap.store() ; gap_QPc.store()
            self.out('gaps' , gap )
            self.out('gaps_QPc' , gap_QPc )
            self.out('QP_corrections' , bd_QPcorr )         
               



    def clean_remoteFolder_DFT(self):
            from aiida import orm
            self.report(orm.CalcJobNode)

            cleaned_calcs = []

            for WC_DFT in self.ctx.WCrecord_DFT:
                for called_descendant_DFT in WC_DFT.called_descendants:
                    if isinstance(called_descendant_DFT, orm.CalcJobNode):
                        self.report(called_descendant_DFT)
                        self.report(called_descendant_DFT.outputs.remote_folder)
                        try:
                            called_descendant_DFT.outputs.remote_folder._clean() # pylint: disable=protected-access
                            cleaned_calcs.append(called_descendant_DFT.pk)
                        except (IOError, OSError, KeyError):
                            pass
