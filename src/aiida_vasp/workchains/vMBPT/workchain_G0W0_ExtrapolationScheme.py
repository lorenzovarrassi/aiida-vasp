# pylint: disable=too-many-arguments

import math
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


from .utils_calcfunctions import get_closest_EncutNband_multiple , get_EncutNbandFitParams_completeBasis_quadratic , input_magnetic_moment_tomagmom
from .workchain_base import VaspDFTGWWorkChain



# The workchain determines the quasiparticle energies (and gaps) extrapolated to infinity basis-set.
# Up to Four different G0W0 calculations are launched (via the VaspDFTGWWorkChain workchain), with different number of bands included.
# The cutoffs and number of bands of these calculations are constrained by the complete basis hypothesis, 
# that is by including all orbitals that the plane-wave basis set allows to calculate.
#
class VaspG0W0BasisExtrWorkChain(WorkChain):
    _next_workchain_string = 'vasp.vasp'
    _next_workchain = WorkflowFactory(_next_workchain_string)

    @classmethod
    def define(cls, spec):
            super(VaspG0W0BasisExtrWorkChain, cls).define(spec)        

            spec.expose_inputs(cls._next_workchain , exclude=('kpoints','parameters', )) 
            spec.expose_inputs(VaspDFTGWWorkChain  , exclude=('kpoints','parameters', 'ns_parameters' , 'ns_continuationJob' , 'ns_option')) 


            spec.input('ns_extrapolation.mode'                  , valid_type=Str       , required=False , default=lambda: Str("final") , help='either final , memory-conserving , standard, custom.') 
            spec.input('ns_extrapolation.encut_chi_low'         , valid_type=Bool      , required=False , default=lambda: Bool(False)  , help='if not specified, ENCUTGW defined as 0.63 x ENCUT ; - if true ENCUTGW = 0.50 x ENCUT.' )
            spec.input('ns_extrapolation.nbands_stride'         , valid_type=Int       , required=False , help='minimum nbands steps used to increase the number of bands in the fit for the final (an other) modes')
            spec.input('ns_extrapolation.cutoff_fractions'      , valid_type=ArrayData , required=False , help='specify the fractions of the cutoff of the first calculation to be used for the extrapolation; it overrides the standard mode')
            spec.input('ns_extrapolation.cutoff_starting_value' , valid_type=Int       , required=False , help='specify to value of cutoff of the first calculation for the extrapolation; it overrides standard value, which is determined from DFTgr'  )   
            spec.input('ns_extrapolation.r2_threshold'          , valid_type=Float     , required=False , default=lambda: Float(0.85) , help='R2 threshold for the extrapolation; if the R2 of the fit is below this value for at least one of the gaps or QP corrections, an additional G0W0 calculation is performed (up to num_calc_touse_for_extrapolation) and the last three calculations are used for the extrapolation (i.e. a moving window of 3 calculations). Default is 0.85')
            
            spec.input('ns_parameters.magnetic_moment_onsite' , valid_type=Dict       , required=False , help='Starting collinear on-site magnetic moment ; Syntax is {ElName:value}')
            spec.input('ns_parameters.nomega'                 , valid_type=Int        , required=False , default=lambda: Int(96) , help='number of frequency points for the chi and sigma calculation in G0W0 runs. Default is 96.') 
            spec.input('ns_parameters.encut_chi'              , valid_type=Float      , required=False , help='cutoff energy for the response function in eV - ENCUTGW variable in VASP') 

            spec.input('ns_parallelization.kpar'              , valid_type=Int        , required=False , default=lambda: Int(1)      , help='kpar value to be used in G0W0 calculations')

            spec.input('kpoints'                              , valid_type=DataFactory('core.array.kpoints') , help='K-mesh used for VASP G0W0 and DFT runs; get_kpoints_mesh() must work.' )     

            spec.input('ns_reference.DFTgr_RemoteData'        , valid_type=RemoteData , help='Folder of the starting-point DFT ground-state; the workflows copies the starting point WAVECAR and CHGCAR from it; the CHGCAR is used only for magnetic collinear calcs.')
            spec.input('ns_reference.DFTgr_NGarray'           , valid_type=ArrayData  , help='FFT grid used to determine the complete basis compatible with a given cutoff.')
            spec.input('ns_reference.DFTgr_ENMAXarray'        , valid_type=ArrayData  , help='Array containing the ENMAX of all employed POTCARs.') 
            spec.input('ns_reference.DFTgr_kpoints'	          , valid_type=DataFactory('core.array.kpoints') , help='k-points used to determine the complete basis compatible with a given cutoff. Must contain explicit mesh : get_kpoints() must work.' )
            
            spec.input('ns_option.GW_max_iteration' ,  valid_type=Int  , required=False , default=lambda: Int(1) , help='maximum number of times the workchain will restart a crashed G0W0 runs' ) 
            spec.input('ns_option.verbose'          ,  valid_type=Bool , required=False , default=lambda: Bool(True))      


            spec.output('ENMAX_referenceValue'       , valid_type=Float     , help='reference cutoff values used for determining the cutoff fractions in the extrapolation')
            spec.output('ENMAX_array'                , valid_type=ArrayData , help='array of cutoff fractions values used for the extrapolation')
            spec.output('pairs_nbands_encuts'        , valid_type=XyData    , help='nbands - cutoff (eV) pairs used in the extrapolation')
                        
            spec.output('extrapolated', valid_type=Dict , help='extrapolated gaps and QuasiParticle corrections.') 
            spec.output('extrapolated_bands', valid_type=Dict , help='extrapolated bands.') 
            spec.output('ns_maximum_num_calcgaps'     , valid_type=Dict , help='Dict with direct - indirect - Gamma gaps used for the extrapolation.')
            spec.output('ns_QPc'      , valid_type=Dict , help='Dict with the QP corrections (of the states associated to the direct - indirect - gamma gaps) used for the extrapolation.')
            spec.output('ns_gaps'     , valid_type=Dict , help='Dict with the (direct - indirect - gamma) gaps used for the extrapolation.')


            spec.exit_code(402,'ONE_OR_MORE_GW_FAILED',message='One or more GW calculations failed.')
            spec.exit_code(403,'NON_EXISTENT_MODE'    ,message='The inserted mode does not exist - Please choose between custom , final , standard , memory-conserving.')

      
      #Original and correct
            spec.outline(
                cls.initialize,
                
                #Prepares the inputs of the several G0W0 data points which will be used for the extropolation.
                #the cutoffs - number of bands of these points are chosen in order to satisfy the complete basis constraint.
                cls.determine_completeBasis_encutNband,
                
                #Submit in parallel the DFT+G0W0 workflows.
                cls.prepare_run_wc_DFT_G0W0_firstThreeG0W0s,

                #If the extrapolation from the first 3 G0W0s show a R2 value below the threshold (0.85) 
                #for at least one of the gaps or QP corrections, run additional G0W0 calculations (up to num_calc_touse_for_extrapolation)
                #and use the last three calculations for the extrapolation (i.e. a moving window of 3 calculations) 
                while_(cls.are_r2_under_threshold)(
                    cls.prepare_run_wc_DFT_G0W0_additionalG0W0s,
                ),

                cls.elaborate_extrapolate_results,
                )

  
    def initialize(self):
        self.report('\n Started Basis-Extrapolation procedure with potentials: '+str(self.inputs.potential_mapping.get_dict())+"\n")
        
        self.ctx.finishedWC_DFTgr     = []
        self.ctx.finishedWC_DFT_G0W0  = []
        

    def prepare_run_wc_DFT_G0W0_firstThreeG0W0s(self): 
            """
            The workchain launches 3 (the precise value is defined by self.ctx.num_calc_used_for_extrapolation) istances of VaspDFTGWWorkChain (each represent a complete G0W0 data point). 
            This function prepares the correct input for each call of VaspDFTGWWorkChain.
            """
            #Preparing inputs for each workchain
            self.ctx.inputs_array = []
            self.ctx.runningWC_DFT_G0W0 = {}

            #We prepare the inputs for each call of VaspDFTGWWorkChain, which differ only for the encut and nbands values (and the encut_chi value, which is a fraction of encut)
            #  All inputs, as declared in the define of VaspDFTGWWorkChain, are saved inside the list self.ctx.inputs_array
            #  The idea is that the first entry in inputs_array corresponds to the first call of VaspDFTGWWorkChain AND the FIRST G0W0 data point for the extrapolation.
            #We prepare the inputs for all G0W0 data points which could be used, equal to self.ctx.max_num_runnable_G0W0_calcs
            #  However, only the first self.ctx.num_calc_used_for_extrapolation will be actually launched in this function.
            #  The other calculations will be launched only if the R2 of the extrapolation is below the threshold for at least one of the gaps.
            for ecutNbIdx in range( self.ctx.max_num_runnable_G0W0_calcs ): 
                self.ctx.inputs_array.append( AttributeDict() )
                self.ctx.inputs_array[ecutNbIdx].update(self.exposed_inputs(VaspDFTGWWorkChain))
                self.ctx.inputs_array[ecutNbIdx].clean_workdir    = Bool(False)
                #self.ctx.finishedWC_DFTgr_NSP[-1].outputs.kpoints does not work for VASP G0W0s: it doesn't use VASP automatic generation but define manually the points inside KPOINTS - may give error in screened_2e.F -> use inputs values, which employs VASP automatic generation
                #self.ctx.finishedWC_DFTgr_NSP[-1].outputs.kpoints does not work for determine_completeBasis_encutNband: it requires explicit k-mesh -> in 
                self.ctx.inputs_array[ecutNbIdx]['kpoints'] = self.inputs.kpoints
                self.ctx.inputs_array[ecutNbIdx]['ns_parameters']       = AttributeDict() 
                self.ctx.inputs_array[ecutNbIdx]['ns_parallelization']  = AttributeDict() 
                self.ctx.inputs_array[ecutNbIdx]['ns_option']           = AttributeDict()
                self.ctx.inputs_array[ecutNbIdx]['ns_reference']        = AttributeDict()
                
                #This part is different among the different entry of inputs_array - because each entry has a different encut - bnads
                self.ctx.inputs_array[ecutNbIdx].ns_parameters.nbands   = Int(   self.ctx.EncutNbands_completeBasis[ecutNbIdx]["nbands"])
                self.ctx.inputs_array[ecutNbIdx].ns_parameters.encut    = Float( self.ctx.EncutNbands_completeBasis[ecutNbIdx]["encut"] )
                
                #These parameters are instead identical between the different calls of VaspDFTGWWorkChain ; same NOMEGA, same MAGMOM
                self.ctx.inputs_array[ecutNbIdx].ns_parameters.nomega   = self.inputs.ns_parameters.nomega
                if ('magnetic_moment_onsite' in self.inputs['ns_parameters'] ): 
                    self.ctx.inputs_array[ecutNbIdx].ns_parameters.magnetic_moment_onsite   = self.inputs.ns_parameters.magnetic_moment_onsite
                if ('nbandsgw' in self.inputs['ns_parameters'] ):   self.ctx.inputs_array[ecutNbIdx].ns_parameters.nbandsgw = self.inputs.ns_parameters.nbandsgw

                #Same paralleization options
                self.ctx.inputs_array[ecutNbIdx].ns_parallelization.kpar  = self.inputs.ns_parallelization.kpar
                self.ctx.inputs_array[ecutNbIdx].ns_parallelization.lreal = Bool(self.inputs.ns_parallelization.lreal)
                
                #Same DFT ground state as starting point
                self.ctx.inputs_array[ecutNbIdx].ns_reference.DFTgr_RemoteData = self.inputs.ns_reference.DFTgr_RemoteData        #self.ctx.finishedWC_DFTgr[-1].outputs.remote_folder

                self.ctx.inputs_array[ecutNbIdx].ns_option.maximum_iterations  = self.inputs.ns_option.GW_max_iteration
                self.ctx.inputs_array[ecutNbIdx].ns_option.verbose             = self.inputs.ns_option.verbose
                self.ctx.inputs_array[ecutNbIdx].ns_option.calculationLabel    = Str("extrapolation point "+str(ecutNbIdx) )               
                self.ctx.inputs_array[ecutNbIdx].ns_option.run_G0W0            = Bool(True) 

                #now let's manage the ENCUTGW terms:
                #By default they are kp at 0.63*ENCUT - if the encut_chi_low flag is activated, just 0.50
                self.ctx.inputs_array[ecutNbIdx].ns_parameters.encut_chi = Float( self.ctx.EncutNbands_completeBasis[ecutNbIdx]["encut"] * 0.63 )
                if self.inputs.ns_extrapolation.encut_chi_low.value == True :
                        self.ctx.inputs_array[ecutNbIdx].ns_parameters.encut_chi = Float( self.ctx.EncutNbands_completeBasis[ecutNbIdx]["encut"] * 0.50 )

            for ecutNbIdx in range( self.ctx.num_calc_touse_for_extrapolation ): 
                #Actually launching each workchain. The workchain are launched in parallel
                #See for details https://aiida.readthedocs.io/projects/aiida-core/en/v2.0.1/topics/workflows/usage.html
                #for ecutNbIdx in range(len(self.ctx.EncutNbands_completeBasis)): 
                self.ctx.runningWC_DFT_G0W0[ecutNbIdx] =  self.submit(VaspDFTGWWorkChain   , **self.ctx.inputs_array[ecutNbIdx]) 
                key = f'WC_DFT_G0W0_{ecutNbIdx}'
                self.to_context(**{key: self.ctx.runningWC_DFT_G0W0[ecutNbIdx]})

                if self.inputs.ns_option.verbose:
                    self.report("\n  [preparing the VaspDFTGWWorkChain inputs]"
                    +"\n  > encut/nbands pair index ecutNbIdx:"+str(ecutNbIdx)
                    +"\n  > self.ctx.EncutNbands_completeBasis[ecutNbIdx]"+str(self.ctx.EncutNbands_completeBasis[ecutNbIdx])
                    +"\n  > self.ctx.EncutNbands_completeBasis[ecutNbIdx][nbands]"+str(self.ctx.EncutNbands_completeBasis[ecutNbIdx]["nbands"])
                    +"\n  > self.ctx.EncutNbands_completeBasis[ecutNbIdx][encut] "+str(self.ctx.EncutNbands_completeBasis[ecutNbIdx]["encut"] ) 
                    +"\n  > workchain node :"+str(self.ctx.runningWC_DFT_G0W0[ecutNbIdx])+"\n\n"  )


    def determine_completeBasis_encutNband(self): 
        """
        The procedure extrapolates the G0W0, QP energies with respect to the number of bands.
        In order to do so, 3-4 different G0W0 calculations are run (each one through a call of VaspDFTGWWorkChain, see later) with different number of bands.
        The encut and number of bands of each calculations are constrained by the complete basis hypothesis, i.e. the number of bands is equal to all orbitals 
        that the plane-wave basis set (determined by encut) allows to calculate.
        This functions determines the list of the encut-nbands pairs respecting the constraint.
        
        The starting point is the maximum ENMAX between all POTCARs used (DFTgr_ENMAXmax)
        
        """
        #[Part1] We extract the required data from the previous DFT ground state for the complete basis determination 
        #DFTgr_NGarray contains the fft dimensions along x , y , z.
        #DFTgr_ENMAXmax contains the maximum ENMAX between all POTCARs used
        DFTgr_NGarray   = self.inputs.ns_reference.DFTgr_NGarray 
        DFTgr_ENMAXmax  = Int( max( self.inputs.ns_reference.DFTgr_ENMAXarray.get_array('ENMAXarray') ) )  #maximum ENMAX between all POTCARs used (DFTgr_ENMAXmax)
        DFTgr_kpts      = self.inputs.ns_reference.DFTgr_kpoints
        DFTgr_cell      = self.inputs.structure
        GW_mpithrd_num  = Int( (self.inputs.options.get_dict()['resources']['num_machines'] * self.inputs.options.get_dict()['resources']['num_mpiprocs_per_machine']) // self.inputs.ns_parallelization.kpar.value  )   
       


        #The extrapolation is governed by 4 parameters:
        # 1.1) encut_atENMAX     - the encut energy used for the first point in the extrapolation
        #                          conventionally we set it to 1.00*DFTgr_ENMAXmax, where DFTgr_ENMAXmax is the maximum ENMAX between all POTCARs used.
        # 1.2) num_calc_touse_for_extrapolation - the maximum number of calculations used for the extrapolation (typically 4, but could be more with a very low nbands_stride for very large volumes)
        # 1.3) nbands_stride     - the minimum number of bands step used to increase the number of bands in the fit.
        # 1.4) cutoff_fractions  - the fractions of DFTgr_ENMAXmax used for the extrapolation; is an additional, older modes which is altarnative to the 3).
        # Let's define them one-by-one.

        #1.1] Let's handle the case in which the user wants to override the standard value of DFTgr_ENMAXmax and has thus passed a custom value
        #Then let's save the value (original or overridden) in the ctx.
        if hasattr(self.inputs, 'ns_extrapolation.cutoff_starting_value'):
            DFTgr_ENMAXmax = self.inputs.ns_extrapolation.cutoff_starting_value
            self.report("overriding ENMAX extracting from DFT ground state with one defined in self.inputs")
        self.ctx.encut_atENMAX = Float(DFTgr_ENMAXmax)


        #1.2] Let's handle the case in which the user wants to override the standard value of num_calc_touse_for_extrapolation and has thus passed a custom value
        self.ctx.r2_threshold = self.inputs.ns_extrapolation.r2_threshold.value
        self.ctx.num_calc_touse_for_extrapolation = 3
        self.ctx.max_num_runnable_G0W0_calcs      = 4
        #The definition of cutoff_fractions overries the standard mode of determination of the encut-nbands pairs.
        #  num_calc_touse_for_extrapolation should still be = 3, the min is to avoid errors if the user passes just 2 fractions, which means that only 2 G0W0 calculations will be performed.
        #  max_num_runnable_G0W0_calcs is instead = len(cutoff_fractions).
        if hasattr(self.inputs.ns_extrapolation, 'cutoff_fractions') :
            self.ctx.num_calc_touse_for_extrapolation = min( self.ctx.num_calc_touse_for_extrapolation , len(self.inputs.ns_extrapolation.cutoff_fractions.get_array('cutoff_fractions'))  )
            self.ctx.max_num_runnable_G0W0_calcs      = len(self.inputs.ns_extrapolation.cutoff_fractions.get_array('cutoff_fractions'))


        #1.3] Let's handle the case in which the user wants to override the standard value of nbands_stride and has thus passed a custom value
        #The standard value is nbands_stride = #MPI-threads        
        if hasattr(self.inputs.ns_extrapolation, 'nbands_stride'): self.ctx.nbands_stride = self.inputs.ns_extrapolation.nbands_stride.value
        else:                                                      self.ctx.nbands_stride = GW_mpithrd_num

        #1.4] Let's handle the case in which the user wants to use the mode defined by cutoff_fractions.
        if hasattr(self.inputs.ns_extrapolation, 'cutoff_fractions'):
            self.ctx.cutoff_fractions = self.inputs.ns_extrapolation.cutoff_fractions.get_array('cutoff_fractions')


        #The complete basis constraint allows to consider a single variables between the encut and nbands as indipendent, and is typically encut;
        #nbands=nbands(encut) is determined by counting how many plane waves are included inside a sphere (of radius determined by encut) 
        #centered in the brillouin zone on a given k-point.
        #
        #We have however a problem: when we launch a VASP calculations, VASP automatically rounds the given NBANDS to the closest multiple of #MPI-threads/KPAR.
        #This is problematic, because IT BREAKS the complete basis hypothesis for that G0W0 calculation.
        #If the extrapolation is applied in "final mode", we:
        # 1) invert the relation nbands=nbands(encut) (determined by the complete-basis-hypothesis) to encut=encut(nbands). The solution is not analytical,
        #    we will use the inverted relation to determine the encut corresponding to a given nbands respecting the complete-basis-hypothesis
        # 2) determine nbands corresponding to the DFTgr_ENMAXmax encut value - this value is nbands_atENMAX; 
        # Note that get_closest_EncutNband_multiple returns a a list of dict [{'encut'; <Aiida-Float> , 'nbands': <Aiida-Int>}, thus we select nbands with ['nbands']} , <string representing log>] ; the [0] select the {'encut':..,'nbands':..} dict
        params_fit = get_EncutNbandFitParams_completeBasis_quadratic(DFTgr_kpts , DFTgr_cell , DFTgr_NGarray , DFTgr_ENMAXmax)
        nbands_atENMAX = get_closest_EncutNband_multiple(DFTgr_kpts , DFTgr_cell , DFTgr_NGarray , DFTgr_ENMAXmax , params_fit , self.ctx.nbands_stride , self.ctx.encut_atENMAX , Str("encut") , flag_twoSidesRounding=False )[0]['nbands']

        #1.3-Redux] Let's ensure that nbands_stride is at least 0.05% of nbands_atENMAX - this avoid too many calculations for large volumes and different calculations with very similar nbands,
        # where numerical noise could be dominant.
        #Finally, let's ensure that nbands_stride is a multiple of #MPI-threads - because VASP rounds NBANDS to the closest multiple of #MPI-threads/KPAR, and that would create problems
        #while reading WAVECAR/WAVEDER.
        self.ctx.minimum_nbands_stride = int(0.05*nbands_atENMAX)
        self.ctx.nbands_stride = max(self.ctx.nbands_stride , self.ctx.minimum_nbands_stride)
        self.ctx.nbands_stride = math.ceil( self.ctx.nbands_stride / GW_mpithrd_num) * GW_mpithrd_num

        # Now we can determine the encut-nbands pairs to be used in the extrapolation.
        # By default we the first calculation is at (DFTgr_ENMAXmax , nbands_atENMAX) and the other calculations are determined by increasing nbands in steps of 0.2*nbands_atENMAX
        # and determining the corresponding encut by the inverted relation encut=encut(nbands) - in order to satisfy the complete basis hypothesis.
        # The number of calculations is determined by num_calc_touse_for_extrapolation.
        # If the user has specified the cutoff_fractions, we use them instead of the previous way to determine the encut-nbands pairs.
        # The results are saved in self.ctx.EncutNbands_completeBasis , which is a list of dict [{'encut': <Aiida-Float> , 'nbands': <Aiida-Int>} , ...] while _logger is a string containing a log of the determinations.
        self.ctx.EncutNbands_completeBasis = List()
        self.ctx.EncutNbands_completeBasis_logger = str("")
        if not hasattr(self.inputs.ns_extrapolation, 'cutoff_fractions'):
            nbands_array = [nbands_atENMAX + i * self.ctx.nbands_stride for i in range(self.ctx.max_num_runnable_G0W0_calcs)]
            for nb in nbands_array:
                [tmp_EB , tmp_EB_logger] = get_closest_EncutNband_multiple(DFTgr_kpts , DFTgr_cell , DFTgr_NGarray , DFTgr_ENMAXmax , params_fit , self.ctx.nbands_stride , nb , Str("nbands") , flag_twoSidesRounding=Bool(True) )
                self.ctx.EncutNbands_completeBasis.append(tmp_EB)
                self.ctx.EncutNbands_completeBasis_logger = self.ctx.EncutNbands_completeBasis_logger + tmp_EB_logger + "\n"
        
        else:
            ENMAX_array = np.multiply(self.ctx.cutoff_fractions , DFTgr_ENMAXmax)
            for ec in ENMAX_array:
                [tmp_EB , tmp_EB_logger] = get_closest_EncutNband_multiple(DFTgr_kpts , DFTgr_cell , DFTgr_NGarray , DFTgr_ENMAXmax , params_fit , Int(GW_mpithrd_num) , ec , Str("encut") , flag_twoSidesRounding= Bool(True) )
                self.ctx.EncutNbands_completeBasis.append(tmp_EB)
                self.ctx.EncutNbands_completeBasis_logger = self.ctx.EncutNbands_completeBasis_logger + tmp_EB_logger + "\n"


        #Inputs handling - printing the input data if requested.
        if self.inputs.ns_option.verbose and self.inputs.ns_extrapolation.mode.value != 'final':
            self.report(
             "\n[preparatory-1 - input of DFT ground state]-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---"
            +"\n  Inside routine determine_completeBasis_encutNband - list input data determined from DFTgr"
            +'\n  > nNGX, NGY, NGZ array:'+str(DFTgr_NGarray)
            +'\n  > max ENMAX           :'+str(DFTgr_ENMAXmax)
            +'\n  > kpoints mesh        :'+str(DFTgr_kpts)
            +'\n  > cell                :'+str(DFTgr_cell)
            +'\n  > GW_mpithrd_num      :'+str(GW_mpithrd_num)
            +'\n  > nbands_stride         :'+str(self.ctx.nbands_stride)
            +'\n  > mode                :'+str(self.inputs.ns_extrapolation.mode)
           +"\n[1 - input handling]--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -"
            +'\n[2 - determining (encut,nbands) -> determining corrected (encut,nbands)]--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- '
            +"\n  > mode            :"+str(self.inputs.ns_extrapolation.mode.value)
            +"\n  > DFTgr_ENMAXmax  :"+str(DFTgr_ENMAXmax)
            +"\n  -> in mode!=final cutoff_fractions :"+str(self.ctx.cutoff_fractions)
            +"\n               not corrected encuts :"+str(ENMAX_array)
            +"\n"+self.ctx.EncutNbands_completeBasis_logger
			+"\n  > corrected encut-nbands couples  :"+str(self.ctx.EncutNbands_completeBasis)
			+'\n[2 - determining (encut,nbands) -> determining corrected (encut,nbands)]--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ')
        if self.inputs.ns_option.verbose and self.inputs.ns_extrapolation.mode.value == 'final':
            self.report("\n[preparatory-1 - input of DFT ground state]-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---"
            +"\n  Inside routine determine_completeBasis_encutNband - list input data determined from DFTgr"
            +'\n  > nNGX, NGY, NGZ array:'+str(DFTgr_NGarray)
            +'\n  > max ENMAX           :'+str(DFTgr_ENMAXmax)
            +'\n  > kpoints mesh        :'+str(DFTgr_kpts)
            +'\n  > cell                :'+str(DFTgr_cell)
            +'\n  > GW_mpithrd_num     :'+str(GW_mpithrd_num)
            +'\n  > nbands_stride         :'+str(self.ctx.nbands_stride)
            +'\n  > mode                :'+str(self.inputs.ns_extrapolation.mode)
             
            +"\n[1 - input handling]--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---"
            +'\n[2 - determining (encut,nbands) -> determining corrected (encut,nbands)]--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---'
            +"\n  > mode            :"+str(self.inputs.ns_extrapolation.mode.value)
            +"\n  > DFTgr_ENMAXmax  :"+str(DFTgr_ENMAXmax)
            +"\n"+self.ctx.EncutNbands_completeBasis_logger
			+"\n  > corrected encut-nbands couples       :"+str(self.ctx.EncutNbands_completeBasis)
			+'\n[2 - determining (encut,nbands) -> determining corrected (encut,nbands)]--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---\n\n')




    @staticmethod
    def _extract_gaps_from_outputs_into_dicts(runningWC_DFT_G0W0 , spinComp):
        """Collect Indirect/Direct/Gamma gaps and QP corrections for all calculations."""

        gap_keys = ("G0W0_Dir", "G0W0_Ind", "G0W0_Gam")
        qpc_keys = ("HOMO_Dir", "HOMO_Ind", "HOMO_Gam", "LUMO_Dir", "LUMO_Ind", "LUMO_Gam")

        ns_gaps     = {gap_k: [runningWC_DFT_G0W0[WC_idx].outputs.gaps.get_dict()[spinComp][gap_k]     for WC_idx in runningWC_DFT_G0W0] for gap_k in gap_keys}
        ns_gaps_QPc = {qpc_k: [runningWC_DFT_G0W0[WC_idx].outputs.gaps_QPc.get_dict()[spinComp][qpc_k] for WC_idx in runningWC_DFT_G0W0] for qpc_k in qpc_keys}
        return ns_gaps, ns_gaps_QPc

    @staticmethod
    def _extrapolate_gaps_from_dicts(ar_nbandsInput, ns_gap, ns_QPc, spinComp, str_log , use_num_calc_for_extrapolation=3 ):
        """
        Extrapolate the Direct-Indirect-Gamma gaps and QP corrections with respect to 1/(Number of bands).
        """
        gap_keys = ["G0W0_Dir", "G0W0_Ind", "G0W0_Gam"]
        qpc_keys = ["HOMO_Dir", "HOMO_Ind", "HOMO_Gam", "LUMO_Dir", "LUMO_Ind", "LUMO_Gam"]

        extrapolated_gap = {"r2": {}}
        extrapolated_QPc = {"r2": {}}
    
        num_calc = min( len(ar_nbandsInput) , use_num_calc_for_extrapolation )

        # Fit gaps
        for key in gap_keys:
            extr_x = (1 / np.array(ar_nbandsInput[-num_calc:])).reshape(-1, 1)
            extr_y = ns_gap[spinComp][key][-num_calc:]
            reg = LinearRegression().fit(extr_x, extr_y)
            extrapolated_gap[key]       = Float(reg.intercept_)
            extrapolated_gap["r2"][key] = Float(reg.score(extr_x, extr_y))

        # Fit QP corrections
        for key in qpc_keys:
            extr_x = (1 / np.array(ar_nbandsInput[-num_calc:])).reshape(-1, 1)
            extr_y = ns_QPc[spinComp][key][-num_calc:]
            reg = LinearRegression().fit(extr_x, extr_y)    
            extrapolated_QPc[key]       = Float(reg.intercept_)
            extrapolated_QPc["r2"][key] = Float(reg.score(extr_x, extr_y))

        # Logging
        str_log_spinSpecific = str_log + "\n  >> [0] nbands used: "+str(ar_nbandsInput[-num_calc:])+"out of nbands array"+str(ar_nbandsInput)+"\n  >>     actually inverse nbands is used: "+str((1 / np.array(ar_nbandsInput[-num_calc:])).reshape(-1, 1))
        for key in gap_keys:
            str_log_spinSpecific += f"\n  >> [1] bandGap_{key}_ar: {ns_gap[spinComp][key]}"
            str_log_spinSpecific += f"\n  >>     bandGap_{key}_extrapolated: {extrapolated_gap[key].value} (r^2: {extrapolated_gap['r2'][key].value})"
        for key in qpc_keys:
            str_log_spinSpecific += f"\n  >> [2] QPc_{key}_ar: {ns_QPc[spinComp][key]}"
            str_log_spinSpecific += f"\n  >>     QPc_{key}_extrapolated: {extrapolated_QPc[key].value} (r^2: {extrapolated_QPc['r2'][key].value})"

        return extrapolated_gap, extrapolated_QPc, str_log_spinSpecific

    @staticmethod
    def _extrapolate_bands_into_dict(ar_nbandsInput , runningWC_DFT_G0W0 ):
        ##[Part 4] Extrapolate ALL QPshifts [for all kpts, spin and bands<tmp_min_nbandsGW] and pass r2  
        #for Metals and semimetals nbandsgw could be NOT defined by the workchain - in that case we resort to the safest definition, i.e. extrapolating all bands.
        tmp_min_nbandsGW = min(ar_nbandsInput)  
        try:   tmp_min_nbandsGW = int(min( [box.ns_parameters.nbandsgw for box in self.ctx.inputs_array] ))
        except:pass 
        
        #First we extract the G0W0 bands in the AiiDA format - then extract as a list of array - and finally inlude only bands up to nbandsGW.
        bands_G0W0_AiiDA        = [ runningWC_DFT_G0W0[WC_idx].outputs.bands_G0W0  for WC_idx in runningWC_DFT_G0W0]
        bands_G0W0_differentCalcStacked              = [bnd.get_bands() for bnd in bands_G0W0_AiiDA]
        bands_G0W0_upToNBANDSGW_differentCalcStacked = np.stack( [bnd[:,:tmp_min_nbandsGW] for bnd in bands_G0W0_differentCalcStacked] )
        #The same for DFT bands
        bands_DFT_AiiDA        = [ runningWC_DFT_G0W0[WC_idx].outputs.bands_DFT  for WC_idx in runningWC_DFT_G0W0]
        bands_DFT_differentCalcStacked              = [bnd.get_bands() for bnd in bands_DFT_AiiDA]
        bands_DFT_upToNBANDSGW_differentCalcStacked = np.stack( [bnd[:,:tmp_min_nbandsGW] for bnd in bands_DFT_differentCalcStacked] )
        bands_QPc_upToNBANDSGW_differentCalcStacked = bands_G0W0_upToNBANDSGW_differentCalcStacked - bands_DFT_upToNBANDSGW_differentCalcStacked
        #bands_G0W0_differentCalcStacked has shape List[ (#kpts , #bands) ]
        #bands_G0W0_upToNBANDSGW_differentCalcStacked  has shape (#GWcalc , #kpts , #bands)
        #bands_G0W0_extrapolated has shape (#kpts , #bands)

        #Let's extrapolate now the QP corrections for all kpoints, spins and bands<tmp_min_nbandsGW
        bands_G0W0_extrapolated    = np.zeros(np.shape(bands_G0W0_upToNBANDSGW_differentCalcStacked[0,:,:]))  #The zero refers to the Calculation index; bands_G0W0_upToNBANDSGW contains the bands of more than one calculations
        bands_G0W0_extrapolated_r2 = np.zeros(np.shape(bands_G0W0_upToNBANDSGW_differentCalcStacked[0,:,:])) 
        ar_inverseNbands = 1/np.array(ar_nbandsInput) #Remember that ar_nbandsInput contains all NBANDS of the various calculations used for the extrapolation.
        for idx in np.ndindex(np.shape(bands_G0W0_extrapolated)): 
            reg_b = LinearRegression().fit( ar_inverseNbands.reshape(-1, 1) , bands_G0W0_upToNBANDSGW_differentCalcStacked[:,idx[0],idx[1]] )
            bands_G0W0_extrapolated[idx]    = reg_b.intercept_
            bands_G0W0_extrapolated_r2[idx] = reg_b.score(ar_inverseNbands.reshape(-1, 1), bands_G0W0_upToNBANDSGW_differentCalcStacked[:,idx[0],idx[1]] )

  
        bands_QPc_extrapolated    = np.zeros(np.shape(bands_QPc_upToNBANDSGW_differentCalcStacked[0,:,:]))
        bands_QPc_extrapolated_r2 = np.zeros(np.shape(bands_QPc_upToNBANDSGW_differentCalcStacked[0,:,:]))
        for idx in np.ndindex(np.shape(bands_QPc_extrapolated)): 
            reg_b = LinearRegression().fit( ar_inverseNbands.reshape(-1, 1) , bands_QPc_upToNBANDSGW_differentCalcStacked[:,idx[0],idx[1]] )
            bands_QPc_extrapolated[idx]    = reg_b.intercept_
            bands_QPc_extrapolated_r2[idx] = reg_b.score(ar_inverseNbands.reshape(-1, 1), bands_QPc_upToNBANDSGW_differentCalcStacked[:,idx[0],idx[1]] )

        #Save the extrapolated in the proper format and output that
        extrapolated_bands = {}
        extrapolated_bands["bands_G0W0"] = bands_G0W0_extrapolated ; extrapolated_bands["bands_G0W0_r2"] = bands_G0W0_extrapolated_r2 
        extrapolated_bands["bands_QPc"]  = bands_QPc_extrapolated ;  extrapolated_bands["bands_QPc_r2"]  = bands_QPc_extrapolated_r2
        return extrapolated_bands


    def are_r2_under_threshold(self):
        ar_nbandsInput  = [ self.ctx.runningWC_DFT_G0W0[WC_idx].inputs.ns_parameters['nbands'].value  for WC_idx in self.ctx.runningWC_DFT_G0W0 ]

        str_log = '\n VaspG0W0BasisExtrWorkChain pk='+str(self.node.pk)+" checking if the additional VaspDFTGWWorkChain is required\n only the gaps (and not QPc) are checked; threshold is = "+str(self.inputs.ns_extrapolation.r2_threshold.value) 
        if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
            ns_gaps_spinUp , ns_gaps_QPc_spinUp = self._extract_gaps_from_outputs_into_dicts( self.ctx.runningWC_DFT_G0W0 , 'spinUp')
            ns_gaps_spinDw , ns_gaps_QPc_spinDw = self._extract_gaps_from_outputs_into_dicts( self.ctx.runningWC_DFT_G0W0 , 'spinDw')
            ns_gap     = Dict(dict = {"spinUp":ns_gaps_spinUp     , "spinDw":  ns_gaps_spinDw     })
            ns_gap_QPc = Dict(dict = {"spinUp":ns_gaps_QPc_spinUp , "spinDw":  ns_gaps_QPc_spinDw })
        else:
            ns_gaps_spinUp , ns_gaps_QPc_spinUp = self._extract_gaps_from_outputs_into_dicts( self.ctx.runningWC_DFT_G0W0 ,  'spinUp')
            ns_gap     = Dict(dict = {"spinUp":ns_gaps_spinUp})
            ns_gap_QPc = Dict(dict = {"spinUp":ns_gaps_QPc_spinUp}) 


        ## Determining extrapolated G0W0 bandgaps and QP shifts through fitting nbands/gaps_G0W        if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
        if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
            extrapolated_gap_spinUp , extrapolated_QPc_spinUp , str_log_spinUp = self._extrapolate_gaps_from_dicts( ar_nbandsInput , ns_gap , ns_gap_QPc , 'spinUp' , str_log )
            extrapolated_gap_spinDw , extrapolated_QPc_spinDw , str_log_spinDw = self._extrapolate_gaps_from_dicts( ar_nbandsInput , ns_gap , ns_gap_QPc , 'spinDw' , str_log )
            flag_is_extrapolation_converged = ( (extrapolated_gap_spinUp["r2"]["G0W0_Dir"] >= self.ctx.r2_threshold) and
                                                (extrapolated_gap_spinUp["r2"]["G0W0_Ind"] >= self.ctx.r2_threshold) and
                                                (extrapolated_gap_spinUp["r2"]["G0W0_Gam"] >= self.ctx.r2_threshold) and
                                                (extrapolated_gap_spinDw["r2"]["G0W0_Dir"] >= self.ctx.r2_threshold) and
                                                (extrapolated_gap_spinDw["r2"]["G0W0_Ind"] >= self.ctx.r2_threshold) and
                                                (extrapolated_gap_spinDw["r2"]["G0W0_Gam"] >= self.ctx.r2_threshold) )
            
        else:
            extrapolated_gap_spinUp , extrapolated_QPc_spinUp , str_log_spinUp = self._extrapolate_gaps_from_dicts( ar_nbandsInput , ns_gap , ns_gap_QPc , 'spinUp' , str_log )
            flag_is_extrapolation_converged = ( (extrapolated_gap_spinUp["r2"]["G0W0_Dir"] >= self.ctx.r2_threshold) and
                                                (extrapolated_gap_spinUp["r2"]["G0W0_Ind"] >= self.ctx.r2_threshold) and
                                                (extrapolated_gap_spinUp["r2"]["G0W0_Gam"] >= self.ctx.r2_threshold) )

        str_log = str_log + "\n\n > [5] Extrapolation results (r2 values):"
        if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
            str_log = str_log + "\n\n > [spin UP]"+str_log_spinUp + "\n\n > [spin DW]"+str_log_spinDw
        else:
            str_log = str_log + "\n\n > [spin UP]"+str_log_spinUp   
        self.report(str_log)

        if flag_is_extrapolation_converged:
            str_log = str_log + "\n  > No additional calc is is required, extrapolation is converged!!!"
            self.report(str_log)
            return Bool(False)
        elif (len(self.ctx.runningWC_DFT_G0W0) >= self.ctx.max_num_runnable_G0W0_calcs) :
            str_log = str_log + "\n  > Maximum number of calculations reached, cannot perform additional calculations!!!"
            self.report(str_log)
            return Bool(False)
        else:
            str_log = str_log + "\n  > Additional calc is is required, extrapolation is NOT converged!!!"
            self.report(str_log)
            return Bool(True)        

    def prepare_run_wc_DFT_G0W0_additionalG0W0s(self):
        print("\n\n [prepare_run_wc_DFT_G0W0_additionalG0W0s \n\n")
        print(len(self.ctx.runningWC_DFT_G0W0) , self.ctx.max_num_runnable_G0W0_calcs)
        print(len(self.ctx.EncutNbands_completeBasis) , self.ctx.num_calc_touse_for_extrapolation)

        ecutNbIdx = len(self.ctx.runningWC_DFT_G0W0)
        self.ctx.runningWC_DFT_G0W0[ecutNbIdx] =  self.submit(VaspDFTGWWorkChain   , **self.ctx.inputs_array[ecutNbIdx]) 
        key = f'WC_DFT_G0W0_{ecutNbIdx}'
        self.to_context(**{key: self.ctx.runningWC_DFT_G0W0[ecutNbIdx]})



    def elaborate_extrapolate_results(self):
        """
        Extract the Direct - indirect - Gamma gaps and Quasiparticle corrections from the G0W0 data points; extrapolate them to the infinite basis-set limit; returns them.
        """
                    
              
        #[Part1] outputting the ENMAX values, used as input in this workflow, might result useful (this saves the effort of reconstructing it later).
        self.ctx.encut_atENMAX.store()
        ##self.ctx.ENMAXarray.store() 
        self.out('ENMAX_referenceValue' ,  self.ctx.encut_atENMAX                       )
        self.out('ENMAX_array'          , self.inputs.ns_reference.DFTgr_ENMAXarray     )    

        ##[Part2] Extracting nbands/gaps for fit and doing sanity checks.
        ##self.ctx.runningWC_DFT_G0W0 is a Dict with form {1:WorkChain_Node[..] , 2:WorkChain_Node[..] ; for WC_idx in self.ctx.runningWC_DFT_G0W0 runs on index 1, 2, etc
        ar_isFinishedOk     = [ self.ctx.runningWC_DFT_G0W0[WC_idx].is_finished_ok      for WC_idx in self.ctx.runningWC_DFT_G0W0 ]
        #ar_encutInput is the equivlaent of ar_nbandsInput for the cutoff (ENCUT)
        ar_nbandsInput      = [ self.ctx.runningWC_DFT_G0W0[WC_idx].inputs.ns_parameters['nbands'].value  for WC_idx in self.ctx.runningWC_DFT_G0W0 ]
        ar_encutInput       = [ self.ctx.runningWC_DFT_G0W0[WC_idx].inputs.ns_parameters['encut'].value   for WC_idx in self.ctx.runningWC_DFT_G0W0 ]
        str_log =("\n > [1] ar_isFinishedOk= "+str(ar_isFinishedOk) 
                 +"\n   [2] PAW potentials= "+str(self.inputs.potential_mapping.get_dict()) 
                 +"\n > [2] ar_nbandsInput= " +str(ar_nbandsInput) +"\n > [2] ar_encutInput=  " +str(ar_encutInput) )
        if not all(ar_isFinishedOk):  return self.exit_codes.ONE_OR_MORE_GW_FAILED
        #Let's save in the proper format and output those data
        ar_encut_nbands  = XyData()
        ar_encut_nbands.set_x(np.array(ar_encutInput)                  ,'encut'      ,'eV')
        ar_encut_nbands.set_y(np.array(ar_nbandsInput)                 ,'nbands'     ,''  )    
        ar_encut_nbands.store()        
        self.out('pairs_nbands_encuts' , ar_encut_nbands )        
        

        #[Part2] Extracting gaps and QP corrections (related to gaps) for all calculations.
        #If the calculation is spin-polarized, apply elaborate_results_gaps to both spin component - and save the gaps for each spin-component separately.
        if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
            ns_gaps_spinUp , ns_gaps_QPc_spinUp = self._extract_gaps_from_outputs_into_dicts( self.ctx.runningWC_DFT_G0W0 , 'spinUp')
            ns_gaps_spinDw , ns_gaps_QPc_spinDw = self._extract_gaps_from_outputs_into_dicts( self.ctx.runningWC_DFT_G0W0 , 'spinDw')
            ns_gap     = Dict(dict = {"spinUp":ns_gaps_spinUp     , "spinDw":  ns_gaps_spinDw     })
            ns_gap_QPc = Dict(dict = {"spinUp":ns_gaps_QPc_spinUp , "spinDw":  ns_gaps_QPc_spinDw })
        else:
            ns_gaps_spinUp , ns_gaps_QPc_spinUp = self._extract_gaps_from_outputs_into_dicts( self.ctx.runningWC_DFT_G0W0 ,  'spinUp')
            ns_gap     = Dict(dict = {"spinUp":ns_gaps_spinUp})
            ns_gap_QPc = Dict(dict = {"spinUp":ns_gaps_QPc_spinUp}) 
        ns_gap.store() ; ns_gap_QPc.store()
        self.out('ns_gaps' , ns_gap     )
        self.out('ns_QPc'  , ns_gap_QPc )

        ##[Part3] Determining extrapolated G0W0 bandgaps and QP shifts through fitting nbands/gaps_G0W0
        if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
            extrapolated_gap_spinUp , extrapolated_QPc_spinUp , str_log_spinUp = self._extrapolate_gaps_from_dicts( ar_nbandsInput , ns_gap , ns_gap_QPc , 'spinUp' , str_log )
            extrapolated_gap_spinDw , extrapolated_QPc_spinDw , str_log_spinDw = self._extrapolate_gaps_from_dicts( ar_nbandsInput , ns_gap , ns_gap_QPc , 'spinDw' , str_log )
            extrapolated = Dict(dict = { "gaps"    : { "spinUp":extrapolated_gap_spinUp , "spinDw":extrapolated_gap_spinDw } ,
                                         "gaps_QPc": { "spinUp":extrapolated_QPc_spinUp , "spinDw":extrapolated_QPc_spinDw } })
            #extrapolated = { "gaps"    : { "spinUp":extrapolated_gap_spinUp , "spinDw":extrapolated_gap_spinDw } ,
            #                 "gaps_QPc": { "spinUp":extrapolated_QPc_spinUp , "spinDw":extrapolated_QPc_spinDw } }
            str_log = str_log +"\n [1 Spin Component]" +str_log_spinUp +"\n [2 Spin Component]" +str_log_spinDw
        else:
            extrapolated_gap_spinUp , extrapolated_QPc_spinUp , str_log_spinUp = self._extrapolate_gaps_from_dicts( ar_nbandsInput , ns_gap , ns_gap_QPc , 'spinUp' , str_log )
            extrapolated = Dict(dict = { "gaps"    : { "spinUp":extrapolated_gap_spinUp } ,
                                         "gaps_QPc": { "spinUp":extrapolated_QPc_spinUp } }) 
            #extrapolated = { "gaps"    : { "spinUp":extrapolated_gap_spinUp } ,
            #                 "gaps_QPc": { "spinUp":extrapolated_QPc_spinUp } }
            str_log = str_log +"\n [1 Spin Component]" +str_log_spinUp
        self.report(str_log)
        extrapolated.store()    
        self.out('extrapolated' , extrapolated )
    
        ##[Part4] Determining extrapolated ALL QP shifts [for all kpts, spin and bands<tmp_min_nbandsGW] and pass r2
        extrapolated_bands = self._extrapolate_bands_into_dict( ar_nbandsInput , self.ctx.runningWC_DFT_G0W0 )
        extrapolated_bands = Dict(extrapolated_bands)
        extrapolated_bands.store()
        self.out('extrapolated_bands' , extrapolated_bands )    




    def clean_remoteFolder_DFT(self):
            from aiida import orm
            self.report(orm.CalcJobNode)

            cleaned_calcs = []
            for WC_DFTgr in self.ctx.finishedWC_DFTgr:
                for called_descendant_DFTgr in WC_DFTgr.called_descendants:
                    if isinstance(called_descendant_DFTgr, orm.CalcJobNode):
                        self.report(called_descendant_DFTgr)
                        self.report(called_descendant_DFTgr.outputs.remote_folder)
                        try:
                            called_descendant_DFTgr.outputs.remote_folder._clean() # pylint: disable=protected-access
                            cleaned_calcs.append(called_descendant_DFTgr.pk)
                        except (IOError, OSError, KeyError):
                            pass

            for WC_DFT in self.ctx.finishedWC_DFT_G0W0:
                for called_descendant_DFT in WC_DFT.called_descendants:
                    if isinstance(called_descendant_DFT, orm.CalcJobNode):
                        self.report(called_descendant_DFT)
                        self.report(called_descendant_DFT.outputs.remote_folder)
                        try:
                            called_descendant_DFT.outputs.remote_folder._clean() # pylint: disable=protected-access
                            cleaned_calcs.append(called_descendant_DFT.pk)
                        except (IOError, OSError, KeyError):
                            pass

