import numpy as np
from copy import deepcopy
import itertools
from aiida import orm
from aiida.orm import Code, Int, Float, Str, Dict, Bool , List , RemoteData , ArrayData , BandsData , XyData
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine  import WorkChain, calcfunction , ToContext , append_ , submit, while_ , if_ , run
from aiida.common.extendeddicts  import AttributeDict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


from .workchain_G0W0_base import VaspDFTGWWorkChain

from aiida import load_profile
load_profile()

#Miscellaneous utils functions
def _get_kmesh_from_kdensity(latVec , KSPACING , flag_roundInsteadCeil=True ):
        """
        Calculate the k-point mesh dimensions for a given lattice and k-point spacing.
        Args:
          - latVec (array-like): A 3x3 array representing the lattice vectors of the unit cell.
          - KSPACING (in Angstrom^{-1}) represents the smallest allowed spacking between k-points in the BZ; SMALLER values produce DENSER meshes;
            Conversely, The output number of divisions Ni is chosen as the maximum integer that satisfies |b_i| / Ni <= KSPACING
        """

        latVec = np.array( latVec )
        recLatVec= np.zeros((3,3))
        Vol= np.abs( np.dot(latVec[0,:] , np.cross(latVec[1,:],latVec[2,:])) )
        recLatVec[0,:]= np.cross(latVec[1,:],latVec[2,:])  /Vol
        recLatVec[1,:]= np.cross(latVec[2,:],latVec[0,:])  /Vol
        recLatVec[2,:]= np.cross(latVec[0,:],latVec[1,:])  /Vol

        # List of dim=3 with dimensions of the 3 rec.vectors.
        rec_cell_norm = np.array( [np.linalg.norm( recLatVec[x,:]) for x in range(3)] )
        
        #Exact (and thus fractional) K-mesh corresponding to the EXACT KSPACING
        #  For example, for rec_cell_norm = [0.319, 0.319, 0.319] :    
        #  KSPACING = 0.5 -> array([4.0084, 4.0084, 4.0084])
        #  KSPACING = 0.3 -> array([6.6807, 6.6807, 6.6807])
        Kmesh_ideal_fractional = np.array(rec_cell_norm) * 2*np.pi / KSPACING

        if flag_roundInsteadCeil:   Kmesh = [ max(1.0,np.round(k)) for k in Kmesh_ideal_fractional ]
        else:                       Kmesh = np.ceil( Kmesh_ideal_fractional )
        Kmesh = np.array( Kmesh ).astype(int)
        return Kmesh

def _get_kspacing_from_kmesh(latVec, kmesh): 
        #k-mesh = 2pi * |bi|/ k-density  ->
        kmesh_tmp = deepcopy( kmesh )
        latVec = np.array( latVec )
        recLatVec= np.zeros((3,3))
        Vol= np.abs( np.dot(latVec[0,:] , np.cross(latVec[1,:],latVec[2,:])) )
        recLatVec[0,:]= np.cross(latVec[1,:],latVec[2,:])  /Vol
        recLatVec[1,:]= np.cross(latVec[2,:],latVec[0,:])  /Vol
        recLatVec[2,:]= np.cross(latVec[0,:],latVec[1,:])  /Vol

        rec_cell_norm = [np.linalg.norm( recLatVec[x,:]) for x in range(3)]    
        return [2*np.pi * rec_cell_norm[idx] / kmesh_tmp[idx] for idx in range(len(rec_cell_norm)) ]
        


class VaspMBPTKptsConvWorkChain(WorkChain):
    _next_workchain = WorkflowFactory('vasp.vasp')

    @classmethod
    def define(cls, spec):
            super(VaspMBPTKptsConvWorkChain , cls).define(spec)

            spec.expose_inputs(cls._next_workchain     , exclude=('kpoints','parameters','settings','potential_family','potential_mapping')) 
            spec.expose_inputs(VaspDFTGWWorkChain      , exclude=('kpoints','ns_reference') ) 
             
            spec.input( 'ns_kpoints.convergence_threshold'       , valid_type=Float , required=False , default=lambda:Float(0.1), help="minimum converge value in eV" ) 
            spec.input( 'ns_kpoints.kMesh.startingValue'         , valid_type=List  , required=False , help="Starting l-mesh for the k-point density convergence If not specified, a k-mesh based on the density kDensity.startingValue will be used" )
            spec.input( 'ns_kpoints.kMesh.maxValue'              , valid_type=List  , required=False , help="Maximum k-mesh for the k-point density convergence. If not specified, a k-mesh based on the density kDensity.maxValue will be used" ) 
            spec.input( 'ns_kpoints.kMesh.step'                  , valid_type=List  , required=False , default=lambda:List([1,1,1]), help="Step size for the k-point mesh." ) 
            spec.input( 'ns_kpoints.kDensity.minimum_constraint' , valid_type=Float , required=False , default=lambda:Float(0.4), help="Constraint on the minimum k-point density - that converged k-mesh must guarantee." )
            spec.input( 'ns_kpoints.kDensity.startingValue'      , valid_type=Float , required=False , default=lambda:Float(0.4), help="Starting value for the k-point density convergence." ) 
            spec.input( 'ns_kpoints.kDensity.maxValue'           , valid_type=Float , required=False , default=lambda:Float(0.1), help="Maximum value for the k-point density convergence." ) 
            spec.input( 'ns_kpoints.kDensity.step'               , valid_type=Float , required=False , default=lambda:Float(0.05), help="Step size for the k-point density convergence." ) 

            spec.input('ns_opt_converge.converge_G0W0gap' , valid_type=Bool , required=False , default=lambda:Bool(True)  )
            spec.input('ns_opt_converge.converge_mBSEgap' , valid_type=Bool , required=False , default=lambda:Bool(False) )
            spec.input('ns_opt_converge.use_Gradient'     , valid_type=Bool , required=False , default=lambda:Bool(True)  )
            
            #spec.output( 'final_kDensity' , valid_type= List )
            spec.output( 'kmesh_converged'    , valid_type= DataFactory('core.array.kpoints') , required=False)
            
            spec.exit_code(402,'UNDEFINED_BEHAVIOUR',message="Something is wrong; the iteration of the k-point convergence returned no calc or internal consistency has been lost.")
            spec.exit_code(403,'NO_STARTINGDFT_FOR_SP' , message="Spin-Polarized(SP) DFT should start either from NSP or Given DFT.")
            spec.exit_code(403,'NO_STARTINGDFT_FOR_GW' , message="G0W0 should start either from NSP or SP DFT.") 
            spec.exit_code(404,'CONVERGENCE_NOT_FOUND' , message='Convergence has not been reached; please relax the threshold / increase the range studied / check the calculations.')
            

            
            
            spec.outline(
                cls.initialize,
                while_(cls.monitor_convergence)(
                     cls.prepare_run_MBPT,
                ),
                cls.elaborate_results
            
            )
        
    def initialize(self):
        self.ctx.control = AttributeDict() ; 
        
        self.ctx.control['kmesh_converged'] = None ; self.ctx.control['kdensity_converged'] = None ; 
        self.ctx.control['G0W0_gapType_toConverge'] = "G0W0_Dir"  #Can be "G0W0_Dir" , "G0W0_Ind" , "G0W0_Gam"
        self.ctx.WC_MBPT = []

            
        ##[ The Kpoint part ]
        #There are two possible ways to control the convergence:
        # 1) through the k-mesh : in this case, both the kmesh.startingValue and kmesh.maxValue of kMesh must be specified; only the step is optional (default=1)
        # 2) through the k-point density : this case is activated only if BOTH kMesh.startingValue and kMesh.maxValue are NOT specified; 
        #                                  and is more flexible, as there are no mandatory values, all values have a default values
        #    Two additional notes:
        #    - if kmesh.startingValue is specified, and not kmesh.maxValue that kmesh is used as starting k-mesh, but the increase of the k-mesh is done through the k-point density 
        #      and not through the k-mesh step
        #    - in this second case, the minimum_constraint of kDensity is always considered, i.e. the startingValue of kDensity is automatically increased if it is < minimum_constraint
  
        self.ctx.control.iteration_counter = -1
        self.ctx.control['control_way'] = ''  #can be 'kmesh' or 'kdensity' depending on how the convergence is controlled
        self.ctx.control['kmesh']       = [] #List of k-meshes to be tested
        self.ctx.control['kdensity']    = [] #List of kdensities to be tested
        # self.ctx.control['Delta_k']     = [] #This will contain the difference between two consecutive k-point densities
        # self.ctx.control['Delta_obj']   = AttributeDict() #obj stands for objective; This will contain the difference between two consecutive objective functions (i.e. G0W0 gap, mBSE gap)
        # self.ctx.control.Delta_obj      = ( {'SpinUp':[]} if ('magnetic_moment_onsite' in self.inputs['ns_parameters'])   else {'SpinUp':{},'SpinDw':[]} )
        
        
        ##[1]Now Let's check which of the two ways is used
        if ('kMesh' in self.inputs.ns_kpoints) and ('startingValue' in self.inputs.ns_kpoints.kMesh) and ('maxValue' in self.inputs.ns_kpoints.kMesh):
            self.ctx.control['control_way'] = 'kmesh'
            str_log = ("\n [Initializing K-points convergence]"+"\n > Both kmesh.startingValue and kmesh.maxValue are specified"+
                       "\n   -> Controlling k-point convergence through k-mesh."+
                       "\n   > kMesh.startingValue: " + str(self.inputs.ns_kpoints.kMesh.startingValue.get_list())+
                       "\n   > kMesh.maxValue: " + str(self.inputs.ns_kpoints.kMesh.maxValue.get_list()))
        else:
            self.ctx.control['control_way'] = 'kdensity'
            str_log = ("\n [Initializing K-points convergence]"+"\n > BOTH kmesh.startingValue and kmesh.maxValue are NOT specified"+
                       "\n   -> Controlling k-point convergence through k-point density.")

        #[2] Initialize the starting k-mesh
        if ('kMesh' in self.inputs.ns_kpoints) and ('startingValue' in self.inputs.ns_kpoints.kMesh) :
            kmesh_start    = self.inputs.ns_kpoints.kMesh.startingValue.get_list()
            kdensity_start = max( _get_kspacing_from_kmesh(self.inputs.structure.cell, kmesh_start) )
            str_log = ("\n [Initializing K-points convergence]"+"\n > Starting k-mesh is specified by the user ("+str(self.inputs.ns_kpoints.kMesh.startingValue.get_list())+") - using as starting k-mesh")
        else:
            # If kMesh is not specified, use kDensity.startingValue
            kdensity_start = max( self.inputs.ns_kpoints.kDensity.minimum_constraint.value , self.inputs.ns_kpoints.kDensity.startingValue.value )
            kmesh_start    = _get_kmesh_from_kdensity(self.inputs.structure.cell, kdensity_start)
            str_log = ("\n [Initializing K-points convergence]"+"\n > Starting k-mesh is not specified by the user; using k-mesh corresponding to kDensity.startingValue")
            str_log = str_log + ("\n   > kDensity.startingValue:      "+str(self.inputs.ns_kpoints.kDensity.startingValue.value)+" A^{-1}"+
                                 "\n   > kDensity.minimum_constraint: "+str(self.inputs.ns_kpoints.kDensity.minimum_constraint.value)+" A^{-1}"+
                                 "\n   > corresponding to k-mesh:     "+np.array2string(kmesh_start , separator=" , ").replace('\n', '') )   

        
        ##[2] Generate the other k-meshes to be tested 
        if self.ctx.control['control_way'] == 'kmesh':
            #First append the starting one
            self.ctx.control['kmesh'].append(    np.array(kmesh_start    , dtype=int)   )
            self.ctx.control['kdensity'].append( np.array(kdensity_start , dtype=float) )          
            #And generate the first candidate
            tmp_new_candidate_kmesh =  self.ctx.control['kmesh'][-1] + np.array(self.inputs.ns_kpoints.kMesh.step.get_list() , dtype=int )
            #Then check that candidate and generate new candidates
            while np.any(tmp_new_candidate_kmesh < self.inputs.ns_kpoints.kMesh.maxValue.get_list()):
                 self.ctx.control['kmesh'].append(    np.array(tmp_new_candidate_kmesh , dtype=int)  )
                 self.ctx.control['kdensity'].append( np.array(_get_kspacing_from_kmesh(self.inputs.structure.cell, tmp_new_candidate_kmesh) , dtype=float) )              
                 tmp_new_candidate_kmesh =  self.ctx.control['kmesh'][-1] + np.array(self.inputs.ns_kpoints.kMesh.step.get_list() , dtype=int )
                
        elif self.ctx.control['control_way'] == 'kdensity':
                 kdensity_step  = max( self.inputs['ns_kpoints']['kDensity']['step'].value , 0.0025)
                 list_kdensity  = np.arange( min(kdensity_start , self.inputs['ns_kpoints']['kDensity']['maxValue'].value) , 
                                             max(kdensity_start , self.inputs['ns_kpoints']['kDensity']['maxValue'].value) , 
                                             kdensity_step )
                 list_kmesh     = [_get_kmesh_from_kdensity(self.inputs.structure.cell , KS) for KS in list_kdensity]   
                 
                 #A kmesh is composed by integers, therefore it's density does not exactly corresponds to the values list_kdensity; it's defined as the kmesh with the closest density to the ones in list_kdensity
                 #Therefore if kdensity is very small, there could be identical entries in list_kmesh; these duplicates correspond to different kdensity values which round to the same kmesh
                 #Moreover, np.unique also re-order it
                 self.ctx.control['kmesh'] , _ = np.unique( list_kmesh , axis=0 , return_index=True ) 
                 #np.unique returns an array a lists; we want a list of arrays, thus
                 self.ctx.control['kmesh']     = [np.array(kmesh) for kmesh in self.ctx.control['kmesh'] ]
                 self.ctx.control['kdensity']  = [ np.array(_get_kspacing_from_kmesh(self.inputs.structure.cell, kmesh) , dtype=float) for kmesh in self.ctx.control['kmesh'] ]
                 self.ctx.control['kdensity_notRounded'] = list_kdensity
                 str_log = str_log + ( "\n   > kdensity_step:               "+str(kdensity_step) )
        str_log = str_log + ( "\n   > kmesh list:    "+str(self.ctx.control['kmesh'])    )
        str_log = str_log + ( "\n   > kdensity list: "+str(self.ctx.control['kdensity']) )
        self.report(str_log)



    def prepare_run_MBPT(self):
        # Build inputs for the base workchain
        self.ctx.inputs_G0W0base = AttributeDict()
        self.ctx.inputs_G0W0base.ns_option , self.ctx.inputs_G0W0base.ns_parameters , self.ctx.inputs_G0W0base.ns_reference = AttributeDict() , AttributeDict() , AttributeDict()
        #self.ctx.inputs_DFTgr_NSP.update(self.exposed_inputs(self._next_workchain))
        self.ctx.inputs_G0W0base.update(self.exposed_inputs(VaspDFTGWWorkChain))
        self.ctx.inputs_G0W0base.clean_workdir = Bool(False)            
        self.ctx.inputs_G0W0base.potential_family  = self.inputs.potential_family
        self.ctx.inputs_G0W0base.potential_mapping = self.inputs.potential_mapping
        self.ctx.inputs_G0W0base.ns_option.run_1DFTgr       = Bool(True)
        self.ctx.inputs_G0W0base.ns_option.run_2DFTvo_3G0W0 = Bool(True)

        #Finally, kpoint related stuff        
        self.ctx.inputs_G0W0base.kpoints = DataFactory('core.array.kpoints')()
        self.ctx.inputs_G0W0base.kpoints.set_kpoints_mesh(  self.ctx.control['kmesh'][self.ctx.control.iteration_counter] )
           
        # Reference WAVECAR/CHGCAR if user provided (optional)
        if ('ns_reference' in self.inputs) and ('DFTgr_RemoteData' in self.inputs['ns_reference']):
            self.ctx.inputs_G0W0base.ns_reference = AttributeDict()
            self.ctx.inputs_G0W0base.ns_reference.DFTgr_RemoteData = self.inputs.ns_reference.DFTgr_RemoteData


        # ns_parameters (copy through what you use)
        if 'encut' in self.inputs.ns_parameters:        self.ctx.inputs_G0W0base.ns_parameters.encut = self.inputs.ns_parameters.encut
        if 'nbands' in self.inputs.ns_parameters:       self.ctx.inputs_G0W0base.ns_parameters.nbands = self.inputs.ns_parameters.nbands
        if 'nomega' in self.inputs.ns_parameters:       self.ctx.inputs_G0W0base.ns_parameters.nomega = self.inputs.ns_parameters.nomega
        if 'encut_chi' in self.inputs.ns_parameters:    self.ctx.inputs_G0W0base.ns_parameters.encut_chi = self.inputs.ns_parameters.encut_chi
        elif 'encut' in self.inputs.ns_parameters:  
            # sensible default if not provided
            self.ctx.inputs_G0W0base.ns_parameters.encut_chi = Float(self.inputs.ns_parameters.encut.value * 0.5)
        # Spin-polarization if requested
        if 'magnetic_moment_onsite' in self.inputs.ns_parameters:
            self.ctx.inputs_G0W0base.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite

                                     
        self.report("\n [wkc_KptsConv][Preliminary DFT-GroundState calc - NonSpinPolarized - From scratch]\n  > Lauching DFT(NonSpinPolarized) using workchain_DFT_G0W0 on k-mesh "
                            +np.array2string(   self.ctx.control['kmesh'][self.ctx.control.iteration_counter] , separator=" , ").replace('\n', '')+"\n")  
        running_G0W0base = self.submit(VaspDFTGWWorkChain , **self.ctx.inputs_G0W0base) 
        return ToContext(WC_MBPT=append_(running_G0W0base))
         


    
    
    
    def monitor_convergence(self): 
        # #counter starts at -1 and changes BEFORE a calc is launched
        # #1°control: counter starts=-1  -> [conv. check] -> increased to 0 -> launched 1° G0W0/mBSE
        # #2°control: counter starts= 0  -> [conv. check] -> increased to 1 -> launched 2° G0W0/mBSE
        # #3°control: counter starts= 1  -> [conv. check] -> increased to 2 -> launched 3° G0W0/mBSE


        #Initialize Initialize
        flag_is_converged= False
                
        str_log =("\n [wkc_KptsConv][monitor_convergence] iteration_counter="+str(self.ctx.control.iteration_counter) + " before launching MBPT calculation num="+str(self.ctx.control.iteration_counter+1)+
                                "\n  Remember : iteration_counter starts at (i-1)th  -> [conv. check] -> increased to i-th -> launched i-th G0W0/mBSE"
                                )
               
        if self.ctx.control['control_way'] == 'kmesh':
            #[K-mesh only Check - 1]  Need two calculations to compare last-two gaps.
            # Consider that the counter is increased BEFORE launching the calculation, and starts at -1; so when counter=0 here we are at the conv.check of the second calculation.
            if self.ctx.control.iteration_counter <= 0:
                str_log = str_log + ("\n  > Not enough calculations to perform convergence check; launching next k-mesh calculation.");  self.report(str_log)
                self.ctx.control.iteration_counter += 1
                if self.ctx.control.iteration_counter >= len(self.ctx.control['kmesh']):
                    self.report("\n  > Reached maximum number of k-meshes to be tested; convergence not found.\n")
                    return self.exit_codes.CONVERGENCE_NOT_FOUND    
                return True
            
            #Let's define some variables to make the following less verbose
            thr = self.inputs.ns_kpoints.convergence_threshold.value
            gap_type = self.ctx.control.get('G0W0_gapType_toConverge', 'G0W0_Dir')  # default to direct gap
            
            WF_G0W0_gaps = [ wc.outputs.gaps.get_dict() for wc in self.ctx.WC_MBPT ]
            WF_G0W0_gaps_toCompare = {} ; delta_gap = {}
            WF_G0W0_gaps_toCompare['spinUp'] = [ gaps['spinUp'][ gap_type ]  for gaps in WF_G0W0_gaps ]
            str_log = str_log + ("\n  > G0W0 gaps to be compared - SpinUp: " + str(WF_G0W0_gaps_toCompare["spinUp"]) )
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):  
                WF_G0W0_gaps_toCompare['spinDw'] = [ gaps['spinDw'][ gap_type ]  for gaps in WF_G0W0_gaps ]
                str_log = str_log + ("\n  > G0W0 gaps to be  compared - SpinDw: " + str(WF_G0W0_gaps_toCompare["spinDw"]) )

            delta_gap['spinUp'] = abs( WF_G0W0_gaps_toCompare['spinUp'][-1] - WF_G0W0_gaps_toCompare['spinUp'][-2] )
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):  
                delta_gap['spinDw'] = abs( WF_G0W0_gaps_toCompare['spinDw'][-1] - WF_G0W0_gaps_toCompare['spinDw'][-2] )
                
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):  
                flag_is_converged = ( delta_gap['spinUp'] < thr ) and ( delta_gap['spinDw'] < thr )
                str_log = str_log + ("\n  > (Delta (SpinUp) among last two data points): " + str( np.round(delta_gap['spinUp'],decimals=4) ) + " - threshold: " + str(thr) )
                str_log = str_log + ("\n  > (Delta (SpinUp) among last two data points): " + str( np.round(delta_gap['spinDw'],decimals=4) ) + " - threshold: " + str(thr) )
            else:
                flag_is_converged = ( delta_gap['spinUp'] < thr )
                str_log = str_log + ("\n  > (Delta (SpinUp) among last two data points): " + str( np.round(delta_gap['spinUp'],decimals=4) ) + " - threshold: " + str(thr) )
            str_log = str_log + ("  --> Convergence is reached?: " + str(flag_is_converged) )
            
            
            if flag_is_converged:
                str_log = str_log + ("\n  --> K-Convergence reached with k-mesh: " + np.array2string( self.ctx.control['kmesh'][self.ctx.control.iteration_counter] , separator=" ").replace('\n', '')+"\n") 
                self.report(str_log)
                self.ctx.control['kmesh_converged']   = DataFactory('core.array.kpoints')()
                self.ctx.control['kmesh_converged'].set_kpoints_mesh(   self.ctx.control['kmesh'][self.ctx.control.iteration_counter] )

                return False
            else:
                str_log = str_log + ("\n  > K-point convergence NOT reached; launching next k-mesh calculation.\n")
                self.report(str_log)
                self.ctx.control.iteration_counter += 1
                if self.ctx.control.iteration_counter >= len(self.ctx.control['kmesh']):
                    self.report("\n  > Reached maximum number of k-meshes to be tested; convergence not found.")
                    return self.exit_codes.CONVERGENCE_NOT_FOUND
                return True




        if self.ctx.control['control_way'] == 'kdensity':
            # #counter starts at -1 ; it's increased AFTER the convergence check + but BEFORE launching the calculation
            # #The counter increase is the LAST thing done before returgnin
            # #1°control: counter starts=-1  -> [conv. check] -> increased to 0 -> launched 1° G0W0/mBSE
            # #2°control: counter starts= 0  -> [conv. check] -> increased to 1 -> launched 2° G0W0/mBSE
            # #3°control: counter starts= 1  -> [conv. check] -> increased to 2 -> launched 3° G0W0/mBSE
            # #4°control: counter starts= 2  -> convergence is checked; if not changed to 3 -> launched 3° G0W0/mBSE
            # #[..]
            # #we have N k-density to test:
            # #N-1° control: counter starts= N-3 -> changed to N-2 -> launched N-1° G0W0/mBSE
            # #N°   control: counter starts= N-2 -> changed to N-1 -> launched N° G0W0/mBSE
            # #N+1° control  counter starts= N-1 -> exit with error  
    
        
            #Let's define two helpers functions that will be used to compute the gradient and the extrapolated value of the GWgap
            def _compute_GWgap_gradient( inputs , control , WC_G0W0 , flag_debug = True):
                #Initialize stuff
                Delta_GWgap  = AttributeDict()   ;  Delta_GWgap['spinUp']    = AttributeDict() ; 
                Gradient_GWgap = AttributeDict() ;  Gradient_GWgap['spinUp'] = AttributeDict() ;
                idx_toIterate = list( itertools.product( ['spinUp'] , ['G0W0_Dir', 'G0W0_Ind','G0W0_Gam']) )                
                if ('magnetic_moment_onsite' in inputs['ns_parameters']):  
                    idx_toIterate = list( itertools.product( ['spinUp','spinDw'] , ['G0W0_Dir', 'G0W0_Ind','G0W0_Gam']) )
                    Delta_GWgap['spinDw']    = AttributeDict() 
                    Gradient_GWgap['spinDw'] = AttributeDict() 
                
                #Determine the Two deltas : on GWgap and on kspacaing
                Delta_consecutive_kdensity = np.array(self.ctx.control['kdensity'])[:-1] - np.array(self.ctx.control['kdensity'])[1:]
                for idx in idx_toIterate:
                    Delta_GWgap[idx[0]][idx[1]]= ( np.array([G0W0istance.outputs.gaps.get_dict()[idx[0]][idx[1]]  for G0W0istance in WC_G0W0 ])[:-1] 
                                                 - np.array([G0W0istance.outputs.gaps.get_dict()[idx[0]][idx[1]]  for G0W0istance in WC_G0W0])[1:]   )
                
                for idx in idx_toIterate:
                    Gradient_GWgap[idx[0]][idx[1]] = []
                    for idx_delta in range(len(  Delta_GWgap[idx[0]][idx[1]]  )):
                            
                        #Quick way would be Gradient_GWgap[idx[0]][idx[1]] = Delta_GWgap[idx[0]][idx[1]][idx_delta] / control.Delta_kDensity[idx_delta , :]
                        #Problem of this expression: if two different (consecutive) k-meshes have same number of division along an axis; it always happens for 2D.
                        #Thus we iterate over each direction of control.Delta_kDensity[idx_delta , :]; if Delta_kDensity is = 0 on a direction (i.e. same number of division along that direction)
                        #We take care explicitly of that
                                            
                        tmp_Gradient_singleIteration = [0,0,0]
                        for k_Delta_axisIdx , k_Delta in enumerate( Delta_consecutive_kdensity[idx_delta] ) :
                            if k_Delta == 0:  tmp_Gradient_singleIteration[k_Delta_axisIdx] = 0 
                            else:             tmp_Gradient_singleIteration[k_Delta_axisIdx] = Delta_GWgap[idx[0]][idx[1]][idx_delta] / Delta_consecutive_kdensity[idx_delta, k_Delta_axisIdx ]
                        Gradient_GWgap[idx[0]][idx[1]].append( tmp_Gradient_singleIteration )           #QUI HO UN DUBBIO - MA GIA HO CONTROLLATO
                return Gradient_GWgap  
    
            def _compute_GWgap_extrapolated( inputs , control , WC_G0W0 , flag_debug = True , fit_poly_degree=1 , fit_number_calc_usedForExtr = 3):
                #[Part 1]Initialize stuff
                idx_toIterate = list( itertools.product( ['spinUp'] , ['G0W0_Dir', 'G0W0_Ind','G0W0_Gam']) )
                GWgap_Extrapolated = AttributeDict()      ;  GWgap_Extrapolated['spinUp']      = AttributeDict() ;
                GWgap_Extrapolated_coef = AttributeDict() ;  GWgap_Extrapolated_coef['spinUp'] = AttributeDict() ;
                GWgap_DataForExtr  = AttributeDict()      ;  GWgap_DataForExtr['spinUp']       = AttributeDict() ;
                Delta_GWgap  = AttributeDict()            ;  Delta_GWgap['spinUp']             = AttributeDict() ; 
                if ('magnetic_moment_onsite' in inputs['ns_parameters']):  
                    idx_toIterate = list( itertools.product( ['spinUp','spinDw'] , ['G0W0_Dir', 'G0W0_Ind','G0W0_Gam']) )
                    GWgap_Extrapolated['spinDw'] = AttributeDict() ; GWgap_Extrapolated_coef['spinDw'] = AttributeDict() ;
                    GWgap_DataForExtr['spinDw']  = AttributeDict() ; Delta_GWgap['spinDw']             = AttributeDict() ; 
 
                Delta_consecutive_kdensity = np.array(self.ctx.control['kdensity'])[:-1] - np.array(self.ctx.control['kdensity'])[1:]
    
                #[Part 2]Prepare variables for the interpolation; we perform a separate fit with respect to the densities along the three axis.
                for idx in idx_toIterate:
                    GWgap_DataForExtr[idx[0]][idx[1]] = np.array([G0W0istance.outputs.gaps.get_dict()[idx[0]][idx[1]] for G0W0istance in WC_G0W0])    
                Kdens_DataForExtr = []
                Kdens_DataForExtr.append( np.array(Delta_consecutive_kdensity)[ :len(GWgap_DataForExtr[idx[0]][idx[1]])  , 0 ] )  #first len(GWgap_DataForExtr[idx[0]][idx[1]]) elements of k-density along first axis
                Kdens_DataForExtr.append( np.array(Delta_consecutive_kdensity)[ :len(GWgap_DataForExtr[idx[0]][idx[1]])  , 1 ] )  #first len(GWgap_DataForExtr[idx[0]][idx[1]]) elements of k-density along second axis
                Kdens_DataForExtr.append( np.array(Delta_consecutive_kdensity)[ :len(GWgap_DataForExtr[idx[0]][idx[1]])  , 2 ] )  #first len(GWgap_DataForExtr[idx[0]][idx[1]]) elements of k-density along third axis
    
                
                #[Part 3]We perform a sort of "moving interpolation" ;use only the three last points for the interpolation
                for idx in idx_toIterate:
                    GWgap_DataForExtr[idx[0]][idx[1]] = GWgap_DataForExtr[idx[0]][idx[1]][-fit_number_calc_usedForExtr:]
                Kdens_DataForExtr[0] = Kdens_DataForExtr[0][-fit_number_calc_usedForExtr:]
                Kdens_DataForExtr[1] = Kdens_DataForExtr[1][-fit_number_calc_usedForExtr:]
                Kdens_DataForExtr[2] = Kdens_DataForExtr[2][-fit_number_calc_usedForExtr:]
    
                #[Part 4]Finally we interpolate
                poly = PolynomialFeatures(degree=fit_poly_degree, include_bias=False)   
                poly_features_xaxis = poly.fit_transform( Kdens_DataForExtr[0].reshape(-1, 1) )
                poly_features_yaxis = poly.fit_transform( Kdens_DataForExtr[1].reshape(-1, 1) )
                poly_features_zaxis = poly.fit_transform( Kdens_DataForExtr[2].reshape(-1, 1) )
    
                #fit polynomial regression model
                poly_reg_model_xaxis , poly_reg_model_yaxis , poly_reg_model_zaxis = LinearRegression() , LinearRegression() , LinearRegression()                  
                    
                for idx in idx_toIterate:   
                    poly_reg_model_xaxis.fit(  poly_features_xaxis , GWgap_DataForExtr[idx[0]][idx[1]]  )
                    poly_reg_model_yaxis.fit(  poly_features_yaxis , GWgap_DataForExtr[idx[0]][idx[1]]  )
                    poly_reg_model_zaxis.fit(  poly_features_zaxis , GWgap_DataForExtr[idx[0]][idx[1]]  )
              
                    GWgap_Extrapolated[idx[0]][idx[1]]      = [ poly_reg_model_xaxis.intercept_  ,  poly_reg_model_yaxis.intercept_  ,  poly_reg_model_zaxis.intercept_ ] 
                    GWgap_Extrapolated_coef[idx[0]][idx[1]] = [ poly_reg_model_xaxis.coef_       ,  poly_reg_model_yaxis.coef_       ,  poly_reg_model_zaxis.coef_ ]  
                return GWgap_Extrapolated , GWgap_Extrapolated_coef
           
                        
            #[K-Density only Check - 1] check that allows to return ; If we haven’t done ≥2 calcs yet, just advance beacause the checks uses three points for stability
            if self.ctx.control.iteration_counter <= 1:
                self.ctx.control.iteration_counter += 1
                if self.ctx.control.iteration_counter >= len(self.ctx.control['kmesh']):
                    self.report("\n  > Reached maximum number of k-meshes to be tested; convergence not found.")
                    return self.exit_codes.CONVERGENCE_NOT_FOUND
                
                return True
           

            #This Block determines the flag_is_converged value; does not return anything ---------------------------------
            #Control of convergence starts after completions of first three calcs.
            #remember that the counter is increased as the last thing before returning, and AFTER this check
            #thus before the conv.check of the 3° iteration of the cycle is=1; and after the 3°run is =2
            GWgap_Gradient         = _compute_GWgap_gradient( self.inputs     , self.ctx.control , self.ctx.WC_MBPT )
            GWgap_Extrapolated , _ = _compute_GWgap_extrapolated( self.inputs , self.ctx.control , self.ctx.WC_MBPT )
            
     
            ##[Check - 2] for convergence - using gradient
            thr = self.inputs.ns_kpoints.convergence_threshold.value
            flag_gapType = self.ctx.control['G0W0_gapType_toConverge']
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                flag_is_converged_gradient = ((  np.all( np.abs( GWgap_Gradient['spinUp'][flag_gapType][-1]) <= thr ) ) and
                                              (  np.all( np.abs( GWgap_Gradient['spinDw'][flag_gapType][-1]) <= thr ) ) )
            else:
                flag_is_converged_gradient = np.all( np.abs( GWgap_Gradient['spinUp'][flag_gapType][-1]) <= thr )


            ##[Check - 3]  for convergence - using deviation from extrapolation
            GWgap_DataForExtr  = AttributeDict() 
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):  
                GWgap_DataForExtr['spinUp'] , GWgap_DataForExtr['spinDw']  = AttributeDict() , AttributeDict() 
                idx_toIterate = list( itertools.product( ['spinUp','spinDw'] , ['G0W0_Dir', 'G0W0_Ind','G0W0_Gam']) )
            else: 
                GWgap_DataForExtr['spinUp'] = AttributeDict() ;
                idx_toIterate = list( itertools.product( ['spinUp'] , ['G0W0_Dir', 'G0W0_Ind','G0W0_Gam']) )
            for idx in idx_toIterate:
                GWgap_DataForExtr[idx[0]][idx[1]] = np.array([G0W0istance.outputs.gaps.get_dict()[idx[0]][idx[1]] for G0W0istance in self.ctx.WC_G0W0]) 
                
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                flag_is_converged_extr = (np.all(  np.abs( np.array(GWgap_Extrapolated["spinUp"][flag_gapType]) - np.array(GWgap_DataForExtr['spinUp'][flag_gapType][-1]) ) <= thr) and
                                          np.all(  np.abs( np.array(GWgap_Extrapolated["spinDw"][flag_gapType]) - np.array(GWgap_DataForExtr['spinDw'][flag_gapType][-1]) ) <= thr) )
            else:
                flag_is_converged_extr =  np.all(  np.abs( np.array(GWgap_Extrapolated["spinUp"][flag_gapType]) - np.array(GWgap_DataForExtr['spinUp'][flag_gapType][-1]) ) <= thr)
    
            if self.inputs.ns_opt_converge.use_Gradient: flag_is_converged = flag_is_converged_gradient
            else:                                        flag_is_converged = flag_is_converged_extr
               
            self.report("\n [wkc_KptsConv][monitor_convergence"    
                             +"\n > iteration_counter: " + str(self.ctx.control.iteration_counter)
                             +"\n > kdensity:          " + str(self.ctx.control['kdensity'])
                             +"\n > GWgap_DataForExtr:  " + str(GWgap_DataForExtr)
                             +"\n > GWgap_Gradient:     " + str(GWgap_Gradient)
                             +"\n > GWgap_Extrapolated: " + str(GWgap_Extrapolated)
                             +"\n > flag_is_converged_gradient: " + str(flag_is_converged_gradient)
                             +"\n > flag_is_converged_extr:     " + str(flag_is_converged_extr))
            ##End of the block that determines the flag_is_converged value; that block does not return anything ----------


            #[Check - 4] This block decides whether to return True or False or an exit_code ------------------------------------------
            # the check on the iteration_counter is repeated here for robustness (it was already done at the beginning as check-1)
            if (self.ctx.control.iteration_counter <= 1) or flag_is_converged  == False:
                self.ctx.control.iteration_counter += 1
                if self.ctx.control.iteration_counter >= len(self.ctx.control['kmesh']):
                    self.report("\n  > Reached maximum number of k-meshes to be tested; convergence not found.")
                    return self.exit_codes.CONVERGENCE_NOT_FOUND
                return True
            elif flag_is_converged == True:
                self.ctx.control['kmesh_converged']   = DataFactory('core.array.kpoints')()
                self.ctx.input_DFTG0W0.kpoints.set_kpoints_mesh(   self.ctx.control['kmesh'][self.ctx.control.iteration_counter] )
                return False
            else:
                return self.exit_codes.CONVERGENCE_NOT_FOUND



    
    
    
            
    def elaborate_results(self):
        kmesh_final = np.array(self.ctx.control['kmesh'][self.ctx.control.iteration_counter], dtype=int)
        node_kpoints = DataFactory('core.array.kpoints')()
        node_kpoints.set_kpoints_mesh(kmesh_final)
        node_kpoints.store()
        self.out('kmesh_converged', node_kpoints )
    
        #kdensity = _get_kspacing_from_kmesh(self.inputs.structure.cell, kmesh_final)  # list of 3 floats
       # self.out('final_kDensity', Float(float(np.mean(kdensity))))
    
     
       




