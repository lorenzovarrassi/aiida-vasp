import numpy as np
from copy import deepcopy
import itertools
from aiida import orm
from aiida.orm import Code, Int, Float, Str, Dict, Bool , List , RemoteData , ArrayData , BandsData , XyData
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine  import WorkChain, calcfunction , ToContext , append_ , submit, while_ , if_ , submit, run
from aiida_vasp.utils.workchains import prepare_process_inputs
from aiida_vasp.utils.aiida_utils import get_data_class
from aiida.common.extendeddicts  import AttributeDict
from aiida_vasp.utils.workchains import site_magnetization_to_magmom
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


import warnings
from .workchain_base import VaspDFTGWWorkChain
from .workchain_G0W0_master import VaspG0W0CompleteWorkChain

from aiida import load_profile
load_profile()

#Miscellaneous utils functions
def _getKmesh_from_Kspacing(latVec , KSPACING , flag_roundInsteadCeil=True ):
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

def _getKspacing_from_Kmesh(latVec, kmesh): 
        #k-mesh = 2pi * |bi|/ k-density  ->
        
        latVec = np.array( latVec )
        recLatVec= np.zeros((3,3))
        Vol= np.abs( np.dot(latVec[0,:] , np.cross(latVec[1,:],latVec[2,:])) )
        recLatVec[0,:]= np.cross(latVec[1,:],latVec[2,:])  /Vol
        recLatVec[1,:]= np.cross(latVec[2,:],latVec[0,:])  /Vol
        recLatVec[2,:]= np.cross(latVec[0,:],latVec[1,:])  /Vol

        rec_cell_norm = [np.linalg.norm( recLatVec[x,:]) for x in range(3)]    
        return [2*np.pi * rec_cell_norm[idx] / kmesh[idx] for idx in range(len(rec_cell_norm)) ]
        


class VaspG0W0KptsConvWorkChain(WorkChain):
    _next_workchain = WorkflowFactory('vasp.vasp')

    @classmethod
    def define(cls, spec):
            super(VaspG0W0KptsConvWorkChain , cls).define(spec)

            spec.expose_inputs(cls._next_workchain     , exclude=('kpoints','parameters','settings','potential_family','potential_mapping')) 
            spec.expose_inputs(VaspDFTGWWorkChain      , exclude=('kpoints','ns_parameters','ns_reference') ) 
            
            
            spec.input( 'ns_kpoints.use_explicitDivisions_instead_Density' , valid_type=Bool  , required=False , default=lambda:Bool(False),help="TODO")
            spec.input( 'ns_kpoints.convergence_threshold'       , valid_type=Float , required=False , default=lambda:Float(0.1), help="minimum converge value in eV" ) 
            spec.input( 'ns_kpoints.kMesh.startingValue'         , valid_type=List  , required=False , default=lambda:List([4 ,4 ,4] ), help="constraint on the minimum k-point density - that converged k-mesh must guarantee." )
            spec.input( 'ns_kpoints.kMesh.maxValue'              , valid_type=List  , required=False , default=lambda:List([20,20,20]), help="maximum value for the k-point density convergence." ) 
            spec.input( 'ns_kpoints.kMesh.step'                  , valid_type=Float , required=False , default=lambda:Float(0.1), help="minimum constraint that the converged k-mesh must guarantee." ) 
            spec.input( 'ns_kpoints.kDensity.minimum_constraint' , valid_type=Float , required=False , default=lambda:Float(0.2), help="constraint on the minimum k-point density - that converged k-mesh must guarantee." )
            spec.input( 'ns_kpoints.kDensity.startingValue'      , valid_type=Float , required=False , default=lambda:Float(0.4), help="starting value for the k-point density convergence." ) 
            spec.input( 'ns_kpoints.kDensity.maxValue'           , valid_type=Float , required=False , default=lambda:Float(0.2), help="maximum value for the k-point density convergence." ) 
            spec.input( 'ns_kpoints.kDensity.step'               , valid_type=Float , required=False , default=lambda:Float(0.1), help="minimum constraint that the converged k-mesh must guarantee." ) 


            spec.input('ns_parameters.encut'                  , valid_type=Float      , required=False , help='cutoff energy for the wavefunction in eV. ENCUT variable in VASP.')  #ns stands for namespace
            spec.input('ns_parameters.nbands'                 , valid_type=Int        , required=False , help='total number of bands included in the DFT and G0W0 runs. NBANDS variable in VASP.'  )   
            spec.input('ns_parameters.nomega'                 , valid_type=Int        , required=False , help='total number of bands included in the DFT and G0W0 runs. NBANDS variable in VASP.'  )   
            spec.input('ns_parameters.magnetic_moment_onsite' , valid_type=Dict       , required=False , help='Starting collinear on-site magnetic moment ; Syntax is {ElName:value}')

            spec.input('ns_reference.DFTgr_RemoteData'        , valid_type=RemoteData , required=False , help='the DFT ground state wavefunction (WAVECAR) and CHGCAR will be copied from this RemoteData folder as a starting point' )
            
            spec.input('ns_opt_converge.converge_G0W0gap' , valid_type=Bool , required=False , default=lambda:Bool(True)  )
            spec.input('ns_opt_converge.converge_mBSEgap' , valid_type=Bool , required=False , default=lambda:Bool(False) )
            spec.input('ns_opt_converge.use_Gradient'     , valid_type=Bool , required=False , default=lambda:Bool(True)  )
            
            spec.output( 'final_kDensity' , valid_type=Float )
            spec.output( 'final_kMesh'    , valid_type= DataFactory('core.array.kpoints') , required=False)
            
            spec.exit_code(404,'CONVERGENCE NOT FOUND'    ,message='Convergence has not been reached; please relax the threshold / increase the range studied / check the calculations.')

            
            
            spec.outline(
                cls.initialize,
                while_(cls.monitor_convergence)(
                    cls.prepare_run_DFTgr_NSP , 
                    cls.prepare_run_DFTgr_SP  ,
                    cls.prepare_run_DFT_G0W0  ,
                ),
                cls.elaborate_results
            
            )
        
    def initialize(self):

        self.ctx.control = AttributeDict() ; 
        self.ctx.control.iteration_counter = -1
        self.ctx.control['Delta_k']   = AttributeDict()
        self.ctx.control['Delta_obj'] = AttributeDict() ; 
        self.ctx.control.Delta_obj['spinUp'] = AttributeDict() ;
        if ('magnetic_moment_onsite' in self.inputs['ns_parameters']): self.ctx.control.Delta_obj['spinDw'] = AttributeDict() ;


       


        if self.inputs.ns_kpoints.use_explicitDivisions_instead_Density :

        else:
            kspacing_start = max( self.inputs['ns_kpoints']['kDensity']['minimum_constraint'].value , self.inputs['ns_kpoints']['kDensity_startingValue'].value )
            kspacing_max   = max( self.inputs['ns_kpoints']['kDensity']['maxValue'].value , kspacing_start) 
            kspacing_step  = max( self.inputs['ns_kpoints']['convergence_threshold'].value , 0.05)
            
            list_kspacing  = np.arange( kspacing_start , kspacing_max , kspacing_step )
            
            list_kmesh = [_getKmesh_from_Kspacing(self.inputs.structure.cell , KS) for KS in list_kspacing]               
            self.ctx.list_kmesh = np.insert(np.array(list_kmesh , dtype=float) , 0 , list_kspacing , axis=1)
            #self.ctx.list_kmesh[idx,0] = k-point density of idx entry ; self.ctx.list_kmesh[idx,1:] = k-point mesh of idx entry
            
            _ , list_kmesh_uniqueidx   = np.unique( self.ctx.list_kmesh[:,1:] , axis=0 , return_index=True )      
            self.ctx.list_kmesh_unique = self.ctx.list_kmesh[list_kmesh_uniqueidx , :]
            
            #a kmesh is composed by integers, therefore it's density does not exactly corresponds to the values list_kspacing; it's defined as the kmesh with the closest density to the ones in list_kspacing
            #However, to calculate the gradient Delta[gap]/Delta[kdensity] = (gap[idx]-gap[idx-1])/kdensity[idx]-kdensity[i-1]) we prefer to use the density precisely corresponding to the kmesh.
            self.ctx.control['Delta']['kdensity_notRounded'] = [_getKspacing_from_Kmesh(self.inputs.structure.cell , kmesh ) for kmesh in self.ctx.list_kmesh_unique[:,1:]] 

            str_log = ("\n [Initializing K-points convergence]"+
                            "\n > Controlling k-point convergence through density along dimensions in rec.space (in A^{-1})."
                            "\n [input] ns_kpoints.kDensity_minimum_constraint:"+str(self.inputs.ns_kpoints.kDensity.minimum_constraint)+
                            "\n [input] ns_kpoints.kDensity_startingValue:     "+str(self.inputs.ns_kpoints.kDensity.startingValue)     +
                            "\n [input] ns_kpoints.kDensity_step:              "+str(self.inputs.ns_kpoints.kDensity.step)              +                        
                            "\n [input] ns_kpoints.convergence_threshold:      "+str(self.inputs.ns_kpoints.convergence_threshold)+" eV"+
                            "\n > kDensities to-test: "+str(self.ctx.list_kmesh_unique[:,0]) +
                            "\n > corresponding to:   "+np.array2string(self.ctx.list_kmesh_unique[:,1:] , separator=" , ").replace('\n', ''))
             #               "\n > precise k-density from mesh: "+ np.array2string(self.ctx.control['Delta']['kdensity_notRounded'] , separator=" , " , precision=1).replace('\n', '') )
     
                    
                    
    def prepare_run_DFTgr_NSP(self):
        #Flowchart implemented: 
        #[no referenceData is passed][no MagnMom is NOT passed] -> (non-Spin-polarized from scratch)
        #[no referenceData is passed][MagMom is passed]	        -> (non-Spin-polarized from scratch) -> (spin-polarized reading previous CHGCAR)
        #[referenceData is passed][no MagnMom is NOT passed]    -> (non-Spin-polarized reading CHGCAR)
        #[referenceData is passed][MagMom is passed]	        -> (Spin-polarized reading CHGCAR) 
    

        self.ctx.inputs_DFTgr_NSP = AttributeDict()
        self.ctx.inputs_DFTgr_NSP.ns_option , self.ctx.inputs_DFTgr_NSP.ns_parameters , self.ctx.inputs_DFTgr_NSP.ns_reference = AttributeDict() , AttributeDict() , AttributeDict()
        self.ctx.inputs_DFTgr_NSP.update(self.exposed_inputs(self._next_workchain))
        self.ctx.inputs_DFTgr_NSP.clean_workdir = Bool(False)            
        self.ctx.inputs_DFTgr_NSP.potential_family  = self.inputs.potential_family
        self.ctx.inputs_DFTgr_NSP.potential_mapping = self.inputs.potential_mapping
        self.ctx.inputs_DFTgr_NSP.ns_option.run_G0W0 = Bool(False)
                
        self.ctx.inputs_DFTgr_NSP.kpoints = DataFactory('array.kpoints')()
        self.ctx.inputs_DFTgr_NSP.kpoints.set_kpoints_mesh(  np.array(  self.ctx.list_kmesh_unique[self.ctx.control.iteration_counter,1:] , dtype=int))
        
        self.ctx.inputs_DFTgr_NSP.ns_parameters.encut  = self.inputs.ns_parameters.encut
        self.ctx.inputs_DFTgr_NSP.ns_parameters.nbands = self.inputs.ns_parameters.nbands 
        
        
        if  ("DFTgr_RemoteData" not in self.inputs["ns_reference"]): # (non-Spin-polarized from scratch)
            self.report("\n [wkc_KptsConv][Preliminary DFT-GroundState calc - NonSpinPolarized - From scratch]\n               > Lauching DFT(NonSpinPolarized) using workchain_DFT_G0W0 on k-mesh "
                        +np.array2string(self.ctx.list_kmesh_unique[self.ctx.control.iteration_counter,1:] , separator=" , ").replace('\n', '')+"\n")
            runningWC_DFT_NSP = self.submit(VaspDFTGWWorkChain , **self.ctx.inputs_DFTgr_NSP) 
            return ToContext(WC_NSP=append_(runningWC_DFT_NSP))
            
        elif (("DFTgr_RemoteData"       in self.inputs["ns_reference"])      and 
              ("magnetic_moment_onsite" not in self.inputs["ns_parameters"]) ):
            self.report("\n [wkc_KptsConv][Preliminary DFT-GroundState calc - Restarting from"+str(self.inputs.ns_reference.DFTgr_RemoteData)+"]\n               > Lauching DFT(NonSpinPolarized) using VaspDFTGWWorkChain\n")
            self.ctx.inputs_DFTgr_NSP.ns_reference.DFTgr_RemoteData  =  self.inputs.ns_reference.DFTgr_RemoteData
            runningWC_DFT_NSP = self.submit(VaspDFTGWWorkChain , **self.ctx.inputs_DFTgr_NSP) 
            return ToContext(WC_NSP=append_(runningWC_DFT_NSP))
                
    def prepare_run_DFTgr_SP(self):
        self.ctx.inputs_DFTgr_SP = self.ctx.inputs_DFTgr_NSP

        if (("DFTgr_RemoteData"       not in self.inputs["ns_reference"]) and
            ("magnetic_moment_onsite" in self.inputs["ns_parameters"])    ):
            self.report("\n [wkc_KptsConv][Preliminary DFT-GroundState calc - Continuing previous NonSpinPolarized]\n               > Lauching DFT(SpinPolarized) using VaspDFTGWWorkChain\n")
            self.ctx.inputs_DFTgr_SP.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite 
            self.ctx.inputs_DFTgr_SP.ns_reference.DFTgr_RemoteData        = self.ctx.WC_NSP[-1].outputs.RemoteData_DFT
            runningWC_DFT_SP = self.submit(VaspDFTGWWorkChain , **self.ctx.inputs_DFTgr_SP) 
            return ToContext(WC_final=append_(runningWC_DFT_SP))            
  
        elif (("DFTgr_RemoteData"     in self.inputs["ns_reference"])  and
            ("magnetic_moment_onsite" in self.inputs["ns_parameters"]) ):
            self.report("\n [wkc_KptsConv][Preliminary DFT-GroundState calc - Restarting from"+str(self.inputs.ns_reference.DFTgr_RemoteData)+"]\n               > Lauching DFT(SpinPolarized) using VaspDFTGWWorkChain on k-mesh "
                        +np.array2string(self.ctx.list_kmesh_unique[self.ctx.control.iteration_counter,1:] , separator=" , ").replace('\n', '')+"\n")
            self.ctx.inputs_DFTgr_SP.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite 
            self.ctx.inputs_DFTgr_SP.ns_reference.DFTgr_RemoteData       =  self.inputs.ns_reference.DFTgr_RemoteData
            runningWC_DFT_SP = self.submit(VaspDFTGWWorkChain , **self.ctx.inputs_DFTgr_SP) 
            return ToContext(WC_final=append_(runningWC_DFT_SP))          
            
        else:
            self.ctx.WC_final = []
            self.ctx.WC_final.append( self.ctx.WC_NSP[-1] )
            
        print("self.ctx.WC_NSP"   , self.ctx.WC_NSP  )
        print("self.ctx.WC_final" , self.ctx.WC_final)
     
    def prepare_run_DFT_G0W0(self):
        self.ctx.input_DFTG0W0 = AttributeDict()
        self.ctx.input_DFTG0W0.ns_option , self.ctx.input_DFTG0W0.ns_parameters , self.ctx.input_DFTG0W0.ns_reference = AttributeDict() , AttributeDict() , AttributeDict()
        self.ctx.input_DFTG0W0.update(self.exposed_inputs( VaspDFTGWWorkChain ))
        #self.ctx.input_DFTG0W0.update(self.exposed_inputs(self._next_workchain))
        self.ctx.input_DFTG0W0.clean_workdir = Bool(False)            
        self.ctx.input_DFTG0W0.potential_family  = self.inputs.potential_family
        self.ctx.input_DFTG0W0.potential_mapping = self.inputs.potential_mapping
        self.ctx.input_DFTG0W0.ns_option.run_G0W0 = Bool(True)
               
        self.ctx.input_DFTG0W0.kpoints = DataFactory('array.kpoints')()
        self.ctx.input_DFTG0W0.kpoints.set_kpoints_mesh(  np.array(  self.ctx.list_kmesh_unique[self.ctx.control.iteration_counter,1:] , dtype=int))
        
        self.ctx.input_DFTG0W0.ns_parameters.encut  = self.inputs.ns_parameters.encut
        self.ctx.input_DFTG0W0.ns_parameters.nbands = self.inputs.ns_parameters.nbands  
        self.ctx.input_DFTG0W0.ns_parameters.nomega = self.inputs.ns_parameters.nomega 
        self.ctx.input_DFTG0W0.ns_parameters.encut_chi = Float( self.ctx.input_DFTG0W0.ns_parameters.encut * 0.50 )
        
        if ("magnetic_moment_onsite" in self.inputs["ns_parameters"]):
            self.ctx.input_DFTG0W0.ns_parameters.magnetic_moment_onsite = self.inputs.ns_parameters.magnetic_moment_onsite 
         
        self.ctx.input_DFTG0W0.ns_reference.DFTgr_RemoteData =  self.ctx.WC_final[-1].outputs.RemoteData_DFT
        
        self.report("\n [wkc_KptsConv][G0W0 - Launching VaspDFTGWWorkChain on k-mesh "
                    +np.array2string(self.ctx.list_kmesh_unique[self.ctx.control.iteration_counter,1:] , separator=" , ").replace('\n', '')+"\n")
        runningWC_G0W0 = self.submit(VaspDFTGWWorkChain , **self.ctx.input_DFTG0W0) 
        return ToContext(WC_G0W0=append_(runningWC_G0W0))       
    
    
    
    
    
    def monitor_convergence(self): 
        #counter starts at -1
        #1°control: counter starts=-1  -> changed to 0 -> launched 1° G0W0/mBSE
        #2°control: counter starts= 0  -> changed to 1 -> launched 2° G0W0/mBSE
        #3°control: counter starts= 1  -> changed to 2 -> launched 3° G0W0/mBSE
        #4°control: counter starts= 2  -> convergence is checked; if not changed to 3 -> launched 3° G0W0/mBSE
        #[..]
        #we have N k-density to test:
        #N-1° control: counter starts= N-3 -> changed to N-2 -> launched N-1° G0W0/mBSE
        #N°   control: counter starts= N-2 -> changed to N-1 -> launched N° G0W0/mBSE
        #N+1° control  counter starts= N-1 -> exit with error
        
        def _compute_GWgap_gradient( inputs , control , WC_G0W0 , flag_debug = True):
            Gradient_GWgap = AttributeDict() ;  Gradient_GWgap['spinUp'] = AttributeDict() ;
            if ('magnetic_moment_onsite' in inputs['ns_parameters']): Gradient_GWgap['spinDw'] = AttributeDict() 
            
            
            if ('magnetic_moment_onsite' in inputs['ns_parameters']):  idx_toIterate = list( itertools.product( ['spinUp','spinDw'] , ['G0W0_dir', 'G0W0_Ind','G0W0_Gam']) )
            else: idx_toIterate = list( itertools.product( ['spinUp'] , ['G0W0_Dir', 'G0W0_Ind','G0W0_Gam']) )
            
            control.Delta_kDensity = np.array(control['Delta']['kdensity_notRounded'])[:-1,:] - np.array(control['Delta']['kdensity_notRounded'])[1:,:]
            for idx in idx_toIterate:
                control.Delta_GWgap[idx[0]][idx[1]]= (np.array([G0W0istance.outputs.gaps.get_dict()[idx[0]][idx[1]]   for G0W0istance in WC_G0W0 ])[:-1] 
                                                               - np.array([G0W0istance.outputs.gaps.get_dict()[idx[0]][idx[1]] for G0W0istance in WC_G0W0])[1:]   )
     
            
            for idx in idx_toIterate:
                Gradient_GWgap[idx[0]][idx[1]] = []
                for idx_delta in range(len(  control.Delta_GWgap[idx[0]][idx[1]]  )):
                        
                    #Quick way would be Gradient_GWgap[idx[0]][idx[1]] = control.Delta_GWgap[idx[0]][idx[1]][idx_delta] / control.Delta_kDensity[idx_delta , :]
                    #Problem of this expression: if two different (consecutive) k-meshes have same number of division along an axis; it always happens for 2D.
                    #Thus we iterate over each direction of control.Delta_kDensity[idx_delta , :]; if Delta_kDensity is = 0 on a direction (i.e. same number of division along that direction)
                    #We take care explicitly of that
                    
                    print("DEBUG - control.Delta_GWgap[idx[0]][idx[1]]", control.Delta_GWgap[idx[0]][idx[1]])
                    print("DEBUG - control.Delta_kDensity[idx_delta", control.Delta_kDensity[:,0])
                    
                    tmp_Gradient_singleIteration = [0,0,0]
                    for k_Delta_axisIdx , k_Delta in enumerate( control.Delta_kDensity[idx_delta, :] ) :
                        if k_Delta == 0:  tmp_Gradient_singleIteration[k_Delta_axisIdx] = 0 
                        else:             tmp_Gradient_singleIteration[k_Delta_axisIdx] = control.Delta_GWgap[idx[0]][idx[1]][idx_delta] / control.Delta_kDensity[idx_delta, k_Delta_axisIdx ]
                    Gradient_GWgap[idx[0]][idx[1]].append( tmp_Gradient_singleIteration )           #QUI HO UN DUBBIO - MA GIA HO CONTROLLATO
            if flag_debug: self.report("\n > self.ctx.control at the end of gradient computation",control)      


            return Gradient_GWgap  

        def _compute_GWgap_extrapolated( inputs , control , WC_G0W0 , flag_debug = True , fit_poly_degree=1 , fit_number_calc_usedForExtr = 3):
            #[Part 1]Initialized
            GWgap_Extrapolated = AttributeDict()      ;  GWgap_Extrapolated['spinUp'] = AttributeDict() ;
            GWgap_Extrapolated_coef = AttributeDict() ;  GWgap_Extrapolated_coef['spinUp'] = AttributeDict() ;
            GWgap_DataForExtr  = AttributeDict()      ;  GWgap_DataForExtr['spinUp'] = AttributeDict() ;
            if ('magnetic_moment_onsite' in inputs['ns_parameters']): 
                GWgap_Extrapolated['spinDw']      = AttributeDict() 
                GWgap_Extrapolated_coef['spinDw'] = AttributeDict()
                GWgap_DataForExtr['spinDw']       = AttributeDict() 
                        
            if ('magnetic_moment_onsite' in inputs['ns_parameters']):  idx_toIterate = list( itertools.product( ['spinUp','spinDw'] , ['G0W0_dir', 'G0W0_Ind','G0W0_Gam']) )
            else: idx_toIterate = list( itertools.product( ['spinUp'] , ['G0W0_Dir', 'G0W0_Ind','G0W0_Gam']) )

  
            #[Part 2]Prepare variables for the interpolation; we perform a separate fit with respect to the densities along the three axis.
            for idx in idx_toIterate:
                GWgap_DataForExtr[idx[0]][idx[1]] = np.array([G0W0istance.outputs.gaps.get_dict()[idx[0]][idx[1]] for G0W0istance in WC_G0W0])    
            Kdens_DataForExtr = []
            Kdens_DataForExtr.append( np.array(control['Delta']['kdensity_notRounded'])[ :len(GWgap_DataForExtr[idx[0]][idx[1]])  , 0 ] )  #first len(GWgap_DataForExtr[idx[0]][idx[1]]) elements of k-density along first axis
            Kdens_DataForExtr.append( np.array(control['Delta']['kdensity_notRounded'])[ :len(GWgap_DataForExtr[idx[0]][idx[1]])  , 1 ] )  #first len(GWgap_DataForExtr[idx[0]][idx[1]]) elements of k-density along second axis
            Kdens_DataForExtr.append( np.array(control['Delta']['kdensity_notRounded'])[ :len(GWgap_DataForExtr[idx[0]][idx[1]])  , 2 ] )  #first len(GWgap_DataForExtr[idx[0]][idx[1]]) elements of k-density along third axis

            
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
       
        flag_is_converged = False
        flag_gapType = "G0W0_Dir"

          
        if (self.ctx.control.iteration_counter >  1)  :  #control of convergence starts after completions of first three calcs.


            self.ctx.control.GWgap_Gradient         = _compute_GWgap_gradient( self.inputs     , self.ctx.control , self.ctx.WC_G0W0 )
            self.ctx.control.GWgap_Extrapolated , _ = _compute_GWgap_extrapolated( self.inputs , self.ctx.control , self.ctx.WC_G0W0 )
        
 
            ##[Check for convergence - using gradient]
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                flag_is_converged_gradient = ((  np.all( np.abs( self.ctx.control.GWgap_Gradient['spinUp'][flag_gapType][-1]) <= self.inputs.ns_kpoints.convergence_threshold) ) and
                                     (  np.all( np.abs( self.ctx.control.GWgap_Gradient['spinDw'][flag_gapType][-1]) <= self.inputs.ns_kpoints.convergence_threshold) ) )
            else:
                flag_is_converged_gradient = np.all( np.abs( self.ctx.control.GWgap_Gradient['spinUp'][flag_gapType][-1]) <= self.inputs.ns_kpoints.convergence_threshold )
            
            
            ##[Check for convergence - using deviation from extrapolation]
            GWgap_DataForExtr  = AttributeDict() 
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):  
                GWgap_DataForExtr['spinUp'] , GWgap_DataForExtr['spinDw']  = AttributeDict() , AttributeDict() 
                idx_toIterate = list( itertools.product( ['spinUp','spinDw'] , ['G0W0_dir', 'G0W0_Ind','G0W0_Gam']) )
            else: 
                GWgap_DataForExtr['spinUp'] = AttributeDict() ;
                idx_toIterate = list( itertools.product( ['spinUp'] , ['G0W0_Dir', 'G0W0_Ind','G0W0_Gam']) )
            for idx in idx_toIterate:
                GWgap_DataForExtr[idx[0]][idx[1]] = np.array([G0W0istance.outputs.gaps.get_dict()[idx[0]][idx[1]] for G0W0istance in self.ctx.WC_G0W0]) 
            
            if ('magnetic_moment_onsite' in self.inputs['ns_parameters']):
                flag_is_converged_extr = (np.all(  np.abs( self.ctx.control.GWgap_Extrapolated["spinUp"][flag_gapType] - GWgap_DataForExtr['spinUp'][flag_gapType][-1] ) <= self.inputs.ns_kpoints.convergence_threshold) and
                                          np.all(  np.abs( self.ctx.control.GWgap_Extrapolated["spinDw"][flag_gapType] - GWgap_DataForExtr['spinDw'][flag_gapType][-1] ) <= self.inputs.ns_kpoints.convergence_threshold) )
            else:
                flag_is_converged_extr =  np.all(  np.abs( self.ctx.control.GWgap_Extrapolated["spinUp"][flag_gapType] - GWgap_DataForExtr['spinUp'][flag_gapType][-1] ) <= self.inputs.ns_kpoints.convergence_threshold )


            if self.inputs.ns_opt_converge.use_Gradient: flag_is_converged = flag_is_converged_gradient
            else:                                        flag_is_converged = flag_is_converged_extr
            
            self.report("\n [wkc_KptsConv][monitor_convergence"    
                         +"\n > iteration_counter: " + str(self.ctx.control.iteration_counter)
                         +"\n > kdensity_notRounded:" + str(self.ctx.control['Delta']['kdensity_notRounded'])
                         +"\n > GWgap_DataForExtr:  " + str(GWgap_DataForExtr)
                         +"\n > GWgap_Gradient:     " + str(self.ctx.control.GWgap_Gradient)
                         +"\n > GWgap_Extrapolated: " + str(self.ctx.control.GWgap_Extrapolated)
                         +"\n > flag_is_converged_gradient: " + str(flag_is_converged_gradient)
                         +"\n > flag_is_converged_extr:     " + str(flag_is_converged_extr))

 
        if self.ctx.control.iteration_counter >= ( len(self.ctx.list_kmesh_unique) - 1) :
            return self.exit_codes.ERROR_RETURNED_NO_CALCULATION  # pylint: disable=no-member
       
        if (self.ctx.control.iteration_counter <= 1) or flag_is_converged == False:
            self.ctx.control.iteration_counter = self.ctx.control.iteration_counter + 1
            return True 
        else:
            return False
        



    
        
    
    
    
            
    def elaborate_results(self):
            
            self.out( 'final_kDensity' , self.ctx.list_kmesh_unique[self.ctx.control.iteration_counter,0]  )
            self.out( 'final_kMesh'    , self.ctx.list_kmesh_unique[self.ctx.control.iteration_counter,1:] )
            
            print("in last function - end")
            
 
   




