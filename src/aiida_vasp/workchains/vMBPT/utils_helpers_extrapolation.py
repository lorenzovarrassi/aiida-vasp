# pylint: disable=too-many-arguments

import numpy as np
from copy import deepcopy
from aiida.common.extendeddicts import AttributeDict
from aiida import orm
from aiida.orm import Code, Int, Float, Str, Dict, Bool , List , RemoteData , ArrayData , BandsData , XyData
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import WorkChain, calcfunction , ToContext , append_ , submit, while_
from aiida_vasp.utils.workchains import prepare_process_inputs
import warnings
from sklearn.linear_model import LinearRegression
from aiida_vasp.utils.workchains import site_magnetization_to_magmom


#This file contains miscellaneous calcfunctions and functions used by the workchains.






#<->-------------------------------------------------------------------------------------------------------------------------------------------------------------------------<->
#<->-------------------------------------------------------------------------------------------------------------------------------------------------------------------------<->
#Suppose we have an arbitrary couple of  encut-nbands values connected by the complete basis hypothesis (from now on CBH; to understand it see PHYSICAL REVIEW B 90, 075125 (2014) ).
#This creates a problem: given an arbitrary NBANDS value in the INCAR, VASP automatically rounds it to the nearest multiple of (#mpi-threads)/KPAR. This may break the complete-basis relation if ENCUT is not corrected correspondingly.
#Thus to respect the CBH we have to determine (and use) the ENCUT cutoff values corresponding (through the complete-basis hypothesis) to nbands roundes to the nearest multiple of (#mpi-threads)/KPAR.
# The function get_closest_EncutNband_multiple, given an arbitrary cutoff value (encut_original) returns the cutoff-nbands couple (linked by the complete-basis hypothesis) with the nbands multiple of
# (#mpi-threads)/KPAR closest (from higher up) to nbands correspondonding to encut_original. 

#       BTW under the complete basis hypothesis is possible to determine one from the another:
#       (encut , nbands) = (encut , get_PWNum_fromEncut_underCBH(encut , [..]) ) or from nbands as (encut , nbands) = (get_nbands_fromEncut_underCBH_fitInversion(nbands , [..]) , nbands)



@calcfunction
def get_closest_EncutNband_multiple(kpoints , structure , FFT_NG_gridpoints , POTCAR_ENMAX_max , param_quad , 
                                    nbands_divisor_constraint , starting_parameter_value , input_type , 
                                    flag_twoSidesRounding=lambda:Bool(True), 
                                    label_str_for_log=lambda:Str() ,
                                    verbose=lambda:Bool(True)      ):
    """
    given an arbitrary ENCUT cutoff value OR an arbitrary NBANDS value and a k-point lists
    the function a) calculates the other value for each k-point under the complete basis hypothesis
                    in order to have a list of cutoff-nbands values (one for each k-point) constrained by the CBH.
                 b) nbands is rounded to (#mpi-threads)/KPAR closest to the original value. 
    The original value is passed to starting_parameter_value; the type of the original value (if it represents a cutoff or a nbands) is determined by input_type.
    get_closest_EncutNband_multiple returns a a list of dicts {'encut'; <Aiida-Float> , 'nbands': <Aiida-Int>}, one for each k-point.

    Parameters
    ----------
    kpoints : KpointsData
        List of kpoints to calculate the (encut, nbands).
    structure : StructureData
    FFT_NG_gridpoints : List
        Dimension of the FFT grid used by VASP in the DFT calculations.
    POTCAR_ENMAX_max : Int
        correponds to the maximum ENMAX parameter among the POTCARs used
    param_quad : List
        list of the paratemeters determined form the fit.
    nbands_divisor_constraint : Int 
        Number of mpi_thread the calculation is using; VASP rounds NBANDS to the nearest multiple of (#mpi-threads)/KPAR. 
    starting_parameter_value : Float
        Represent the starting value (cutoff or NBANDS variable).
    input_type : Str
        It can take ONLY the values Str("encut") or Str("nbands"). if "encut" is passed, the functions returns the (encut, nbands) under the CBH with nbands rounded as discussed above.
        if Str("nbands") is passed, starting_parameter_value is considered as a NBANDS variable; if Str("encut") is passed, is considered as a cutoff.
    flag_twoSidesRounding : Bool, optional
    label_str_for_log   : Str, optional
        label that it's added to the log string returned
    verbose : Bool, optional    
    It returns a List([ output_array , string_log ])
    where output_array = AttributeDict() with keys "encut", "nbands"
    """
    # Additional functions needed for get_EncutNbandFitParams_completeBasis_quadratic ---------------------   

    def _get_recLatVec(latVec):
        """
        Determines the reciprocal lattice vector from the direct (in real space) lattice vectors.

        """
        recLatVec= np.zeros((3,3)) 
        Vol= np.abs( np.dot(latVec[0,:] , np.cross(latVec[1,:],latVec[2,:])) )
        recLatVec[0,:]= np.cross(latVec[1,:],latVec[2,:])  /Vol
        recLatVec[1,:]= np.cross(latVec[2,:],latVec[0,:])  /Vol
        recLatVec[2,:]= np.cross(latVec[0,:],latVec[1,:])  /Vol
        return recLatVec

    def _get_LPCT(NG):
        """
        corresponds to SUBROUTINE INILPC (NGX,NGY,NGZ,LPCTX,LPCTY,LPCTZ) in the source code
        LPCTX,LPCTY,LPCTZ are loop counters that label the number of the reciprocal lattice vectors in the x,y,z directions, respectively. 
        For the x direction the reciprocal lattice vectors corresponding to the first,second,...,ngxth elements in all of the reciprocal 
        lattice arrays are 0,1,..,(NGX/2),-((NGX/2-1),..,-1 times the x reciprocal lattice vector
        """
        
        LPCTX=np.zeros(NG[0])
        LPCTY=np.zeros(NG[1])
        LPCTZ=np.zeros(NG[2])

        for NX in range(1,int(NG[0]/2)+2):        #range does not include last term, thus to inclide NG[0]/2+1 we should include NG[0]/2+2 in range
                LPCTX[NX-1] = NX - 1               #[NX-1] because Python array inexing starts from 0, in Fortran from 1
        for NX in range(int(NG[0]/2)+2, NG[0]+1): #same reason of above range
                LPCTX[NX-1] = NX -1 - NG[0]
                
        for NY in range(1,int(NG[1]/2)+2):        #range does not include last term, thus to inclide NG[0]/2+1 we should include NG[0]/2+2 in range
                 LPCTY[NY-1] = NY - 1              #[NX-1] because Python array inexing starts from 0, in Fortran from 1
        for NY in range(int(NG[1]/2)+2, NG[1]+1): 
                 LPCTY[NY-1] = NY - 1 - NG[1]
                 
        for NZ in range(1,int(NG[2]/2)+2):        #range does not include last term, thus to inclide NG[0]/2+1 we should include NG[0]/2+2 in range
                 LPCTZ[NZ-1] = NZ - 1              #[NX-1] because Python array inexing starts from 0, in Fortran from 1
        for NZ in range(int(NG[2]/2)+2, NG[2]+1): 
                 LPCTZ[NZ-1] = NZ - 1 - NG[2]                
                
        return LPCTX , LPCTY , LPCTZ
         
    def _get_PWNum_fromEncut_underCBH_singleKPTS(kpt , recLatVec , encut , NG , LPCTX , LPCTY , LPCTZ): #CBH stands for Complete Basis Hypothesis, see PRB90, 075125 (2014) 
        #The functions determines the dimension of the plane wave basis set for a specific k-point and ENCUT.
        #Is compatible with the values determined by VASP in the source code file wave.F
    
        #c_RytoeV = 13.605826
        #c_2pi    = 6.2831853
        #c_AUtoA  = 0.52917725   
        #c_HSQDTM = c_RytoeV*c_AUtoA*c_AUtoA
        #The values below corresponds to the ones VASP uses.
        c_2pi   = 6.2831853071795862 
        c_HSQDTM= 3.8100198740807945
        
        recLatVec = np.transpose(recLatVec)   
        
       
        RC_I2 = np.zeros(NG[1]*NG[2] , dtype=np.int8)
        RC_I3 = np.zeros(NG[1]*NG[2] , dtype=np.int8)  
        IND=0
        for N3 in range(NG[2]):
            for N2 in range(NG[1]):
                RC_I2[IND] = N2
                RC_I3[IND] = N3
                IND = IND + 1

        NLBOXI  = 0
        counter = 0
        for NC in range(NG[1]*NG[2]):  
            IN_THIS_ROW = 0         
             
            N2 = RC_I2[NC]  #N2=GRID%RC%I2(NC)   
            N3 = RC_I3[NC]  #N3=GRID%RC%I3(NC) 
         
            for N1 in range(NG[0]): 
                NLBOXI=NLBOXI+1

                G1=LPCTX[N1] + kpt[0]
                G2=LPCTY[N2] + kpt[1]
                G3=LPCTZ[N3] + kpt[2]
                GX=(G1*recLatVec[0,0]    +G2*recLatVec[0,1]   +G3*recLatVec[0,2]  )*c_2pi
                GY=(G1*recLatVec[1,0]    +G2*recLatVec[1,1]   +G3*recLatVec[1,2]  )*c_2pi
                GZ=(G1*recLatVec[2,0]    +G2*recLatVec[2,1]   +G3*recLatVec[2,2]  )*c_2pi   
                
                Energy= c_HSQDTM*((GX**2)+(GY**2)+(GZ**2))     
                if (Energy < encut):  
                    counter=counter+1 
        
        return counter 
   
    def get_PWNum_fromEncut_underCBH(kptsArray , latVec , encut , NG , verbose=True):
        #kptsArray should have form e.g. [[0,0,0]] , [[0, 0, 0], [0.33, 0.33, 0.33]] 
        pwNum_array=np.zeros (  np.shape(kptsArray)[0]  )
        rlv  = _get_recLatVec(latVec)
        LPCT = _get_LPCT(NG)   
        for index, kpt in enumerate(kptsArray):
            pwNum_array[index] = _get_PWNum_fromEncut_underCBH_singleKPTS(kpt , rlv , encut , NG , LPCT[0] , LPCT[1] , LPCT[2])
    
        if verbose:
            print("\n[3.1 inside subroutine get_PWNum_fromEncut_underCBH]"
            +"\n  > calculating number of plane waves for kpts ="+str(*kptsArray)           #* to print different elements of list on same line
            +"\n  >                                   for encut="+str(encut)
            +"\n  > PW number per ktps ="+str(pwNum_array.astype(int)) +"\n")
    
        return int(max(pwNum_array))

    def get_nbands_fromEncut_underCBH_fitInversion(nband , param_quad , verbose=True): 
        """
        Given the parameters of the fit in the form param_quad[0] + param_quad[1]*x + param_quad[2]*x**2
        For a given bands return the corresponding encut under the hypothesis of complete basis (invert the encut-nband relation).
        """
        a=param_quad[2] #coefficient of quadratic term
        b=param_quad[1] #coefficient of linear term
        c=param_quad[0] #offset
        if isinstance(nband,int):
            encut = ( np.sqrt( nband -   c + (0.25*b**2/a)  ) -  0.5*b*(1/np.sqrt(a)) ) * (1/np.sqrt(a))
            return encut
        elif isinstance(nband,list) or isinstance(nband, np.ndarray):
            encut = []
            for Nb in nband:
                encut.append( ( np.sqrt( Nb -   c + (0.25*b**2/a)  ) -  0.5*b*(1/np.sqrt(a)) ) * (1/np.sqrt(a)) )
            return encut
        else:
            print("ERROR: WRONG format for nband in get_nbands_fromEncut_underCBH_fitInversion: either an int, array or numpy array.")
            return None  

    #Extracting the python-type variables from the AiiDA-type variables.
    latVec      = np.array(structure.cell)    #np.array to avoid error TypeError: list indices must be integers or slices, not tuple when using latVec[0,:]
    kptsArray   = kpoints.get_kpoints()
    NG          = np.array( FFT_NG_gridpoints.get_array('NGarray') , dtype = int)
    param_array = np.array( param_quad.get_list() )
    divisor_constraint = nbands_divisor_constraint.value
  
    ##[Step 0]
    input_type_normalized = input_type.value if hasattr(input_type, "value") else str(input_type)
    if input_type_normalized == "encut":
        encut_0_original = starting_parameter_value.value
        #Extracting the nbands associated by the complete-basis hypothesis (CBH) to the input encut
        nband_0_original = get_PWNum_fromEncut_underCBH(kptsArray , latVec , encut_0_original, NG , verbose=False) 
    elif input_type_normalized == "nbands":
        nband_0_original = int( starting_parameter_value.value )
        encut_0_original = get_nbands_fromEncut_underCBH_fitInversion(nband_0_original , param_array)
    else:
        raise ValueError("wrong input_type passed: must either be 'encut' or 'nbands' and must be a AiiDA variable.")
       
       
    #[Step 1]Determing the multiple of divisor_constraint closest to nband_0_original - first pass (VASP impose a constraint on NBANDS, as it must be a multiple of total core number)
    #        then we invert the (fitted) relation nbands=nbands(encut) under CBH to determine encut associated to nband_1_closestMultiple
    if nband_0_original % divisor_constraint == 0 :   #nband_0_original is a perfect multiple of divisor_constraint
        nband_1_closestMultiple = nband_0_original
    elif flag_twoSidesRounding : nband_1_closestMultiple = round(nband_0_original / divisor_constraint) * divisor_constraint   #can both round up and round down
    else :                       nband_1_closestMultiple = (nband_0_original // divisor_constraint + 1) * divisor_constraint   #always round up
    #[Step 1] Inverting the relation
    nband_1_closestMultiple = int( nband_1_closestMultiple )
    encut_1_CBH = get_nbands_fromEncut_underCBH_fitInversion( nband_1_closestMultiple , param_array)
    #[Step 1] In case of NaN, inverting through stupid (but functional) linear piecewise interpolation
    if not encut_1_CBH == encut_1_CBH:
        linInt_array_ENCUT = np.multiply( np.array([0.6 , 0.75 , 1.00 , 1.25 , 1.587 , 1.8]) , POTCAR_ENMAX_max.value)  #larger than EMAX_fraction because value can be rounded outside usual Encut-Nbands values, introducing errors.
        linInt_array_Nb    = np.array([get_PWNum_fromEncut_underCBH(kptsArray , latVec , encut, NG , False) for encut in linInt_array_ENCUT])
        encut_1_CBH = np.interp(nband_1_closestMultiple, linInt_array_Nb, linInt_array_ENCUT)  #https://numpy.org/doc/stable/reference/generated/numpy.interp.html
        #index_firstLarger  = np.where( NbAr == next(e for e in NbAr if e>NbObj) )[0][0]
        #index_lastLower    = index_firstLarger-1
        if verbose:
            print("\n  >  encut under CBH from corrected nband is NaN - probsbly fit provides a x^0 or x^2 negative parameter. Falling back to linear interpolation..."
            +"\n  >> values used for linear interpolation"+str(linInt_array_ENCUT)+" "+str(linInt_array_Nb)
            +"\n  >> linear interpolation for nband_closestMultiple = "+str(nband_1_closestMultiple)
            +"\n  -> [1.1] linear interpolated encut = "+str(encut_1_CBH))



    #[Step 2] Refining the values - the inversion done in Step 1 may be imprecise, and CBH could be accurately respect; this way we accurately refine the value predicted in step 1
    flag_More_precise_values = True
    if not flag_More_precise_values:
        nband_2_closestMultiple = nband_1_closestMultiple
        encut_2_CBH = encut_1_CBH
    else:
        linInt_array_encut = np.array([encut_1_CBH])
        linInt_array_nb    = np.array([get_PWNum_fromEncut_underCBH(kptsArray , latVec , encut_1_CBH, NG , False)])
        nband_2_closestMultiple = nband_1_closestMultiple
        TMP_COUNTER = 0
        while (nband_2_closestMultiple < min(linInt_array_nb) or  nband_1_closestMultiple > max(linInt_array_nb) ) and TMP_COUNTER < 200:
            TMP_COUNTER = TMP_COUNTER + 1
            if nband_1_closestMultiple < min(linInt_array_nb) :
                tmp_encut = min(linInt_array_encut)-0.5
                tmp_nb    = get_PWNum_fromEncut_underCBH(kptsArray , latVec , tmp_encut , NG , False)
                linInt_array_encut = np.insert(linInt_array_encut , 0 , tmp_encut )
                linInt_array_nb    = np.insert(linInt_array_nb    , 0 , tmp_nb    )
            if nband_2_closestMultiple > max(linInt_array_nb) :
                tmp_encut = max(linInt_array_encut)+0.5
                tmp_nb    = get_PWNum_fromEncut_underCBH(kptsArray , latVec , tmp_encut , NG , False)
                linInt_array_encut = np.insert(linInt_array_encut , len(linInt_array_encut) , tmp_encut )
                linInt_array_nb    = np.insert(linInt_array_nb    , len(linInt_array_nb )   , tmp_nb    )

        encut_2_CBH = np.interp(nband_2_closestMultiple, linInt_array_nb, linInt_array_encut)  #https://numpy.org/doc/stable/reference/generated/numpy.interp.html

        string_log=("\n  [2.1 routine get_closest_EncutNband_multiple for "+str(label_str_for_log.value)+" ]"
        +"\n   nbands - encut pairs under Complete Basis Hyp."
        +"\n     [0] Starting non-corrected parameter    :"+str(starting_parameter_value.value)
        +"\n      |  Starting parameters represent a:"+str(input_type_normalized)+" value."
        +"\n      ↓  Starting encut-nbands under CBH     : ("+str(encut_0_original)+" , "+str(nband_0_original)+")"
        +"\n     [1] Imposing constraint: nbands must be multiple of:"+str(divisor_constraint)
        +"\n      ↓  Corrected nbands                     :"+str(nband_1_closestMultiple)
        +"\n     [2] encut under CBH from corrected nbands :"+str(encut_1_CBH)
        +"\n      ↓   (fit - param_array :"+str(param_array)+")"
        +"\n     [3] nbands - refining from linear interpol. :"+str(nband_2_closestMultiple)
        +"\n         encut  - refining from linear interpol. :"+str(encut_2_CBH)
        +"\n          ( CBH nbands- x data for lin.interp:"+str(linInt_array_nb)+")"
        +"\n          ( CBH encut - y data for lin.interp:"+str(linInt_array_encut)+")")
    output_dict =  {"encut": float(encut_2_CBH), "nbands": int(nband_2_closestMultiple)}

    return List([ output_dict , string_log ])
#def get_closest_EncutNband_multiple(kpoints , structure , FFT_NG_gridpoints , POTCAR_ENMAX_max , param_quad , nbands_divisor_constraint , starting_parameter_value , input_type , verbose=True , flag_twoSidesRounding=True ):

@calcfunction
def get_EncutNbandFitParams_completeBasis_quadratic(kpoints , structure , FFT_NG_gridpoints , POTCAR_ENMAX_max , verbose=Bool(True)):        
    """
    The complete basis constraint allows to consider a single variables between the encut and nbands as indipendent, 
    and tipically is thought as a function nbands = nbands(encut). 
    nbands=nbands(encut) is determined by counting how many plane waves are included inside a sphere 
    (of radius determined by encut) centered in the brillouin zone on a given k-point.
    (This is exactly what _get_PWNum_fromEncut_underCBH_singleKPTS does).
    
    We have however a problem: in our workchain is often useful to INVERT this relation, i.e. find the encut=encut(nbands)
    In order to do so we fit a quadratic curve to 4 points determined analytically. Our test showed how this is usually very accurate.

    """

    # Additional functions needed for get_EncutNbandFitParams_completeBasis_quadratic ---------------------   
    def _get_recLatVec(latVec):
        """
        Determines the reciprocal lattice vector from the direct (in real space) lattice vectors.
        """
        recLatVec= np.zeros((3,3)) 
        Vol= np.abs( np.dot(latVec[0,:] , np.cross(latVec[1,:],latVec[2,:])) )
        recLatVec[0,:]= np.cross(latVec[1,:],latVec[2,:])  /Vol
        recLatVec[1,:]= np.cross(latVec[2,:],latVec[0,:])  /Vol
        recLatVec[2,:]= np.cross(latVec[0,:],latVec[1,:])  /Vol
        return recLatVec

    def _get_LPCT(NG):
        """
        corresponds to SUBROUTINE INILPC (NGX,NGY,NGZ,LPCTX,LPCTY,LPCTZ) in the source code
        LPCTX,LPCTY,LPCTZ are loop counters that label the number of the reciprocal lattice vectors in the x,y,z directions, respectively. 
        For the x direction the reciprocal lattice vectors corresponding to the first,second,...,ngxth elements in all of the reciprocal 
        lattice arrays are 0,1,..,(NGX/2),-((NGX/2-1),..,-1 times the x reciprocal lattice vector
        """
        
        LPCTX=np.zeros(NG[0])
        LPCTY=np.zeros(NG[1])
        LPCTZ=np.zeros(NG[2])

        for NX in range(1,int(NG[0]/2)+2):        #range does not include last term, thus to inclide NG[0]/2+1 we should include NG[0]/2+2 in range
                LPCTX[NX-1] = NX - 1               #[NX-1] because Python array inexing starts from 0, in Fortran from 1
        for NX in range(int(NG[0]/2)+2, NG[0]+1): #same reason of above range
                LPCTX[NX-1] = NX -1 - NG[0]
                
        for NY in range(1,int(NG[1]/2)+2):        #range does not include last term, thus to inclide NG[0]/2+1 we should include NG[0]/2+2 in range
                 LPCTY[NY-1] = NY - 1              #[NX-1] because Python array inexing starts from 0, in Fortran from 1
        for NY in range(int(NG[1]/2)+2, NG[1]+1): 
                 LPCTY[NY-1] = NY - 1 - NG[1]
                 
        for NZ in range(1,int(NG[2]/2)+2):        #range does not include last term, thus to inclide NG[0]/2+1 we should include NG[0]/2+2 in range
                 LPCTZ[NZ-1] = NZ - 1              #[NX-1] because Python array inexing starts from 0, in Fortran from 1
        for NZ in range(int(NG[2]/2)+2, NG[2]+1): 
                 LPCTZ[NZ-1] = NZ - 1 - NG[2]                
                
        return LPCTX , LPCTY , LPCTZ

    def _get_PWNum_fromEncut_underCBH_singleKPTS(kpt , recLatVec , encut , NG , LPCTX , LPCTY , LPCTZ):
        """ CBH stands for Complete Basis Hypothesis, see PRB90, 075125 (2014) 
        This function calculates the dimension of the plane wave basis set for a given ENCUT and (single) k-point; it's designed to be compatible with the values determined by VASP (see source code wave.F).
        """
        #c_RytoeV = 13.605826
        #c_2pi    = 6.2831853
        #c_AUtoA  = 0.52917725   
        #c_HSQDTM = c_RytoeV*c_AUtoA*c_AUtoA
        #The values below are the actual ones used by VASP.
        c_2pi   = 6.2831853071795862 
        c_HSQDTM= 3.8100198740807945
        
        recLatVec = np.transpose(recLatVec)   

        #OLD- previous - this could cause a bug if NG[1] or NG[2] exceed 127
        #RC_I2 = np.zeros(NG[1]*NG[2] , dtype=np.int8)
        #RC_I3 = np.zeros(NG[1]*NG[2] , dtype=np.int8)  
        RC_I2 = np.zeros(NG[1]*NG[2] , dtype=np.int32)
        RC_I3 = np.zeros(NG[1]*NG[2] , dtype=np.int32) 
        IND=0
        for N3 in range(NG[2]):
            for N2 in range(NG[1]):
                RC_I2[IND] = N2
                RC_I3[IND] = N3
                IND = IND + 1

        NLBOXI  = 0
        counter = 0
        for NC in range(NG[1]*NG[2]):  
            IN_THIS_ROW = 0         
             
            N2 = RC_I2[NC]  #N2=GRID%RC%I2(NC)   
            N3 = RC_I3[NC]  #N3=GRID%RC%I3(NC) 
         
            for N1 in range(NG[0]): 
                NLBOXI=NLBOXI+1

                G1=LPCTX[N1] + kpt[0]
                G2=LPCTY[N2] + kpt[1]
                G3=LPCTZ[N3] + kpt[2]
                GX=(G1*recLatVec[0,0]    +G2*recLatVec[0,1]   +G3*recLatVec[0,2]  )*c_2pi
                GY=(G1*recLatVec[1,0]    +G2*recLatVec[1,1]   +G3*recLatVec[1,2]  )*c_2pi
                GZ=(G1*recLatVec[2,0]    +G2*recLatVec[2,1]   +G3*recLatVec[2,2]  )*c_2pi    
                Energy= c_HSQDTM*((GX**2)+(GY**2)+(GZ**2))     
                if (Energy < encut):  
                    counter=counter+1 
        
        return counter 
    
    def get_PWNum_fromEncut_underCBH(kptsArray , latVec , encut , NG , verbose=True):
        """
        Apply _get_PWNum_fromEncut_underCBH_singleKPTS to all k-points inside kptsArray.

        """
        #kptsArray are compatible to a list of lists, e.g. [[0,0,0]] , [[0, 0, 0], [0.33, 0.33, 0.33]] 
        if verbose:
            print("")
            print("[2.1 get_PWNum_fromEncut_underCBH routine inside get_EncutNbandFitParams_completeBasis_quadratic]")
            print("  > calculating number of plane waves for encut=",encut) 
            print("  >                                   for kpts =",*kptsArray) #* to print different elements of list on same line
    
        pwNum_array=np.zeros (  np.shape(kptsArray)[0]  )
        rlv  = _get_recLatVec(latVec)
        LPCT = _get_LPCT(NG)   
        for index, kpt in enumerate(kptsArray):
            pwNum_array[index] = _get_PWNum_fromEncut_underCBH_singleKPTS(kpt , rlv , encut , NG , LPCT[0] , LPCT[1] , LPCT[2])
    
        if verbose:
            print("  > PW number per ktps =",pwNum_array.astype(int) )

        return int(max(pwNum_array))    
    
    def fit_EncutNband_completeBasis_quadratic(encut_array , nband_array , verbose=False):
        # Fit function to (encut , nband) couples
        def fitFun_quadratic(x, p_0, p_1, p_2):	  return p_0 +  p_1*x + p_2*(x**2)
        

        import scipy.optimize as sp
        from matplotlib import pyplot
        param_quad, _  = sp.curve_fit(fitFun_quadratic, encut_array, nband_array)
        #param_exp, _   = sp.curve_fit(fitFun_exp      , encut_array, nband_array)

        x_line = np.arange(400,1200, 1)
        y_line_quad  = fitFun_quadratic(x_line , param_quad[0], param_quad[1], param_quad[2])
        
        #often gives OptimizeWarning: Covariance of the parameters could not be estimated
        #if you don't like it https://stackoverflow.com/questions/50371428/scipy-curve-fit-raises-optimizewarning-covariance-of-the-parameters-could-not

        return [param_quad[0], param_quad[1], param_quad[2]]
    # Additional functions needed for get_EncutNbandFitParams_completeBasis_quadratic now finished. -------
    # Let's go back to the (very short) body of the function. -------------------------------------------   


    c_RytoeV = 13.605826
    c_2pi    = 6.2831853
    c_AUtoA  = 0.52917725
   
    latVec      = np.array(structure.cell)    #np.array to avoid error TypeError: list indices must be integers or slices, not tuple when using latVec[0,:]
    kptsArray   = kpoints.get_kpoints()
    NG          = np.array( FFT_NG_gridpoints.get_array('NGarray') , dtype = int)
    Encut_array = np.multiply([0.9 , 1.25 , 1.5] , POTCAR_ENMAX_max.value ) # To determine the fit we use three points, corresponding to the memory conserving procedure; this is a bit arbitrary, CAN BE IMPROVED.
                                                                     # np.array to avoid error TypeError: 'numpy.float64' object cannot be interpreted as an integer
                                                                     # when used to define dimensions, for example in LPCTX=np.zeros(NG[0]+1)
    
    array_PWnum_atVariousEncut = [get_PWNum_fromEncut_underCBH(kptsArray , latVec , i, NG , verbose=False) for i in Encut_array]
    warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")
    p_quad = fit_EncutNband_completeBasis_quadratic(Encut_array , array_PWnum_atVariousEncut , verbose=False)
    
    if verbose:
       print("\n[preparatory-2 determining the ENCUT-NBANDS curve under CBH via fit]--- --- --- --- --- --- ---"
       +"\n > inside function fit_EncutNband_completeBasis_quadratic"
       +"\n >> Encut_array used for fit  :"+str(Encut_array)
       +"\n >> Nbands_array used for fit :"+str(array_PWnum_atVariousEncut)
       +"\n >> params resulting from fit :"+str(p_quad)
       +"\n[2 determining the ENCUT-NBANDS curve]--- --- --- --- --- --- --- --- --- --- --- --- --- --- -")
    
    return List(p_quad)
#<->-------------------------------------------------------------------------------------------------------------------------------------------------------------------------<->
#<->-------------------------------------------------------------------------------------------------------------------------------------------------------------------------<->



#Miscellaneous utils function
def input_magnetic_moment_tomagmom(structure , magnetic_moment_onsite):
    """
    Convert a Dict containing the magnetic moment per site with a format VASP-like like magnetic_moment_onsite = {'Cr1':3.0 , 'Cr2':-3.0 , 'Cr3':-3.0 , 'Cr4':3.0}
    To the one requested by AiiDA.
    
    The problem here is that AiiDA often changes order of elements inside POSCAR with respect to POSCAR supplied;
    #Solution: get_ase().get_chemical_symbols()  is identical to the order chosen by AiiDA;
    we reorder the magnetic_moment_onsite supplied based on this, and define MAGMOM from that.
    """
        
    elements_sorted = structure.get_ase().get_chemical_symbols()


    #Costruct for ease of use an intermediate representation of the onsite magnetization:
    #ES:  magnetic_moment_onsite = {'Cr1':3.0 , 'Cr2':-3.0 , 'Cr3':-3.0 , 'Cr4':3.0}
    #then:magmom_intermRep = [['Cr', 1, 3.0], ['Cr', 2, -3.0], ['Cr', 3, -3.0], ['Cr', 4, 3.0]]
    magmom_intermRep = []
    for el in magnetic_moment_onsite:
        if el[-1].isdigit(): magmom_intermRep.append([ el[:-1] , int(el[-1]) , magnetic_moment_onsite[el] ])
        else:                magmom_intermRep.append([ el      , 1           , magnetic_moment_onsite[el] ])

        
    flag_MAGMOM = {'site_magnetization': {'sphere': {'x': {'site_moment': { }}}}}
    for el_idx,el in enumerate(elements_sorted):
            flag_MAGMOM['site_magnetization']['sphere']['x']['site_moment'][el_idx+1] = {'tot':0}  
    magmom_intermRep_notYetFound = deepcopy(magmom_intermRep)
    for el_sorted_idx , el_sorted in enumerate(elements_sorted):
        for el_magmom_idx , _ in enumerate(magmom_intermRep_notYetFound) : 
            if el_sorted == magmom_intermRep_notYetFound[el_magmom_idx][0]:
                flag_MAGMOM['site_magnetization']['sphere']['x']['site_moment'][el_sorted_idx+1]['tot'] =  magmom_intermRep_notYetFound.pop(el_magmom_idx)[2]
                break    
    return [flag_MAGMOM , site_magnetization_to_magmom(flag_MAGMOM)]
