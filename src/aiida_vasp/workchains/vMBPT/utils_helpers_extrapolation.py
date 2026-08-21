# pylint: disable=too-many-arguments

from copy import deepcopy
from aiida_vasp.utils.workchains import site_magnetization_to_magmom


#This file contains miscellaneous calcfunctions and functions used by the workchains.
#Note: the extrapolation calcfunctions (get_closest_EncutNband_multiple,
#get_EncutNbandFitParams_completeBasis_quadratic) that used to live here moved to
#aiida-vasp-gwconv in Phase 2 - this function stays since it's used by aiida-vasp's
#own bucket-A code (DFT/mBSE spin-polarized INCAR construction), not just by
#convergence/extrapolation workchains.


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
