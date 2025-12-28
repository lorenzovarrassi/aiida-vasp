# -*- coding: utf-8 -*-
from xml.etree import cElementTree as ET
import numpy as np
import scipy
from scipy import optimize
from pymatgen.io.vasp.outputs import Eigenval , Vasprun
from pymatgen.electronic_structure.core import Spin
from os.path import exists
import os 
from copy import deepcopy
import spglib 
import pymatgen.electronic_structure.bandstructure
from itertools import islice
import itertools
import re
import shutil
import argparse



class bandsObj_: ##<----------------------------------------------------------->
#1] we cannot pymatgen.electronic_structure.bandstructure module because occupation is not saved as matrix, but determined from Efermi
#   but want to keep close-compatibility with that https://pymatgen.org/pymatgen.electronic_structure.bandstructure.html
#   bands form is determined from https://pymatgen.org/pymatgen.io.vasp.outputs.html#pymatgen.io.vasp.outputs.Eigenval
#2] Eigenvalues as a dict of {(spin): np.ndarray(shape=(nkpt, nbands, 2))}. This representation is based on actual ordering in VASP.
#   Which means non-spin.polarized is saved as dict as {Spin(1): ndarray [nkpt, nbands , 2]} - spin.polarized is saved as dict as{Spin(1): ndarray [nkpt, nbands , 2] , Spin(-1): ndarray [nkpt, nbands , 2]}
#   OPPOSITE ORDER FROM OLD-VERSION

    def __init__(self , bands_up=None , bands_dw=None , occup_up=None , occup_dw=None , nelect=None , kpts_list=None , kpts_mesh=None , kpts_shift=None , bands_indexs=None, spglib_structure=None) :      
        self.nelect           = nelect
        self.bands_indexs     = bands_indexs
        self.kpts_mesh        = kpts_mesh
        self.kpts_shift       = kpts_shift
        self.kpts_list        = kpts_list
        self.spglib_structure = spglib_structure
        
        #if bands_up==None and bands_dw==None: 
        #    self.eigenval = None
        #elif bands_dw==None :
        if np.any(bands_up!=None) and np.any(occup_up!=None) and np.any(bands_dw==None): 
            self.is_spin_polarized = False #left due to compatibility with pymatgen structure 
            assert np.shape( bands_up )[0] == len( kpts_list )      , 'kpoints number dimension mismatch between bands_up and kpts_list'
            assert np.shape( bands_up )    == np.shape( occup_up )  , 'dimension mismatch between bands_up and occup_up'
            self.eigenval = {Spin(1): np.zeros( [np.shape(bands_up)[0],np.shape(bands_up)[1],2] )  }
            self.eigenval[Spin(1)][:,:,0] = bands_up
            self.eigenval[Spin(1)][:,:,1] = occup_up 
        elif np.any(bands_up!=None) and np.any(bands_dw!=None) and np.any(occup_up!=None) and np.any(occup_dw!=None) : 
            self.is_spin_polarized = True #left due to compatibility with pymatgen structure 
            assert np.shape( bands_up )[1] == len( kpts_list )       , 'kpoints number dimension mismatch between bands_up and kpts_list '
            assert np.shape( bands_dw )[1] == len( kpts_list )       , 'kpoints number dimension mismatch between bands_dw and kpts_list'
            assert np.shape( bands_up )    == np.shape( occup_up )   , 'dimension mismatch between bands_up and occup_up'
            assert np.shape( bands_dw )    == np.shape( occup_dw )   , 'dimension mismatch between bands_up and occup_up'
            self.bands = {Spin(1): np.zeros( [np.shape(bands_up)[0],np.shape(bands_up)[1],2] ) , Spin(-1): np.zeros( [np.shape(bands_up)[0],np.shape(bands_up)[1],2] )}
            self.eigenval[Spin(1)][:,:,0]  = bands_up
            self.eigenval[Spin(1)][:,:,1]  = occup_up 
            self.eigenval[Spin(-1)][:,:,0] = bands_dw
            self.eigenval[Spin(-1)][:,:,1] = occup_dw 
        
    def read_file_EIGENVAL(self, path ):
        try:
            EigenvalObject = Eigenval(path)
        except:
            raise ValueError("error while opening Eigenval file.")

        self.nelect = EigenvalObject.nelect
        self.bands_indexs = {"first":1,"last":EigenvalObject.nbands}
        self.kpts_list        = EigenvalObject.kpoints        
        self.kpts_mesh        = None
        self.kpts_shift       = None
        self.spglib_structure = None  #informaiton about the structure is not saved into EIGENVAL files.

        # we keep the same format of AiiDA BandsData (see https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/data_types.html#topics-data-types-materials-bands)
        # for non-spin polarized calcs, nkpoints * nbands ; for spin polarized calcs (nspins=2)* nkpoints * nstates
        if   EigenvalObject.ispin == 1:
            self.is_spin_polarized = False #left due to compatibility with pymatgen structure 
            bands_up = EigenvalObject.eigenvalues[Spin(1)][:,:,0] 
            occup_up = EigenvalObject.eigenvalues[Spin(1)][:,:,1]
            self.eigenval = {Spin(1): np.zeros( [np.shape(bands_up)[0],np.shape(bands_up)[1],2] )  }
            self.eigenval[Spin(1)][:,:,0] = bands_up 
            self.eigenval[Spin(1)][:,:,1] = occup_up 
        elif EigenvalObject.ispin == 2:
            self.is_spin_polarized = True #left due to compatibility with pymatgen structure 
            bands_up = EigenvalObject.eigenvalues[Spin(1)][:,:,0]   ;  occup_up = EigenvalObject.eigenvalues[Spin(1)][:,:,1]
            bands_dw = EigenvalObject.eigenvalues[Spin(-1)][:,:,0]  ;  occup_dw = EigenvalObject.eigenvalues[Spin(-1)][:,:,1]
            self.bands = {Spin(1): np.zeros( [np.shape(bands_up)[0],np.shape(bands_up)[1],2] ) , Spin(-1): np.zeros( [np.shape(bands_up)[0],np.shape(bands_up)[1],2] )}
            self.eigenval[Spin(1)][:,:,0]  = bands_up
            self.eigenval[Spin(1)][:,:,1]  = occup_up 
            self.eigenval[Spin(-1)][:,:,0] = bands_dw
            self.eigenval[Spin(-1)][:,:,1] = occup_dw 
        else:
            raise ValueError("spin - noncollinear calculation are not supported, sorry.")
                   
    def read_file_VASPRUN(self, path):
        try:
            vasprunObject = Vasprun(path , parse_projected_eigen=False , parse_potcar_file=False)
        except:
            raise ValueError("error while opening vasprun.xml file.")
            
          
        self.eigenval = vasprunObject.eigenvalues          
        self.bands_indexs={}
        self.bands_indexs["first"]=1 ;  self.bands_indexs["last"]=np.shape(self.eigenval[Spin.up])[1]
        self.is_spin_polarized = vasprunObject.is_spin
        self.kpts_mesh  = vasprunObject.kpoints.kpts
        self.kpts_shift = vasprunObject.kpoints.kpts_shift  
        self.kpts_list  = vasprunObject.actual_kpoints
         
        
        self.spglib_structure = {}
        self.spglib_structure["lattice"]   = vasprunObject.final_structure.lattice.matrix
        self.spglib_structure["positions"] = vasprunObject.final_structure.frac_coords
        tmp_unique_species = [] ; tmp_numbers = []
        for species, itertoolsGrouper in itertools.groupby( vasprunObject.final_structure, key=lambda s: s.species):
            if species in tmp_unique_species:
                ind = tmp_unique_species.index(species)
                tmp_numbers.extend([ind + 1] * len(tuple( itertoolsGrouper )))
            else:
                tmp_unique_species.append(species)
                tmp_numbers.extend([len(tmp_unique_species)] * len(tuple( itertoolsGrouper )))
        self.spglib_structure["numbers"]   = tmp_numbers  #Explanation: species in POSCAR, each integer identifies a different species
                                                        #ES: numbers = [1, 2, 2, 2]        # Al, Ni, Ni, Ni 
        self.spglib_structure["cell"] = ( self.spglib_structure["lattice"] ,  self.spglib_structure["positions"] ,  self.spglib_structure["numbers"] )
                         
    def read_file_OUTCAR_spinUnpol(self, path , setMaxOccupationsFromTwoToOne=True , setGWDataAsprimary='GW' ):
        f = open(path, "r")
        
        #[1] some constants
        f.seek(0,0)	 #restart file index from 0 byte. Seek counts in byte, not lines
        GWBandsStartIndexInLine=[]    ; DFTBandsStartIndexInLine = []
        kpts_shift_IndexInLine = -100 ; kpts_IBKPT_IndexInLine = [] ;
        for lineCount, lineString in enumerate(f,0):
            if( lineString.find(' E-fermi') >=0 ): c_fermiEn = float(lineString.split()[2])
            if( lineString.find('NKPTS')    >=0 ): c_nkpts   = int(lineString.split()[3])
            #if( lineString.find('NBANDS=')  >=0 ): self.bands_indexs = {"first":1,"last": int(lineString.split()[14]) }    
            if( lineString.find('NELECT')   >=0 ): self.nelect = lineString.split()[2]
            if( lineString.find('QP shifts <psi_nk| G(iteration)W_0 |psi_nk>') >=0 ): GWBandsStartIndexInLine.append(lineCount)
            if( lineString.find('average (electrostatic) potential at core')   >=0 ): DFTBandsStartIndexInLine.append(lineCount)                
    
            #[2] k-point related 
            if( lineString.find('generate k-points for:') >=0 ): self.kpts_mesh = [int(lineString.split()[3]) , int(lineString.split()[4]) , int(lineString.split()[5])]
            if( lineString.find('Shift w.r.t. Gamma in fractional coordinates') >=0 ): kpts_shift_IndexInLine = lineCount
            if( lineString.find('Subroutine IBZKPT returns following result')   >=0 ): kpts_IBKPT_IndexInLine.append(lineCount)
            #if( lineString.find('irreducible k-points:')   >=0 ):                      kpts_IBKPT_numK.append(int(lineString.split()[1]))
    
        f.seek(0,0)	 #restart file index from 0 byte. Seek counts in byte, not lines
        self.kpts_list = []
        for lineCount, lineString in enumerate(f,0):
            if kpts_IBKPT_IndexInLine[0]+6 < lineCount < kpts_IBKPT_IndexInLine[0]+c_nkpts+7: 
                self.kpts_list.append( [float(lineString.split()[0]) , float(lineString.split()[1]) , float(lineString.split()[2])])
                
        #[3] bands related
        #    starting from determining k-point lines of each band
        kptsKeys = []                                   #Determine k-points keys to be searched in the OUTCAR:
        for i in range(1,c_nkpts+1):            
            kptsKeys.append(" k-point +"+str(i)+" :")   #Different Vasp version have different number of space between "k-point" and the number, so we use regular expression.
            
        kptsLineIndex = np.zeros(c_nkpts    , dtype=int)
        kptsCoord     = np.zeros((c_nkpts,3), dtype=float)
        kptsIndex=0
        f.seek(0,0)	 #restart file index from 0 byte. Seek counts in byte, not lines 
        for lineCount, lineString in enumerate(f,0):
            if lineCount>=max(DFTBandsStartIndexInLine , GWBandsStartIndexInLine)[0] and (re.search( kptsKeys[kptsIndex] , lineString ) != None):
            # We check only lines after max(DFTBandsStartIndexInLine , GWBandsStartIndexInLine)[0], in order to avoid lines like  k-point   1 :   0.0000 0.0000 0.0000  plane waves:    4367
                   kptsLineIndex[kptsIndex] = int(lineCount) #remember: lineCount starts from 0, but string couting starts from 1.
                   kptsCoord[kptsIndex][0]  = float(lineString.split()[3]) ; kptsCoord[kptsIndex][1]=float(lineString.split()[4]) ; kptsCoord[kptsIndex][2]=float(lineString.split()[5]);
                   kptsIndex = kptsIndex + 1
            if (kptsIndex >= c_nkpts): break                   #we have already meet and stored all kpts; continuing would cause and OutOfBound for kptsCoord; consider that kptsLineIndex and kptsCoord starts from 0.
        #del kptsKeys , kptsIndex , lineCount, lineString  


        #How many Bands are printed in the output file? <---------------------------- >
        #c_nbands_printed = c_nbands  #For EIGENVAL
        self.is_spin_polarized = False
        if len(GWBandsStartIndexInLine) > 0:           
            c_nbands_printed = kptsLineIndex[1]-kptsLineIndex[0] - 4 #this assumes more than one kpts.
        elif len(DFTBandsStartIndexInLine) > 0:
            c_nbands_printed = kptsLineIndex[1]-kptsLineIndex[0] - 3 #this assumes more than one kpts.
        self.eigenval_DFT = { Spin(1): np.zeros( [c_nkpts,c_nbands_printed,2] ) }
        if len(GWBandsStartIndexInLine) > 0: 
            self.eigenval_GW  = { Spin(1): np.zeros( [c_nkpts,c_nbands_printed,2] ) }
            self.eigenval_Z   = { Spin(1): np.zeros( [c_nkpts,c_nbands_printed,2] ) }
            self.eigenval_QPc = { Spin(1): np.zeros( [c_nkpts,c_nbands_printed,2] ) }


        print('GWBandsStartIndexInLine: ',GWBandsStartIndexInLine)
        print('DFTBandsStartIndexInLine:',DFTBandsStartIndexInLine)

        if len(GWBandsStartIndexInLine) > 0:           
            for kptIndex in range(c_nkpts):
                f.seek(0,0) #restart file index from 0th byte
                for line in islice(f, kptsLineIndex[kptIndex]+3, kptsLineIndex[kptIndex]+c_nbands_printed+3):
                    line_splitted = line.split()
                    self.eigenval_DFT[Spin(1)][ kptIndex , int(line_splitted[0])-1 , 0] = float(line_splitted[1])
                    self.eigenval_GW[Spin(1)][  kptIndex , int(line_splitted[0])-1 , 0] = float(line_splitted[2])
                    self.eigenval_Z[Spin(1)][   kptIndex , int(line_splitted[0])-1 , 0] = float(line_splitted[6])
                    self.eigenval_QPc[Spin(1)][ kptIndex , int(line_splitted[0])-1 , 0] = float(line_splitted[2])-float(line_splitted[1])
                    self.eigenval_DFT[Spin(1)][ kptIndex , int(line_splitted[0])-1 , 1] = float(line_splitted[7])
                    self.eigenval_GW[Spin(1)][  kptIndex , int(line_splitted[0])-1 , 1] = float(line_splitted[7])
                    self.eigenval_Z[Spin(1)][   kptIndex , int(line_splitted[0])-1 , 1] = float(line_splitted[7])
                    self.eigenval_QPc[Spin(1)][ kptIndex , int(line_splitted[0])-1 , 1] = float(line_splitted[7])

        elif len(DFTBandsStartIndexInLine) > 0:        
            for kptIndex in range(c_nkpts):
            #for kptIndex in range(990,c_nkpts):
                f.seek(0,0) #restart file index from 0th byte
                for line in islice(f, kptsLineIndex[kptIndex]+2, kptsLineIndex[kptIndex]+c_nbands_printed+2):
                    line_splitted = line.split()
                    self.eigenval_DFT[Spin(1)][ kptIndex , int(line_splitted[0])-1 , 0] = float(line_splitted[1])
                    self.eigenval_DFT[Spin(1)][ kptIndex , int(line_splitted[0])-1 , 1] = float(line_splitted[2])
                    
        
        
        if len(GWBandsStartIndexInLine)>0   and setGWDataAsprimary=='GW':  self.eigenval = self.eigenval_GW
        elif len(GWBandsStartIndexInLine)>0 and setGWDataAsprimary=='QPc': self.eigenval = self.eigenval_QPc
        elif len(GWBandsStartIndexInLine)>0 and setGWDataAsprimary=='DFT': self.eigenval = self.eigenval_DFT
        elif len(GWBandsStartIndexInLine)>0 and setGWDataAsprimary=='Z':   self.eigenval = self.eigenval_Z
        elif len(DFTBandsStartIndexInLine) > 0: self.eigenval = self.eigenval_DFT
        self.bands_indexs={} ; self.bands_indexs["first"]=1 ; self.bands_indexs["last"]=np.shape(self.eigenval[Spin.up])[1]

           
        if setMaxOccupationsFromTwoToOne:
            self.eigenval_DFT[Spin.up][:,:,1] = self.eigenval_DFT[Spin.up][:,:,1] / 2
            self.eigenval_GW[Spin.up][:,:,1]  = self.eigenval_GW[Spin.up][:,:,1]  / 2
            self.eigenval_QPc[Spin.up][:,:,1] = self.eigenval_QPc[Spin.up][:,:,1] / 2
            if self.is_spin_polarized:
                self.eigenval_DFT[Spin(-1)][:,:,1] = self.eigenval_DFT[Spin(-1)][:,:,1] / 2
                self.eigenval_GW[Spin(-1)][:,:,1]  = self.eigenval_GW[Spin(-1)][:,:,1]  / 2
                self.eigenval_QPc[Spin(-1)][:,:,1] = self.eigenval_QPc[Spin(-1)][:,:,1] / 2  
          
                
        path_directory = os.path.split(os.path.abspath(f.name))
        pcs_POSCAR  = pymatgen.core.structure.Structure.from_file( os.path.join(path_directory[0],'POSCAR') )
        self.spglib_structure = {}
        self.spglib_structure["lattice"]   = pcs_POSCAR.lattice.matrix
        self.spglib_structure["positions"] = pcs_POSCAR.frac_coords
        tmp_unique_species = [] ; tmp_numbers = []
        for species, itertoolsGrouper in itertools.groupby( pcs_POSCAR, key=lambda s: s.species):
            if species in tmp_unique_species:
                ind = tmp_unique_species.index(species)
                tmp_numbers.extend([ind + 1] * len(tuple( itertoolsGrouper )))
            else:
                tmp_unique_species.append(species)
                tmp_numbers.extend([len(tmp_unique_species)] * len(tuple( itertoolsGrouper )))
        self.spglib_structure["numbers"]   = tmp_numbers  #Explanation: species in POSCAR, each integer identifies a different species
                                                           #ES: numbers = [1, 2, 2, 2]        # Al, Ni, Ni, Ni 
        self.spglib_structure["cell"] = ( self.spglib_structure["lattice"] ,  self.spglib_structure["positions"] ,  self.spglib_structure["numbers"] )
        
        
        
        
        
        
        
        
        
    def read_file_WAVECAR(self, path , verboseFlag=False , usePOSCARinSamePath=False):
              print("{<PARSING WAVECAR>} <---------------------------->")
              print("   at path= ",path)
              if not exists(path ):
                  raise IOError("file at the specified path does NOT exist.")

              with open(path, "rb") as f:
                  # Read from 1° record  record length , number of spin components, rtag (aka precision)
                  # They are integer but stored as float64.
                  c_reclen_bit, c_nspin, rtag = np.fromfile(f, dtype=np.float64, count=3).astype(np.int_)
                  #Rewrite precision from bit to byte.
                  c_reclen = int(c_reclen_bit / 8)

                  #only record length-number of spin components-rtag are saved inside 1°record;
                  #Vasp writes WDES%NKPTS , WDES%NB_TOT , WDES%ENMAX , ((LATT_INI%A(I,J),I=1,3),J=1,3) , EFERMI to 2°record;
                  #thus let's pad to second record and read them.
                  np.fromfile(f, dtype=np.float64, count=(c_reclen - 3)  )
                  c_numk, c_numb, c_encut = np.fromfile(f, dtype=np.float64, count=3).astype(np.int_)
                  c_latvec      = np.fromfile(f, dtype=np.float64, count=9).reshape((3, 3))
                  c_efermi      = np.fromfile(f, dtype=np.float64, count=1)[0]
                  if verboseFlag:
                      print("-constants read from WAVECAR prolog:\n  spin comp ",c_nspin,"\n  num kpts  ",c_numk,"\n  num bands ",c_numb,"\n  encut (eV)",c_encut)
                      print("  efermi    ",c_efermi)
                      print("  lattice vec",c_latvec[0,:],"\n             ",c_latvec[1,:],"\n             ",c_latvec[2,:],"\n")

                  # padding to end of record REC=2 / beginning of record REC=3
                  # c_reclen -13 because we have read 13 element from REC=1 start
                  np.fromfile(f, dtype=np.float64, count=(c_reclen - 13) )
                  eigenvalues= np.zeros([c_nspin , c_numk , c_numb , 2] , dtype=np.float64) #[spin,nk,nb,0]=eigenvalues - [spin,nk,nb,1]=occupation
                  self.kpts_mesh  = None
                  self.kpts_shift = None
                  self.kpts_list  = []
                  for idx_spin in range(c_nspin):
                      if verboseFlag: print("-outer level loop: spin component {}".format(idx_spin))
                      for idx_kpt in range(c_numk):
                          if verboseFlag: print("--1° level loop: kpt {}".format(idx_kpt))

                          #Read the number of plane waves for this specific kpts for this spin channel
                          num_pw = int(np.fromfile(f, dtype=np.float64, count=1)[0])
                          
                          #Read the kpt associated to this index
                          #Sometimes kpts that should have 0 coefficients have very small but non-zero value,
                          #ES [1.26237862e-15 1.26237862e-15 1.26237862e-15] for Gamma [0,0,0]. We handle them manually.
                          kpoint =     np.fromfile(f, dtype=np.float64, count=3)
                          kpoint[abs(kpoint) < 1E-10] = 0
                          self.kpts_list.append(list(kpoint))
                          if verboseFlag: print("  kpoint {} with {: 3} plane waves".format(kpoint,num_pw))

                          #Vasp writes REAL(W%CELTOT(I,K,ISP),q) AIMAG(W%CELTOT(I,K,ISP)) =W%FERTOT(I,K,ISP)
                          #Where CELTOT=eigenvalues FERTOT=occupation; we are not interested in imaginary part.
                          temp_eig = np.fromfile(f, dtype=np.float64, count=(3*c_numb) ).reshape((c_numb, 3))
                          eigenvalues[idx_spin , idx_kpt,:,:]= np.array(temp_eig[:,[0,2]]  , dtype=np.float64)

                          # padding to end of record(s) containing the eigenvalues for a single kpt, considering that we have already read 4+3*nb float64
                          # eigenvaluesmay span several records, thus we add padding to reach the end of the last record spanned by the eigenvalues of kpt idx_kpt
                          np.fromfile(f, dtype=np.float64, count=((c_reclen -4 -3*c_numb) % c_reclen)  )

                          # padding used to skip completely the plane-wave coefficients.
                          # To read them we should have done something like
                          #for inb in range(c_numb):
                          #        data = np.fromfile(f, dtype=np.complex64, count=nplane)
                          #        np.fromfile(f, dtype=np.float64, count=c_reclen - nplane)
                          if rtag==45200:
                              np.fromfile(f, dtype=np.complex64 , count=(c_reclen*c_numb) )
                          elif rtag==45210:
                              np.fromfile(f, dtype=np.complex128, count=(c_reclen*2*c_numb) )
                          else: 
                              print(rtag)
                              raise IOError("rtag value (precision) not recognized")
                  del temp_eig , kpoint , idx_spin , idx_kpt
                  f.close()
                  #return eigenvalues , kpts_list,  c_numk , c_numb , c_encut


                  if c_nspin == 1:
                      self.is_spin_polarized = False
                      self.spglib_structure  = None
                      self.bands_indexs = {"first":1,"last":c_numb}
                      #self.bands  = eigenvalues[0,:,:,0]
                      #self.occupations = eigenvalues[0,:,:,1]
                      self.eigenval  = { Spin(1): np.zeros( [c_numk,c_numb,2] ) }
                      self.eigenval[Spin(1)] = eigenvalues[0,:,:,:]
                      #WAVECAR invece di occupazione 2 ha salvato 1; raddoppiamo per coerenza con il resto
                      #self.eigenval[Spin(1)][:,:,1] = self.eigenval[Spin(1)][:,:,1]*2                  
                      #self.nelect      = int(sum( self.eigenval[Spin(1)][0,:,1])*2 )*2
                      print("   TOTEST: nelect is reconstructed and not read, \n           check it's correct please.")
                  elif c_nspin==2:
                      print("   TOTEST: SPIN POLARIZED NON YET TESTED")
                      self.is_spin_polarized = True
                      self.spglib_structure  = None
                      self.bands_indexs = {"first":1,"last":c_numb}
                      #self.bands  = np.array([ eigenvalues[0,:,:,0] , eigenvalues[1,:,:,0] ])
                      #self.occupations = np.array([ eigenvalues[0,:,:,1] ,  eigenvalues[1,:,:,1] ])
                      self.eigenval  = { Spin(1): np.zeros( [c_numk,c_numb,2] ) , Spin(-1): np.zeros( [c_numk,c_numb,2] ) }
                      self.eigenval[Spin(1)]  = eigenvalues[0,:,:,:]
                      self.eigenval[Spin(-1)] = eigenvalues[1,:,:,:]
                      #self.nelect      = int(sum( self.eigenval[Spin(1)][0,:,1])*2 ) + int(sum( self.eigenval[Spin(-1)][0,:,1])*2 )
                      print("  TOTEST: nelect is reconstructed and not read, check it's correct please.")
                  else:
                      raise ValueError("spin - noncollinear calculation are not supported, sorry.")
                  print("{<END PARSING WAVECAR>} <------------------------>")      
                  
    def write_file_WAVECAR(self, path , verboseFlag=False):
        if not exists( path ):
            raise IOError("file at the specified path does NOT exist.")
            
        # some checks
        assert (np.shape( self.eigenval[Spin(1)] )[0] == len(self.kpts_list)) , "input bandsObj is not consistent: #kpts is different between self.kpoints and self.bands."
        assert (self.bands_indexs["first"] == 1) , "self.eigenval do not start from 1°bands."
        numKpts = kptsNum = len(self.kpts_list)
        numNbnd = np.shape( self.eigenval[Spin(1)] )[1]
        if self.is_spin_polarized == False : numSpin = 1 
        else: numSpin = 2  
    
    
        if verboseFlag == True:
            print("write_file_WAVECAR: trying to write eigenvalues+occupation array of dimension " , [numSpin , numKpts , numNbnd , 2] , "to path" , path)
        eigenvalues = np.zeros([numSpin , numKpts , numNbnd , 2] , dtype=np.float64) #[spin,nk,nb,0]=eigenvalues - [spin,nk,nb,1]=occupation
        if numSpin ==1:
            eigenvalues[0,:,:,0] = self.eigenval[Spin(1)][:,:,0]    #self.bands
            eigenvalues[0,:,:,1] = self.eigenval[Spin(1)][:,:,1]    #self.occupations
        elif numSpin ==2:
            eigenvalues[0,:,:,0] = self.eigenval[Spin(1) ][:,:,0]   #self.bands[0,:,:]
            eigenvalues[0,:,:,1] = self.eigenval[Spin(1) ][:,:,1]   #self.occupations[0,:,:]
            eigenvalues[1,:,:,0] = self.eigenval[Spin(-1)][:,:,0]   #self.bands[1,:,:]
            eigenvalues[1,:,:,1] = self.eigenval[Spin(-1)][:,:,1]   #self.occupations[1,:,:]


        with open( path, "r+b") as f:
            # Read from 1° record  record length , number of spin components, rtag (aka precision)
            # They are integer but stored as float64.
            c_reclen_bit, c_nspin, rtag = np.fromfile(f, dtype=np.float64, count=3).astype(np.int_)
            #Rewrite precision from bit to byte.
            c_reclen = int(c_reclen_bit / 8)
    
            #only record length-number of spin components-rtag are saved inside 1°record; 
            #Vasp writes WDES%NKPTS , WDES%NB_TOT , WDES%ENMAX , ((LATT_INI%A(I,J),I=1,3),J=1,3) , EFERMI to 2°record; 
            #thus let's pad to second record and read them.
            np.fromfile(f, dtype=np.float64, count=(c_reclen - 3)  )
            c_numk, c_numb, c_encut = np.fromfile(f, dtype=np.float64, count=3).astype(np.int_)
            c_latvec      = np.fromfile(f, dtype=np.float64, count=9).reshape((3, 3))
            c_efermi      = np.fromfile(f, dtype=np.float64, count=1)[0]
        
            if verboseFlag:
                print("-constants read from WAVECAR prolog:\n  spin comp ",c_nspin,"\n  num kpts  ",c_numk,"\n  num bands ",c_numb,"\n  encut (eV)",c_encut)
                print("  efermi    ",c_efermi)
                print("  lattice vec",c_latvec[0,:],"\n             ",c_latvec[1,:],"\n             ",c_latvec[2,:],"\n")
            if [c_nspin , c_numk , c_numb] != [ np.shape(eigenvalues)[0] ,  np.shape(eigenvalues)[1] ,  np.shape(eigenvalues)[2] ]:
                raise ValueError("spin number and/or kpts number and/or band number of input eigenvalue variable does NOT match WAVECAR internal dimensions.")
    
            # padding to end of record REC=2 / beginning of record REC=3
            # c_reclen -13 because we have read 13 element from REC=1 start
            np.fromfile(f, dtype=np.float64, count=(c_reclen - 13) )
            for idx_spin in range(c_nspin):
                if verboseFlag: print("-outer level loop: spin component {}".format(idx_spin))
                for idx_kpt in range(c_numk):
                    if verboseFlag: print("--1° level loop: kpt {}".format(idx_kpt))
                    
                    num_pw = int(np.fromfile(f, dtype=np.float64, count=1)[0])
                    kpoint =     np.fromfile(f, dtype=np.float64, count=3)
                    if verboseFlag: print("  kpoint {} with {: 3} plane waves".format(kpoint,num_pw))            
                    
                    # Vasp writes REAL(W%CELTOT(I,K,ISP),q) AIMAG(W%CELTOT(I,K,ISP)) =W%FERTOT(I,K,ISP)
                    # Where CELTOT=eigenvalues FERTOT=occupation; we are not interested in imaginary part.
                    temp= np.array((eigenvalues[idx_spin , idx_kpt,:,0], np.zeros(c_numb ,dtype=np.float64) , eigenvalues[idx_spin , idx_kpt,:,1]))
                    temp_toWrite= np.concatenate([ temp[:,i] for i in range(c_numb) ])
                    f.write(temp_toWrite.tobytes())
            
                    # padding to end of record(s) containing the eigenvalues for a single kpt, considering that we have already read 4+3*nb float64
                    # eigenvaluesmay span several records, thus we add padding to reach the end of the last record spanned by the eigenvalues of kpt idx_kpt
                    np.fromfile(f, dtype=np.float64, count=((c_reclen -4 -3*c_numb) % c_reclen)  )
            
                    # padding used to skip completely the plane-wave coefficients.
                    # To read them we should have done something like 
                    #for inb in range(c_numb):
                    #        data = np.fromfile(f, dtype=np.complex64, count=nplane)
                    #        np.fromfile(f, dtype=np.float64, count=c_reclen - nplane)
                    if rtag==45200:
                        np.fromfile(f, dtype=np.complex64 , count=(c_reclen*c_numb) )
                    elif rtag==45210:
                        np.fromfile(f, dtype=np.complex128, count=(c_reclen*2*c_numb) )
                    else: raise IOError("rtag value (precision) not recognized")
        del temp_toWrite , idx_spin , idx_kpt
        f.close()


    def apply_QPcorr( bandsObj_DFT , bandsObj_interp ):
        # A bit of input checking; without these check, the other calculations do not make snese.
        theshold_kpts = 0.0001
        bands_overlapping_range  = {"first": max(bandsObj_DFT.bands_indexs["first"] , bandsObj_interp.bands_indexs["first"]) ,
                                    "last": min(bandsObj_DFT.bands_indexs["last"]   , bandsObj_interp.bands_indexs["last"] ) }
        assert np.any(np.abs(np.array(bandsObj_DFT.kpts_list) - np.array(bandsObj_interp.kpts_list)) < theshold_kpts) , "kpts lish meshes are different."
        assert bandsObj_DFT.is_spin_polarized == bandsObj_interp.is_spin_polarized , "different number of spin component."
        assert bands_overlapping_range["first"] < bands_overlapping_range["last"]  , "the input objects span no overlapping bands."
        
        assert (Spin(-1) in bandsObj_DFT.eigenval and bandsObj_DFT.is_spin_polarized)       or (not Spin(-1) in bandsObj_DFT.eigenval and not bandsObj_DFT.is_spin_polarized)       , "Presence of minority spin bands and is_spin_polarized flags are contradictory in bandsObj_DFT."
        assert (Spin(-1) in bandsObj_interp.eigenval and bandsObj_interp.is_spin_polarized) or (not Spin(-1) in bandsObj_interp.eigenval and not bandsObj_interp.is_spin_polarized) , "Presence of minority spin bands and is_spin_polarized flags are contradictory in bandsObj_interp."

    
   
        bandsObj_corrected = deepcopy( bandsObj_DFT )
        
        fb= bandsObj_interp.bands_indexs["first"] -1   #-1 because ["first"] is described in Vasp formalism, where bands start from 1; but python array starts from 0.   
        lb= min ( bandsObj_interp.bands_indexs["last"] -1+1 , bandsObj_DFT.bands_indexs["last"] -1+1 ) 
                                                       #+1 because with slicing a:b, as in range(a,b) b is not comprised (only b-1) and we want to include bandsObj_interp.bands_indexs["last"].
                                                       #min because without an error could happen, here I given an example:
                                                       #kcorr/DFT2.vasprun has NBANDS=96 , kcorr/GW.vasprun.xml has NBANDS=768  ,  and CdTe_666_notShifted has NBANDS=48
                                                       #thus bandsObj_interp inherits NBANDS=96 (to calculate QPc, aka GW-DFT of kcorr, you're limited by kcorr/DFT2.vasprun.xml NBANDS)
                                                       #NBANDS=48 is lowest value that is a multiple of #MPIproc and is  >= NELECT + NBANDSV + buffer.value(10)"
  
        if not bandsObj_interp.is_spin_polarized :
            bandsObj_corrected.eigenval[Spin(1)][:,fb:lb,0] =    bandsObj_DFT.eigenval[Spin(1)][:,fb:lb,0]   +   bandsObj_interp.eigenval[Spin(1)][:,:lb,0]
        else:
            bandsObj_corrected.eigenval[Spin(1) ][:,fb:lb,0] =    bandsObj_DFT.eigenval[Spin(1) ][:,fb:lb,0]   +   bandsObj_interp.eigenval[Spin(1) ][:,:lb,0]
            bandsObj_corrected.eigenval[Spin(-1)][:,fb:lb,0] =    bandsObj_DFT.eigenval[Spin(-1)][:,fb:lb,0]  +   bandsObj_interp.eigenval[Spin(-1)][:,:lb,0]

        return bandsObj_corrected
                                              
    def _determine_BZ_IBZ_grid(self , kpts_shift=None ):
      #[0] get_ir_reciprocal_mesh is strange:  16 IBZ kpts in BZ have not contiguos index {0-3,7-11,14-16,21,51-52,58}  [saved in spg_map_IBZ]
      #[1] all points in BZ (spg_map and spg_map_modified) have one of those indexs - is the IBZ index which they are equivalent (by symmetry operations) to.
        
        print("   {<EXTRACTING kpoints_BZ/_IBZ and mapping>} <--->") 
        print("     Structure supplied has spacegroup = ",spglib.get_spacegroup( self.spglib_structure["cell"], symprec=1e-1))  #this as sanity check
        if kpts_shift is not None and (np.array(kpts_shift) != np.array([0,0,0])).all() :
            print("     A shift for the kmesh is supplied. The current method do not work for shifted meshes, sorry")
            return None
    
        def _determine_nova(q):
            #> [8] indip e- in periodic pag 5, Fieschi - De Renzi: per ogni vettore k che cade sul bordo zona ce n'è un altro k' pari k+G.
            #> we are working in direct.coordinates, which from https://www.vasp.at/wiki/index.php/KPOINTS are defined as (x1b1 + x2b2 + x3b3)
            # reciprocal.lattice.vectors are exactly defined in the same manner (x1b1 + x2b2 + x3b3), thus in frac.coordinates are G1[(x1,x2,x3)=(1,0,0)] etc
            #> we define nova(q) q + {all possible 1°shell G}
            from itertools import permutations 
            transl_singleG = list(set(    list(permutations([1, 0, 0]))+list(permutations([-1, 0, 0]))    ))
            transl_doubleG = list(set(    list(permutations([1, 1, 0]))+list(permutations([-1, 1, 0]))+list(permutations([-1, -1, 0]))    ))
            transl_tripleG = list(set(    list(permutations([1, 1, 1]))+list(permutations([-1, 1, 1]))+list(permutations([-1, -1, 1]))+list(permutations([-1, -1, -1]))    ))
            transl_tot = transl_singleG+transl_doubleG+transl_tripleG
            return(   np.unique(q + np.array(transl_tot) , axis=0 )   )
    
        
    
        #[part 1] define BZ - IBZ stuff
        #[1.1] mappings spg_map_BZtoIBZ:  
        #  spg_map_BZtoIBZ:  spg_direct_IBZ  = spg_direct_BZ[spg_map_BZtoIBZ]  
        #  spg_map_IBZtoBZ_symConnected: it's not possible to fully reconstruct IBZ to BZ, because we miss informations (namely we would require the pointGroupSymmetryOp to reconstruct all informations).
        #                                spg_map is non-injective mapping from BZ to IBZ- associate different kpts linked by symOp to same IBZ kpt.
        #                                definition: spg_map[i] = spg_map_BZtoIBZ[j] -> spg_map_IBZtoBZ_symConnected[i] = j
        spg_map, spg_grid_BZ  = spglib.get_ir_reciprocal_mesh(self.kpts_mesh, self.spglib_structure["cell"], is_shift=[0, 0, 0]) 
        spg_map_BZtoIBZ = np.unique(spg_map)
        spg_grid_IBZ    = spg_grid_BZ[spg_map_BZtoIBZ]
        spg_direct_BZ   = spg_grid_BZ / self.kpts_mesh
        spg_direct_IBZ  = spg_direct_BZ[spg_map_BZtoIBZ]
        assert np.all( spg_direct_BZ[spg_map_BZtoIBZ] == spg_direct_IBZ ) 
        assert np.all( abs( np.sort(self.kpts_list)  - np.sort(spg_direct_IBZ)  ) <1E-4   ) , "ERROR: The kpoints list saved in self.kpoints and the list (of kpts in the IBZ) returned by spglib differ."

        spg_map_IBZtoBZ_symConnected = np.array( [np.where(map_value == spg_map_BZtoIBZ)[0][0]  for map_value in spg_map])


        spg_map_modified        = deepcopy(spg_map).tolist()
        spg_direct_BZmodified  = deepcopy(spg_direct_BZ).tolist()
        ##[1] which kpts are on the BZ edge, as reconstructed by spglib - checked manually and seems ok.
        kBZ_onBZedge =[]  ;  kBZ_onBZedge_mapIdx = [] 
        for kBZ_mapIdx , kBZ in  zip(spg_map , spg_direct_BZ):
             if (0.5 in kBZ):
                    kBZ_onBZedge.append(kBZ)
                    kBZ_onBZedge_mapIdx.append(kBZ_mapIdx)

        ##[2] construct nova for these k-points
        for kBZ_mapIdx , kBZ in zip(kBZ_onBZedge_mapIdx , kBZ_onBZedge):
            kBZ_onBZedge_nova      = _determine_nova(kBZ)  
            kBZ_onBZedge_nova_inBZ = [x for x in kBZ_onBZedge_nova if np.all(abs(x)<=0.5)]         
    
            #[3] if k of nova is NOT comprised in BZ list, add.
            for k_nova in kBZ_onBZedge_nova_inBZ:
                if list(k_nova) not in spg_direct_BZmodified:
                    print("    ",k_nova,"not present -> ADDING.")
                    spg_direct_BZmodified.append( list(k_nova)  )
                    spg_map_modified.append( kBZ_mapIdx      )
                else:
                    print("testing if",k_nova," already present... nope.")
        del k_nova , kBZ_mapIdx , kBZ , kBZ_onBZedge_nova , kBZ_onBZedge_nova_inBZ   
        spg_map_IBZtoBZmodified_symConnected = np.array( [np.where(map_value == spg_map_BZtoIBZ)[0][0]  for map_value in spg_map_modified])
              
        return spg_map_BZtoIBZ , spg_map_IBZtoBZ_symConnected , spg_map_IBZtoBZmodified_symConnected , spg_direct_IBZ , spg_direct_BZ , spg_direct_BZmodified
    
    def map_bands_fromIBZ_toBZ(self):
#        #[0] get_ir_reciprocal_mesh is strange:  16 IBZ kpts in BZ have not contiguos index {0-3,7-11,14-16,21,51-52,58}  [saved in spg_map_IBZ]
#        #[1] all points in BZ (spg_map and spg_map_modified) have one of those indexs - is the IBZ index which they are equivalent (by symmetry operations) to.
#        #[2] data read from vasprun has IBZ kpoints ordered as in self.kpoints, not as in get_ir_reciprocal_mesh; thus we sort/reorganized the IBZ indexs in order to match self.kpoints order [saved in spg_map_IBZ_selfkptsOrder]	
#        #[3] for each kpts in spg_map_modified value we compare maps of IBZ(self.kpoints) spg_map_IBZ_selfkptsOrder  IN ORDER TO DETERMINE which kpt bands to copy        
        print("{<MAPPING bands IBZ -> BZ>} <------------------>")
        spg_map_BZtoIBZ , spg_map_IBZtoBZ_symConnected , spg_map_IBZtoBZmodified_symConnected , spg_direct_IBZ , spg_direct_BZ , spg_direct_BZmodified = self._determine_BZ_IBZ_grid( )

        assert (Spin(-1) in self.eigenval and self.is_spin_polarized) or (not Spin(-1) in self.eigenval and not self.is_spin_polarized) , "Presence of minority spin bands and is_spin_polarized flags are contradictory."
        tmp_bands_up = np.zeros( [len(spg_map_IBZtoBZmodified_symConnected) , np.shape( self.eigenval[Spin(1)][:,:,0] )[1]] )
        tmp_occup_up = np.zeros( [len(spg_map_IBZtoBZmodified_symConnected) , np.shape( self.eigenval[Spin(1)][:,:,1] )[1]] )
        if self.is_spin_polarized: 
            tmp_bands_dw = np.zeros( [len(spg_map_IBZtoBZmodified_symConnected) , np.shape( self.eigenval[Spin(-1)][:,:,0] )[1]] )
            tmp_occup_dw = np.zeros( [len(spg_map_IBZtoBZmodified_symConnected) , np.shape( self.eigenval[Spin(-1)][:,:,1] )[1]] )


        #[3] of above.
        for map_value_Idx , map_value in enumerate(spg_map_IBZtoBZmodified_symConnected):
            if self.is_spin_polarized:
                tmp_bands_up[map_value_Idx , :]  = self.eigenval[Spin(1)][map_value , : , 0 ]
                tmp_occup_up[map_value_Idx , :]  = self.eigenval[Spin(1)][map_value , : , 1 ]
                tmp_bands_dw[map_value_Idx , :]  = self.eigenval[Spin(-1)][map_value , : , 0]
                tmp_occup_dw[map_value_Idx , :]  = self.eigenval[Spin(-1)][map_value , : , 1]
            else:
                tmp_bands_up[map_value_Idx , :]  = self.eigenval[Spin(1)][map_value , : , 0]
                tmp_occup_up[map_value_Idx , :]  = self.eigenval[Spin(1)][map_value , : , 1]

        if self.is_spin_polarized:
            bandsOjb_BZmodified = bandsObj_(bands_up=tmp_bands_up , bands_dw=tmp_bands_dw , occup_up=tmp_occup_up , occup_dw=tmp_occup_dw , 
                                            nelect=None , bands_indexs=self.bands_indexs, spglib_structure=self.spglib_structure           ,
                                            kpts_list=spg_direct_BZmodified , kpts_mesh=self.kpts_mesh , kpts_shift=self.kpts_shift       )  
        else:
            bandsOjb_BZmodified = bandsObj_(bands_up=tmp_bands_up , bands_dw=None , occup_up=tmp_occup_up , occup_dw=None           , 
                                            nelect=None , bands_indexs=self.bands_indexs, spglib_structure=self.spglib_structure     ,
                                            kpts_list=spg_direct_BZmodified , kpts_mesh=self.kpts_mesh , kpts_shift=self.kpts_shift )  
        return bandsOjb_BZmodified
   
    @staticmethod
    def interpolation_UsingEdgedBZ(bandsObj_coarsegrid , finegrid_kpts_list , finegrid_kpts_mesh, finegrid_kpts_shift , finegrid_spglib_structure , finegrid_is_spin_polarized , interp_method='linear'):
      assert finegrid_is_spin_polarized == bandsObj_coarsegrid.is_spin_polarized  
      
      if bandsObj_coarsegrid.is_spin_polarized:
          bands_interp =  {}
          bands_interp[Spin(-1)]= np.zeros( [len(finegrid_kpts_list) , np.shape(bandsObj_coarsegrid.eigenval[Spin(-1)])[1]] , dtype=np.float64)
          bands_interp[Spin(1)] = np.zeros( [len(finegrid_kpts_list) , np.shape(bandsObj_coarsegrid.eigenval[Spin.up])[1]]  , dtype=np.float64)
      else:
          bands_interp =  {}
          bands_interp[Spin(1)] = np.zeros( [len(finegrid_kpts_list) , np.shape(bandsObj_coarsegrid.eigenval[Spin.up])[1]] , dtype=np.float64)
      bandsObj_coarsegrid_mappedToBZ = bandsObj_coarsegrid.map_bands_fromIBZ_toBZ()
      
      
      
      if bandsObj_coarsegrid.is_spin_polarized:        
          for bIdx in range(np.shape(bandsObj_coarsegrid_mappedToBZ.eigenval[Spin.up])[1]):
              bands_interp[Spin(1)][:,bIdx] = scipy.interpolate.griddata( points= bandsObj_coarsegrid_mappedToBZ.kpts_list , 
                                                                          values= bandsObj_coarsegrid_mappedToBZ.eigenval[Spin.up][: , bIdx , 0] ,
                                                                          xi= finegrid_kpts_list , method=interp_method)
              bands_interp[Spin(-1)][:,bIdx] = scipy.interpolate.griddata( points= bandsObj_coarsegrid_mappedToBZ.kpts_list , 
                                                                           values= bandsObj_coarsegrid_mappedToBZ.eigenval[Spin(-1)][: , bIdx , 0] ,
                                                                           xi= finegrid_kpts_list , method=interp_method)
      else:            
           for bIdx in range(np.shape(bandsObj_coarsegrid_mappedToBZ.eigenval[Spin.up])[1]):
               bands_interp[Spin(1)][:,bIdx] = scipy.interpolate.griddata( points= bandsObj_coarsegrid_mappedToBZ.kpts_list , 
                                                                           values= bandsObj_coarsegrid_mappedToBZ.eigenval[Spin.up][: , bIdx , 0] ,
                                                                           xi= finegrid_kpts_list , method=interp_method )   
      return bands_interp
   
    @staticmethod
    def interpolation_nearestCorrectionForNaN(bandsObj_coarsegrid , finegrid_kpts_list , finegrid_kpts_mesh, finegrid_kpts_shift , finegrid_spglib_structure , finegrid_is_spin_polarized , interp_method='linear'):
          assert finegrid_is_spin_polarized == bandsObj_coarsegrid.is_spin_polarized  
          
          if bandsObj_coarsegrid.is_spin_polarized:
              bands_interp =  {} ; bands_interp_nn = {}
              bands_interp[Spin(-1)]= np.zeros( [len(finegrid_kpts_list) , np.shape(bandsObj_coarsegrid.eigenval[Spin(-1)])[1]] , dtype=np.float64)
              bands_interp[Spin(1)] = np.zeros( [len(finegrid_kpts_list) , np.shape(bandsObj_coarsegrid.eigenval[Spin.up])[1]]  , dtype=np.float64)
              bands_interp_nn[Spin(-1)]= np.zeros( [len(finegrid_kpts_list) , np.shape(bandsObj_coarsegrid.eigenval[Spin(-1)])[1]] , dtype=np.float64)
              bands_interp_nn[Spin(1)] = np.zeros( [len(finegrid_kpts_list) , np.shape(bandsObj_coarsegrid.eigenval[Spin.up])[1]]  , dtype=np.float64)
          else:
              bands_interp =  {} ; bands_interp_nn = {}
              bands_interp[Spin(1)] = np.zeros( [len(finegrid_kpts_list) , np.shape(bandsObj_coarsegrid.eigenval[Spin.up])[1]] , dtype=np.float64)
              bands_interp_nn[Spin(1)] = np.zeros( [len(finegrid_kpts_list) , np.shape(bandsObj_coarsegrid.eigenval[Spin.up])[1]] , dtype=np.float64)
      
      
          if bandsObj_coarsegrid.is_spin_polarized:        
              for bIdx in range(np.shape(bandsObj_coarsegrid.eigenval[Spin.up])[1]):
                  bands_interp[Spin(1)][:,bIdx] = scipy.interpolate.griddata( points= bandsObj_coarsegrid.kpts_list , 
                                                                              values= bandsObj_coarsegrid.eigenval[Spin.up][: , bIdx , 0] ,
                                                                              xi= finegrid_kpts_list , method=interp_method)
                  bands_interp[Spin(-1)][:,bIdx] = scipy.interpolate.griddata( points= bandsObj_coarsegrid.kpts_list , 
                                                                               values= bandsObj_coarsegrid.eigenval[Spin(-1)][: , bIdx , 0] ,
                                                                               xi= finegrid_kpts_list , method=interp_method)
                  bands_interp_nn[Spin(1)][:,bIdx] = scipy.interpolate.griddata( points= bandsObj_coarsegrid.kpts_list , 
                                                                                 values= bandsObj_coarsegrid.eigenval[Spin.up][: , bIdx , 0] ,
                                                                                 xi= finegrid_kpts_list , method="nearest")
                  bands_interp_nn[Spin(-1)][:,bIdx] = scipy.interpolate.griddata( points= bandsObj_coarsegrid.kpts_list , 
                                                                                  values= bandsObj_coarsegrid.eigenval[Spin(-1)][: , bIdx , 0] ,
                                                                                  xi= finegrid_kpts_list , method="nearest")
          else:            
               for bIdx in range(np.shape(bandsObj_coarsegrid.eigenval[Spin.up])[1]):
                   bands_interp[Spin(1)][:,bIdx] = scipy.interpolate.griddata( points= bandsObj_coarsegrid.kpts_list , 
                                                                               values= bandsObj_coarsegrid.eigenval[Spin.up][: , bIdx , 0] ,
                                                                               xi= finegrid_kpts_list , method=interp_method )   
                   bands_interp_nn[Spin(1)][:,bIdx] = scipy.interpolate.griddata( points= bandsObj_coarsegrid.kpts_list , 
                                                                                  values= bandsObj_coarsegrid.eigenval[Spin.up][: , bIdx , 0] ,
                                                                                  xi= finegrid_kpts_list , method="nearest" ) 
                   
                     
          if  np.any( np.isnan(bands_interp[Spin(1)]) ) and interp_method == "linear":  #if  there is at least one NaN inside hosting_l
              print("Warning - NaN raised from linear interpolation - trying to approximate their results..")
              bands_interp_corrected = deepcopy(bands_interp)
      
              kpt_NaN_idx_fineGrid    = np.where( bands_interp_corrected[Spin.up][:,0] != bands_interp_corrected[Spin.up][:,0]) #if a is Nan, a==a is False and a!=a is True
              kpt_nonNaN_idx_fineGrid = np.where( bands_interp_corrected[Spin.up][:,0] == bands_interp_corrected[Spin.up][:,0]) #if a is Nan, a==a is False
              kpt_NaN_fineGrid    = np.array(finegrid_kpts_list)[kpt_NaN_idx_fineGrid]
              kpt_nonNaN_fineGrid = np.array(finegrid_kpts_list)[kpt_nonNaN_idx_fineGrid]     
              
            
              l2distance_NaN_nonNaNkpt = []; 
              for NaNkptIdx , NaNkpt  in enumerate(kpt_NaN_fineGrid):
                   tmp = {}; tmp["NaNkpt"] = NaNkpt    
                   tmp["nonNaN_kpts"]= kpt_nonNaN_fineGrid
                   tmp["l2distance"] = np.array([np.linalg.norm( NaNkpt - nonNaNkpt )   for nonNaNkpt in tmp["nonNaN_kpts"]])  
                   tmp_idxSort = np.argsort( tmp["l2distance"] )
                   tmp["l2distance_sorted"]  = tmp["l2distance"][tmp_idxSort]
                   tmp["nonNaN_kpts_sorted"] = tmp["nonNaN_kpts"][tmp_idxSort]      
              
                   idx_NaNkpt_overFineKptgrid            = np.flatnonzero(  np.equal(NaNkpt , np.array(finegrid_kpts_list)).all(1)  ) 
                   idx_closestNonNaNkpts_overFineKptgrid = np.flatnonzero(  np.equal(tmp["nonNaN_kpts_sorted"][0] , np.array(finegrid_kpts_list)).all(1)  ) 
                   print("Working on kpt (which raised NaN):",NaNkpt,"with index over fineGrid",idx_NaNkpt_overFineKptgrid,)
            
                   l2distance_NaN_nonNaNkpt.append(tmp) 
                   if bandsObj_coarsegrid.is_spin_polarized:
                       bands_interp_corrected[Spin(1)][idx_NaNkpt_overFineKptgrid , : ] =  bands_interp_nn[Spin(1)][idx_NaNkpt_overFineKptgrid , :]           \
                                                                                        +  bands_interp[Spin(1)][idx_closestNonNaNkpts_overFineKptgrid , : ]  \
                                                                                        -  bands_interp_nn[Spin(1)][idx_closestNonNaNkpts_overFineKptgrid , :] 
                       bands_interp_corrected[Spin(-1)][idx_NaNkpt_overFineKptgrid , : ] =  bands_interp_nn[Spin(-1)][idx_NaNkpt_overFineKptgrid , :]           \
                                                                                         +  bands_interp[Spin(-1)][idx_closestNonNaNkpts_overFineKptgrid , : ]  \
                                                                                         -  bands_interp_nn[Spin(-1)][idx_closestNonNaNkpts_overFineKptgrid , :]           
                   else:
                       bands_interp_corrected[Spin(1)][idx_NaNkpt_overFineKptgrid , : ] =  bands_interp_nn[Spin(1)][idx_NaNkpt_overFineKptgrid , :]           \
                                                                                        +  bands_interp[Spin(1)][idx_closestNonNaNkpts_overFineKptgrid , : ]  \
                                                                                        -  bands_interp_nn[Spin(1)][idx_closestNonNaNkpts_overFineKptgrid , :]                 
          else:
              bands_interp_corrected = bands_interp
          return bands_interp_corrected    
   
    
  
    def get_cbm(self ,  c_unoccThreshold = 0.45):
        tmp_HOMOar_up = [] ; tmp_HOMOar_dw = []
       
        for kptIdx in range( len(self.kpts_list) ):
                tmp_HOMOar_up.append( np.where(  (self.eigenval[Spin(1)][kptIdx,1:,1]  < c_unoccThreshold) !=  
                                                 (self.eigenval[Spin(1)][kptIdx,:-1,1] < c_unoccThreshold) )[0][0] )
        if self.is_spin_polarized:
            for kptIdx in range( len(self.kpts_list) ):
                   tmp_HOMOar_dw.append( np.where(  (self.eigenval[Spin(-1)][kptIdx,1:,1]  < c_unoccThreshold) !=  
                                                    (self.eigenval[Spin(-1)][kptIdx,:-1,1] < c_unoccThreshold) )[0][0] )
        
        if self.is_spin_polarized: bndIdx_HOMOar = {Spin(1): np.array(tmp_HOMOar_up) , Spin(-1): np.array(tmp_HOMOar_dw)}
        else:                      bndIdx_HOMOar = {Spin(1): np.array(tmp_HOMOar_up) }            
        return bndIdx_HOMOar  

    def get_kpt_index(self , kpt , threshold=0.0001):
        diff = np.abs( np.array(self.kpts_list) - kpt )
        return np.where ( np.all(diff < threshold , axis=1))[0][0]

   

def run_interpolate_and_modifyWAVECAR(path_sparse_GW, path_dense_DFT_reference, path_dense_DFT_toInterp="./", nbandsgw_dense=-1):
    input = {"path_sparse_GW":           path_sparse_GW,
             "path_dense_DFT_reference": path_dense_DFT_reference,
             "path_dense_DFT_toInterp":  path_dense_DFT_toInterp if path_dense_DFT_toInterp else "./",
             "nbandsgw_dense":           nbandsgw_dense if nbandsgw_dense is not None else -1,       }

    bObj_SPARSE_OUTCAR = bandsObj_() ;                  bObj_SPARSE_OUTCAR.read_file_OUTCAR_spinUnpol( path_sparse_GW , setGWDataAsprimary='QPc' )
    bObj_DENSE_VASPRUN_reference = bandsObj_() ;        bObj_DENSE_VASPRUN_reference.read_file_VASPRUN( path_dense_DFT_reference )
    bObj_DENSE_WAVECAR_toReceiveInterp = bandsObj_()  ; bObj_DENSE_WAVECAR_toReceiveInterp.read_file_WAVECAR(  path_dense_DFT_toInterp  )


    NBANDS_DENSE_DFT_toRecInt = np.shape( bObj_DENSE_WAVECAR_toReceiveInterp.eigenval[Spin.up] )[1]
    NBANDS_DENSE_DFT_REF      = np.shape( bObj_DENSE_VASPRUN_reference.eigenval[Spin.up] )[1]
    NBANDS_SPARSE_GW = np.shape( bObj_SPARSE_OUTCAR.eigenval_GW[Spin.up] )[1]
    if input["nbandsgw_dense"] <=0 :
        NBANDSGW_DENSE = min( NBANDS_DENSE_DFT_REF , NBANDS_SPARSE_GW , NBANDS_DENSE_DFT_toRecInt )
    else:
        NBANDSGW_DENSE = min( input["nbandsgw_dense"] , NBANDS_DENSE_DFT_REF , NBANDS_SPARSE_GW , NBANDS_DENSE_DFT_REF )
    print("\n\n{<Used arguments for script>} <-------------------->")
    print("  Inputs - path_sparse_GW: ", path_sparse_GW)
    print("  Inputs - path_dense_DFT_reference: ", path_dense_DFT_reference)
    print("  Inputs - path_dense_DFT_toInterp: " , path_dense_DFT_toInterp)
    print("  Inputs - nbandsgw_dense: ", input["nbandsgw_dense"])
    print("  Actual - nbandsgw_dense: ", NBANDSGW_DENSE)
    print("{<End Used arguments for script>} <-------------->\n")

    print("\n{<Debug: Gamma.kpt pre interp.QPc & reference ----->")
    np.set_printoptions(linewidth=np.inf , precision=4)  # prevent line wrapping when printing numpy array
    print( bObj_DENSE_VASPRUN_reference.eigenval[Spin.up][0,:NBANDSGW_DENSE,0])
    print( bObj_DENSE_WAVECAR_toReceiveInterp.eigenval[Spin.up][0,:NBANDSGW_DENSE,0])
    print("{<End Gamma.kpt pre interp.QPc & reference -------->\n\n")



    bands_DENSE_GWinterp = bandsObj_.interpolation_UsingEdgedBZ(  bandsObj_coarsegrid = bObj_SPARSE_OUTCAR,
                                                                  finegrid_kpts_list  = bObj_DENSE_VASPRUN_reference.kpts_list  ,
                                                                  finegrid_kpts_mesh  = bObj_DENSE_VASPRUN_reference.kpts_mesh  ,
                                                                  finegrid_kpts_shift = bObj_DENSE_VASPRUN_reference.kpts_shift ,
                                                                  finegrid_spglib_structure  = bObj_DENSE_VASPRUN_reference.spglib_structure  ,
                                                                  finegrid_is_spin_polarized = bObj_DENSE_VASPRUN_reference.is_spin_polarized ,
                                                                  interp_method='linear'  )
    print("\n\n{<QPc Interpolated values>} <-----------..--------->")
    # Print header (column indices starting from 1) + # Print each row with 4 decimals
    arr = bands_DENSE_GWinterp[Spin.up][:,:NBANDSGW_DENSE]
    header = " kpt" + " ".join([f"{i:10d}" for i in range(1, arr.shape[1] + 1)])
    print(header)
    for kpt_idx , kpt_bands in enumerate( arr ):
        print(f"{kpt_idx:4}"+ " : " + " ".join([f"{val:10.4f}" for val in kpt_bands]))
    print("\n")
    for kpt_idx , kpt in enumerate( bObj_DENSE_VASPRUN_reference.kpts_list ):
        print(f"{kpt_idx:4}"+ " : " + "".join(str(kpt)))
    print("{<End QPc Interpolated values>} <------------------>\n\n")

    bObj_DENSE_WAVECAR_toReceiveInterp.eigenval[Spin.up][:,:NBANDSGW_DENSE,0] = bObj_DENSE_WAVECAR_toReceiveInterp.eigenval[Spin.up][:,:NBANDSGW_DENSE,0] + bands_DENSE_GWinterp[Spin.up][:,:NBANDSGW_DENSE]
    print("{<Debug: Gamma.kpt post interp.QPc & reference ----->")
    print( bObj_DENSE_VASPRUN_reference.eigenval[Spin.up][0,:NBANDSGW_DENSE,0])
    print( bObj_DENSE_WAVECAR_toReceiveInterp.eigenval[Spin.up][0,:NBANDSGW_DENSE,0])



    bObj_DENSE_WAVECAR_toReceiveInterp.write_file_WAVECAR( path_dense_DFT_toInterp  )



if __name__ == "__main__":
    ## [Interface]
    parser = argparse.ArgumentParser()
    parser.add_argument("-ps"  , "--path_sparse_GW" , type=str , required=True )
    parser.add_argument("-pdr" , "--path_dense_DFT_reference" , type=str , required=True  )
    parser.add_argument("-pdi" , "--path_dense_DFT_toInterp"  , type=str , required=False , default="./")
    parser.add_argument("-ngw" , "--nbandsgw_dense"           , type=int , required=False , default=-1)

    input = {} ; args = parser.parse_args()
    if args.path_sparse_GW is not None:           input["path_sparse_GW"] = args.path_sparse_GW
    if args.path_dense_DFT_reference is not None: input["path_dense_DFT_reference"] = args.path_dense_DFT_reference
    if args.path_dense_DFT_toInterp is not None:  input["path_dense_DFT_toInterp"] = args.path_dense_DFT_toInterp
    else: input["path_dense_DFT_toInterp"] = "./"
    if args.nbandsgw_dense is not None: input["nbandsgw_dense"] = args.nbandsgw_dense
    else: input["nbandsgw_dense"] = -1

    path_sparse_GW           = os.path.join(input["path_sparse_GW"]  , 'OUTCAR.3')
    path_dense_DFT_reference = os.path.join(input["path_dense_DFT_reference"] ,'vasprun.xml')
    path_dense_DFT_toInterp  = os.path.join(input["path_dense_DFT_toInterp"]   ,'WAVECAR')

    run_interpolate_and_modifyWAVECAR(
        path_sparse_GW=args.path_sparse_GW,
        path_dense_DFT_reference=args.path_dense_DFT_reference,
        path_dense_DFT_toInterp=args.path_dense_DFT_toInterp,
        nbandsgw_dense=args.nbandsgw_dense,
    )

