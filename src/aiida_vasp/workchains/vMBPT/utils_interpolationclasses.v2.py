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
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pymatgen
from pymatgen.core import Structure
from pymatgen.electronic_structure.core import Spin


@dataclass(frozen=True)
class InstanceHistory:
    timestamp_loaded: datetime
    path_loaded: Path | None = None
    comment: str | None = None
@dataclass(frozen=True)
class KpointsData:
    kpts:  np.ndarray                 # shape (nkpts,3)
    mesh:  list[int]   | None = None
    shift: list[float] | None = None
@dataclass(frozen=True)
class BandsState:
    history: InstanceHistory
    kpoints: KpointsData | None = None
    structure: pymatgen.core.structure.Structure | None = None
    eigenval:   dict[Spin, np.ndarray] | None = None  # energies as  [nkpts, nbands]
    occupation: dict[Spin, np.ndarray] | None = None  # energies as  [nkpts, nbands]
    spin_keys:  tuple[Spin, ...] = (Spin.up,)         # defaults to non–spin-polarized
    
    # Optional miscellaneous data: alternate eigenvalue sets (GW, Z, etc.)
    misc: dict[str, dict[Spin, np.ndarray]] = field(default_factory=dict)

    
    
class BandsState_IO :      
    def parse_outcar_spinUnpol( path: str  | Path , poscar_path: str | Path | None = None ,
                                setMaxOccupationsFromTwoToOne: bool = True ,                               
                                setGWDataAsprimary: str = "GW"             ) -> BandsState:        
        "Parse an OUTCAR for non–spin-polarized DFT or GW calculations. Extracts eigenvalues, occupations, k-points, and structure."
        
        if not Path(path).exists():
            raise FileNotFoundError("OUTCAR not found!")
    
    
        #[1]Pre-scan for tags -------------------------#
        with open(Path(path), "r") as f:
            c_nkpts, kpts_shift_line = None, -100
            lineidx_starts_gw, lineidx_starts_dft, lineidx_kpts_ibz = [], [], []
            for idx, line in enumerate(f):
                if   "NKPTS"  in line: c_nkpts  = int(line.split()[3])
                elif "NELECT" in line: c_nelect = line.split()[2]
                elif "QP shifts <psi_nk| G(iteration)W_0 |psi_nk>" in line:  lineidx_starts_gw.append(idx)
                elif "average (electrostatic) potential at core"   in line:  lineidx_starts_dft.append(idx)
                elif "generate k-points for:"                      in line:  bs_kpts_mesh = list(map(int, line.split()[3:6]))
                elif "Shift w.r.t. Gamma in fractional coordinates" in line: kpts_shift_line = idx
                elif "Subroutine IBZKPT returns following result"  in line:  lineidx_kpts_ibz.append(idx)
        if c_nkpts is None:      raise ValueError("Could not find NKPTS in OUTCAR.")
        if not lineidx_kpts_ibz: raise ValueError("Could not find IBZKPT section in OUTCAR.")
    
        #[2]Read k-point coordinates ------------------#
        with open(Path(path), "r") as f:
            bs_kpts_list  = []
            for i, line in enumerate(f):
                if lineidx_kpts_ibz and lineidx_kpts_ibz[0] + 6 < i < lineidx_kpts_ibz[0] + c_nkpts + 7:
                    bs_kpts_list .append(  list(map(float, line.split()[:3]  ))  )
        bs_kpts_list = np.array(bs_kpts_list )
    
    
        #[3]Find band-block line positions ------------#
        # Build a regex that matches both "k-point  1 :" and "k-point +1 :"
        def _kpt_regex(i: int) -> re.Pattern:
            return re.compile(rf"\s*k-point\s*\+?\s*{i}\s*:")
        
        kpts_lines  = np.zeros(c_nkpts, dtype=int)
        kpts_coords = np.zeros((c_nkpts, 3), dtype=float)
        
        #Start searching only at the first relevant block (DFT or GW)
        #skipping the previous part of the OUTCAR - otherwise we could match some spurious text
        lineidx_start_after = 0
        if lineidx_starts_gw:    lineidx_start_after = lineidx_starts_gw[0]
        elif lineidx_starts_dft: lineidx_start_after = lineidx_starts_dft[0]
        
        # Extract the "k-point header" lines: one header per k-point, marking the start of each QP/DFT block.
        # Assumptions: - `lineidx_start_after` is the first line index after which k-point headers appear.
        #              - OUTCAR header lines look like: " k-point   1 :   0.0000 0.0000 0.0000  plane waves: 4367"
        with open(path, "r") as f:
            # Iterate over the file starting from `lineidx_start_after` (memory-safe, streaming slice)
            for ln, line in enumerate(islice(f, lineidx_start_after, None), start=lineidx_start_after):
                s = line.strip()
                #Fast path: skip anything that doesn't look like a k-point header
                if not s.startswith("k-point"): continue    
            
                #Normalize possible " +1 " formatting and split into tokens
                #After this, we expect: ['k-point', '1', ':', 'kx', 'ky', 'kz', ...]
                parts = s.replace("+", "").split()  # normalize "+" and split
                if len(parts) < 6 or parts[2] != ":": continue
    
                # Parse the k-point index as written in the OUTCAR(1-based in OUTCAR)
                ik = int(parts[1])
                #Only store if the index is within the expected [1, c_nkpts]
                #Just not for robustness, not strictly needed.
                if 1 <= ik <= c_nkpts:
                    #Record the line number where this k-point block starts
                    #Then record the fractional k-point coordinates
                    kpts_lines[ik - 1]  = ln
                    kpts_coords[ik - 1] = [float(parts[3]), float(parts[4]), float(parts[5])]
                    #Early exit: stop as soon as we have found all k-point headers
                    if ik == c_nkpts: break
    
    
        #[4]Determine number of printed bands ---------#
        if lineidx_starts_gw:    c_nbands_printed = kpts_lines[1] - kpts_lines[0] - 4
        elif lineidx_starts_dft: c_nbands_printed = kpts_lines[1] - kpts_lines[0] - 3
    
    
        #[5]Allocate arrays for eigenvalues -----------#
        bs_spin_keys = (Spin.up,)      
        eigenval = {}
        if lineidx_starts_gw:
            for key in ("DFT", "GW", "Z", "QPC", "occ"):
                eigenval[key] = {s: np.zeros((c_nkpts, c_nbands_printed)) for s in bs_spin_keys}
        elif lineidx_starts_dft:
            for key in ("DFT", "occ"):
                eigenval[key] = {s: np.zeros((c_nkpts, c_nbands_printed)) for s in bs_spin_keys}
    
    
        #[6]Parse eigenvalue blocks -------------------#
        #Each k-point block in OUTCAR starts after the "k-point N :" line found earlier in kpts_lines.
        #We reopen the file for each k-point to avoid iterator exhaustion (islice cannot rewind).
        #This is slower but robust and simple.
        if lineidx_starts_gw:
            for ik in range(c_nkpts):
                with open(path, "r") as f:  # reopen each time; small overhead, but safe and robust
                #Extract the lines containing band information for this k-point
                #'+3' skips "k-point" header and first two non-data lin
                    for line in islice(f , kpts_lines[ik]+3 , kpts_lines[ik]+c_nbands_printed+3 ):
                            ls = line.split()
                            if len(ls) < 8: continue
                            ib = int(ls[0]) - 1  # band index (1-based in OUTCAR)
                            eigenval['DFT'][Spin.up][ik, ib] = float(ls[1])
                            eigenval['GW'][Spin.up][ik, ib]  = float(ls[2])
                            eigenval['Z'][Spin.up][ik, ib]   = float(ls[6])
                            eigenval['QPC'][Spin.up][ik, ib] = float(ls[2]) - float(ls[1])
                            eigenval['occ'][Spin.up][ik, ib] = float(ls[7])
        elif lineidx_starts_dft:
            for ik in range(c_nkpts):
                with open(path, "r") as f:  # reopen each time; small overhead, but safe
                        for line in islice(f , kpts_lines[ik]+2 , kpts_lines[ik]+c_nbands_printed+2 ):
                            ls = line.split()
                            if len(ls) < 3: continue
                            ib = int(ls[0]) - 1
                            eigenval['DFT'][Spin.up][ik, ib] = float(ls[1])
                            eigenval['occ'][Spin.up][ik, ib] = float(ls[2])
    
    
        #[7]Normalize occupations ---------------------#
        if setMaxOccupationsFromTwoToOne:
            for sp in bs_spin_keys: eigenval['occ'][sp] /= 2.0          
    
    
        #[8]Populate primary and misc datasets --------#
        if lineidx_starts_gw:
            primary_key = setGWDataAsprimary.strip().upper()     # ensure case-insensitive matching
            if primary_key not in eigenval: primary_key = "DFT"  # fallback
            bs_eigenval = eigenval[primary_key]
        else:
            bs_eigenval = eigenval["DFT"]
    
    
        #[9]Optionally construct the structure data from the POSCAR; this is not strictly required
        if (poscar_path is not None) and Path(poscar_path).exists():
                bs_structure = Structure.from_file(poscar_path)
        else:
                bs_structure = None
    
    
        # [10] Construct BandsState -------------------#
        bs_kpoints = KpointsData(kpts=bs_kpts_list, mesh=tuple(bs_kpts_mesh), shift=None)
        bs_misc    = {k: v for k, v in eigenval.items() if k not in ("occ",)}
        return BandsState(
            history=InstanceHistory(timestamp_loaded=datetime.now(), path_loaded=path, comment="Data from parsed OUTCAR at path_loaded"),
            spin_keys=bs_spin_keys , eigenval=bs_eigenval, occupation=eigenval["occ"],
            structure=bs_structure, kpoints=bs_kpoints ,
            misc=bs_misc,                              )
    
    def parse_bands_from_vasprun(path: str | Path) -> BandsState:
        """
        Parse a standard vasprun.xml file and return a BandsState.
        Functional, immutable, and pymatgen-compatible.
        """
        vasprunObject = Vasprun( str(path) , parse_projected_eigen=False , parse_potcar_file=False)
        #except: raise ValueError("error while opening vasprun.xml file.")
            
        bs_structure = vasprunObject.final_structure
        bs_kpts = KpointsData( kpts=np.array(vasprunObject.actual_kpoints),
                               mesh = vasprunObject.kpoints.kpts[0]    if vasprunObject.kpoints.kpts  is not None else None,
                               shift= vasprunObject.kpoints.kpts_shift if vasprunObject.kpoints.shift else None,           )
        
        bs_spin_keys = (Spin.up, Spin.down) if vasprunObject.is_spin else (Spin.up,)
    
        # Extract eigenvalues and occupations
        bs_eigenval, bs_occupation = {}, {}
        for spin in bs_spin_keys:
            bs_eigenval[spin]   = vasprunObject.eigenvalues[spin][:, :, 0]
            bs_occupation[spin] = vasprunObject.eigenvalues[spin][:, :, 1]
    
        return BandsState(
            history=InstanceHistory(timestamp_loaded=datetime.now(), path_loaded=path, comment="Data from parsed vasprun at path_loaded"),
            spin_keys=bs_spin_keys, eigenval=bs_eigenval, occupation=bs_occupation     ,
            structure=bs_structure , kpoints=bs_kpts,                                  )

    def parse_bands_from_WAVECAR(path: str | Path , poscar_path: str | Path | None = None , flag_verbose: bool = True  ) -> BandsState:
        """
        Parse a VASP WAVECAR manually (functional version, no py4vasp).
        Reads eigenvalues and occupations directly from binary records.
        Returns a fully immutable BandsState.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"WAVECAR not found at {path}")
    
        with open(path, "rb") as f:
            #[1]Read from 1° record  record length , number of spin components, rtag (aka precision)
            #They are integer but stored as float64.
            c_reclen_bit, c_nspin, rtag = np.fromfile(f, dtype=np.float64, count=3).astype(np.int_)
            #Rewrite precision from bit to byte.
            c_reclen = int(c_reclen_bit / 8)
    
            #[2]Only record length-number of spin components-rtag are saved inside 1°record;
            #Vasp writes WDES%NKPTS , WDES%NB_TOT , WDES%ENMAX , ((LATT_INI%A(I,J),I=1,3),J=1,3) , EFERMI to 2°record;
            #thus let's pad to second record and read them.
            np.fromfile(f, dtype=np.float64, count=(c_reclen - 3)  )
            c_numk, c_numb, c_encut = np.fromfile(f, dtype=np.float64, count=3).astype(np.int_)
            c_latvec      = np.fromfile(f, dtype=np.float64, count=9).reshape((3, 3))
            c_efermi      = np.fromfile(f, dtype=np.float64, count=1)[0]
            if flag_verbose:
                print("-constants read from WAVECAR prolog:\n  spin comp ",c_nspin,"\n  num kpts  ",c_numk,"\n  num bands ",c_numb,"\n  encut (eV)",c_encut)
                print("  efermi    ",c_efermi)
                print("  lattice vec",c_latvec[0,:],"\n             ",c_latvec[1,:],"\n             ",c_latvec[2,:],"\n")
    
            #[3]Padding to end of record REC=2 / beginning of record REC=3
            # c_reclen -13 because we have read 13 element from REC=1 start
            np.fromfile(f, dtype=np.float64, count=(c_reclen - 13) )
            
            #[4]First initialize the stuff, then read it!
            eigenvalues= np.zeros([c_nspin , c_numk , c_numb , 2] , dtype=np.float64) #[spin,nk,nb,0]=eigenvalues - [spin,nk,nb,1]=occupation
            kpts_list  = []
            for idx_spin in range(c_nspin):
                if flag_verbose: print("-outer level loop: spin component {}".format(idx_spin))
                for idx_kpt in range(c_numk):
                      if flag_verbose: print("--1° level loop: kpt {}".format(idx_kpt))
    
                      #Read the number of plane waves for this specific kpts for this spin channel
                      num_pw = int(np.fromfile(f, dtype=np.float64, count=1)[0])
                      
                      #Read the kpt associated to this index
                      #Sometimes kpts that should have 0 coefficients have very small but non-zero value,
                      #ES [1.26237862e-15 1.26237862e-15 1.26237862e-15] for Gamma [0,0,0]. We handle them manually.
                      kpoint = np.fromfile(f, dtype=np.float64, count=3)
                      kpoint[abs(kpoint) < 1E-10] = 0
                      kpts_list.append(list(kpoint))
                      if flag_verbose: print("  kpoint {} with {: 3} plane waves".format(kpoint,num_pw))
    
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
        #[5]Construct first BandsState elements which can be constructed from WAVECAR
        bs_spin_keys  = (Spin.up, Spin.down) if c_nspin == 2 else (Spin.up,)
        bs_eigenval   = {spin: eigenvalues[i, :, :, 0] for i, spin in enumerate(bs_spin_keys)}
        bs_occupation = {spin: eigenvalues[i, :, :, 1] for i, spin in enumerate(bs_spin_keys)}
        bs_kpoints    = KpointsData(kpts=np.array(kpts_list) , mesh=None , shift=None)
        bs_structure  = None
        
        #[6]Optionally construct the structure data from the POSCAR; this is not strictly required
        if (poscar_path is not None) and Path(poscar_path).exists():
                bs_structure = Structure.from_file(poscar_path)
            
        return BandsState(
                history   = InstanceHistory(timestamp_loaded=datetime.now(), path_loaded=path, comment="Data from parsed WAVECAR at path_loaded") ,
                eigenval=bs_eigenval , occupation=bs_occupation , spin_keys=bs_spin_keys       ,
                structure = bs_structure , kpoints=bs_kpoints,                                 )
    



   def parse_bands_from_WAVECAR_NEW(path: str | Path,
                             poscar_path: str | Path | None = None,
                             flag_verbose: bool = True) -> BandsState:
    """
    Parse bands from WAVECAR using pymatgen.Wavecar (robust, version-safe).
    Returns a fully populated BandsState with the same structure as
    parse_outcar_spinUnpol() and parse_bands_from_vasprun().
    """

    from pymatgen.io.vasp.outputs import Wavecar

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"WAVECAR not found at {path}")

    if flag_verbose:
        print(f"[pymatgen] Reading eigenvalues from WAVECAR: {path}")


    #1] Load WAVECAR
    w = Wavecar(str(path))

    #2] constants
    nspin   = w.nspin
    nkpts   = w.nkpts
    nbands  = w.nbands
    kpts    = np.array(w.kpoints)
    eigs    = w.eigs          # shape (nspin, nkpts, nbands)
    occs    = w.occupancies   # shape (nspin, nkpts, nbands)
    if flag_verbose:
        print(f"  spin components : {nspin}")
        print(f"  k-points        : {nkpts}")
        print(f"  bands           : {nbands}")
        print(f"  fermi energy    : {w.efermi}")


    #3] Build BandsState fields
    # spin keys
    bs_spin_keys = (Spin.up, Spin.down) if nspin == 2 else (Spin.up,)

    # eigenvalues & occupations dictionaries
    bs_eigenval   = {spin: eigs[i] for i, spin in enumerate(bs_spin_keys)}
    bs_occupation = {spin: occs[i] for i, spin in enumerate(bs_spin_keys)}

    # kpoints
    bs_kpoints = KpointsData(
        kpts=kpts,
        mesh=None,     # a WAVECAR does NOT store the mesh, only explicit kpts
        shift=None
    )

    # 4] Optional structure
    if poscar_path is not None and Path(poscar_path).exists():
        if flag_verbose:
            print(f"[pymatgen] Loading structure from POSCAR: {poscar_path}")
        bs_structure = Structure.from_file(poscar_path)
    else:
        bs_structure = None

    # 5] Construct BandsState
    bs = BandsState(
        history=InstanceHistory( imestamp_loaded=datetime.now(),
                                 path_loaded=path,
                                 comment="Data from parsed WAVECAR via pymatgen" ),
        kpoints=bs_kpoints,
        structure=bs_structure,
        eigenval=bs_eigenval,
        occupation=bs_occupation,
        spin_keys=bs_spin_keys,
        misc={}   # WAVECAR has no GW/DFT auxiliary sets     )
    return bs











    def write_bands_to_WAVECAR(state: BandsState, path: str | Path, flag_verbose: bool = True):
        """
        Write eigenvalues and occupations from a BandsState object back into an existing WAVECAR.
        Keeps binary format intact, only overwriting the eigenvalue+occupation blocks.
        """  
        #[Preparatory-1]Extract constants from BandsState
        c_numSpin = len(state.spin_keys)
        c_numKpts = np.shape(state.eigenval[Spin.up])[0]
        c_numNbnd = np.shape(state.eigenval[Spin.up])[1]
    
        if flag_verbose: print(f"write_file_WAVECAR: writing eigenvalues+occupations { [c_numSpin, c_numKpts, c_numNbnd, 2] } to {path}")
    
        #[Preparatory-2]Pack into [spin, k, band, 2] array for easier indexing
        eigenvalues = np.zeros([c_numSpin, c_numKpts, c_numNbnd, 2], dtype=np.float64)
        for i, spin in enumerate(state.spin_keys):
            eigenvalues[i, :, :, 0] = state.eigenval[spin]
            eigenvalues[i, :, :, 1] = state.occupation[spin]
    
        
        #Binary rewriting section (unchanged)
        with open(Path(path), "r+b") as f:
            #Read WAVECAR header
            #Read from 1° record  record length , number of spin components, rtag (aka precision)
            #They are integer but stored as float64.
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
            
            if flag_verbose:
                print("<Constants read from WAVECAR prolog:\n  spin comp ",c_nspin,"\n  num kpts  ",c_numk,"\n  num bands ",c_numb,"\n  encut (eV)",c_encut)
                print("  efermi    ",c_efermi)
                print("  lattice vec",c_latvec[0,:],"\n             ",c_latvec[1,:],"\n             ",c_latvec[2,:],"\n")
                print("<Costants from BandsState whose eigenvalues will replaces ones in WAVECAR\n  spin comp ",np.shape(eigenvalues)[0],"\n  num kpts  ",np.shape(eigenvalues)[1],"\n  num bands ",np.shape(eigenvalues)[2])
            if [c_nspin , c_numk , c_numb] != [ np.shape(eigenvalues)[0] ,  np.shape(eigenvalues)[1] ,  np.shape(eigenvalues)[2] ]:
                raise ValueError("spin number and/or kpts number and/or band number of input eigenvalue variable does NOT match WAVECAR internal dimensions.")
    
            #Padding to end of record REC=2 / beginning of record REC=3
            #c_reclen -13 because we have read 13 element from REC=1 start
            np.fromfile(f, dtype=np.float64, count=(c_reclen - 13) )
            for idx_spin in range(c_nspin):
                    if flag_verbose: print("-outer level loop: spin component {}".format(idx_spin))
                    for idx_kpt in range(c_numk):
                        if flag_verbose: print("--1° level loop: kpt {}".format(idx_kpt))
                        
                        num_pw = int(np.fromfile(f, dtype=np.float64, count=1)[0])
                        kpoint =     np.fromfile(f, dtype=np.float64, count=3)
                        if flag_verbose: print("  kpoint {} with {: 3} plane waves".format(kpoint,num_pw))            
                        
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

class BandsState_InterpOp :
    @staticmethod
    def apply_QP_correction(  dft_BS: BandsState , interpolatedQP_BS: BandsState ,
                              kpt_tolerance: float = 1e-4 , flag_save_original_State_in_misc = True ,
                              nb_apply_first = 0 ,  # First band (0-indexed) - this is hardcoded because the actual vasp version always interpolate all bands from the first one
                                                    # However other codes allow to control also the full range of bands to interpolate
                                                    # This is used as index of a numpy array, so the first band has index=0
                              flag_pad_qp_beyond_interp_with_last = True , 
                              flag_verbose: bool = True,
                              ) -> BandsState:
        """
        Apply quasiparticle (QP) corrections stored in `interp_state` to the DFT eigenvalues
        of `dft_state`, returning a *new* corrected BandsState.
      
         
       1) interpolatedQP_BS provides `nb_interp_last` correction bands indexed [0..nb_interp_last-1]
       2)  they are applied to DFT bands inside dft_BS
           starting at index nb_apply_first: i.e dft_BS.eigenval[:, nb_apply_first + i] += QPcorr[:, i]
        
            Overflow handling:
            - if nb_apply_first + nb_interp_last > nb_dft_last => clip application and WARN (no error)
            Padding:
            - if enabled and the applied range ends before the last DFT band => pad remaining with last available correction

       `interpolatedQP_BS.occupation` may be None (typical if it represents ΔE_QP only).
       """
        
    
        #[1] Input checks ------------------------------------------------------
        # Check k-points consistency
        if dft_BS.kpoints is None or interpolatedQP_BS.kpoints is None:
            raise ValueError("kpoints is None in one of the BandsState objects.")
        if not np.allclose(dft_BS.kpoints.kpts, interpolatedQP_BS.kpoints.kpts, atol=kpt_tolerance):
            raise ValueError("K-point grids differ between DFT and QP interpolation states.")
        # Check spin consistency
        if len(dft_BS.spin_keys) != len(interpolatedQP_BS.spin_keys):
            raise ValueError("Spin polarization differs between DFT and interpolated data.")
    
        #[2] Determine band ranges --------------------------------------------
        # nb_dft_last    = last band index of the bands WILL BE APPLIED QP TO (the DFT)
        # nb_interp_last = last band index of the bands THAT ARE INTERPOLATED AND WILL BE APPLIED (QPc)
        
        nb_dft_last    = dft_BS.eigenval[dft_BS.spin_keys[0]].shape[1]
        nb_interp_last = interpolatedQP_BS.eigenval[interpolatedQP_BS.spin_keys[0]].shape[1]
        
        if nb_apply_first < 0 : nb_apply_first=0
        if nb_apply_first >= nb_dft_last :
            raise ValueError("`nb_apply_first` exceeds available band count in dft_BS.")
       
        
        #2.1] NOW determine application window on DFT bands
        nb_apply_last          = nb_apply_first + nb_interp_last         # exclusive
        nb_apply_last_clipped  = min(nb_apply_last , nb_dft_last)        # exclusive (clipped)
        nb_apply_nbands        = nb_apply_last_clipped - nb_apply_first  # how many bands we will actually apply

        if flag_verbose:
            print( "[apply_QP_correction] "
                    f"nb_dft_last={nb_dft_last}, nb_interp_last={nb_interp_last}, "
                    f"apply: DFT[{nb_apply_first}:{nb_apply_last_clipped}) <- QP[0:{nb_apply_nbands})"    )
            if nb_apply_last > nb_apply_last_clipped:
                print(  f"[apply_QP_correction] WARNING: overflow: nb_apply_first+nb_interp_last={nb_apply_last} "
                        f"> nb_dft_last={nb_dft_last}. Clipping applied range."                   )
              
        
        #[3] Edge-case : Nothing to apply if there are no correction bands ----
        if nb_interp_last <= 0:
            if flag_verbose:
                print(f"[apply_QP_correction] nb_interp_last={nb_interp_last}: nothing to apply.")
            # Return a copy of dft_BS BandsState 
            return BandsState( structure=dft_BS.structure, kpoints=dft_BS.kpoints,
                               spin_keys=dft_BS.spin_keys,
                               eigenval=deepcopy(dft_BS.eigenval),
                               occupation=deepcopy(dft_BS.occupation),
                               misc=deepcopy(dft_BS.misc),
                history=InstanceHistory(timestamp_loaded=datetime.now(),
                                        path_loaded=dft_BS.history.path_loaded,
                                        comment=(dft_BS.history.comment or "") + " | apply_QP_correction: no QP bands"), )

        
        #[4] Compute corrected eigenvalues ------------------------------------
        bs_corrected_eigenval   = {}
        bs_corrected_occupation = {}
             
        for sp in dft_BS.spin_keys:
            dft_energies = dft_BS.eigenval[sp][: , :]   # (nk, nb_dft_last)
            qp_corr = np.zeros(dft_energies.shape, dtype=dft_energies.dtype)
        
            #[4.1] apply corrections into target window
            qp_corr[:, nb_apply_first:nb_apply_last_clipped] = interpolatedQP_BS.eigenval[sp][:, :nb_apply_nbands]

            #[4.2] optional padding above applied region
            # nb_apply_last_clipped < nb_dft_last, i.e. corrections are available for less bands than the one on DFT
            #       pad corrections beyond available interp bands. i.e. repeat the LAST available correction band 
            #       (the last correction band has index nb_interp_last-1)
            #       This is controlled by an input flag
            if flag_pad_qp_beyond_interp_with_last and (nb_interp_last > 0) and (nb_apply_last_clipped < nb_dft_last):
                qp_corr[:, nb_apply_last_clipped:nb_dft_last] = interpolatedQP_BS.eigenval[sp][:, [nb_interp_last - 1]]

            bs_corrected_eigenval[sp] = dft_energies + qp_corr
        

        #[5] The occupations --------------------------------------------------
        # pass through DFT occupations (can be None, or dict with None entries.
        # or dict of arrays)
        # If occupations were None, keep occupation=None (not a dict of Nones)
        if dft_BS.occupation is None:
            bs_corrected_occupation = None
        else:
            # If occupations is a dict of None, keep that
            for sp in dft_BS.spin_keys:
                if dft_BS.occupation[sp] is None:
                    bs_corrected_occupation[sp] = None
                else:
                    bs_corrected_occupation[sp] = dft_BS.occupation[sp]
            

        #[6] Construct misc object depending on the flag ----------------------
        if flag_save_original_State_in_misc : 
            bs_misc = { "DFT": dft_BS.eigenval , 
                        "QP_correction": interpolatedQP_BS.eigenval }
        else : 
            bs_misc = {}
    
        #[7] Assemble corrected BandsState ------------------------------------------
        comment_text = "BandsState obtained from adding misc[QP_corrections] to misc[DFT], where DFT data obtained from path=" +\
                       str(dft_BS.history.path_loaded)+" and QP_corrections from path="+str(interpolatedQP_BS.history.path_loaded)
        corrected_state = BandsState(
            history=InstanceHistory( timestamp_loaded=datetime.now(), path_loaded=dft_BS.history.path_loaded, comment=comment_text ),
            structure=dft_BS.structure , kpoints=dft_BS.kpoints,
            spin_keys=dft_BS.spin_keys ,
            eigenval=bs_corrected_eigenval     ,
            occupation=bs_corrected_occupation ,
            misc=bs_misc ,                     )    
        return corrected_state

    @staticmethod
    def _structure_to_spglibcompatible_structure(bands_state):
        spglib_structure = {}
        spglib_structure["lattice"]   = bands_state.structure.lattice.matrix
        spglib_structure["positions"] = bands_state.structure.frac_coords
        tmp_unique_species = [] ; tmp_numbers = []
        for species, itertoolsGrouper in itertools.groupby( bands_state.structure, key=lambda s: s.species):
            if species in tmp_unique_species:
                ind = tmp_unique_species.index(species)
                tmp_numbers.extend([ind + 1] * len(tuple( itertoolsGrouper )))
            else:
                tmp_unique_species.append(species)
                tmp_numbers.extend([len(tmp_unique_species)] * len(tuple( itertoolsGrouper )))
        spglib_structure["numbers"]   = tmp_numbers  #Explanation: species in POSCAR, each integer identifies a different species
                                                        #ES: numbers = [1, 2, 2, 2]        # Al, Ni, Ni, Ni 
        return ( spglib_structure["lattice"] ,  spglib_structure["positions"] ,  spglib_structure["numbers"] )
  
    @staticmethod              
    def _determine_BZ_IBZ_grid(bands_state, symprec=1e-5, flag_verbose=False):
        """
        Determine the mapping between IBZ and full BZ k-points for a BandsState object.
        Returns dict with:
            'map_BZ_to_IBZ'  : (nBZ,) int array :  For each BZ k-point, gives the index (0..nIBZ-1) of the equivalent IBZ point.
            'map_IBZ_to_BZ'  : (nBZ,) int array :  For each entry in BZ, gives which IBZ point index generated it (redundant, but symmetric).
            'kpts_IBZ'       : (nIBZ,3) array of fractional coordinates.
            'kpts_BZ'        : (nBZ,3) array of fractional coordinates.
        """
    
        def _determine_nova(q):
            #> [8] indip e- in periodic pag 5, Fieschi - De Renzi: per ogni vettore k che cade sul bordo zona ce n'è un altro k' pari k+G.
            #> we are working in direct.coordinates, which from https://www.vasp.at/wiki/index.php/KPOINTS are defined as (x1b1 + x2b2 + x3b3)
            # reciprocal.lattice.vectors are exactly defined in the same manner (x1b1 + x2b2 + x3b3), thus in frac.coordinates are G1[(x1,x2,x3)=(1,0,0)] etc
            #> we define nova(q) = q + {all possible 1°shell G}
            from itertools import permutations 
            transl_singleG = list(set(    list(permutations([1, 0, 0]))+list(permutations([-1, 0, 0]))    ))
            transl_doubleG = list(set(    list(permutations([1, 1, 0]))+list(permutations([-1, 1, 0]))+list(permutations([-1, -1, 0]))    ))
            transl_tripleG = list(set(    list(permutations([1, 1, 1]))+list(permutations([-1, 1, 1]))+list(permutations([-1, -1, 1]))+list(permutations([-1, -1, -1]))    ))
            transl_tot = transl_singleG+transl_doubleG+transl_tripleG
            return(   np.unique(q + np.array(transl_tot) , axis=0 )   )
    

    
        # --- [1] Construct a spglib compatible structure object ------------------
        # spglib has a structure ( structure: np.array , atoms : np.array , atomtype : np.array )    
        import spglib    
        spglib_str  = BandsState_InterpOp._structure_to_spglibcompatible_structure( bands_state )
        kpts_mesh = np.array(bands_state.kpoints.mesh, dtype=int)
        if flag_verbose:
            print("   [IBZ] Determining full BZ grid and IBZ->BZ mapping:")
            print("   [IBZ] Structure supplied has spacegroup = ",spglib.get_spacegroup( spglib_str , symprec=1e-1))  #this as sanity chec
    
    
        # --- [2] Build reciprocal mesh in BZ and IBZ via spglib ------------------
        #[0] Given a kmesh (a kmesh is the list with the numbers of division, i.e. [6,6,6], not the kpts list) + the structure
        #   mapping, grid = get_ir_reciprocal_mesh(mesh, cell, is_shift=[0, 0, 0])
        #   grid gives the mesh points in fractional coordinates in reciprocal space. 
        #   mapping has dimension = #(kpts.list of BZ point); the k-points in the IBZ have an index inr range (0,#IBZ.kptslist-1) 
        #   The kpts outside the IBZ have (inside the mapping vector) the index of the kpt in the IBZ the map to.
        #[1] all points in BZ (spg_map and spg_map_modified) have one of those indexs - is the IBZ index which they are equivalent (by symmetry operations) to.
        #[2] Thus the indexes of the point of the IBZ are found by np.unique(mapping)
        spg_map, spg_grid_BZ = spglib.get_ir_reciprocal_mesh(kpts_mesh , spglib_str , is_shift=[0, 0, 0])
    
        #  spg_map_BZtoIBZ:  spg_direct_IBZ  = spg_direct_BZ[spg_map_BZtoIBZ]  
        #  spg_map_IBZtoBZ_symConnected: it's not possible to fully reconstruct IBZ to BZ, because we miss informations (namely we would require the pointGroupSymmetryOp to reconstruct all informations).
        #                                spg_map is non-injective mapping from BZ to IBZ- associate different kpts linked by symOp to same IBZ kpt.
        #                                definition: spg_map[i] = spg_map_BZtoIBZ[j] -> spg_map_IBZtoBZ_symConnected[i] = j
        spg_map_BZtoIBZ = np.unique(spg_map)
        spg_grid_IBZ    = spg_grid_BZ[spg_map_BZtoIBZ]
        # Convert integer grid coordinates to fractional (direct) k-point coords by dividing by the mesh.
        # Example: if mesh = [6,6,6], a grid point [3,0,1] becomes [0.5, 0.0, 1/6].
        spg_direct_BZ   = spg_grid_BZ / kpts_mesh
        spg_direct_IBZ  = spg_direct_BZ[spg_map_BZtoIBZ]
        spg_map_IBZtoBZ_symConnected = np.array( [np.where(map_value == spg_map_BZtoIBZ)[0][0]  for map_value in spg_map])
        # Sanity checks:
        # 1) The IBZ k-points picked out from the BZ list must equal the separately computed IBZ list.
        # 2) Ensure the IBZ k-points we got from spglib match the BandsState list (up to small numerical noise).
        #    We sort both lists so that order differences don’t matter.
        assert np.all( spg_direct_BZ[spg_map_BZtoIBZ] == spg_direct_IBZ ) 
        assert np.all( abs( np.sort(bands_state.kpoints.kpts)  - np.sort(spg_direct_IBZ)  ) <1E-4   ) , "ERROR: The kpoints list saved in self.kpoints and the list (of kpts in the IBZ) returned by spglib differ."
    
    
        # Create editable copies of the spglib BZ map and BZ k-point list.
        # We’ll augment these when we detect “duplicate” points on the BZ boundary (k and k+G).        
        spg_map_modified       = deepcopy(spg_map).tolist()
        spg_direct_BZmodified  = deepcopy(spg_direct_BZ).tolist()
        ##[1] which kpts are on the BZ edge, as reconstructed by spglib - checked manually and seems ok.
        # [Edge detection] Identify k-points that lie exactly on the BZ boundary (i.e., any component is ±0.5).
        # For the interpolation we need the kpts (and the energies on) all boundaries; if a kpts is on a BZ boundary, 
        # another kpts  related by a reciprocal-lattice vector G may sit on another BZ boundary
        kBZ_onBZedge =[]  ;  kBZ_onBZedge_mapIdx = [] 
        for kBZ_mapIdx , kBZ in  zip(spg_map , spg_direct_BZ):
             if (0.5 in kBZ):
                    kBZ_onBZedge.append(kBZ)
                    kBZ_onBZedge_mapIdx.append(kBZ_mapIdx)

        ##[2] construct nova for these k-points k + first-shell reciprocal vectors (±1,0,0 permutations, etc.).
        # Keep only those images that still lie inside the first BZ (|component| ≤ 0.5).
        for kBZ_mapIdx , kBZ in zip(kBZ_onBZedge_mapIdx , kBZ_onBZedge):
            kBZ_onBZedge_nova      = _determine_nova(kBZ)  
            kBZ_onBZedge_nova_inBZ = [x for x in kBZ_onBZedge_nova if np.all(abs(x)<=0.5)]         
    
            #[3] if k of nova is NOT comprised in BZ list, add.
            # If a nova point is inside the BZ but missing from our BZ list, add it and map it to
            # the same IBZ representative (same symmetry label as its generator).
            for k_nova in kBZ_onBZedge_nova_inBZ:
                if list(k_nova) not in spg_direct_BZmodified:
                    print("    ",k_nova,"not present -> ADDING.")
                    spg_direct_BZmodified.append( list(k_nova)  )
                    spg_map_modified.append( kBZ_mapIdx      )
                else:
                    print("testing if",k_nova," already present... nope.")
        # Clean up loop variables (not strictly necessary, but keeps namespace tidy).
        del k_nova , kBZ_mapIdx , kBZ , kBZ_onBZedge_nova , kBZ_onBZedge_nova_inBZ   
        # Rebuild the BZ→IBZ mapping indices for the modified lists, again expressed as indices into the
        # unique IBZ set order. This mirrors spg_map_IBZtoBZ_symConnected but for the augmented BZ.
        spg_map_IBZtoBZmodified_symConnected = np.array( [np.where(map_value == spg_map_BZtoIBZ)[0][0]  for map_value in spg_map_modified])
         
        # Return:
    # - spg_map_BZtoIBZ: indices (into spg_grid_BZ) of the IBZ representatives (size nIBZ)
    # - spg_map_IBZtoBZ_symConnected: for each original BZ point, which IBZ index it maps to (size nBZ_original)
    # - spg_map_IBZtoBZmodified_symConnected: same as above but for the augmented BZ (size nBZ_modified)
    # - spg_direct_IBZ: fractional coords of IBZ k-points (nIBZ, 3)
    # - spg_direct_BZ: fractional coords of original BZ k-points (nBZ_original, 3)
    # - spg_direct_BZmodified: fractional coords of augmented BZ k-points (nBZ_modified, 3)
        return spg_map_BZtoIBZ , spg_map_IBZtoBZ_symConnected , spg_map_IBZtoBZmodified_symConnected , spg_direct_IBZ , spg_direct_BZ , spg_direct_BZmodified

    @staticmethod
    def _map_bands_fromIBZ_toEdgedBZ( ibz_BS: BandsState , flag_verbose: bool = True ) -> BandsState:
        """
        Map band structure from irreducible Brillouin zone (IBZ) to full/modified BZ.
        
        Uses spglib symmetry operations to expand IBZ bands to the full BZ grid,
        including equivalent points on zone edges (k' = k + G).
        
        Parameters
        ----------
        ibz_BS : BandsState
            Band structure with IBZ k-points only
        flag_verbose : bool
            Print diagnostic information
            
        Returns
        -------
        BandsState
            Band structure expanded to modified BZ grid
            
        Notes
        -----
        - get_ir_reciprocal_mesh returns non-contiguous IBZ indices
        - All BZ points map to one IBZ index (their symmetry equivalent)
        - Data from vasprun is ordered as in self.kpoints, not as spglib orders IBZ
        """
        
        if flag_verbose:
            print("{<MAPPING bands IBZ -> BZ>} <------------------>")
        
        #[1] Determine BZ/IBZ grids and mappings -> then unpack with original variable names
        bz_mapping = BandsState_InterpOp._determine_BZ_IBZ_grid( ibz_BS , flag_verbose=flag_verbose )
        (spg_map_BZtoIBZ,
         spg_map_IBZtoBZ_symConnected,
         spg_map_IBZtoBZmodified_symConnected,
         spg_direct_IBZ,
         spg_direct_BZ,
         spg_direct_BZmodified) = bz_mapping

       
        
        #[2] Allocate arrays for mapped BZ data ===
        n_bz_mod = len(spg_map_IBZtoBZmodified_symConnected)
        n_bands  = ibz_BS.eigenval[ibz_BS.spin_keys[0]].shape[1]
        bz_eigenval = {} ; bz_occupation = {}
    
        #[3] Map IBZ data to BZ using symmetry connections ===
        # For each BZ point, copy data from its symmetry-equivalent IBZ point
        for sp in ibz_BS.spin_keys:
            bz_eigenval[sp]    = np.zeros((n_bz_mod, n_bands))
            bz_occupation[sp] = np.zeros((n_bz_mod, n_bands)) 
            for map_value_Idx, map_value in enumerate(spg_map_IBZtoBZmodified_symConnected):
                bz_eigenval[sp][map_value_Idx, :] = ibz_BS.eigenval[sp][map_value, :]
                bz_occupation[sp][map_value_Idx, :] = ibz_BS.occupation[sp][map_value, :]
      
        #[4] Construct new BandsState with BZ data ===
        bz_kpoints = KpointsData(  kpts=spg_direct_BZmodified , mesh=ibz_BS.kpoints.mesh , shift=ibz_BS.kpoints.shift )   
      
        bz_BS = BandsState( structure=ibz_BS.structure , kpoints=bz_kpoints ,
                            history=InstanceHistory( timestamp_loaded=datetime.now() , path_loaded=None ),
                            eigenval=bz_eigenval ,  occupation=bz_occupation, spin_keys=ibz_BS.spin_keys,
                            misc=deepcopy(ibz_BS.misc) )
        return bz_BS 

    @staticmethod
    def _interpolate_bands_fromCoarseToFineEdgedKmesh( coarsegrid_BS: BandsState , finegrid_KpointsData: KpointsData , 
                                             flag_interp_method: str = 'linear' , 
                                             flag_apply_delta_correction: bool = True, flag_verbose: bool = True,
                                           ) -> tuple[ dict[Spin, np.ndarray], dict[Spin, np.ndarray], dict[Spin, np.ndarray], bool]:
        """
        - Interpolate bands using scipy.interpolate.griddata
        - Straight interpolation could give NaN (this if some kpts on the dense grid are outside 
          the convex hull defined by the kpoints of the sparse grid)
          In this case NaN correction using nearest-neighbor fallback, i.e.
          E_NaN(k) ≈ E_nn(k)  for all k with NaN ; where E_nn(k) = E_{nearest neighbor of k}
          This correction is ALWAYS ENABLED; if flag_verbose is True, a warning is printed.
        - An additional refinement correction is applied to the E_NaN(k) ≈ E_nn(k)
          E_NaN(k) ≈ E_nn(k) + [E_linear(nearest neighbor of k) - E_nn(nearest neighbor of k)]
        
        This is a *numerical* interpolation kernel — it returns raw NumPy
        arrays of interpolated eigenvalues, not full BandsState objects.
        A wrapper should construct BandsState objects from these arrays.
        
        Parameters
        ----------
        coarsegrid_BS : BandsState          - Coarse grid band structure (IBZ)
        finegrid_kpts_list : np.ndarray     - Target fine grid k-points, shape (n_kpts, 3)
        finegrid_kpts_mesh : tuple          - Fine grid mesh dimensions
        finegrid_kpts_shift : tuple or None - Fine grid shift 
        finegrid_spglib_structure : dict    - Structure for fine grid
        flag_apply_delta_correction : bool 
        flag_verbose : bool
        flag_interp_method : str            - Primary interpolation method ('linear', 'rbf' , 'regular')
                             where the supported methods are:
                            - 'linear' or 'cubic' : scipy.interpolate.griddata (unstructured)
                            - 'rbf'               : scipy.interpolate.RBFInterpolator (smooth, extrapolates)
                            - 'regular'           : scipy.interpolate.RegularGridInterpolator (structured grid only)

        Returns
        -------
        bands_interp : dict[Spin, np.ndarray]                    Raw interpolation results, shape (c_nkfine, c_nbands)
        bands_interp_NaNreplacedbynn : dict[Spin, np.ndarray]    Same as `bands_interp`, but with NaNs replaced by 
                                                                 nearest-neighbor values.
        bands_interp_NaNreplacedbynn_DeltaCorr : dict[Spin, np.ndarray]   Final Δ-corrected results (preferred for downstream use).
        flag_hasNaN : bool                                       True if any NaN values were encountered during interpolation.  
            
        Notes
        -----
        - Each band index is interpolated independently in k-space.
        - This function assumes identical band ordering between coarse  and fine grids (no band crossings tracked).
        - Δ-correction can slightly smooth discontinuities near convex-hull boundaries when coarse data are sparse.
        - 
        """

        #[1] Initialize Initialize
        c_nbands    = coarsegrid_BS.eigenval[coarsegrid_BS.spin_keys[0]].shape[1]
        c_nkfine    = len(finegrid_KpointsData.kpts)
        c_spin_keys = coarsegrid_BS.spin_keys
        bands_interp    = {sp: np.zeros((c_nkfine, c_nbands), dtype=np.float64) for sp in coarsegrid_BS.spin_keys }
        bands_interp_nn = {sp: np.zeros((c_nkfine, c_nbands), dtype=np.float64) for sp in coarsegrid_BS.spin_keys }
        bands_interp_NaNreplacedbynn = {sp: np.copy(bands_interp[sp]) for sp in coarsegrid_BS.spin_keys}

        coarsegrid_BS_mappedToBZ =  BandsState_InterpOp._map_bands_fromIBZ_toEdgedBZ( coarsegrid_BS )
        coarsegrid_BS_toInterp = coarsegrid_BS_mappedToBZ

        #[2]Perform both primary and nearest-neighbor interpolation
        #OLD DRIVER
        # for sp in c_spin_keys:
        #     for bIdx in range(c_nbands):
        #         # Primary interpolation method
        #         bands_interp[sp][:, bIdx] = scipy.interpolate.griddata(
        #             points=coarsegrid_BS_toInterp.kpoints.kpts ,          # Data point coordinates
        #             values=coarsegrid_BS_toInterp.eigenval[sp][:, bIdx],  # Data values
        #             xi=finegrid_KpointsData.kpts      ,          # Points at which to interpolate data
        #             method=flag_interp_method         )
        #         # Nearest-neighbor fallback
        #         bands_interp_nn[sp][:, bIdx] = scipy.interpolate.griddata(
        #             points=coarsegrid_BS_toInterp.kpoints.kpts ,          # Data point coordinates
        #             values=coarsegrid_BS_toInterp.eigenval[sp][:, bIdx],  # Data values
        #             xi=finegrid_KpointsData.kpts      ,          # Points at which to interpolate data
        #             method="nearest"                  ) 
        for sp in c_spin_keys:
            kpts_fine   = finegrid_KpointsData.kpts
            kpts_coarse = coarsegrid_BS_toInterp.kpoints.kpts
            for bIdx in range(c_nbands):
                vals_b = coarsegrid_BS_toInterp.eigenval[sp][:, bIdx]
                
                # Nearest-neighbor fallback - done in any casae
                bands_interp_nn[sp][:, bIdx] = scipy.interpolate.griddata(
                    points=kpts_coarse , # Data point coordinates
                    xi=kpts_fine       , # Points at which to interpolate data
                    values=vals_b      , # Data values
                    method="nearest"   )                 
                
                # Primary interpolation method
                if flag_interp_method == 'linear' :
                    bands_interp[sp][:, bIdx] = scipy.interpolate.griddata(
                        points=kpts_coarse , # Data point coordinates
                        xi=kpts_fine       , # Points at which to interpolate data            
                        values=vals_b      , # Data values
                        method="linear" )
                elif flag_interp_method == 'rbf:gaussian' :
                    rbf_interp = scipy.interpolate.RBFInterpolator( kpts_coarse, vals_b, kernel="gaussian", epsilon=0.2, smoothing=1e-4 )
                    bands_interp[sp][:, bIdx] = rbf_interp(kpts_fine)
                elif (flag_interp_method == 'rbf:cubic') or (flag_interp_method == 'rbf'):
                    rbf_interp = scipy.interpolate.RBFInterpolator( kpts_coarse, vals_b, kernel="cubic", smoothing=1e-4 )
                    bands_interp[sp][:, bIdx] = rbf_interp(kpts_fine) 
                
                elif (flag_interp_method == 'regular') or (flag_interp_method == 'rbf'):
                    kpts_coarse = np.asarray(kpts_coarse, dtype=float)
                    vals_b = np.asarray(vals_b, dtype=float)
                    # Extract unique grid axes (sorted)
                    kx = np.unique(kpts_coarse[:, 0])
                    ky = np.unique(kpts_coarse[:, 1])
                    kz = np.unique(kpts_coarse[:, 2])
                    mesh_shape = (len(kx), len(ky), len(kz))
                    if np.prod(mesh_shape) != len(kpts_coarse): raise ValueError("K-points do not form a regular 3D grid " )
                    grid_val = np.full(mesh_shape, np.nan, dtype=float)
            
                    # --- Fill 3D grid by mapping each (kx_i, ky_i, kz_i) triplet to its index ---
                    # This guarantees that we use numpy indexing, not list indexing
                    for i, (kx_i, ky_i, kz_i) in enumerate(kpts_coarse):
                        ix = np.searchsorted(kx, kx_i)
                        iy = np.searchsorted(ky, ky_i)
                        iz = np.searchsorted(kz, kz_i)
                        grid_val[ix, iy, iz] = vals_b[i]
            
                    # --- Construct interpolator and evaluate on fine grid ---
                    interp_reg = scipy.interpolate.RegularGridInterpolator( 
                                                (kx, ky, kz), grid_val,
                                                method="linear", bounds_error=False, fill_value=None, )
                    bands_interp[sp][:, bIdx] = interp_reg(np.asarray(kpts_fine, dtype=float))

        #[3] Replace NaN values using the nearest-neighbor interpolation values for the same indexes===
        bands_interp_NaNreplacedbynn = {sp: deepcopy(bands_interp[sp]) for sp in c_spin_keys}
        nan_mask = {sp: np.array([]) for sp in c_spin_keys}
        for sp in c_spin_keys:
            nan_mask[sp] = np.isnan(bands_interp_NaNreplacedbynn[sp])
            if np.any(nan_mask[sp]):
                if flag_verbose : print(f"Warning: NaN values detected for spin {sp} — Replacing NaN values with nearest-neighbor correction.")
                bands_interp_NaNreplacedbynn[sp][ nan_mask[sp] ] = bands_interp_nn[sp][ nan_mask[sp] ]
                
        #[4] We do not stop to replacement of NaN with nearest-neighbors
        # But a “delta correction” that approximated missing linear interpolation results based 
        # on how the nearest-neighbor and linear values differ at a nearby valid k-point.
        # Formula: E_NaN​(k_NaN​) ≈ E_nn​(k_NaN​) + [E_linear​(k_closest​)−E_nn​(k_closest​)]
        # That is:  Take the nearest-neighbor value at the NaN
        #           Add the offset between linear and nearest-neighbor interpolation at the closest valid non-NaN k-point.
        bands_interp_NaNreplacedbynn_DeltaCorr = {sp: deepcopy(bands_interp_NaNreplacedbynn[sp]) for sp in c_spin_keys}
        if flag_apply_delta_correction :
            for sp in c_spin_keys:
                if np.any(nan_mask[sp]):
                    nan_indices   = np.where( np.isnan(bands_interp[sp][:, 0]))[0]
                    valid_indices = np.where(~np.isnan(bands_interp[sp][:, 0]))[0]
                    for idx_NaN in nan_indices:
                        k_NaN = finegrid_KpointsData.kpts[idx_NaN]
                        distances = np.linalg.norm(finegrid_KpointsData.kpts[valid_indices] - k_NaN, axis=1)
                        idxValid_closest_toNaN = valid_indices[np.argmin(distances)]
                        bands_interp_NaNreplacedbynn_DeltaCorr[sp][idx_NaN, :] = (
                            bands_interp_nn[sp][idx_NaN, :]
                            + ( bands_interp[sp][idxValid_closest_toNaN, :] - bands_interp_nn[sp][idxValid_closest_toNaN, :] ) )        
        flag_hasNaN = any(np.any(nan_mask[sp]) for sp in c_spin_keys)
        if flag_verbose: print("The original interpolation has NaN?:"+str(flag_hasNaN))
        
        return bands_interp , bands_interp_NaNreplacedbynn , bands_interp_NaNreplacedbynn_DeltaCorr , flag_hasNaN

    @staticmethod
    def interpolate_BandsState_fromCoarseToFineKmesh( coarsegrid_BS_QP: BandsState , finegrid_BS_toReceiveQP: BandsState ,
                                                 flag_interp_method: str = 'linear'   , 
                                                 flag_apply_delta_correction = True   ) ->  tuple[BandsState, BandsState]:
        """
        Wrapper that calls the numerical interpolation kernel `_interpolate_bands_fromCoarseToFineEdgedKmesh`,
        It receives two BandStates objects -
        (1) coarsegrid_BS_QP : coarsegrid_BS_QP.eigenval will be iterpolated BY CALLING _interpolate_bands_fromCoarseToFineEdgedKmesh
                               on finegrid_BS_toReceiveQP.kpoints 
        (2) finegrid_BS_toReceiveQP : it uses only finegrid_BS_toReceiveQP.kpoints as the finegrid to interpolate on
                                      does not apply (or touch at all) finegrid_BS_toReceiveQP.occupations
        and it constructs and returns a single BandsState object:
        (1) finegrid_BS_interpolatedQP : interpolated QP corrections
                                         IMPORTANT: its occupation are none! QP corrections does not need occupations
         """
        bands_interp , bands_interp_NaNreplacedbynn , bands_interp_NaNreplacedbynn_DeltaCorr , flag_hasNaN = BandsState_InterpOp._interpolate_bands_fromCoarseToFineEdgedKmesh( 
                    coarsegrid_BS_QP, finegrid_BS_toReceiveQP.kpoints, flag_interp_method, flag_apply_delta_correction)         
        comment_text = "BandsState obtained from interpolating coarsegrid_BS to the kpts-mesh="+str(finegrid_BS_toReceiveQP.kpoints.mesh) +\
                       " where coarsegrid_BS data obtained from path="+str(coarsegrid_BS_QP.history.path_loaded) +\
                       " - The interpolation has found and corrected NaN:"+str(flag_hasNaN)
        finegrid_BS_interpolatedQP = BandsState(
                                history = InstanceHistory( timestamp_loaded=datetime.now(), 
                                                           path_loaded=coarsegrid_BS_QP.history.path_loaded, 
                                                           comment=comment_text )   ,
                                structure  = finegrid_BS_toReceiveQP.structure      , 
                                kpoints    = finegrid_BS_toReceiveQP.kpoints        ,
                                spin_keys  = finegrid_BS_toReceiveQP.spin_keys      ,
                                occupation = None,  # <<< IMPORTANT: QP corrections don't need occupations
                                eigenval   = bands_interp_NaNreplacedbynn_DeltaCorr )       
        return finegrid_BS_interpolatedQP

    @staticmethod
    def run_interpolate_and_writeWAVECAR( obj_sparse_GW: Path | BandsState, 
                                          obj_dense_fromWAVECAR_toReceiveInterp: Path | BandsState,
                                          obj_dense_DFT_ref: Path | BandsState | None = None,
                                          nbandsgw_dense: int = -1            ,
                                          flag_interp_method: str = "linear"  ,
                                          flag_applyQP_toWAVECAR: bool = True , 
                                          flag_apply_delta_correction : bool = True,                                          
                                          flag_verbose: bool = True,  ) -> BandsState:
        """
        Interpolate sparse GW quasiparticle corrections onto a dense DFT k-mesh,
        and optionally apply them to a target WAVECAR (overwriting the WAVECAR energies in place). 
        Note that the wavefunction coefficients are untouched , which is consistent with a G0W0 type calc.
        when only the energies are updated and the orbitals are left fixed at starting point level).
    
        Key variables:
        - BS_sparse_QPc : BandsState
            BandState containing the QPcorrections ON A THE SPARSE K-POINTS
            IT WHICH WILL BE INTERPOLATED AND LATER APPLIED
            It is constructed from path_sparse_GW  (Path to G0W0 OUTCAR file with G0W0) 
        
        - BS_dense_WAVECAR_toReceiveInterp : BandsState
            The **dense-grid BandsState** associated with the target WAVECAR file.
            
            Conceptually, this object must originate from a WAVECAR file — since, when
            `flag_applyQP_toWAVECAR=True`, the routine will overwrite the eigenvalue
            records of that WAVECAR in place with the corrected quasiparticle energies.
            The underlying wavefunction coefficients are left untouched.

            A full BandsState object is required here (rather than a simple file path)
            to allow internal consistency checks: specifically, the routine verifies that
            the k-point grid and DFT eigenvalues of this object are compatible with those
            of `BS_dense_DFT_ref` if the latter is provided.  This ensures that the GW
            corrections are applied on an exactly matching fine-mesh reference.

        - BS_dense_DFT_ref : BandsState
            The **reference dense-grid BandsState** providing the DFT eigenvalues that
            serve as the baseline for the quasiparticle correction
            
            It's an optional argument; he behavior depends on whether `BS_dense_DFT_ref` 
            is explicitly provided:
            if BS_dense_DFT_ref is supplied:
                - A control si done between the Kmesh grid and DFT energies of of BS_dense_DFT_ref 
                  and BS_dense_WAVECAR_toReceiveInterp - which should match
                - The DFT.energies ON THE FINE GRID will be taken from BS_dense_DFT_ref
                - The QPCorrection will be taken from BS_sparse_QPc and interpolated
                - The resulting (DFT.energies+QPcorrections) will be saved into 
                  another object BS_dense_wAppliedInterpQP
                  The BS_dense_wAppliedInterpQP object will be associated to the SAME
                   WAVECAR associated to BS_dense_WAVECAR_toReceiveInterp
            if BS_dense_DFT_ref is None:
                - The DFT.energies ON THE FINE GRID will be taken from BS_dense_WAVECAR_toReceiveInterp
                  Thus a single source.
                - The QPCorrection will be taken from BS_sparse_QPc and interpolated
                - The resulting (DFT.energies+QPcorrections) will be saved into the WAVECAR
                  associated to BS_dense_WAVECAR_toReceiveInterp
                
        Parameters
        obj_sparse_GW : Path/BandsState            BandsState OR Path to OUTCAR file with G0W0 data
                                                   -> ON A THE SPARSE K-POINTS.
        obj_dense_DFT_ref : Path/BandsState        BandsState OR Path to vasprun.xml containing the 
                                                   DFT reference bands -> on THE DENSE K-POINTS
        obj_dense_fromWAVECAR_toReceiveInterp :    BandsState OR Path to Path to (DFT) WAVECAR on
                                                   -> ONTHE DENSE K-POINTS to be corrected in place.
        nbandsgw_dense : int        Number of bands to include in the interpolation and correction.
                                    If <= 0, automatically uses the smallest band count among all inputs.
        flag_interp_method : str    Interpolation method passed to scipy.interpolate.griddata ('linear' or 'cubic').
        flag_verbose : bool         Print diagnostic output.
        """
    
        ##-----------------------------------------------------------------------------------------------------------
        ##[PRELIMINARY-1] Parse all three inputs ------------------------------
        # If a BandsState is passed directly use it; if a Path object is passed,
        # construct an BandState object frim it 
        if flag_verbose: print("\n{<Step 1: Parsing input files>}")
        # [1.1] Sparse GW object
        if isinstance(obj_sparse_GW, BandsState):
            BS_sparse_QPc = obj_sparse_GW
        elif isinstance(obj_sparse_GW, (str, Path)):
            path_poscar_sparse = Path(obj_sparse_GW).parent / "POSCAR"
            BS_sparse_QPc = BandsState_IO.parse_outcar_spinUnpol( obj_sparse_GW, 
                                                                  poscar_path=path_poscar_sparse , 
                                                                  setGWDataAsprimary="QPC"       )
        else: raise TypeError("obj_sparse_GW must be a BandsState or a Path to an OUTCAR file.")
        #[1.2] Dense WAVECAR object
        if isinstance(obj_dense_fromWAVECAR_toReceiveInterp, BandsState):
            BS_dense_WAVECAR_toReceiveInterp = obj_dense_fromWAVECAR_toReceiveInterp
        elif isinstance(obj_dense_fromWAVECAR_toReceiveInterp, (str, Path)):
            path_poscar_dense = Path(obj_dense_fromWAVECAR_toReceiveInterp).parent / "POSCAR"
            BS_dense_WAVECAR_toReceiveInterp = BandsState_IO.parse_bands_from_WAVECAR( obj_dense_fromWAVECAR_toReceiveInterp , 
                                                                                       poscar_path=path_poscar_dense         )
        else: raise TypeError("obj_dense_WAVECAR_toReceiveInterp must be a BandsState or Path to a WAVECAR.")
        #[1.3] Dense DFT reference object (optional)
        if obj_dense_DFT_ref is None:
            BS_dense_DFT_ref = BS_dense_WAVECAR_toReceiveInterp
            if flag_verbose: print("No DFT reference provided; using dense WAVECAR as reference.")
        elif isinstance(obj_dense_DFT_ref, BandsState):
            BS_dense_DFT_ref = obj_dense_DFT_ref
        elif isinstance(obj_dense_DFT_ref, (str, Path)):
            BS_dense_DFT_ref = BandsState_IO.parse_bands_from_vasprun(obj_dense_DFT_ref)
        else: raise TypeError("obj_dense_DFT_ref must be a BandsState, Path, or None.")


        ##[PRELIMINARY-2] Determine constants and number of bands -------------------
        # Determine how many bands will be included in the interpolation and comparison.
        # The number is limited to the smallest band count among the sparse GW data,
        # dense DFT reference, and dense WAVECAR (unless manually specified).
        c_NBANDS_dense_WAVECAR = BS_dense_WAVECAR_toReceiveInterp.eigenval[Spin.up].shape[1]
        c_NBANDS_sparse_GW     = BS_sparse_QPc.eigenval[Spin.up].shape[1]
        c_NBANDS_dense_DFT_ref = ( BS_dense_DFT_ref.eigenval[Spin.up].shape[1] if BS_dense_DFT_ref is not None  
                                   else c_NBANDS_dense_WAVECAR  )
        if nbandsgw_dense <= 0: NBANDSGW_DENSE = min(c_NBANDS_dense_DFT_ref, c_NBANDS_dense_WAVECAR , c_NBANDS_sparse_GW)
        else:                   NBANDSGW_DENSE = min(nbandsgw_dense, c_NBANDS_dense_DFT_ref, c_NBANDS_dense_WAVECAR , c_NBANDS_sparse_GW)

        ##[PRELIMINARY-3] Check if BS_dense_WAVECAR_toReceiveInterp actually --
        #                 from a WAYECAR --------------------------------------
        if BS_dense_WAVECAR_toReceiveInterp.history.path_loaded is not None:
            path_check = Path(BS_dense_WAVECAR_toReceiveInterp.history.path_loaded)
            if "WAVECAR" not in path_check.name:
                warnings.warn(f"Target BandsState does not appear to originate from a WAVECAR: {path_check.name}")
          
        
        ##[PRELIMINARY-4] Check consistency of dense grids and energies -------
        # 4.1] Ensure that the k-point meshes of the dense reference and the dense WAVECAR
        # correspond exactly. If they differ, the interpolation and application steps
        # may produce inconsistent results.
        if BS_dense_DFT_ref is not None:
           if not np.allclose( BS_dense_WAVECAR_toReceiveInterp.kpoints.kpts ,
                               BS_dense_DFT_ref.kpoints.kpts, atol=1e-6,     ):
               warnings.warn( "The k-point grids of BS_dense_WAVECAR_toReceiveInterp and "
                               "BS_dense_DFT_ref differ. Interpolation results may be inconsistent."  )
        # 4.2] Check DFT eigenvalue similarity within the selected band window
           try:
                eig_dft_ref = BS_dense_DFT_ref.eigenval[Spin.up][:, :NBANDSGW_DENSE]
                eig_wavecar = BS_dense_WAVECAR_toReceiveInterp.eigenval[Spin.up][:, :NBANDSGW_DENSE]
                if eig_dft_ref.shape == eig_wavecar.shape: diff_matrix = np.abs(eig_dft_ref - eig_wavecar)
                if np.any(diff_matrix > 1e-3): warnings.warn("Detected difference > 1e-3 eV between the dense grid eigenvalues provided!")
           except Exception as err:  warnings.warn( f"Could not verify DFT eigenvalue consistency between reference and WAVECAR: {err}" )
    
        ##[PRELIMINARY-5] Brief intermezzo: the prints
        if flag_verbose: print( "\n{{<Used arguments>}}\n"
                                " sparse_GW:" +str(Path(BS_sparse_QPc.history.path_loaded).absolute())+"\n"
                                " dense_WAVECAR:"+str(Path(BS_dense_WAVECAR_toReceiveInterp.history.path_loaded).absolute())+"\n"
                                " dense_DFT_ref:"+str(Path(BS_dense_DFT_ref.history.path_loaded).absolute())+"\n"
                                " nbandsgw_dense (requested): "+str(nbandsgw_dense)+"\n"
                                " nbandsgw_dense (used): " +str(NBANDSGW_DENSE)+"\n"
                                "{{<End arguments>}}\n" )
 
    
        ##-----------------------------------------------------------------------------------------------------------
        #[1] Interpolate GW corrections from sparse to dense k-mesh -----------
        if flag_verbose:
            print("\n{<Step 2: Interpolating GW corrections>}")
        BS_dense_interpolatedQP = BandsState_InterpOp.interpolate_BandsState_fromCoarseToFineKmesh(
                                                            coarsegrid_BS_QP = BS_sparse_QPc,
                                                            finegrid_BS_toReceiveQP = BS_dense_DFT_ref ,
                                                            flag_interp_method = flag_interp_method    ,
                                                            flag_apply_delta_correction = flag_apply_delta_correction )
              
        
        
        #[2] Make interpolatedQP internally consistent, then resize to WAVECAR NBANDS
        if flag_verbose:
            print("\n{<Step 2b: Resizing interpolated QP corrections to match WAVECAR NBANDS>}")

        nb_interp = BS_dense_interpolatedQP.eigenval[Spin.up].shape[1]
        nb_target = BS_dense_WAVECAR_toReceiveInterp.eigenval[Spin.up].shape[1]

        #2.2] Resize/pad QP corrections up (or down) to match the WAVECAR band count
        BS_dense_interpolatedQP = BandsState_InterpOp.resize_nbandsState_toTargetBand(
                                        state=BS_dense_interpolatedQP,
                                        nbands_target=nb_target      ,
                                        fill_mode="repeat_last"      ,
                                        update_history=True          )
            
        #[2] Apply interpolated corrections to target WAVECAR bands -----------
        if flag_verbose: print("\n{<Step 3: Applying interpolated GW corrections to WAVECAR>}")
        corrected_BS_WAVECAR = BandsState_InterpOp.apply_QP_correction(
                                                dft_BS  =  BS_dense_WAVECAR_toReceiveInterp,
                                                interpolatedQP_BS  =  BS_dense_interpolatedQP,
                                                flag_save_original_State_in_misc=True,   )
    
    
        #[4] Write modified WAVECAR -------------------------------------------
        if flag_applyQP_toWAVECAR:
            if flag_verbose: print("\n{<Step 4: Writing modified WAVECAR>}")
            if BS_dense_WAVECAR_toReceiveInterp.history.path_loaded is not None:
                path_wavecar = BS_dense_WAVECAR_toReceiveInterp.history.path_loaded
                BandsState_IO.write_bands_to_WAVECAR(corrected_BS_WAVECAR, path_wavecar)
            else: warnings.warn("Cannot write corrected WAVECAR: no valid path found in history.")


        ##-----------------------------------------------------------------------------------------------------------
        #[DEBUG-1] Diagnostic print of Γ-point eigenvalues --------------------
        if flag_verbose:
            print("\n{<Debug: Γ-point eigenvalues summary>}")
            np.set_printoptions(linewidth=np.inf, precision=4, suppress=True)
            idx_gamma = 5
            if BS_dense_DFT_ref is not None :
                print("> [test-1 : should be equal ]")
                print("> [fine mesh] DFT Reference eigenvalues at Γ:")
                print("  ", BS_dense_DFT_ref.eigenval[Spin.up][idx_gamma, :NBANDSGW_DENSE])
            print("> [fine mesh] DFT WAVECAR eigenvalues at Γ:")
            print("  ", BS_dense_WAVECAR_toReceiveInterp.eigenval[Spin.up][idx_gamma, :NBANDSGW_DENSE])
            print("\n> [test-2 : should be equal ]")
            print("> [fine mesh] Interpolated GW QP corrections at Γ:")
            print("  ", BS_dense_interpolatedQP.eigenval[Spin.up][idx_gamma, :NBANDSGW_DENSE])
            print("> [sparse mesh] QPc correction at Γ:")
            print("  ", BS_sparse_QPc.eigenval[Spin.up][idx_gamma, :NBANDSGW_DENSE])

            print("\n> [test-3 : should be equal ]")
            print("> [fine mesh] Corrected DFT+QP eigenvalues at Γ:")
            print("  ", corrected_BS_WAVECAR.eigenval[Spin.up][idx_gamma, :NBANDSGW_DENSE])
            print("> [sparse mesh] G0W0 eigenvalues at Γ:")
            print("  ", BS_sparse_QPc.misc['GW'][Spin.up][idx_gamma, :NBANDSGW_DENSE])


            print("{<End Debug>}\n")
        if flag_verbose: print("{<Done>} <----------------------------------------------->\n")
        return corrected_BS_WAVECAR


    @staticmethod
    def resize_nbandsState_toTargetBand(  state: BandsState, nbands_target: int,
                                          fill_mode: str = "repeat_last",   # currently only this mode is implemented
                                          update_history: bool = True,  ) -> BandsState:
        """ Resize the band-dimension (NBANDS) of a BandsState by truncating or padding
        `eigenval` and `occupation` arrays, returning a NEW immutable BandsState.
    
        Why this exists (practical context)
        In many VASP workflows, the target object you want to modify/compare against
        (e.g. a dense-grid WAVECAR or a dense-grid DFT run) often contains a larger
        number of bands than a reference dataset from which you derived corrections
        (e.g. a sparse G0W0 OUTCAR that printed only the lowest ~N band. Or viceversa.
         
         Behavior
         Case A) nbands_target < nbands_current  (TRUNCATION)
                 Keep only the first nbands_target bands:  E_new[:, :nbands_target] = E_old[:, :nbands_target]
                 Same truncation for occupations.
     
         Case B) nbands_target > nbands_current  (PADDING)
                 We must invent values for bands nbands_current+1 ... nbands_target.
                 In this implementation, fill_mode="repeat_last" means:    E_new[:, nb:] = E_old[:, [nb-1]]
                 i.e. repeat the last available band value for every added band (per k-point).
                 Occupations are padded the same way, if occupation exists.         
        """
     
 
        #2] Initializing the variables
        old_spin_keys = state.spin_keys
        old_eig = state.eigenval
        old_occ = state.occupation if (state.occupation is not None) else None
        new_eig = {}    
        new_occ = {}
        
        #1] Sanity checks on input variables
        if state.eigenval is None:      raise ValueError("state.eigenval is None")
        if nbands_target <= 0:          raise ValueError("nbands_target must be > 0")
        if fill_mode != "repeat_last":  raise ValueError(f"Unsupported fill_mode={fill_mode!r}. Use 'repeat_last'.")
        nk_eig_spUp, nb_eig_spUp = old_eig[old_spin_keys[0]].shape
        if nb_eig_spUp == nbands_target:     return state
        
        
        
        for sp in old_spin_keys :
            #3] Correcting eigenvalues
            old_nk, old_nb = old_eig[sp].shape
            
            if old_nk != nk_eig_spUp or old_nb != nb_eig_spUp:
                raise ValueError(f"Inconsistent eigenval shape for {sp}: {old_eig[sp].shape} vs {(nk_eig_spUp, nb_eig_spUp)}")
       
            if nbands_target < old_nb:  
                new_eig[sp] = deepcopy(  old_eig[sp][:, :nbands_target] )
            else:
                new_eig_sp = np.empty( (old_nk, nbands_target), dtype= old_eig[sp].dtype )
                new_eig_sp[:, :old_nb] =  old_eig[sp]
                new_eig_sp[:, old_nb:] =  old_eig[sp][:, [old_nb - 1]]  # broadcast repeat_last
                new_eig[sp] = new_eig_sp
            
        #4] Correcting occupations
        if old_occ is None:
            new_occ = None
        else:
            for sp in old_spin_keys :
                old_occ_nk, old_occ_nb = old_occ[sp].shape
                
                if old_occ_nk != nk_eig_spUp or old_occ_nb != nb_eig_spUp:
                    raise ValueError(f"Inconsistent occupation shape for {sp}: {old_eig[sp].shape} vs {(nk_eig_spUp, nb_eig_spUp)}")

                if nbands_target < old_occ_nb:  
                    new_occ[sp] = deepcopy(  old_occ[sp][:, :nbands_target] )
                else:
                    new_occ_sp = np.empty((old_occ_nk, nbands_target), dtype= old_occ[sp].dtype)
                    new_occ_sp[:, :old_occ_nb] =  old_occ[sp]
                    new_occ_sp[:, old_occ_nb:] =  old_occ[sp][:, [old_occ_nb - 1]]  # broadcast repeat_last
                    new_occ[sp] = new_occ_sp
        
        
        if update_history:
            comment = (state.history.comment or "")
            comment2 = f"{comment} | resize_nbands: {nb_eig_spUp} -> {nbands_target}"
            new_history = InstanceHistory(
                timestamp_loaded=datetime.now(),
                path_loaded=state.history.path_loaded,
                comment=comment2,        )
        else:
            new_history = state.history

        return BandsState(  history=new_history,
                            kpoints=state.kpoints,
                            structure=state.structure,
                            spin_keys=state.spin_keys,
                            eigenval=new_eig,
                            occupation=new_occ,
                            misc=deepcopy(state.misc),   )






# import shutil
# os.makedirs("./ForWAVECARModified" , exist_ok=True)
# shutil.copyfile("./WAVECAR", "./ForWAVECARModified/WAVECAR")
#
# input={}
# input["path_sparse_GW"]           = "../ReferenceData/2.2-GW"
# input["path_dense_DFT_reference"] = "."
# input["path_dense_DFT_toReceiveInterp"]  = "./ForWAVECARModified/"
# input["nbandsgw_dense"] = -1
#
# path_sparse_GW           = os.path.join(input["path_sparse_GW"]  , 'OUTCAR.3')
# path_dense_DFT_reference = os.path.join(input["path_dense_DFT_reference"]       ,'vasprun.xml')
# path_dense_DFT_toInterp  = os.path.join(input["path_dense_DFT_toReceiveInterp"] ,'WAVECAR'    )
#                                     
# BandsState_DENSE_WAVECAR_interpolated = BandsState_InterpOp.run_interpolate_and_writeWAVECAR( 
#                                                                            obj_sparse_GW     = path_sparse_GW , 
#                                                                            obj_dense_DFT_ref = path_dense_DFT_reference ,
#                                                                            obj_dense_fromWAVECAR_toReceiveInterp = path_dense_DFT_toInterp   , 
#                                                                            flag_interp_method = "rbf:gaussian"  ,
#                                                                            flag_applyQP_toWAVECAR = False     ,
#                                                                            flag_apply_delta_correction = True )                        
                                       
# FLAG_DEBUG = False                                 
# if FLAG_DEBUG :
#     BandsState_sparse_GW = BandsState_IO.parse_outcar_spinUnpol(path_sparse_GW , os.path.join(input["path_sparse_GW"]  , 'POSCAR'))
#     BandsState_DENSE_WAVECAR_toReceiveInterp = BandsState_IO.parse_bands_from_WAVECAR( path_dense_DFT_toInterp )
#     BandsState_DENSE_WAVECAR_toReceiveInterp = BandsState_IO.parse_bands_from_vasprun( path_dense_DFT_reference )
    
#     bObj_SPARSE_OUTCAR = bandsObj_() ;                  
#     bObj_DENSE_VASPRUN_reference = bandsObj_() ;        
#     bObj_DENSE_WAVECAR_toReceiveInterp = bandsObj_()  ; 
    
#     #1st Test BandsState_IO : is parse_outcar_spinUnpol correct?
#     bObj_SPARSE_OUTCAR.read_file_OUTCAR_spinUnpol( path_sparse_GW , setGWDataAsprimary='QPC' )
#     assert np.all( BandsState_sparse_GW.eigenval[Spin.up] == bObj_SPARSE_OUTCAR.eigenval_GW[Spin.up][:,:,0]  )
#     assert np.all( BandsState_sparse_GW.kpoints.kpts      == bObj_SPARSE_OUTCAR.kpts_list)
    
#     #2nd Test BandsState_IO : is parse_bands_from_WAVECAR correct?
#     bObj_DENSE_WAVECAR_toReceiveInterp.read_file_WAVECAR(  path_dense_DFT_toInterp  )
#     assert np.all( BandsState_DENSE_WAVECAR_toReceiveInterp.eigenval[Spin.up] == bObj_DENSE_WAVECAR_toReceiveInterp.eigenval[Spin.up][:,:,0]  )
#     assert np.all( BandsState_DENSE_WAVECAR_toReceiveInterp.kpoints.kpts      == bObj_DENSE_WAVECAR_toReceiveInterp.kpts_list)
    
#     # # #3nd Test BandsState_IO : is parse_bands_from_vasprun correct?
#     bObj_DENSE_VASPRUN_reference.read_file_VASPRUN( path_dense_DFT_reference )

if __name__ == "__main__":
    ## [Interface]
    parser = argparse.ArgumentParser()
    parser.add_argument("-ps"  , "--path_sparse_GW"           , type=str , required=True  , 
                        help="Path to the sparse (aka on the coarse k-mesh) G0W0 calculation folder."   )
    parser.add_argument("-sf"  , "--sparse_GW_filename"       , type=str , required=False , default="OUTCAR.3",
                        help="GW output filename (INSIDE the sparse GW folder) used to parse the QPcorrection that will be interpolated.")

    parser.add_argument("-pdi" , "--path_dense_DFT_toInterp"  , type=str , required=True  , default="./",
                        help="Path to the dense DFT calculation whose WAVECAR will receive the interpolation.") 
    parser.add_argument("-pdr" , "--path_dense_DFT_reference" , type=str , required=False , default=None,
                        help="Optional reference DFT folder used to determine the DFT a (vasprun.xml will be used if present).")
    parser.add_argument("-ngw" , "--nbandsgw_dense"           , type=int , required=False , default=-1)

    #Parsing arguments
    input = {} ; args = parser.parse_args()
    if args.path_sparse_GW is not None:           input["path_sparse_GW"] = args.path_sparse_GW
    if args.sparse_GW_filename is not None:       input["sparse_GW_filename"] = args.sparse_GW_filename
    if args.path_dense_DFT_toInterp is not None:  input["path_dense_DFT_toInterp"]  = args.path_dense_DFT_toInterp
    if args.path_dense_DFT_reference is not None: input["path_dense_DFT_reference"] = args.path_dense_DFT_reference
    if args.nbandsgw_dense is not None:           input["nbandsgw_dense"] = args.nbandsgw_dense
    else: input["nbandsgw_dense"] = -1
    
    #path_sparse_GW           = os.path.join(input["path_sparse_GW"]  , 'OUTCAR.3')
    input["path_sparse_GW_fullPath"]           = os.path.join(input["path_sparse_GW"], input["sparse_GW_filename"])
    input["path_dense_DFT_toInterp_fullPath"]  = os.path.join(input["path_dense_DFT_toInterp"]   ,'WAVECAR')
    if "path_dense_DFT_reference" in input:
        input["path_dense_DFT_reference_fullPath"] =  os.path.join(input["path_dense_DFT_reference"], "vasprun.xml")
    else:
        input["path_dense_DFT_reference_fullPath"] = None  
        
    print("=== [Interpolation Parameters] ===")
    print(f"path_sparse_GW           : {input['path_sparse_GW_fullPath']}")
    print(f"path_dense_DFT_toInterp  : {input['path_dense_DFT_toInterp_fullPath']}")
    print(f"path_dense_DFT_reference : {input['path_dense_DFT_reference_fullPath']}")
    print(f"nbandsgw_dense           : {input['nbandsgw_dense']}")
    print("==================================\n")    
        
        
        
    BandsState_DENSE_WAVECAR_interpolated = BandsState_InterpOp.run_interpolate_and_writeWAVECAR( 
                                                obj_sparse_GW     = input['path_sparse_GW_fullPath'] , 
                                                obj_dense_fromWAVECAR_toReceiveInterp = input["path_dense_DFT_toInterp_fullPath"]   , 
    #                                           #obj_dense_DFT_ref = input["path_dense_DFT_reference_fullPath"] ,
                                                flag_interp_method = "rbf:gaussian"  ,
                                                flag_applyQP_toWAVECAR = True        ,
                                                flag_apply_delta_correction = True   )
    print("Interpolation completed successfully.")                         
    
    # print("\n\n{<QPc Interpolated values>} <-----------..--------->")
    # # Print header (column indices starting from 1) + # Print each row with 4 decimals
    # arr = bands_DENSE_GWinterp[Spin.up][:,:NBANDSGW_DENSE]
    # header = " kpt" + " ".join([f"{i:10d}" for i in range(1, arr.shape[1] + 1)])
    # print(header)
    # for kpt_idx , kpt_bands in enumerate( arr ):
    #     print(f"{kpt_idx:4}"+ " : " + " ".join([f"{val:10.4f}" for val in kpt_bands]))
    # print("\n")
    # for kpt_idx , kpt in enumerate( bObj_DENSE_VASPRUN_reference.kpts_list ):
    #     print(f"{kpt_idx:4}"+ " : " + "".join(str(kpt)))
    # print("{<End QPc Interpolated values>} <------------------>\n\n")
    # bObj_DENSE_WAVECAR_toReceiveInterp.eigenval[Spin.up][:,:NBANDSGW_DENSE,0] = bObj_DENSE_WAVECAR_toReceiveInterp.eigenval[Spin.up][:,:NBANDSGW_DENSE,0] + bands_DENSE_GWinterp[Spin.up][:,:NBANDSGW_DENSE]
    # print("{<Debug: Gamma.kpt post interp.QPc & reference ----->")
    # print( bObj_DENSE_VASPRUN_reference.eigenval[Spin.up][0,:NBANDSGW_DENSE,0])
    # print( bObj_DENSE_WAVECAR_toReceiveInterp.eigenval[Spin.up][0,:NBANDSGW_DENSE,0])

    
