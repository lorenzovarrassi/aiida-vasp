# -*- coding: utf-8 -*-
from xml.etree import cElementTree as ET
import numpy as np
import os
from scipy.optimize import curve_fit
from pymatgen.io.vasp.outputs import Vasprun , BSVasprun
import warnings
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Code, Bool, Str, Int, Dict, Float , KpointsData , RemoteData , FolderData





def _apply_scissor(bandsdata: orm.BandsData, scissor_value: float) -> orm.BandsData:
    """ Apply a rigid scissor correction to conduction bands in a BandsData object.
    The correction is applied to all bands above the highest occupied state.

    Parameters
    ----------
    bandsdata : aiida.orm.BandsData
        DFT band structure.
    scissor_value : float
        Energy shift (in eV) to apply to conduction bands.

    Returns
    -------
    aiida.orm.BandsData
        New BandsData with scissor applied.
    """
    b_band = bandsdata.get_array("bands").copy()
    b_occ = bandsdata.get_array("occupations")

    for k in range(b_band.shape[0]):
        idx_ho = np.where(b_occ[k, :-1] - b_occ[k, 1:] > 0)[0][0]
        b_band[k, idx_ho + 1 :] += scissor_value

    new_bandsdata = bandsdata.clone()
    new_bandsdata.set_array("bands", b_band)
    return new_bandsdata


def _determine_BSE_parameters( bandsdata: orm.BandsData,
                              G0W0_gap: float,
                              energy_window_goal: float = 3,
                              num_bands_included: int   = 20,         
                              num_bands_used_for_transition_matrix: float = 10
                              ) -> tuple[int, int, float, float, float, str]:
    """ Determine NBANDSV, NBANDSO, and OMEGAMAX for BSE calculations
    from DFT (or scissor-corrected) bands.

    Parameters
    ----------
    bandsdata : aiida.orm.BandsData         The band structure used as reference.
    energy_window_goal : float              Target optical window (in eV).
    G0W0_gap : float                        GW gap (in eV). used to determine Scissor correction 
    num_bands_included : int                Safety upper bound for valence/conduction bands to scan.
                                              (for example, the number of bands for which QP corrections are defined)

    Returns
    -------
    NBANDSV, NBANDSO, OMEGAMAX, scissor_value, E_gap_DFT, log_str
    """
    b_band = bandsdata.get_array("bands")
    b_occ  = bandsdata.get_array("occupations")
    n_kpts, n_bands = b_band.shape
    log    = [ f"\n [1] Input BandsData (node pk={bandsdata.pk})\n"]
    log.append(f"  > n_kpts = {n_kpts} , n_bands = {n_bands}\n" )
    output = {}
    #[SAFE-CHECK] Check that occupancy values are in [0, 1]
    if np.any(b_occ < -1e-3) or np.any(b_occ > 1.001):
        warnings.warn("BIG-WARNING-WHICH-IS-BIG : Occupations outside [0,1] detected — please check smearing or parser version.")
        log.append("[BIG-WARNING] Occupations outside [0,1] detected.\n")
    #[1] Determine the highest occupied band index per k-point ---
    #This is a workaround for the case where the highest occupied band is not the same for all k-points
    #idx_HO_forDifferentKpts is the index of the highest occupied band for each k-point
    #NOTE: for aiida_vasp 3.1.0  b_occ shape is (#bands , #kpoints)
    #      for aiida_vasp 4.1.0  b_occ shape is (#kpoints , #bands). 
    idx_HO_forDifferentKpts = [np.where(b_occ[idx_k,:-1] - b_occ[idx_k,1:] > 0)[0][0] for idx_k in range(np.shape(b_occ)[0])]
    if min(idx_HO_forDifferentKpts) == max(idx_HO_forDifferentKpts) : idx_HO =  min(idx_HO_forDifferentKpts)
    else:
         warnings.warn('BIG-WARNING : Highest occupied band is not the same for all k-points. Using the first k-point index as reference.')
         log.append("[BIG-WARNING] Highest occupied band differs among k-points.\n")
         idx_HO = idx_HO_forDifferentKpts[0]
    log.append(f"  > idx_HO = {idx_HO}   (starting from 0, i.e. Python indexing)\n")
    #OLD ROUTINE
    ## Build valence/conduction band windows
    #bval_eachb_max = [ np.max(b_band[:, i]) for i in range(idx_HO - max_bands_in_matrix + 1, idx_HO + 1) ][::-1]
    #cval_eachb_min = [ np.min(b_band[:, i]) for i in range(idx_HO + 1, idx_HO + max_bands_in_matrix + 1) ]
    #bval_eachb_max_delta = bval_ho - np.array(bval_eachb_max)
    #cval_eachb_min_delta = np.array(cval_eachb_min) - bcon_lu
    
    #[2] Adjust number of valence/conduction bands included in the check
    n_valence_available    = idx_HO + 1             #+1 because 0 is indexed and should be included
    n_conduction_available = n_bands - (idx_HO + 1)
    num_bands_available = min(num_bands_included , n_bands) 
    n_valence_used      = min(n_valence_available    , num_bands_included,  )
    n_conduction_used   = min(n_conduction_available , num_bands_included,  )

    #[3] E_VBM =max( array of the values of the band w/ index idx_HO for all k-points   = ValenceBandMaximum )
    #    E_CBM =min( array of the values of the band w/ index idx_HO+1 for all k-points = CondBandMinimum)
    # i.e. the max/min are done along the k-points index for the bands number fixed.
    #Thus, E_VBM and E_CBM are the absolute band energies of the valence band maximum and conduction band minimum respectively
    E_VBM = np.max(b_band[:, idx_HO])
    E_CBM = np.min(b_band[:, idx_HO + 1])
    E_gap_DFT = E_CBM - E_VBM
    log.append( "\n [2] Determining distances from VBM and CBM in BandsData:\n")
    log.append(f"  > max num of v/o bands considered in the scan = {num_bands_included}\n")
    log.append(f"  > n_valence_used = {n_valence_used} , n_conduction_used = {n_conduction_used}\n")
    log.append(f"  > E_VBM = {E_VBM:.4f} eV , E_CBM = {E_CBM:.4f} eV  ->  gap_DFT = {E_gap_DFT:.4f} eV\n")
    
    #[4] Extract valence/conduction band edges for the chosen number of bands ---
    # Each band index corresponds to a single band across all k-points.
    # By taking the min/max of b_band[:, i], we get the global energy range
    # of that band across the entire Brillouin zone.
    # ES    min(b_band[:, i]) -> min energy (band bottom) of i-th band among all kpts
    #       max(b_band[:, i]) -> max energy (band top) of i-th band among all kpts
    # Therefore:
    #   - For valence bands (below the HO band)     we want min to see how deep each band extends below the VBM.
    #   - For conduction bands (above the LU band)  we want max to see how far each band extends above the CBM.
    # Indexes; ve reverse the valbands because we want:
    # E_valbands_max[0]→VBM  E_valbands_max[1]→one band below VBM (idx_VBM-1) ; that the VBM is included as index 0.
    # E_colbands_min[0]→CBM  E_colbands_max[1]→one band above CBM (idx_CBM+1) ; that the CBM is included as index 0.
    E_valbands_max = [np.max(b_band[:, bv_idx]) for bv_idx in range(idx_HO - n_valence_used + 1, idx_HO + 1)]
    E_valbands_max.reverse()
    E_conbands_min = [np.min(b_band[:, bc_idx]) for bc_idx in range(idx_HO + 1, idx_HO + n_conduction_used + 1)]

    #[5] Compute positive energy offsets; NOTE on index: 
    # 1st element (index 0) of DeltaEn_valb_fromVBM corresponds to the VBM itself (THUS ΔE = 0).
    #   Index of DeltaEn_valb_fromVBM is therefore [VBM, VBM-1,VBM-2, etc] 
    # 1st element (index 0) of DeltaEn_valb_fromVBM corresponds to the VBM itself (THUS ΔE = 0).
    #     Index of DeltaE_conb_fromCBM is therefore [CBM, CBM+1,CBM+2, etc] 
    # Finally, a sanity check: Both DeltaEn_valb_fromVBM_val and DeltaE_conb_fromCBM_con must be >= 0
    # NOTE: these are the MIMIMUM energy distances of each he band from VBM/CBM
    #       i.e. the minimum transition (minimum w.r. to varying kpts) for that band to the VBM/CBM
    DeltaE_valb_fromVBM = E_VBM - np.array(E_valbands_max)      # distance of each valence band top below VBM
    DeltaE_conb_fromCBM  = np.array(E_conbands_min) - E_CBM     # distance of each conduction band bottom above CBM
    
    # VALENCE (v00 = VBM, v01 = one band below, ...)    
    # CONDUCTION (c00 = CBM, c01 = one band above, ...)
    colw = 8  # column width (matches %7.4f + space)
    log.append("  > ΔE_valb_from_VBM (eV):\n")
    log.append("   " + "".join(f"{f'c{iv:02d}':>{colw}}" for iv in range(n_valence_used)) + "\n") # Header
    log.append("   " + "".join(f"{v:{colw}.4f}"          for v  in DeltaE_valb_fromVBM)   + "\n") # Values 
    log.append("  > ΔE_conb_from_CBM (eV):\n")
    log.append("   " + "".join(f"{f'c{ic:02d}':>{colw}}" for ic in range(n_conduction_used)) + "\n") # Header
    log.append("   " + "".join(f"{c:{colw}.4f}"          for c  in DeltaE_conb_fromCBM)      + "\n") # Values
    log.append("  > NOTE: each ΔE above is the SMALLEST possible distance of that band from the VBM/CBM,\n"
               "    i.e. taken at whichever k-point brings that band closest to the gap - NOT the value at a fixed k.\n"
               "    This is a conservative (best-case) bound, used so that no band with even a single low-energy\n"
               "    transition anywhere in the BZ is mistakenly excluded later. Repeated identical values across\n"
               "    neighbouring bands (e.g. two equal numbers) indicate degenerate bands.\n")
    if np.any(DeltaE_conb_fromCBM < 0) or np.any(DeltaE_valb_fromVBM < 0):
        warnings.warn("BIG-WARNING : Negative DeltaEn detected; check band ordering or occupations.")

    #[6] Compute DFT and G0W0 gaps, determine scissor shift if needed ---
    SCISSOR = 0.0
    if G0W0_gap is not None:
        SCISSOR = float(G0W0_gap) - float(E_gap_DFT)
        log.append( f"\n [3] Input G0W0 :\n")
        log.append(f"  > G0W0.gap = {G0W0_gap:.3f} eV  --[DFT gap = {E_gap_DFT:.3f} eV]--> scissor = {SCISSOR:.3f} eV")
        if SCISSOR < -1e-3: warnings.warn("BIG-WARNING : Computed scissor shift is negative")
    else:
        log.append(f"  > Using DFT gap only (no G0W0 correction). DFT gap = {E_gap_DFT:.3f} eV")

    #[7] Determine all possible transitions energies (valence × conduction)    #
    # Build a full 2D matrix of single-particle transition energies between
    # valence and conduction bands included in the analysis. 
    # Each element [iv, ic] corresponds to the excitation:
    #     (valence band idx_HO - iv)  →  (conduction band idx_HO + 1 + ic)
    # The transition energy is approximated as:     E_gap_DFT + SCISSOR + ΔE_val(iv) + ΔE_con(ic)
    # where:
    #   - E_gap_DFT  : fundamental DFT gap (E_CBM - E_VBM)
    #   - SCISSOR    : quasiparticle correction (E_G0W0 - E_DFT)
    #   - ΔE_val(iv) : energy offset of the valence band top below VBM
    #   - ΔE_con(ic) : energy offset of the conduction band bottom above CBM
    # Which is an Indipendent.Particle.Approximation transition
    # as BSE contributions lower the transition from IPA starting value, this is a safe bet.
    if energy_window_goal is not None: spectra_energy_window_aboveGap = energy_window_goal
    else: spectra_energy_window_aboveGap = 3.5
    
    transitions_matrix = np.zeros((n_valence_used, n_conduction_used))
    for iv in range(n_valence_used):
        for ic in range(n_conduction_used):
            transitions_matrix[iv, ic] = ( E_gap_DFT + SCISSOR
                                           + DeltaE_valb_fromVBM[iv]
                                           + DeltaE_conb_fromCBM[ic]  )

    # Create a boolean mask of all (lower-bound) transitions lying within the target optical window.
    # 'energyWindow_goal' is the width of the desired spectral window (e.g. 3 eV);
    # hence all transitions below this energy are considered relevant for the BSE kernel.
    # Determine how many valence and conduction bands are required to cover
    # all transitions inside the window:
    #   - if ANY transition from a given valence band is within the window,
    #     that valence band must be included → count over axis=1
    #   - same for conduction bands → count over axis=0
    cutoff = E_gap_DFT + SCISSOR + spectra_energy_window_aboveGap
    mask = transitions_matrix < cutoff
    included_valence    = np.any(mask, axis=1)   # per valence band: at least one in-window transition
    included_conduction = np.any(mask, axis=0)   # per conduction band: at least one in-window transition
    NBANDSO = int(np.sum(included_valence))       # occupied bands needed
    NBANDSV = int(np.sum(included_conduction))    # virtual/unoccupied bands needed

    # The single pair that JUSTIFIES each decision: the deepest included valence band (and the
    # conduction band that puts it in-window), and the highest included conduction band (and the
    # valence band that puts it in-window). Found directly from the mask, not assumed.
    if NBANDSO > 0:
        v_decide      = int(np.max(np.where(included_valence)[0]))
        c_decide_forV = int(np.argmin(transitions_matrix[v_decide, :]))
    else:
        v_decide, c_decide_forV = None, None
    if NBANDSV > 0:
        c_decide      = int(np.max(np.where(included_conduction)[0]))
        v_decide_forC = int(np.argmin(transitions_matrix[:, c_decide]))
    else:
        c_decide, v_decide_forC = None, None

    # Pretty-print the transition matrix. Normally capped at a 10-wide preview, but it is always
    # widened to keep the deciding row/column visible, even when NBANDSO/NBANDSV exceed 10.
    log.append("\n"+" [4] min.transition energy between each (v)alence -> (c)conduction band pair (eV)  [lower bound]"
               "\n"+f"  > lower bound = E_gap_DFT + SCISSOR + ΔE_val(iv) + ΔE_con(ic) ; the smallest energy that pair could possibly have."
               "\n"+f"  > cutoff = E_gap_DFT + SCISSOR + window = {E_gap_DFT:.3f} + {SCISSOR:.3f} + {spectra_energy_window_aboveGap:.3f} = {cutoff:.3f} eV"
               "\n"+f"  > pairs below the cutoff (marked [..]) are guaranteed to be needed to cover the requested window."
               "\n"+f"  > scanning the first {n_valence_used}/{n_conduction_used} val/cond bands (set by num_bands_included).")
    max_display_v = min(n_valence_used, max(10, NBANDSO, (v_decide_forC or 0) + 1))
    max_display_c = min(n_conduction_used, max(10, NBANDSV, (c_decide_forV or 0) + 1))
    # Conduction band header: c00, c01, ...
    header = "\n    "+"       "+ "      ".join(f"c{ic:02d}" for ic in range(max_display_c))
    log.append(header + "\n")
    # Rows: v00, v01, v02 = VBM, VBM-1, VBM-2, ...
    deciding_cells = {(v_decide, c_decide_forV), (v_decide_forC, c_decide)}
    for iv in range(max_display_v):
        cells = []
        for ic in range(max_display_c):
            val = transitions_matrix[iv, ic]
            cells.append(f"[{val:.3f}]" if (iv, ic) in deciding_cells else f"{val:7.3f}")
        log.append(f"    v{iv:02d}: {'  '.join(cells)}\n")

    output['HOMO_band_idx'] = idx_HO
    output['gap_DFT']  = E_gap_DFT
    output['SCISSOR']  = SCISSOR
    output['NBANDSO']  = NBANDSO
    output['NBANDSV']  = NBANDSV
    output['OMEGAMAX'] = cutoff
    log.append(""  + " [5] Results:"
               "\n"+f"  window above gap = {spectra_energy_window_aboveGap:.4f} eV , E_gap_DFT = {E_gap_DFT:.4f} eV , SCISSOR = {SCISSOR:.4f} eV"
               "\n"+f"  > OMEGAMAX = E_gap_DFT + SCISSOR + window = {cutoff:.2f} eV"
               "\n"+f"  > NBANDSV  = {NBANDSV} - NBANDSO = {NBANDSO}\n"     )

    # [Why these values were chosen] - trace each count back to the specific pair that set it,
    # and show the next (excluded) band for contrast, so the cutoff decision is verifiable at a glance.
    log.append("  [Why these values were chosen]\n")
    if v_decide is not None:
        log.append(f"   > NBANDSO={NBANDSO}: deepest valence band still needed is v{v_decide:02d}, via transition"
                    f" (v{v_decide:02d},c{c_decide_forV:02d}) = {transitions_matrix[v_decide, c_decide_forV]:.3f} eV"
                    f" < cutoff {cutoff:.3f} eV.\n")
        if v_decide + 1 < n_valence_used:
            next_c = int(np.argmin(transitions_matrix[v_decide + 1, :]))
            log.append(f"                next valence band v{v_decide+1:02d}'s closest transition is"
                        f" (v{v_decide+1:02d},c{next_c:02d}) = {transitions_matrix[v_decide+1, next_c]:.3f} eV"
                        f" >= cutoff -> excluded.\n")
        else:
            log.append(f"        all {n_valence_used} scanned valence bands are needed; raise num_bands_included to check if more would be required.\n")
    else:
        log.append("   > NBANDSO=0: no valence band has a transition under the cutoff.\n")

    if c_decide is not None:
        log.append(f"   > NBANDSV={NBANDSV}: highest conduction band still needed is c{c_decide:02d}, via transition"
                    f" (v{v_decide_forC:02d},c{c_decide:02d}) = {transitions_matrix[v_decide_forC, c_decide]:.3f} eV"
                    f" < cutoff {cutoff:.3f} eV.\n")
        if c_decide + 1 < n_conduction_used:
            next_v = int(np.argmin(transitions_matrix[:, c_decide + 1]))
            log.append(f"                next conduction band c{c_decide+1:02d}'s closest transition is"
                        f" (v{next_v:02d},c{c_decide+1:02d}) = {transitions_matrix[next_v, c_decide+1]:.3f} eV"
                        f" >= cutoff -> excluded.\n")
        else:
            log.append(f"        all {n_conduction_used} scanned conduction bands are needed; raise num_bands_included to check if more would be required.\n")
    else:
        log.append("   > NBANDSV=0: no conduction band has a transition under the cutoff.\n")

    #Construct the final log string and add an indentation to it
    output['log'] = "".join(log)
    output['log'] =  '   ' +  output['log'].replace('\n', '\n   ')
    return output

    
def _extract_opticalgap_fromWorkchainNode(mBSE_node):
    mBSE_node_energy = mBSE_node.outputs.opticaltransitions.get_array('energy')
    mBSE_node_oscstr = mBSE_node.outputs.opticaltransitions.get_array('osc_strength')
    idxs_oscstr_nonzero = np.where( mBSE_node_oscstr > 0 )[0]
    return ( mBSE_node_energy[ idxs_oscstr_nonzero[0] ] , mBSE_node_oscstr[ idxs_oscstr_nonzero[0] ] )





