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





def apply_scissor(bandsdata: orm.BandsData, scissor_value: float) -> orm.BandsData:
    """
    Apply a rigid scissor correction to conduction bands in a BandsData object.
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


def determine_BSE_parameters( bandsdata: orm.BandsData,
                              G0W0_gap: float,
                              energy_window_goal: float = 3,
                              num_bands_included: int   = 20,                                   
                              ) -> tuple[int, int, float, float, float, str]:
    """
    Determine NBANDSV, NBANDSO, and OMEGAMAX for BSE calculations
    from DFT (or scissor-corrected) bands.

    Parameters
    ----------
    bandsdata : aiida.orm.BandsData         The band structure used as reference.
    energy_window_goal : float              Target optical window (in eV).
    G0W0_gap : float                        GW gap (in eV). used to determine Scissor correction 
    max_bands_in_matrix : int               Safety upper bound for valence/conduction bands to scan.

    Returns
    -------
    NBANDSV, NBANDSO, OMEGAMAX, scissor_value, E_gap_DFT, log_str
    """
    b_band = bandsdata.get_array("bands")
    b_occ  = bandsdata.get_array("occupations")
    n_kpts, n_bands = b_band.shape
    log    = [ f"[info] n_kpts = {n_kpts} , n_bands = {n_bands}\n" ]
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
    log.append(f"[info] idx_HO = {idx_HO}\n")
    #OLD ROUTINE
    ## Build valence/conduction band windows
    #bval_eachb_max = [ np.max(b_band[:, i]) for i in range(idx_HO - max_bands_in_matrix + 1, idx_HO + 1) ][::-1]
    #cval_eachb_min = [ np.min(b_band[:, i]) for i in range(idx_HO + 1, idx_HO + max_bands_in_matrix + 1) ]
    #bval_eachb_max_delta = bval_ho - np.array(bval_eachb_max)
    #cval_eachb_min_delta = np.array(cval_eachb_min) - bcon_lu
    
    #[2] Adjust number of valence/conduction bands included in the check
    n_valence_available    = idx_HO + 1
    n_conduction_available = n_bands - (idx_HO + 1)
    n_valence_used    = min(num_bands_included, n_valence_available )
    n_conduction_used = min(num_bands_included, n_conduction_available )

    #[3] E_VBM =max( array of the values of the band w/ index idx_HO for all k-points )
    #    E_CBM =min( array of the values of the band w/ index idx_HO+1 for all k-points )
    # i.e. the max/min are done along the k-points index for the bands number fixed.
    #Thus, E_VBM and E_CBM are the absolute band energies of the valence band maximum and conduction band minimum respectively
    E_VBM = np.max(b_band[:, idx_HO])
    E_CBM = np.min(b_band[:, idx_HO + 1])
    E_gap_DFT = E_CBM - E_VBM
    log.append(f"[gap] E_VBM = {E_VBM:.4f} eV , E_CBM = {E_CBM:.4f} eV , DFT gap = {E_gap_DFT:.4f} eV\n")
    
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
    DeltaE_valb_fromVBM = E_VBM - np.array(E_valbands_max)    # distance of each valence band top below VBM
    DeltaE_conb_fromCBM  = np.array(E_conbands_min) - E_CBM     # distance of each conduction band bottom above CBM
    # VALENCE (v00 = VBM, v01 = one band below, ...)
    log.append("\n[ΔE_valb_from_VBM] (eV):\n")
    log.append("  ".join(f"v{iv:02d}" for iv in range(n_valence_used)) + "\n")
    log.append("  ".join(f"{v:7.4f}"  for v in DeltaE_valb_fromVBM) + "\n")

    # CONDUCTION (c00 = CBM, c01 = one band above, ...)
    log.append("[ΔE_conb_from_CBM] (eV):\n")
    log.append(" "+"        ".join(f"c{ic:02d}" for ic in range(n_conduction_used)) + "\n")
    log.append(" "+"        ".join(f"{c:7.4f}"  for c in DeltaE_conb_fromCBM) + "\n")
    if np.any(DeltaE_conb_fromCBM < 0) or np.any(DeltaE_valb_fromVBM < 0):
        warnings.warn("BIG-WARNING : Negative DeltaEn detected; check band ordering or occupations.")

    #[6] Compute DFT and G0W0 gaps, determine scissor shift if needed ---
    SCISSOR = 0.0
    if G0W0_gap is not None:
        SCISSOR = float(G0W0_gap) - float(E_gap_DFT)
        log.append(f"DFT gap = {E_gap_DFT:.3f} eV, G0W0 gap = {G0W0_gap:.3f} eV, scissor = {SCISSOR:.3f} eV")
        if SCISSOR < -1e-3: warnings.warn("BIG-WARNING : Computed scissor shift is negative")
    else:
        log.append(f"Using DFT gap only (no G0W0 correction). DFT gap = {E_gap_DFT:.3f} eV")

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
    transitions_matrix = np.zeros((n_valence_used, n_conduction_used))
    for iv in range(n_valence_used):
        for ic in range(n_conduction_used):
            transitions_matrix[iv, ic] = ( E_gap_DFT + SCISSOR 
                                           + DeltaE_valb_fromVBM[iv]
                                           + DeltaE_conb_fromCBM[ic]  )
    # Pretty-print the small transition matrix (max 10×10)
    log.append("\n[transition_matrix - truncated to 10] (eV):\n")
    max_display_v = min(10, n_valence_used)
    max_display_c = min(10, n_conduction_used)
    # Conduction band header: c00, c01, ...
    header = "      " + "      ".join(f"c{ic:02d}" for ic in range(max_display_c))
    log.append(header + "\n")
    # Rows: v00, v01, v02 = VBM, VBM-1, VBM-2, ...
    for iv in range(max_display_v):
        row = "  ".join(f"{transitions_matrix[iv, ic]:7.3f}" for ic in range(max_display_c))
        log.append(f"v{iv:02d}: {row}\n")
    
    # Create a boolean mask of all transitions lying within the target optical window.
    # 'energyWindow_goal' is the width of the desired spectral window (e.g. 3 eV);
    # hence all transitions below this energy are considered relevant for the BSE kernel.
    # Determine how many valence and conduction bands are required to cover
    # all transitions inside the window:
    #   - if ANY transition from a given valence band is within the window,
    #     that valence band must be included → count over axis=1
    #   - same for conduction bands → count over axis=0
    if energy_window_goal is not None: spectra_energy_window_aboveGap = energy_window_goal
    else: spectra_energy_window_aboveGap = 3.5
    
    mask = transitions_matrix < E_gap_DFT + SCISSOR + spectra_energy_window_aboveGap
    output['gap_DFT']  = E_gap_DFT
    output['SCISSOR']  = SCISSOR
    output['NBANDSV']  = np.sum( np.any(mask, axis=1) )
    output['NBANDSO']  = np.sum( np.any(mask, axis=0) )
    output['OMEGAMAX'] =  E_gap_DFT + SCISSOR + spectra_energy_window_aboveGap
    #                     #Note that SCISSOR is defined   
    #                     #SCISSOR = float(G0W0_gap) - float(E_gap_DFT)  if G0W0_gap is not None else 0
    #                     #Thus E_gap_DFT + SCISSOR = G0W0_gap if it's defined, esle E_gap_DFT
    log.append( f"\nSpectra_energy_window_aboveGap = {spectra_energy_window_aboveGap:.2f} eV, "
                f"E_gap_DFT = {E_gap_DFT:.2f} - SCISSOR = {SCISSOR:.2f} eV\n"
                f"[result] OMEGAMAX = {output['OMEGAMAX']:.2f} eV\n"
                f"[result] NBANDSV  = {output['NBANDSV']} - NBANDSO = { output['NBANDSO']}\n"     )
    #Construct the final log string and add an indentation to it
    output['log'] = "".join(log)
    output['log'] =  '   ' +  output['log'].replace('\n', '\n   ')
    return output

    
def _extract_opticalgap_fromWorkchainNode(mBSE_node):
    mBSE_node_energy = mBSE_node.outputs.opticaltransitions.get_array('energy')
    mBSE_node_oscstr = mBSE_node.outputs.opticaltransitions.get_array('osc_strength')
    idxs_oscstr_nonzero = np.where( mBSE_node_oscstr > 0 )[0]
    return ( mBSE_node_energy[ idxs_oscstr_nonzero[0] ] , mBSE_node_oscstr[ idxs_oscstr_nonzero[0] ] )





