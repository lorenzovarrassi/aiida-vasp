# pylint: disable=too-many-arguments

import numpy as np
from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.engine import process_handler

from aiida_vasp.workchains.v2.vasp import VaspWorkChain
from aiida_vasp.workchains.vMBPT.utils_helpers_mBSE import _determine_BSE_parameters
from aiida_vasp.workchains.vMBPT.utils_helpers_extrapolation import input_magnetic_moment_tomagmom

PRECFOCK_VOLUME_THRESHOLD = 250
# Per-k-point BSE screening-restart files, e.g. WFULL0001.tmp, W0001.tmp - never a single
# bare-named file. Shared by __validate_remote_has_required_files (as a prefix-wildcard
# or_group) and __build_parameters (to pick out the extra files to remote-copy), so the
# prefixes are defined once instead of duplicated in two different shapes.
BSE_SCREENING_FILE_PREFIXES = ('WFULL', 'W0')


def _dft_gap_from_bandsdata(bandsdata):
    """Fundamental gap (E_CBM - E_VBM) from a BandsData, same VBM/CBM
    extraction as _determine_BSE_parameters (utils_helpers_mBSE.py) - kept
    as its own small copy here since only the gap value is needed, not the
    full transition-matrix analysis."""
    b_band = bandsdata.get_array('bands')
    b_occ = bandsdata.get_array('occupations')
    idx_ho_per_k = [np.where(b_occ[k, :-1] - b_occ[k, 1:] > 0)[0][0] for k in range(b_occ.shape[0])]
    idx_ho = idx_ho_per_k[0]
    e_vbm = np.max(b_band[:, idx_ho])
    e_cbm = np.min(b_band[:, idx_ho + 1])
    return e_cbm - e_vbm


class VaspAtomicBSEWorkChain(VaspWorkChain):
    """Run exactly one BSE/optical VASP calculation.
    Purpose of workchain:
    1) Hide the complexity of BSE INCAR construction from the user, which is managed by the workchain.
    2) Validate the inputs for consistency and completeness.
       Introduce BSE-specific exit codes for missing/inconsistent inputs, and for missing restart files.
    3) Introduce specific error handling for the BSE case.
      
    Design rationale:
    1) This workchain is a single-step workchain, not a multi-step workflow. It is not intended to manage a sequence of calculations.
       It's essentially a wrapper around a single VaspCalculation, with the added value of input validation, INCAR construction, and error handling.
       Thus it expects a restart_folder input, which is the output of a previous VASP calculation (usually a G0W0 calculation) that produced 
       the necessary WAVECAR/WAVEDER and possibly WFULL*/W0* files.
    2) Subclass VaspWorkChain:
       spec.outline() is inherited unchanged from VaspWorkChain.define() via super().define(spec) and not redefined here. 
       spec.outline() is:
        setup -> init_inputs
              -> if_(run_auto_parallel)(prepare_inputs, perform_autoparallel)
              -> while_(should_run_process)(prepare_inputs, run_process, inspect_process)
              -> results
        Of those steps, only init_inputs() is overridden below; it performs both input validation and INCAR construction. 
        The other steps (setup, prepare_inputs, run_process, results) are inherited as-is from VaspWorkChain.
        An additional process_handler() is added for exit-700/OOM fallback. 
    3) Three optical.algo modes, per https://vasp.at/wiki/Bethe-Salpeter-equations_calculations:
      - 'modelBSE': ALGO=TDHF, model dielectric screening (AEXX/HFSCREEN from screening.static_inverse_diel/range_parameter). 
         No extrarestart files beyond WAVECAR/WAVEDER.
      - 'BSE': ALGO=BSE, screening read from file (per-k-point WFULL* or W0* files, in addition to WAVECAR/WAVEDER).
      - 'IPA': independent-particle optical calc, no BSE/screening machinery at all.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('parameters',        valid_type=orm.Dict,         required=False, default=lambda: orm.Dict(dict={}),
                    help='Ignored - INCAR is built internally by this workchain from its own scalar inputs.',)
        spec.input('restart_folder',    valid_type=orm.RemoteData, required=True)
        spec.input('encut',             valid_type=orm.Float,      required=False, serializer=to_aiida_type,
                   help='ENCUT (eV). If omitted, VASP infers it from the restarted WAVECAR (a warning is logged).')
        spec.input('nbands',            valid_type=orm.Int,        required=False, serializer=to_aiida_type,
                   help='NBANDS. If omitted, VASP infers it from the restarted WAVECAR (a warning is logged).')
        spec.input('encut_chi',         valid_type=orm.Float,      required=False, serializer=to_aiida_type,
                   help='ENCUTGW/ENCUTGWSOFT (eV). Independent of encut - if omitted, VASP determines automatically.')
        spec.input('magnetic_moment_onsite', valid_type=orm.Dict,  required=False, serializer=to_aiida_type,
                   help="Per-site magnetic moments. Presence activates ISPIN=2.")

        spec.input('screening.static_inverse_diel', valid_type=orm.Float, required=False, serializer=to_aiida_type,
                   help="AEXX. Required when optical.algo == 'modelBSE'.")
        spec.input('screening.range_parameter', valid_type=orm.Float, required=False, serializer=to_aiida_type,
                   help="HFSCREEN. Required when optical.algo == 'modelBSE'.")

        spec.input('optical.algo',      valid_type=orm.Str,   required=True,  serializer=to_aiida_type, help="'BSE', 'modelBSE', or 'IPA'.")
        spec.input('optical.ibse',      valid_type=orm.Int,   required=False, default=lambda: orm.Int(2),  serializer=to_aiida_type)
        spec.input('optical.nbseeig',   valid_type=orm.Int,   required=False, default=lambda: orm.Int(50), serializer=to_aiida_type)
        spec.input('optical.nbandsv',   valid_type=orm.Int,   required=False, serializer=to_aiida_type)
        spec.input('optical.nbandso',   valid_type=orm.Int,   required=False, serializer=to_aiida_type)
        spec.input('optical.omegamax',  valid_type=orm.Float, required=False, serializer=to_aiida_type)
        spec.input('optical.optical_energy_window', valid_type=orm.Float, required=False, serializer=to_aiida_type,
                   help='Used to auto-derive nbandsv/nbandso  when they are not both supplied explicitly. Requires optical.bandsdata.')
        spec.input('optical.bandsdata', valid_type=orm.BandsData, required=False,
                   help='Reference DFT band structure. Required when optical_energy_window is actually used.')
        spec.input('optical.scissor',   valid_type=orm.Float, required=False, default=lambda: orm.Float(0.0), serializer=to_aiida_type,
                   help='Scissor shift (eV). Nonzero activates the SCISSOR INCAR tag automatically.')

        spec.input('optimization.kpar', valid_type=orm.Int,  required=False, default=lambda: orm.Int(1),     serializer=to_aiida_type)
        spec.input('optimization.lreal',valid_type=orm.Bool, required=False, default=lambda: orm.Bool(True), serializer=to_aiida_type)
        spec.input('optimization.set_PRECFOCK_to_Fast', valid_type=orm.Bool, required=False,
                   default=lambda: orm.Bool(True), serializer=to_aiida_type)

        spec.input('extraresources_fallback_options', valid_type=orm.Dict, required=False, serializer=to_aiida_type)

        spec.exit_code(406, 'ERROR_MISSING_RESTART_FILES',        message='restart_folder is missing one or more required files.')
        spec.exit_code(407, 'ERROR_INCONSISTENT_SCREENING_INPUTS',message='optical.algo and the screening namespace are inconsistent with each other.')
        spec.exit_code(408, 'ERROR_INCOMPLETE_BSE_BAND_RANGE',    message='Exactly one of optical.nbandsv/optical.nbandso was supplied - both or neither is required.')
        spec.exit_code(409, 'ERROR_MISSING_BSE_BAND_RANGE_INPUTS',message='Neither optical.nbandsv/nbandso nor optical.optical_energy_window were supplied.')
        spec.exit_code(410, 'ERROR_MISSING_BANDSDATA_FOR_ENERGY_WINDOW', message='optical.optical_energy_window was supplied without optical.bandsdata.')

    def __get_omegamax_value(self):
        """Return the Float value of optical.omegamax if supplied, else None."""
        # omegamax is optional, but if supplied it must be a Float.
        # Therefore .get() returns the Float node or None; None has no .value, hence the second line
        # instead of a single .get('omegamax').value chain.
        omegamax = self.inputs.optical.get('omegamax')
        return omegamax.value if omegamax is not None else None
    
    def __get_num_gpu_per_node(self):
        """Return the number of GPUs per node, as specified in the options.custom_scheduler_commands string."""
        opts = self.inputs.get('options')
        if opts is None: return 0
        opts = opts.get_dict()
        if 'custom_scheduler_commands' not in opts: return 0
        tokens = opts['custom_scheduler_commands'].replace('=', ':').split(':')
        if 'gpu' not in tokens: return 0
        return int(tokens[tokens.index('gpu') + 1])

    def __validate_remote_has_required_files(self, remote, required, or_groups=()):
        """Check that the given remote folder contains all required files, and at least one file from each of the or_groups.
        a BSE calculation requires WAVECAR, WAVEDER, and either WFULL* or W0* files. 
        IP and modelBSE calculations require only WAVECAR and WAVEDER.
        If any required files are missing, or if any of the or_groups are not satisfied, report an error and return None.
        In addition return the folder's file listing on success (so callers needing the actual filenames, e.g. for a copy list, 
        don't have to listdir() it a second time)."""
        if remote is None:
            self.report('restart_folder is None')
            return None
        try:
            files = set(remote.listdir())
        except Exception as exc:  # pylint: disable=broad-except
            self.report(f'Could not list restart_folder contents: {exc}')
            return None
        missing = [fname for fname in required if fname not in files]
        if missing:
            self.report(f'restart_folder missing required files: {missing}')
            return None
        for group in or_groups:
            found = any(
                (pattern.endswith('*') and any(f.startswith(pattern[:-1]) for f in files)) or pattern in files
                for pattern in group
            )
            if not found:
                self.report(f'restart_folder missing at least one of: {group}')
                return None
        return files

    def __validate_inputs_entries(self):
        #[1] Input validation: optical.algo and screening namespace must be consistent
        # and nbandsv/nbandso must be both present or both absent.
        algo = self.inputs.optical.algo.value
        if algo not in ('modelBSE', 'BSE', 'IPA'):  
            return self.exit_codes.ERROR_INCONSISTENT_SCREENING_INPUTS

        if algo == 'IPA':
            if any(key in self.inputs.screening for key in ('static_inverse_diel', 'range_parameter')):
                self.report('WARNING: optical.algo is IPA - the screening namespace is ignored.')
        elif algo == 'modelBSE' and \
            (('static_inverse_diel' not in self.inputs.screening) or ('range_parameter' not in self.inputs.screening)):
            return self.exit_codes.ERROR_INCONSISTENT_SCREENING_INPUTS

        nbandsv  = self.inputs.optical.get('nbandsv')
        nbandso  = self.inputs.optical.get('nbandso')

        # If only one of nbandsv/nbandso is supplied, it's an error - both or neither must be supplied.
        if (nbandsv is None) != (nbandso is None):  
            return self.exit_codes.ERROR_INCOMPLETE_BSE_BAND_RANGE

        # If neither nbandsv nor nbandso is supplied they are auto-derived from the reference DFT band structure + optical_energy_window.
        # In that case, optical_energy_window and bandsdata must be supplied.
        # This applies only to the BSE and modelBSE algorithms, not IPA (which doesn't use nbandsv/nbandso at all).
        if (nbandsv is None and nbandso is None) and (algo == 'BSE' or algo == 'modelBSE'):
            if 'optical_energy_window' not in self.inputs.optical: return self.exit_codes.ERROR_MISSING_BSE_BAND_RANGE_INPUTS
            if 'bandsdata' not in self.inputs.optical:             return self.exit_codes.ERROR_MISSING_BANDSDATA_FOR_ENERGY_WINDOW


        #[2] Encut / nbands related warnings: VASP infers encut/nbands from the restarted WAVECAR if they are not supplied, 
        #but this is a potential source of confusion for users -> log an explicit warning.
        #Warn if encut/nbands are not supplied - VASP will infer them from the restarted WAVECAR
        if 'encut' not in self.inputs:  self.report('WARNING: encut not supplied - VASP will infer ENCUT from the restarted WAVECAR.')
        if 'nbands' not in self.inputs: self.report('WARNING: nbands not supplied - VASP will infer NBANDS from the restarted WAVECAR.')

        if (algo == 'BSE' or algo == 'modelBSE'):
            if ('encut_chi' in self.inputs) and ('encut' not in self.inputs):
                self.report('WARNING: encut_chi supplied without encut - ENCUT will be inferred by VASP from the '
                            'restarted WAVECAR while ENCUTGW/ENCUTGWSOFT are set explicitly; double-check this is intended.')
            elif 'encut' in self.inputs and 'encut_chi' not in self.inputs:
                self.report('INFO: encut_chi not supplied - this lets VASP automatically determine ENCUTGW/ENCUTGWSOFT itself.')

    def __resolve_band_range_and_omegamax(self):
        """Resolve NBANDSV/NBANDSO/OMEGAMAX for BSE/modelBSE.
        - Only called from __build_parameters() for BSE/modelBSE - IPA never needs this.
        - It assumes that __validate_inputs_entries() has already been called, so the inputs are consistent and complete, i.e.
        either both nbandsv/nbandso are supplied, or neither is supplied and optical_energy_window + bandsdata are supplied.

        - Nbandsv/Nbandso decision logic:
          -- If nbandsv/nbandso are supplied, they are used directly indipendently of optical_energy_window/bandsdata.
          -- If not, they are auto-derived from the reference band structure + optical_energy_window using _determine_BSE_parameters().
        - Omegamax decision logic:
            -- If optical.omegamax is supplied, it is used directly.
               If optical.omegamax and the run is on GPUs,the supplied value is still used but a warning is logged.
               (VASP GPU best practice recommends leaving OMEGAMAX unset so all NBANDSV/NBANDSO transitions stay in the kernel).
            -- If optical.omegamax is not supplied + nbadnsv/nbandso are supplied, it is left unset (None).
            -- If optical.omegamax is not supplied + the run is on GPUs it is left unset (None) even if nbandsv/nbandso are auto-derived, 
            -- If optical.omegamax is not supplied + nbadnsv/nbandso are not supplied + the run is on CPUs, it is auto-derived from the 
               same _determine_BSE_parameters() call that derives nbandsv/nbandso (i.e. all three are auto-derived together). """

        omegamax         = self.__get_omegamax_value()
        num_gpu_per_node = self.__get_num_gpu_per_node()
        if omegamax is not None and num_gpu_per_node > 0:
            self.report('WARNING: optical.omegamax explicitly supplied while running on GPUs - VASP GPU best '
                        'practice recommends leaving OMEGAMAX unset so all NBANDSV/NBANDSO transitions stay in '
                        'the kernel; double-check this override is intended.')

        nbandsv  = self.inputs.optical.get('nbandsv')
        nbandso  = self.inputs.optical.get('nbandso')

        if nbandsv is not None and nbandso is not None:
            # --- NBANDSV/NBANDSO route: explicit values win outright ---
            # omegamax stays whatever was resolved above (explicit value, or None - never
            # auto-derived here, since that needs the bse_params computed only below).
            nbandsv, nbandso = nbandsv.value, nbandso.value
        else:
            # --- NBANDSV/NBANDSO route: neither supplied -> auto-derive both ---
            dft_gap = _dft_gap_from_bandsdata(self.inputs.optical.bandsdata)
            bse_params = _determine_BSE_parameters(
                bandsdata=self.inputs.optical.bandsdata,
                G0W0_gap=dft_gap + self.inputs.optical.scissor.value,
                energy_window_goal=self.inputs.optical.optical_energy_window.value )
            nbandsv, nbandso = bse_params['NBANDSV'], bse_params['NBANDSO']

            # --- OMEGAMAX route: not supplied -> auto-derive from the same bse_params, unless on GPUs ---
            if omegamax is None and num_gpu_per_node == 0:
                omegamax = bse_params['OMEGAMAX']
        return nbandsv, nbandso, omegamax

    def __build_parameters(self, algo):
        """Build the flat INCAR-tag dict expected directly by VaspCalculation."""
        incar = {'ismear': 0, 'sigma': 0.02, 'prec': 'NORMAL'}


        #[1] parameters common to all three optical.algo modes
        if 'encut' in self.inputs:
            incar['encut'] = self.inputs.encut.value
        if 'nbands' in self.inputs:
            incar['nbands'] = self.inputs.nbands.value

        if 'magnetic_moment_onsite' in self.inputs:
            _, incar['magmom'] = input_magnetic_moment_tomagmom(self.inputs.structure, self.inputs.magnetic_moment_onsite.get_dict())
            incar['ispin'] = 2
        else:
            incar['ispin'] = 1

        #[1.1] Optimization related parameters common to all cases
        incar['lreal'] = 'Auto' if self.inputs.optimization.lreal.value else '.FALSE.'
        # GPU-count-based KPAR override (https://vasp.at/wiki/Best_practices_for_Bethe-Salpeter_calculations)                
        num_gpu_per_node = self.__get_num_gpu_per_node()
        num_nodes = self.inputs.options.get_dict().get('resources', {}).get('num_machines', 1)
        if num_gpu_per_node > 0:
            # options is guaranteed present here: __get_num_gpu_per_node() only returns >0 after
            # reading it itself, so it cannot be absent at this point.
            incar['kpar'] = num_gpu_per_node * num_nodes
            self.report(f'INFO: GPU detected: setting KPAR={incar["kpar"]} (num_gpu_per_node={num_gpu_per_node}, num_nodes={num_nodes}) - OVERRIDING any user-supplied KPAR value.')
        else:
            incar['kpar'] = self.inputs.optimization.kpar.value

        #[2] Parameters specific to excitonic optical calculations (BSE/modelBSE/):
        if (algo == 'BSE' or algo == 'modelBSE'):
            nbandsv, nbandso, omegamax = self.__resolve_band_range_and_omegamax()

            if 'encut_chi' in self.inputs:
                incar['encutgw']     = self.inputs.encut_chi.value
                incar['encutgwsoft'] = self.inputs.encut_chi.value

            incar['ibse']    = self.inputs.optical.ibse.value
            incar['nbseeig'] = self.inputs.optical.nbseeig.value
            incar['nbandso'] = nbandso
            incar['nbandsv'] = nbandsv
            if omegamax is not None:
                incar['omegamax'] = omegamax
            if self.inputs.optical.scissor.value != 0:
                incar['scissor'] = self.inputs.optical.scissor.value

        if algo == 'modelBSE':
            incar['algo']     = 'TDHF'
            incar['antires']  = 0
            incar['lmodelhf'] = '.TRUE.'
            incar['aexx']     = self.inputs.screening.static_inverse_diel.value
            incar['hfscreen'] = self.inputs.screening.range_parameter.value
        elif algo == 'BSE':
            incar['algo']     = 'BSE'
            incar['antires']  = 0

        else:  # 'IPA' - independent-particle optics: no self-consistency, momentum matrix
               # elements from finite differences, no BSE/screening machinery at all.
            incar['algo']    = 'None'
            incar['nelm']    = 1
            incar['loptics'] = '.TRUE.'
            incar['lpead']   = '.TRUE.'
        volume = self.inputs.structure.get_cell_volume()
        if volume > PRECFOCK_VOLUME_THRESHOLD or self.inputs.optimization.set_PRECFOCK_to_Fast.value:
            incar['precfock'] = 'Fast'

        return incar

    def __build_settings(self, extra_copy_files=()):
        """Build the 'settings' dict passed to VaspCalculation, merging into whatever the caller/exposing-parent
        already put in self.ctx.inputs.settings (set by super().init_inputs() from self.inputs.settings) rather
        than overwriting it - same merge pattern as VaspAtomicG0W0WorkChain.__build_settings(). A caller
        composing this class into a larger workflow (e.g. a GroundUp chain) may already have settings of its
        own (e.g. parser_settings); overwriting wholesale would silently drop those. Ensures WAVEDER (always)
        and, for algo=='BSE', the per-k-point WFULL*/W0* screening-restart files actually present in
        restart_folder are remote-copied, on top of whatever else was already requested."""
        settings = dict(self.ctx.inputs.get('settings') or {})
        copy_list = list(settings.get('ADDITIONAL_REMOTE_COPY_LIST', []))
        for name in ('WAVEDER', *extra_copy_files):
            if name not in copy_list:
                copy_list.append(name)
        settings['ADDITIONAL_REMOTE_COPY_LIST'] = copy_list
        return settings

    def init_inputs(self):
        # Perform the superclass input initialization first, so that the inputs namespace is fully populated and ready for validation.
        exit_code = super().init_inputs()
        if exit_code is not None: return exit_code

        #[1] Validate the inputs entries for consistency and completeness.
        exit_code = self.__validate_inputs_entries()
        if exit_code is not None: return exit_code

        algo = self.inputs.optical.algo.value

        self.ctx.inputs.parameters = self.__build_parameters(algo)

        required = ('WAVECAR', 'WAVEDER')
        or_groups = (tuple(f'{prefix}*' for prefix in BSE_SCREENING_FILE_PREFIXES),) if algo == 'BSE' else ()
        files = self.__validate_remote_has_required_files(self.inputs.restart_folder, required, or_groups)
        if files is None:
            return self.exit_codes.ERROR_MISSING_RESTART_FILES

        extra_copy_files = [name for name in files if name.startswith(BSE_SCREENING_FILE_PREFIXES)] if algo == 'BSE' else ()
        self.ctx.inputs.settings = self.__build_settings(extra_copy_files)

        return None

    @process_handler(priority=900)
    def handler_unfinished_calc_generic(self, node):
        """Exit-700 (OOM/walltime) resource fallback - same shape as
        VaspAtomicG0W0WorkChain's, copied not shared. No inspect_process()
        override exists on this class (BSE calcs aren't GW), so this is
        reached through the inherited BaseRestartWorkChain.inspect_process()
        dispatch unmodified."""
        report = super().handler_unfinished_calc_generic(node)
        if report is not None and report.exit_code.status == 0 and 'extraresources_fallback_options' in self.inputs:
            self.report('Calculation did not finish (exit 700) - retrying with extraresources_fallback_options.')
            self.ctx.inputs.metadata['options'] = self.inputs.extraresources_fallback_options.get_dict()
        return report
