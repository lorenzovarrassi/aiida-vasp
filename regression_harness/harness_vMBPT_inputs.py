"""
Golden-file regression harness for VaspmBSEInitScriptWorkChain's pure
input-building logic (workchain_mBSE_base_winterpolation.py).

Purpose
-------
`__prepare_inputs_DFT`, `__prepare_inputs_mBSE_base`,
`__prepare_inputs_G0W0interpolation`, and `__add_inputs_mBSE_incar` are pure
functions over lightweight AiiDA nodes: they build a Python dict (INCAR) or a
shell command string (prepend_text), and never submit/run anything. This
script calls them directly against one fixed, synthetic fixture (no daemon,
no cluster, nothing stored to the database) and captures their output as
JSON. Run it before and after each refactor step; diff the two JSON files.
Any unintended diff is a regression signal.

Usage
-----
    source ~/venv_AiiDA_20251209/bin/activate
    PYTHONPATH=<worktree>/src python regression_harness/harness_vMBPT_inputs.py \\
        regression_harness/golden/<label>.json

Requires the real `lvarras_aiida` AiiDA profile to be loadable (for
`load_profile()` at import time in the target module) but performs no
`.store()` calls anywhere - nothing is written to the database.
"""
import importlib
import json
import os.path
import sys

import numpy as np
from aiida import load_profile
from aiida import orm
from aiida.common.extendeddicts import AttributeDict

load_profile()

from aiida_vasp.workchains.vMBPT.workchain_mBSE_base_winterpolation import (  # noqa: E402
    VaspmBSEInitScriptWorkChain,
)

CLS = VaspmBSEInitScriptWorkChain
MANGLE = "_VaspmBSEInitScriptWorkChain__"


def _get_mangled(name):
    """Fetch a name-mangled double-underscore method as a plain function."""
    return CLS.__dict__[MANGLE + name]


class _Harness:
    """Duck-typed stand-in for `self` - NOT a real WorkChain instance.

    Only carries what the four target methods actually touch: `.inputs`,
    `.ctx`, `.report()`, `._vasp_workchain`/`._vasp_initscript_workchain`
    (class attrs, inherited), `._last_wc_node()` (single-underscore, copied
    directly since it has no Process-internal dependencies), and the
    mangled `__build_options_entry` (called internally by two of the target
    methods, bound the same way).
    """

    _vasp_workchain = CLS._vasp_workchain
    _vasp_initscript_workchain = CLS._vasp_initscript_workchain
    _last_wc_node = CLS._last_wc_node

    def exposed_inputs(self, _proc_cls):
        return AttributeDict()

    def report(self, _msg):
        pass


setattr(_Harness, MANGLE + "build_options_entry", _get_mangled("build_options_entry"))


class _FakeRemoteFolder:
    """Stand-in for a RemoteData restart/reference folder."""

    def __init__(self, path):
        self._path = path

    def get_remote_path(self):
        return self._path

    def __repr__(self):
        return f"<FakeRemoteFolder {self._path}>"


class _FakeDFTNode:
    """Stand-in for a finished `vasp.vasp` process node."""

    is_finished_ok = True

    class outputs:
        pass


FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def build_fixture(variant="remote"):
    """One fixed, representative set of synthetic inputs (same every run).

    `variant` selects which of the two mutually-exclusive GW-reference
    branches is exercised: "remote" (remote_gw_reference_folder) or "local"
    (local_gw_reference_folder) - see __prepare_inputs_G0W0interpolation.
    """
    assert variant in ("remote", "local")
    structure = orm.StructureData(cell=[[4.5, 0, 0], [0, 4.5, 0], [0, 0, 4.5]])
    structure.append_atom(position=(0, 0, 0), symbols="Ga")
    structure.append_atom(position=(2.25, 2.25, 2.25), symbols="N")

    n_kpts, n_bands = 8, 10
    kpts_arr = np.random.RandomState(0).rand(n_kpts, 3)
    kpoints = orm.KpointsData()
    kpoints.set_kpoints(kpts_arr)

    bands = orm.BandsData()
    bands.set_kpointsdata(kpoints)
    band_energies = np.linspace(-5, 5, n_kpts * n_bands).reshape(n_kpts, n_bands)
    occupations = np.zeros((n_kpts, n_bands))
    occupations[:, :5] = 1.0
    bands.set_bands(band_energies, occupations=occupations)

    dft_node = _FakeDFTNode()
    dft_node.outputs.bands = bands

    path_default_script = os.path.join(
        importlib.import_module("aiida_vasp").__path__[0],
        "workchains/vMBPT/utils_interpolationclasses.v2.py",
    )

    ns_interpolation = {
        "use_interpolation": orm.Bool(True),
        "nbandsgw_to_interpolate": orm.Int(8),
        "python_sourcing_env_command": orm.Str("source activate aiida-vasp"),
        "local_initscript": orm.SinglefileData(file=path_default_script),
    }
    if variant == "remote":
        ns_interpolation["remote_gw_reference_folder"] = _FakeRemoteFolder("/remote/scratch/gw_ref")
        ns_interpolation["gw_reference_filename"] = orm.Str("OUTCAR")
    else:
        ns_interpolation["local_gw_reference_folder"] = orm.Str(FIXTURES_DIR)
        ns_interpolation["gw_reference_filename"] = orm.Str("dummy_OUTCAR.3")

    fake_self = _Harness()
    fake_self.inputs = AttributeDict(
        {
            "structure": structure,
            "ns_parameters": AttributeDict(
                {
                    "encut": orm.Float(400.0),
                    "ibse": orm.Int(2),
                    "kpar": orm.Int(1),
                    "nbseeig": orm.Int(50),
                }
            ),
            "ns_reference": AttributeDict({"use_hdf5": orm.Bool(False)}),
            "ns_optimization": AttributeDict(
                {
                    "lreal": orm.Bool(True),
                    "set_PRECFOCK_to_Fast": orm.Bool(True),
                }
            ),
            "ns_BSE": AttributeDict(
                {
                    "static_inverse_diel": orm.Float(0.1),
                    "screening_parameter": orm.Float(0.2),
                    "G0W0_gap": orm.Float(3.5),
                    "optical_energy_window": orm.Float(3.0),
                }
            ),
            "ns_interpolation": AttributeDict(ns_interpolation),
            "ns_option": AttributeDict(
                {
                    "copy_result_locally": orm.Bool(True),
                    "calculation_label": orm.Str("golden-fixture"),
                    "calculation_tag": orm.Str("_golden"),
                    "copy_result_locally_path": orm.Str("/tmp/whatever"),
                }
            ),
            "options": orm.Dict(
                dict={
                    "account": "acct1",
                    "qos": "normal",
                    "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 4},
                    "queue_name": "batch",
                    "max_wallclock_seconds": 3600,
                }
            ),
        }
    )
    fake_self.ctx = AttributeDict(
        {
            "is_spinpol": False,
            "state_WC": AttributeDict(
                {
                    "restart_folders": AttributeDict(
                        {"for_MBSE": _FakeRemoteFolder("/remote/scratch/dft_restart")}
                    ),
                    "submitted": AttributeDict({"DFT": [dft_node], "MBSE": []}),
                }
            ),
        }
    )
    return fake_self


def to_jsonable(obj):
    """Recursively reduce AiiDA nodes / AttributeDicts to plain JSON-able data."""
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if hasattr(obj, "get_dict"):  # orm.Dict
        return to_jsonable(obj.get_dict())
    if isinstance(obj, dict):  # AttributeDict or plain dict
        return {k: to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "value"):  # orm.Int/Float/Str/Bool
        return to_jsonable(obj.value)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "filename"):  # orm.SinglefileData
        return f"<SinglefileData filename={obj.filename!r}>"
    return repr(obj)


def run_harness(variant="remote"):
    fake_self = build_fixture(variant=variant)

    m_dft = _get_mangled("prepare_inputs_DFT")
    m_mbse_base = _get_mangled("prepare_inputs_mBSE_base")
    m_interp = _get_mangled("prepare_inputs_G0W0interpolation")
    m_incar = _get_mangled("add_inputs_mBSE_incar")

    out_dft = m_dft(fake_self)
    out_mbse = m_mbse_base(fake_self)
    out_mbse = m_interp(fake_self, out_mbse)
    out_mbse = m_incar(fake_self, out_mbse)

    return {
        "dft_inputs": to_jsonable(out_dft),
        "mbse_inputs": to_jsonable(out_mbse),
    }


def main():
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} <output_json_path> [remote|local]", file=sys.stderr)
        sys.exit(1)
    out_path = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) == 3 else "remote"
    result = run_harness(variant=variant)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fhandle:
        json.dump(result, fhandle, indent=2, sort_keys=True)
    print(f"Wrote {out_path} (variant={variant})")


if __name__ == "__main__":
    main()
