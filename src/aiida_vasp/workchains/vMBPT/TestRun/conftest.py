"""Local pytest config for the vMBPT TestRun regression suite.

Kept out of the main tests/ tree deliberately: these tests load the *real* AiiDA profile
(via `load_profile()`, same as this codebase's own regression_harness/ scripts) rather than
the ephemeral `aiida_profile_clean` fixture from `aiida.tools.pytest_fixtures`, because the
integration tests need the real, already-configured `vasp@localhost` code and PBE.54 POTCAR
family. Registering the `integration` marker here (instead of in the root pyproject.toml)
keeps that scope local to this directory.
"""


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'integration: real end-to-end VASP regression test (slow; needs vasp@localhost + PBE.54 POTCARs '
        'configured in the loaded AiiDA profile)',
    )
