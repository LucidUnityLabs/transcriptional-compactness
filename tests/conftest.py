"""Test configuration: import paths + legacy-module loading.

The legacy per-experiment scripts are imported UNMODIFIED from their
committed locations so defect-documentation tests pin the historical
behaviour (and so the counterexamples provably fail on the legacy code).
``ot.emd2`` is served by an exact scipy-HiGHS stub when POT is absent so
legacy paths run faithfully without the production dependency.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO / "experiments"
for p in (str(REPO), str(EXPERIMENTS)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _lp_emd2(a, b, M):
    from scipy.optimize import linprog

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    M = np.asarray(M, float)
    n, m = M.shape
    A_eq = np.zeros((n + m, n * m))
    for i in range(n):
        A_eq[i, i * m:(i + 1) * m] = 1.0
    for j in range(m):
        A_eq[n + j, j::m] = 1.0
    res = linprog(M.ravel(), A_eq=A_eq, b_eq=np.concatenate([a, b]),
                  bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError("stub LP failed: " + res.message)
    return float(res.fun)


def _lp_emd(a, b, M):
    from scipy.optimize import linprog

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    M = np.asarray(M, float)
    n, m = M.shape
    A_eq = np.zeros((n + m, n * m))
    for i in range(n):
        A_eq[i, i * m:(i + 1) * m] = 1.0
    for j in range(m):
        A_eq[n + j, j::m] = 1.0
    res = linprog(M.ravel(), A_eq=A_eq, b_eq=np.concatenate([a, b]),
                  bounds=(0, None), method="highs")
    return res.x.reshape(n, m)


try:
    import ot  # noqa: F401
    HAVE_POT = True
except ImportError:
    HAVE_POT = False

if not HAVE_POT:
    stub = types.ModuleType("ot")
    stub.emd2 = _lp_emd2
    stub.emd = _lp_emd
    sys.modules["ot"] = stub


LEGACY_SCRIPTS = {
    "e1": REPO / "experiments" / "E1_within_patient" / "run.py",
    "meta": REPO / "experiments" / "META_patient_level" / "run.py",
    "lung": REPO / "experiments" / "HELDOUT_GSE131907" / "run.py",
    "breast": REPO / "experiments" / "HELDOUT_GSE161529" / "run.py",
    "sens": REPO / "experiments" / "HELDOUT_GSE161529" / "run_sensitivity.py",
    "labelval": REPO / "experiments" / "LABEL_VALIDATION" / "run.py",
}


def load_legacy(name):
    path = LEGACY_SCRIPTS[name]
    spec = importlib.util.spec_from_file_location(f"legacy_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def legacy_e1():
    return load_legacy("e1")


@pytest.fixture(scope="session")
def legacy_meta():
    return load_legacy("meta")


@pytest.fixture(scope="session")
def legacy_labelval():
    return load_legacy("labelval")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "pot_backend: needs the real POT backend (CI-mandatory, "
        "skipped where POT is not installed)")


def pytest_collection_modifyitems(config, items):
    if HAVE_POT:
        return
    if os.environ.get("REQUIRE_POT") == "1":
        # Release validation (CI): the pinned POT backend is a hard
        # requirement.  Failing the whole run (rather than skipping the
        # two real-backend tests, which would run against conftest's
        # scipy stub and look green) is the audit R07 contract.
        raise pytest.UsageError(
            "REQUIRE_POT=1 but POT is not importable: release validation "
            "requires the pinned POT backend (audit R07)")
    skip = pytest.mark.skip(reason="POT not installed: the real-backend "
                                   "integration test is skipped locally and "
                                   "mandatory in release CI")
    for item in items:
        if "pot_backend" in item.keywords:
            item.add_marker(skip)
