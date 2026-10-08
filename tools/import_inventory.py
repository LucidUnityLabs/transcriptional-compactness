"""Print third-party imports across experiments; fail on unclassified imports.

An import inventory is not proof that importing/running a script succeeds.
Keep the explicit distribution mapping reviewed instead of auto-pip-installing
arbitrary names found in source code (audit R06; reference implementation
ported from the audit's appendix and completed for this repository's actual
import surface — `hicstraw` is a direct import in F4/N3 and was unclassified
even in the audit's own KNOWN map).

With ``--require-installed`` every classified third-party module is actually
imported; use only in an environment that installed the reviewed full
profile (requirements/controls.in lock), never to bypass resolution.
"""
import argparse
import ast
import importlib
import sys
from collections import defaultdict
from pathlib import Path

# module name -> distribution providing it
KNOWN = {
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "ot": "POT",
    "matplotlib": "matplotlib", "sklearn": "scikit-learn",
    "networkx": "networkx", "rdata": "rdata",
    "statsmodels": "statsmodels", "scanpy": "scanpy",
    "harmonypy": "harmonypy", "anndata": "anndata", "h5py": "h5py",
    "hicstraw": "hic-straw",
}
ROOT = Path(__file__).resolve().parents[1]
LOCAL_TOPS = {"lib", "experiments", "tools"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-installed", action="store_true")
    args = parser.parse_args()
    imports = defaultdict(set)
    for path in sorted((ROOT / "experiments").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"),
                         filename=str(path))
        local = ({stem.stem for stem in path.parent.glob("*.py")}
                 | LOCAL_TOPS)
        for node in ast.walk(tree):
            names = [alias.name for alias in node.names] \
                if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom)
                and node.module and node.level == 0 else [])
            for name in names:
                top = name.split(".")[0]
                if top not in sys.stdlib_module_names and top not in local:
                    imports[top].add(str(path.relative_to(ROOT)))
    unknown = []
    unavailable = []
    for name, files in sorted(imports.items()):
        distribution = KNOWN.get(name)
        if distribution is None:
            unknown.append(name)
        elif args.require_installed:
            try:
                importlib.import_module(name)
            except Exception as exc:  # noqa: BLE001 - report any failure
                unavailable.append(
                    f"{name}: {type(exc).__name__}: {exc}")
        print(f"{name} -> {distribution or 'UNCLASSIFIED'}: "
              f"{', '.join(sorted(files))}")
    if not imports:
        raise SystemExit("No experiment imports found: run on the actual "
                         "repository checkout")
    if unknown:
        raise SystemExit(f"Review unclassified dependencies: {unknown}")
    if unavailable:
        raise SystemExit(f"Dependency imports failed: {unavailable}")


if __name__ == "__main__":
    main()
