"""Rerun status for every analysis containing the copied OR routine.

The 2026-10-06 audit (C01) requires rerunning every script that embeds
the historical ollivier_ricci_edges implementation under the corrected
method.  The raw data are not redistributed, so this tool enumerates the
affected experiments, detects which inputs each one needs, and writes a
machine-readable status artifact (blocked drivers list their exact
missing inputs in their own results_corrected.json artifacts).

Usage: python3 experiments/rerun_status.py [--write]
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent

#: decision-critical analyses with corrected drivers (TC-1)
CORRECTED_DRIVERS = {
    "META_patient_level": "run_corrected.py",
    "HELDOUT_GSE131907": "run_corrected.py",
    "HELDOUT_GSE161529": "run_corrected.py (+ download_corrected.py)",
    "LABEL_VALIDATION": "run_corrected.py",
}

#: figure generators that read corrected-affected artifacts downstream
FIGURE_GENERATORS = ["FIGGEN_fig4.py", "FIGGEN_fig5.py"]


def data_references(script):
    """Input paths referenced via any data-root variable in a script."""
    text = script.read_text()
    refs = set()
    dynamic = False
    for m in re.finditer(
            r'\b(DATA|DATA_ROOT|data_root)\s*/\s*"([^"]+)"', text):
        refs.add(m.group(2))
    if re.search(r'\b(DATA|DATA_ROOT|data_root)\s*/\s*f"', text) or \
            re.search(r'\b(DATA|DATA_ROOT|data_root)\s*/\s*[A-Za-z_]', text):
        dynamic = True
    out = sorted(refs)
    if dynamic:
        out.append("<dynamic>")
    return out


def main():
    data_root = REPO / "data"
    data_root_exists = data_root.is_dir()
    rows = []
    for run in sorted(HERE.glob("*/run.py")):
        exp = run.parent.name
        text = run.read_text()
        has_legacy_or = ("def ollivier_ricci_edges" in text
                         and "G.subgraph" in text)
        if not has_legacy_or:
            continue
        refs = data_references(run)
        if not data_root_exists:
            missing = refs or ["<entire data/ tree absent>"]
        else:
            missing = [r for r in refs
                       if r != "<dynamic>"
                       and not (data_root / r).exists()]
        status = "blocked_data_absent" if missing else \
            "data_present_needs_rerun"
        if exp in ("E1_within_patient",):
            note = ("origin of the copied OR primitives; kept as the "
                    "historical reference implementation")
        elif exp in CORRECTED_DRIVERS:
            note = f"corrected driver: {CORRECTED_DRIVERS[exp]}"
        else:
            note = ("exploratory/extension analysis; rerun under the "
                    "corrected method after the discovery inputs are "
                    "acquired (see experiments/lib) — historical outputs "
                    "remain committed")
        rows.append({
            "experiment": exp,
            "historical_script": str(run.relative_to(REPO)),
            "contains_legacy_or_routine": True,
            "corrected_driver": CORRECTED_DRIVERS.get(exp),
            "data_references": refs,
            "missing_inputs": missing,
            "status": status,
            "note": note,
        })
    for fig in FIGURE_GENERATORS:
        rows.append({
            "experiment": fig.replace(".py", ""),
            "historical_script": str((HERE / fig).relative_to(REPO)),
            "contains_legacy_or_routine": False,
            "corrected_driver": None,
            "data_references": [],
            "missing_inputs": [],
            "status": "regenerate_after_corrected_reruns",
            "note": "figure generator reading corrected-affected "
                    "artifacts; regenerate PNGs only after the upstream "
                    "corrected artifacts exist and are reviewed",
        })
    report = {
        "method_version": "TC-1",
        "generated_by": "experiments/rerun_status.py",
        "n_affected": len(rows),
        "n_blocked": sum(1 for r in rows
                         if r["status"] == "blocked_data_absent"),
        "experiments": rows,
        "protocol": (
            "Historical analyses stay committed and unmodified. After "
            "the GEO/figshare inputs are acquired under data/ per "
            "REPRODUCING.md: run the four corrected drivers, then rerun "
            "the remaining affected analyses through experiments/lib "
            "(corrected metric), then regenerate FIGGEN figures and "
            "update manuscript numbers from the new artifacts only."),
    }
    out = HERE / "rerun_status.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"wrote {out} "
          f"({report['n_affected']} affected, "
          f"{report['n_blocked']} blocked on absent data)")
    for r in rows:
        print(f"  {r['status']:32s} {r['experiment']}")
    return report


if __name__ == "__main__":
    main()
