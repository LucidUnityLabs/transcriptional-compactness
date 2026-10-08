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
import hashlib
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.publication import load_result, read_set
from lib.runner import load_json_strict

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
    prefix = {'F3_merfish_multi':'merfish', 'N2_visium_hd':'visium_hd',
              'T4_visium_spatial':'visium'}.get(script.parent.name)
    if prefix:
        refs = {f'{prefix}/{r}' for r in refs}
    out = sorted(refs)
    if dynamic:
        out.append("<dynamic>")
    return out


def main():
    data_root = REPO / "data"
    data_root_exists = data_root.is_dir()
    rows = []
    prior_path = HERE / "rerun_status.json"
    prior_rows = {row["experiment"]:row for row in load_json_strict(prior_path)["experiments"]} if prior_path.exists() else {}
    for run in sorted(HERE.glob("*/run.py")):
        exp = run.parent.name
        text = run.read_text()
        has_legacy_or = ("def ollivier_ricci" in text and "G.subgraph" in text)
        adjacent = exp in {"E5_hparam_sweep", "N4_subclone_stratification", "T6_random_hvg_control"}
        if not has_legacy_or and not adjacent:
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
        artifact = run.parent / ("results_corrected.json" if exp in CORRECTED_DRIVERS else "corrected/run_manifest.json")
        if exp == 'HELDOUT_GSE161529':
            bounded = run.parent/'results_primary_excluded_corrected.json'
            if bounded.exists() or bounded.with_suffix('.accepted.json').exists(): artifact = bounded
        evidence = load_result(artifact) if artifact.exists() or artifact.with_suffix('.accepted.json').exists() else {}
        if evidence.get("status") in {"ok", "exploratory_complete"}:
            status = "calculation_complete" if exp in CORRECTED_DRIVERS else "exploratory_complete"
        elif evidence.get("status") in {"running", "failed", "superseded_requires_rerun"}:
            status = evidence["status"]
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
        extra = {}
        if exp == 'HELDOUT_GSE161529' and evidence.get('status') == 'ok':
            paths, accepted = read_set(artifact)
            if accepted['schema'] == 'legacy-flat':
                receipt = load_json_strict(REPO/'config/GSE161529_completed_primary_excluded.json')
                for relative, entry in receipt['files'].items():
                    path = REPO/relative
                    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256'] or path.stat().st_size != entry['bytes']:
                        raise ValueError('conditional accepted breast run bytes changed')
                if evidence['loaded_driver_sha256'] != receipt['loaded_driver_sha256']:
                    raise ValueError('conditional accepted breast driver identity differs')
            status = 'completed_primary_excluded_partial_26_of_27'
            extra = {'completed_scope': 'primary/excluded only', 'original_endpoint_coverage': 'PARTIAL 26/27',
                     'remaining_work': 'other arms/variants; final frozen repeats and controls; affirmative luminal annotation and ER_0001 same-cell restoration',
                     'accepted_run_created_utc': evidence['created_utc'], 'estimable_donors':24}
        if exp == 'T5_proliferating_immune' and evidence.get('status') == 'failed':
            note += '; failed artifact is historical diagnostic evidence, not a pass; corrected rerun remains open'
        if exp == 'LABEL_VALIDATION':
            inherited = prior_rows.get(exp,{})
            ownership = inherited.get('ownership', {'pid':9089, 'started_local':'Wed Oct 7 13:24:45 2026',
                        'owner':'original LABEL task', 'control':'preserve process and original timestamps; no restart/reset'})
            observed = subprocess.run(['ps','-p',str(ownership['pid']),'-o','stat=,command='],capture_output=True,text=True,check=False).stdout.strip()
            if 'experiments/LABEL_VALIDATION/run_corrected.py' in observed:
                status = 'running_owned'; extra = {'ownership':ownership, 'process_observation':observed,
                    'remaining_work':'current owner completion and final frozen repeats; no independent source-successor restart'}
            elif 'ownership' in inherited:
                extra['ownership'] = ownership
        rows.append({
            **extra,
            "experiment": exp,
            "historical_script": str(run.relative_to(REPO)),
            "contains_legacy_or_routine": has_legacy_or,
            "adjacent_identity_precision_solver_correction": adjacent,
            "historical_source_sha256": hashlib.sha256(run.read_bytes()).hexdigest(),
            "corrected_artifact": str(artifact.relative_to(REPO)),
            "artifact_status": evidence.get("status", "absent"),
            "failure_reason": evidence.get("reason") or evidence.get('release_blocker'),
            "corrected_driver": CORRECTED_DRIVERS.get(exp) or f"tools/rerun_extensions.py {exp}",
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
            "status": "corrected_figure_complete" if (HERE / "corrected_figures" / f"{fig.removesuffix('.py')}_corrected.png").exists() else "regenerate_after_corrected_reruns",
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
    if "--check" in sys.argv:
        previous = json.loads(out.read_text())
        if {r["experiment"] for r in previous["experiments"]} != {r["experiment"] for r in rows}:
            raise SystemExit("RERUN_INVENTORY_MISMATCH")
        for row in rows:
            if row["status"] not in {"calculation_complete", "exploratory_complete"}: continue
            document = json.loads((REPO / row["corrected_artifact"]).read_text(), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"nonstandard JSON {token}")))
            if document.get("method_version") != "TC-1": raise SystemExit("METHOD_VERSION_MISMATCH")
            for name, expected in document.get("files", {}).items():
                artifact = (REPO / row["corrected_artifact"]).parent / name
                if not artifact.exists() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                    raise SystemExit(f"OUTPUT_CONTENT_MISMATCH: {artifact}")
        print("rerun inventory, strict JSON, method versions and declared completed output hashes verified; raw-data reproduction remains a separate gate")
        return report
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
