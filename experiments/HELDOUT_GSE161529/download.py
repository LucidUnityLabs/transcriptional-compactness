"""Download per-sample matrix.mtx.gz + barcodes.tsv.gz for the 27 label samples.

Parses GSE161529_family.soft.gz for the GSM -> supplementary mapping, matches
label sample IDs (e.g. TN_B1_0177 -> GSM4909288_TN-B1-MH0177) by
subtype + patient number + optional suffix, curls over HTTPS with retries,
verifies gzip integrity, writes samples/MANIFEST.json.
"""

import gzip
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data"
SOFT = DATA / "GSE161529" / "GSE161529_family.soft.gz"
LABELS = DATA / "GSE161529" / "labels"
OUT = DATA / "GSE161529" / "samples"
OUT.mkdir(exist_ok=True)
FEATURES_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161529/"
                "suppl/GSE161529_features.tsv.gz")


def parse_soft():
    """Return list of (gsm, stem) where stem is like 'GSM4909288_TN-B1-MH0177'."""
    supp = {}
    gsm = None
    with gzip.open(SOFT, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith("^SAMPLE = "):
                gsm = line.split(" = ", 1)[1]
            elif line.startswith("!Sample_supplementary_file_") and gsm:
                url = line.split(" = ", 1)[1]
                fname = url.rsplit("/", 1)[-1]
                if fname.endswith("-matrix.mtx.gz"):
                    supp[gsm] = fname[: -len("-matrix.mtx.gz")]
    return supp


def label_samples():
    s = set()
    for lab in ("cells_primary.tsv.gz", "cells_secondary.tsv.gz"):
        with gzip.open(LABELS / lab, "rt") as f:
            hdr = f.readline()
            for line in f:
                s.add(line.split("\t")[1])
    return sorted(s)


def map_sample(sample, supp):
    """Map label sample ID to (gsm, stem)."""
    m = re.match(r"^(ER|HER2|TN)(_B1)?_(\d{4})(_[A-Z]+\d*)?$", sample)
    if not m:
        return None
    subtype, b1, num, suf = m.group(1), m.group(2) or "", m.group(3), m.group(4) or ""
    want = subtype + ("-B1" if b1 else "")
    suf = suf.lstrip("_")
    cands = []
    for gsm, stem in supp.items():
        body = stem.split("_", 1)[1]
        if not body.startswith(want + "-"):
            continue
        core = body[len(want) + 1:]
        pat = (r"^[A-Za-z]*" + num + r"(-" + suf + r")?$") if suf \
            else (r"^[A-Za-z]*" + num + r"$")
        if re.match(pat, core):
            cands.append((gsm, stem))
    if len(cands) == 1:
        return cands[0]
    return None if not cands else ("AMBIGUOUS", cands)


def curl(url, dest):
    cmd = ["curl", "-f", "-sS", "--retry", "5", "--retry-delay", "5",
           "--retry-all-errors", "-C", "-", "-o", str(dest), url]
    for attempt in range(3):
        r = subprocess.run(cmd)
        if r.returncode == 0:
            return True
        time.sleep(5)
    return False


def main():
    t0 = time.time()
    supp = parse_soft()
    samples = label_samples()
    print(f"soft: {len(supp)} GSMs with matrix; labels: {len(samples)} samples")

    mapping = {}
    for s in samples:
        r = map_sample(s, supp)
        if r is None or r[0] == "AMBIGUOUS":
            raise SystemExit(f"MAPPING FAILURE for {s}: {r}")
        mapping[s] = {"gsm": r[0], "stem": r[1]}
    print("mapping 1:1 for all samples:")
    for s in samples:
        print(f"  {s:14s} -> {mapping[s]['stem']}")

    manifest = {"downloaded_bytes": 0, "files": [], "errors": []}

    # features file (gene ids shared across samples)
    feats_dest = OUT / "GSE161529_features.tsv.gz"
    if not feats_dest.exists() or subprocess.run(["gzip", "-t", str(feats_dest)]).returncode != 0:
        print(f"downloading features ...", flush=True)
        if not curl(FEATURES_URL, feats_dest):
            manifest["errors"].append(FEATURES_URL)
    subprocess.run(["gzip", "-t", str(feats_dest)], check=True)
    manifest["files"].append({"name": feats_dest.name,
                              "bytes": feats_dest.stat().st_size})

    for s in samples:
        gsm = mapping[s]["gsm"]
        stem = mapping[s]["stem"]
        base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4909nnn/{gsm}/suppl/"
        for kind in ("matrix.mtx.gz", "barcodes.tsv.gz"):
            url = base + f"{stem}-{kind}"
            dest = OUT / f"{stem}-{kind}"
            if dest.exists() and subprocess.run(["gzip", "-t", str(dest)]).returncode == 0:
                print(f"  [skip] {dest.name}", flush=True)
            else:
                print(f"  GET {dest.name}", flush=True)
                if not curl(url, dest):
                    manifest["errors"].append(url)
                    continue
                subprocess.run(["gzip", "-t", str(dest)], check=True)
                print(f"    ok {dest.stat().st_size:,} B", flush=True)
            manifest["files"].append({"sample": s, "gsm": gsm,
                                      "name": dest.name,
                                      "bytes": dest.stat().st_size})
            manifest["downloaded_bytes"] += dest.stat().st_size

    with open(OUT / "MANIFEST.json", "w") as f:
        json.dump({"mapping": mapping, "manifest": manifest}, f, indent=2)

    total = sum(x["bytes"] for x in manifest["files"])
    print(f"\nDONE: {len(manifest['files'])} files, {total:,} B total, "
          f"errors={manifest['errors']}, {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
