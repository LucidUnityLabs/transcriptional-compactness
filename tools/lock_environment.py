"""Resolve ONCE on the target interpreter/platform and retain hashed wheels.

Do not run in release CI: CI consumes the reviewed lock. Resolution is an
explicit maintenance operation. No hashes or wheel availability are invented
(audit R06; reference implementation ported verbatim from the audit's
appendix). Example:

    python tools/lock_environment.py \
        --requirements requirements/core.in \
        --wheelhouse wheelhouse/core-cp314-linux-x86_64 \
        --lock requirements/core-cp314-linux-x86_64.lock

Review the generated lock + manifest, install/test from the wheelhouse in a
fresh target venv, and only then commit them as the reviewed lock.
"""
import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    if args.lock.exists() or args.wheelhouse.exists():
        raise SystemExit(
            "Refusing to overwrite a lock/wheelhouse; use new versioned "
            "paths")
    with tempfile.TemporaryDirectory() as temporary:
        subprocess.run([sys.executable, "-m", "pip", "download",
                        "--only-binary=:all:", "--dest", temporary,
                        "-r", str(args.requirements)], check=True)
        wheels = sorted(Path(temporary).glob("*.whl"))
        if not wheels:
            raise SystemExit("No wheels resolved")
        entries = {}
        files = []
        for wheel in wheels:
            with ZipFile(wheel) as archive:
                metadata = [n for n in archive.namelist()
                            if n.endswith(".dist-info/METADATA")]
                if len(metadata) != 1:
                    raise SystemExit(
                        f"Unexpected wheel metadata: {wheel.name}")
                parsed = BytesParser().parsebytes(archive.read(metadata[0]))
            name = re.sub(r"[-_.]+", "-", parsed["Name"]).lower()
            version = parsed["Version"]
            if name in entries:
                raise SystemExit(
                    f"Multiple candidate wheels for {name}; refuse "
                    "ambiguous lock")
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            entries[name] = f"{name}=={version} --hash=sha256:{digest}"
            files.append({"filename": wheel.name, "sha256": digest,
                          "bytes": wheel.stat().st_size})
        args.wheelhouse.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(temporary, args.wheelhouse)
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "# Generated for this interpreter/platform; review before "
            "committing.",
            f"# Python: {sys.version.split()[0]}",
            f"# Platform: {platform.platform()}",
            "# Install only with --require-hashes --no-index "
            "--find-links <wheelhouse>."]
        args.lock.write_text(
            "\n".join(header + [entries[key] for key in sorted(entries)])
            + "\n")
        manifest = {
            "python": sys.version, "platform": platform.platform(),
            "files": files,
            "requirements_input": str(args.requirements),
            "status": "UNREVIEWED_CANDIDATE"}
        args.lock.with_suffix(
            args.lock.suffix + ".manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n")
    print(f"Created {args.lock} and {args.wheelhouse}. Install/test in a "
          "fresh environment, then review and commit.")


if __name__ == "__main__":
    main()
