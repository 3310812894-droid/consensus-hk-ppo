"""Create a source/data archive, excluding runs, outputs, and model files."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

from check_release import ROOT, included_files, main as check_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "consensus_hk_paper_code_v1.zip")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Archive already exists: {output}")
    if ROOT == output.parent or ROOT in output.parents:
        raise ValueError("Keep the distribution ZIP outside the repository")
    check_release()
    files = sorted(included_files())
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, "consensus-hk-ppo/" + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or len(archive.infolist()) != len(files):
            raise RuntimeError("Archive validation failed")
    print(f"Archive: {output.name}; files={len(files)}; bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
