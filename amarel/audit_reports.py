"""Run only the 'reports' stage of the audit from saved tables; needs no data, models or GPU.

Usage: python amarel/audit_reports.py <results_v2 folder, read only> <new output folder>
"""
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from notebook_loader import REPORT_NAMES, load_names  # noqa: E402


def main():
    source, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if output == source or source in output.parents:
        raise SystemExit("The output folder must be outside the results folder.")
    names = {"np": np, "pd": pd, "json": json, "hashlib": hashlib, "Path": Path, "product": product, "AUDIT_DIR": output}
    load_names(REPORT_NAMES, names)
    before = names["tree_manifest"](source)
    names["run_reports"](source)
    fingerprints = names["verify_fingerprints"](source)
    names["save_audit"](fingerprints, "v2_fingerprints")
    assert fingerprints["reproduced"].all(), "A saved configuration does not match its fingerprint."
    assert names["tree_manifest"](source).equals(before), "The Version 2 results changed."
    print(f"{len(fingerprints)} fingerprints reproduced; {len(before)} result files unchanged. Tables: {output}/aggregate")


if __name__ == "__main__":
    main()
