"""Record what each evaluation result was computed with, in results/provenance.csv.

One row per dataset, algorithm and version in results/. evaluate.py replaces the rows of the algorithms it evaluates
and keeps the others, like the plot data files, so the file describes the results as they are now.
"""

import datetime
import hashlib
import os
import subprocess
from importlib import metadata

import yaml

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROVENANCE_FILE = "provenance.csv"
KEY_COLS = ["dataset", "algorithm", "version"]
PACKAGES = ["numpy", "pandas", "scikit-learn", "pyteomics", "koinapy"]


def sha256(path):
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(*paths):
    """Commit of this repository, with "-dirty" if the given paths have uncommitted changes ("" without git)."""
    git = ["git", "-c", "safe.directory=*", "-C", REPO_DIR]
    try:
        commit = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        changes = subprocess.run(
            git + ["status", "--porcelain", "--untracked-files=no", "--", *paths],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return commit + ("-dirty" if changes else "")


def _package_version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _upstream_ref(algo_name, algo_version):
    """upstream_ref of this container_version in algorithms/<algorithm>/versions.log ("" if not there)."""
    path = os.path.join(REPO_DIR, "algorithms", algo_name, "versions.log")
    if not os.path.isfile(path):
        return ""
    with open(path) as f:
        entries = yaml.safe_load(f) or []
    for entry in entries:
        if str(entry.get("container_version")) == algo_version:
            return entry.get("upstream_ref") or ""
    return ""


def evaluation_info(labels_path, proteome, intensity_model, rt_model):
    """The columns that are the same for all algorithms evaluated in one run of evaluate.py."""
    return {
        "evaluated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evaluation_commit": git_commit("evaluation"),
        "evaluation_packages": ", ".join(f"{package} {_package_version(package)}" for package in PACKAGES),
        "labels_sha256": sha256(labels_path),
        "proteome": proteome,
        "intensity_model": intensity_model,
        "rt_model": rt_model,
    }


def provenance_row(dataset_name, algo_name, algo_version, output_path, evaluation):
    """One row of provenance.csv for the predictions in output_path.

    The prediction columns come from prediction_info.yaml next to output.csv, which run.sh writes when it makes
    the predictions (and which is written by hand for algorithms that are run manually).
    """
    info_path = os.path.join(os.path.dirname(output_path), "prediction_info.yaml")
    info = {}
    if os.path.isfile(info_path):
        with open(info_path) as f:
            info = yaml.safe_load(f) or {}
    return {
        "dataset": dataset_name,
        "algorithm": algo_name,
        "version": algo_version,
        "upstream_ref": _upstream_ref(algo_name, algo_version),
        **{column: str(info.get(key) or "") for column, key in [
            ("predicted", "predicted"),
            ("prediction_commit", "commit"),
            ("image_sha256", "image_sha256"),
            ("prediction_notes", "notes"),
        ]},
        "predictions_sha256": sha256(output_path),
        **evaluation,
    }
