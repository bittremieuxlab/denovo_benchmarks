"""Check that the newest versions.log entry of each algorithm has an upstream_ref matching its container.def.

Usage: python check_versions_log.py [algorithm ...]   (default: all algorithms)

- upstream_ref is required.
- "git: <repo>@<commit>": if container.def fetches <repo>, it must use that commit.
- "pip: <package>==<version>" and "docker: <image>": container.def must contain them.

Examples (upstream_ref in versions.log -> container.def -> result):

    "git: https://github.com/statisticalbiotechnology/pairwise.git@adad353"
        git clone https://github.com/statisticalbiotechnology/pairwise.git
        git -C pairwise checkout adad3537405320a8695a15f4a7c4128d43038af0
        -> OK: the recorded commit is checked out.

    "git: https://github.com/nh2tran/DeepNovo.git@bb5840f"
        curl -fL https://github.com/nh2tran/DeepNovo/archive/8b0ec8321d1006770032d069d76234fc95d36af3.tar.gz
        -> "deepnovo: container.def fetches nh2tran/DeepNovo but not at the recorded commit bb5840f"
           (the build uses another commit than the record says).

    "git: https://github.com/statisticalbiotechnology/pairwise.git@adad353"
        git clone https://github.com/statisticalbiotechnology/pairwise.git    (no checkout)
        -> "pairwise: container.def fetches statisticalbiotechnology/pairwise but not at the recorded commit adad353"
           (the build takes whatever the default branch is at build time).

    "git: https://github.com/BEAM-Labs/ContraNovo.git@503fdb3"
        # git clone https://github.com/BEAM-Labs/ContraNovo.git    (comment; the code is in the repository)
        -> OK: container.def does not fetch the repository, so there is nothing to compare.

    "pip: casanovo==5.0.0"
        pip install --no-cache-dir "casanovo==5.0.0" "torch<2.6"
        -> OK.
    "pip: casanovo==5.0.0"
        pip install casanovo
        -> "casanovo: container.def does not contain casanovo==5.0.0"

    "docker: rapidnovor/novor-docker@sha256:2fffc4b0..."
        From: rapidnovor/novor-docker@sha256:2fffc4b0...
        -> OK.
    "docker: rapidnovor/novor-docker@sha256:2fffc4b0..."
        From: rapidnovor/novor-docker:novorai
        -> "novor: container.def does not contain rapidnovor/novor-docker@sha256:2fffc4b0..."

    "vendored: code added by the PepNet authors in f7cf6b0"
        -> OK: only has to be present.

    no upstream_ref, "", or the template's "<reference>"
        -> "pepnet: no upstream_ref in the newest entry"
"""

import os
import re
import sys

ALGORITHMS_DIR = "algorithms"


def newest_entry(versions_log):
    """Return the fields of the first (newest) entry of a versions.log file."""
    entry = {}
    with open(versions_log) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            m = re.match(r'\s*-?\s*(\w+):\s*"([^"]*)"', line)
            if not m:
                continue
            if m.group(1) == "container_version" and entry:
                break
            entry[m.group(1)] = m.group(2)
    return entry


def definition_text(container_def):
    """Return container.def without comments."""
    with open(container_def) as f:
        return "\n".join(re.sub(r"(^|\s)#.*", "", line) for line in f)


def check(algorithm):
    """Return a list of problems with the newest versions.log entry of the algorithm."""
    algo_dir = os.path.join(ALGORITHMS_DIR, algorithm)
    ref = newest_entry(os.path.join(algo_dir, "versions.log")).get("upstream_ref", "").strip()
    if not ref or "<" in ref:
        return ["no upstream_ref in the newest entry"]
    container_def = os.path.join(algo_dir, "container.def")
    if not os.path.exists(container_def):
        return []
    text = definition_text(container_def)
    kind, _, value = ref.partition(":")
    value = value.strip()
    if kind == "git":
        m = re.search(r"github\.com/([\w.-]+/[\w-]+?)(?:\.git)?@([0-9a-f]{7,40})", value)
        if m and m.group(1).lower() in text.lower() and m.group(2) not in text:
            return [f"container.def fetches {m.group(1)} but not at the recorded commit {m.group(2)}"]
    elif kind in ("pip", "docker") and value.split()[0] not in text:
        return [f"container.def does not contain {value.split()[0]}"]
    return []


if __name__ == "__main__":
    algorithms = sys.argv[1:] or sorted(
        a for a in os.listdir(ALGORITHMS_DIR)
        if os.path.exists(os.path.join(ALGORITHMS_DIR, a, "versions.log"))
    )
    failed = False
    for algorithm in algorithms:
        for problem in check(algorithm):
            print(f"{algorithm}: {problem}")
            failed = True
    sys.exit(1 if failed else 0)
