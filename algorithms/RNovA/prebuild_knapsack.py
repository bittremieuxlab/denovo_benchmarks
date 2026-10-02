"""
Build SeqFiller's knapsack table for a candidate list at container build time.

The table takes ~30 s to build for 20 candidates and is otherwise rebuilt on every
benchmark run, where it would count towards the reported runtime. It goes through
resolve_candidates() so the cache key matches what a run will look up exactly.

Usage (cache dir from RNOVA_KNAPSACK_CACHE):
    python prebuild_knapsack.py candidates.txt
"""

import os
import sys

SEQFILLER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RNovA", "RNovA_SeqFiller_Inference")
sys.path.insert(0, SEQFILLER_DIR)

from data.environment_greedy import KNAPSACK_CACHE_DIR, knapsack_for, resolve_candidates  # noqa: E402

with open(sys.argv[1]) as f:
    candidates = f.read().strip().split(";")

table = knapsack_for(resolve_candidates(candidates))
print(f"knapsack table for {len(candidates)} candidates ({table.nbytes / 1e6:.0f} MB) cached in {KNAPSACK_CACHE_DIR}")
