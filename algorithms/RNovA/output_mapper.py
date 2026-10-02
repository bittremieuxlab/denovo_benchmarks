"""
Script to convert predictions from the RNovA SeqFiller output format
to the common output format.

SeqFiller writes ``title,sequence,score`` where ``title`` is the SCANS value set by
input_mapper.py (already ``{filename}:{index}``), ``sequence`` carries modifications
as ``X[UniMod:n]`` for table residues or ``X[delta_mass]`` for delta-mass candidates,
and ``score`` holds one ';'-separated log score per residue.
"""

import argparse
import math
import re

import pandas as pd
from base import OutputMapperBase

# Delta masses a SeqFiller candidate may carry as "X[delta]" -> Unimod accession, for the
# ProForma output. The default candidates (candidates.txt) only use table names
# (C|UniMod:4, M|UniMod:35); this covers delta-mass candidates set via RNOVA_CANDIDATES.
DELTA_TO_UNIMOD = {
    57.02146: 4,     # Carbamidomethyl
    15.99491: 35,    # Oxidation
    0.98402: 7,      # Deamidation
    42.01057: 1,     # Acetyl
    79.96633: 21,    # Phospho
    14.01565: 34,    # Methyl
    28.03130: 36,    # Dimethyl
    42.04695: 37,    # Trimethyl
    114.04293: 121,  # GlyGly
    229.16293: 737,  # TMT6plex
    -17.02655: 28,   # Gln->pyro-Glu
}

DELTA_TOL = 0.005
TOKEN_PATTERN = re.compile(r"([A-Z])(?:\[([^\]]*)\])?")
# Floor for residues SeqFiller scores as -inf (no admissible residue); real scores
# fall roughly within [-10, 3], so this keeps them ranked last.
SCORE_FLOOR = -100.0


def format_mod(mod):
    """One SeqFiller modification label -> ProForma bracket content."""
    if mod.startswith("UniMod:"):
        return "UNIMOD:" + mod.split(":", 1)[1]
    delta = float(mod)
    for mass, accession in DELTA_TO_UNIMOD.items():
        if abs(delta - mass) < DELTA_TOL:
            return f"UNIMOD:{accession}"
    return f"{delta:+.4f}"


class OutputMapper(OutputMapperBase):
    def format_sequence_and_scores(self, sequence, aa_scores):
        tokens = []
        for residue, mods in TOKEN_PATTERN.findall(sequence):
            tokens.append(residue + "".join(f"[{format_mod(m)}]" for m in mods.split("|") if m))
        scores = [float(s) for s in aa_scores.split(";")]
        if len(scores) != len(tokens):
            raise ValueError(f"{len(scores)} scores for {len(tokens)} residues in {sequence}")
        scores = [s if math.isfinite(s) else SCORE_FLOOR for s in scores]
        return "".join(tokens), self._format_scores(scores)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path", required=True, help="The path to the algorithm predictions file."
)
parser.add_argument(
    "--result_path", default="outputs.csv", help="Where to write the common-format predictions."
)
args = parser.parse_args()

output_data = pd.read_csv(args.output_path, dtype={"title": str, "sequence": str, "score": str})
output_data = output_data.rename({"title": "spectrum_id", "score": "aa_scores"}, axis=1)

# A spectrum whose every residue is -inf got no admissible sequence at all:
# leave it unsequenced instead of reporting a filler peptide.
no_solution = output_data["aa_scores"].apply(
    lambda s: all(not math.isfinite(float(x)) for x in s.split(";"))
)
print(f"{int(no_solution.sum())} spectra without an admissible sequence dropped.")
output_data = output_data[~no_solution].copy()

output_mapper = OutputMapper()
output_data = output_mapper.format_output(output_data)
# Peptide score: mean of the per-residue scores. It ranks peptides and picks the winning
# isotope copy.
output_data["score"] = output_data["aa_scores"].apply(
    lambda s: sum(map(float, s.split(","))) / len(s.split(","))
)

# Isotope-shifted copies (input_mapper.py --isotopes): keep the best-scoring one.
if output_data["spectrum_id"].str.contains("|iso", regex=False).any():
    parts = output_data["spectrum_id"].str.rsplit("|iso", n=1, expand=True)
    output_data["spectrum_id"], output_data["isotope"] = parts[0], parts[1].astype(int)
    output_data = output_data.sort_values("score", ascending=False).drop_duplicates("spectrum_id")
    print("isotope shift chosen:", output_data["isotope"].value_counts().sort_index().to_dict())

output_data[["spectrum_id", "sequence", "score", "aa_scores"]].to_csv(args.result_path, index=False)
print(f"{len(output_data)} predictions written to {args.result_path}.")
