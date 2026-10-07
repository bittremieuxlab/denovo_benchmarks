import re
import pandas as pd
from pyteomics import proforma


def get_n_tokens(sequence: str) -> int:
    seq = proforma.parse(sequence)
    n_tokens = len(seq[0])
    if seq[1]["n_term"]:
        n_tokens += len(seq[1]["n_term"])
    if seq[1]["c_term"]:
        n_tokens += len(seq[1]["c_term"])
    return n_tokens

def validate_spectrum_id(spectrum_id: str) -> bool:
    # {filename}:{index}
    SPECTRUM_ID_PATTERN = r"[^:]+:\d+"
    return bool(re.fullmatch(SPECTRUM_ID_PATTERN, spectrum_id))

def validate_sequence(sequence: str) -> bool:
    try:
        seq = proforma.parse(sequence)
    except:
        return False
    return True

def validate_token_scores(scores: str, sequence: str) -> bool:
    n_tokens = get_n_tokens(sequence)
    return len(scores.split(",")) == n_tokens


output_data = pd.read_csv("test_outputs/test_output.csv")

for col in ["sequence", "score", "aa_scores", "spectrum_id"]:
    assert col in output_data, f"`{col}` must be presented."

assert (
    output_data["spectrum_id"].apply(validate_spectrum_id).all()
), "`spectrum_id` does not have expected format `{filename}:{index}`."

assert (
    output_data["sequence"].apply(validate_sequence).all()
), """
    Predicted sequences are not in the output format.
"""

assert output_data.apply(
    lambda row: validate_token_scores(row["aa_scores"], row["sequence"]),
    axis=1,
).all(), """
    Number of per-token scores (','-separated scores in `aa_scores`) 
    must match number of individual tokens in a predicted sequence.
"""
