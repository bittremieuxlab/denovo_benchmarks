import pandas as pd

from evaluation.utils import validate_sequence, validate_spectrum_id, validate_token_scores


output_data = pd.read_csv("test_outputs/test_output.csv")

for col in ["sequence", "score", "aa_scores", "spectrum_id"]:
    assert col in output_data, f"`{col}` must be presented."

assert (
    output_data["spectrum_id"].apply(validate_spectrum_id).all()
), "`spectrum_id` does not have expected format `{filename}:{index}`."

predicted = output_data[output_data["sequence"].notnull()]
valid_sequence = predicted["sequence"].apply(validate_sequence)
valid_scores = predicted[valid_sequence].apply(
    lambda row: validate_token_scores(row["aa_scores"], row["sequence"]), axis=1
)
n_valid = int(valid_scores.sum()) if len(valid_scores) else 0
print(f"Predictions: {len(output_data)} rows, {len(predicted)} predicted sequences, {n_valid} valid.")

# Invalid predictions are reported as a warning
invalid_sequences = predicted.loc[~valid_sequence, "sequence"]
invalid_scores = predicted[valid_sequence].loc[~valid_scores, "sequence"] if len(valid_scores) else []
if len(invalid_sequences):
    print(f"WARNING: {len(invalid_sequences)} predicted sequences are not valid (see the output sequence format in the README), e.g.:")
    for sequence in invalid_sequences[:5]:
        print(f"    {sequence}")
if len(invalid_scores):
    print(f"WARNING: {len(invalid_scores)} predictions have a number of per-token scores (`aa_scores`) "
          "that does not match the number of tokens in the sequence, e.g.:")
    for sequence in invalid_scores[:5]:
        print(f"    {sequence}")

assert n_valid > 0, "No valid predicted sequences in the output."
