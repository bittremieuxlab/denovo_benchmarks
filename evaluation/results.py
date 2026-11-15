"""Save evaluation results by merging them into existing results files."""

import os
import pandas as pd


def _row_keys(data, key_cols):
    """Key of each row: values of key_cols (or the index if key_cols is None), as strings."""
    if key_cols is None:
        return [(str(idx),) for idx in data.index]
    return [tuple(str(value) for value in row) for row in data[key_cols].itertuples(index=False)]


def merge_results(new_data, results_path, key_cols=None):
    """
    Merge new results into the existing results file (if any).

    - existing rows with the same key as a new row are replaced in place
    - new rows with keys not present in the file are appended at the end
    - all other existing rows are kept unchanged

    If key_cols is None, the index is used as the key.
    """
    if not os.path.isfile(results_path):
        return new_data

    # Read everything as strings, so that kept rows are written back unchanged
    old_data = pd.read_csv(
        results_path,
        dtype=str,
        keep_default_na=False,
        index_col=0 if key_cols is None else None,
    )
    new_keys = _row_keys(new_data, key_cols)
    new_rows = {key: new_data.iloc[[i]] for i, key in enumerate(new_keys)}

    merged_rows = []
    for i, key in enumerate(_row_keys(old_data, key_cols)):
        merged_rows.append(new_rows.pop(key) if key in new_rows else old_data.iloc[[i]])
    merged_rows.extend(new_rows.values())

    if not merged_rows:
        return old_data
    return pd.concat(merged_rows)


def save_results(new_data, results_path, key_cols=None):
    """Merge new results into the results file and save it."""
    merged_data = merge_results(new_data, results_path, key_cols)
    merged_data.to_csv(results_path, index=key_cols is None)
