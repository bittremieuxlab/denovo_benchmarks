"""Unit tests for get_sequenced_idx in evaluation/utils.py.

Run from the repository root with the evaluation dependencies from evaluation.def installed, plus pytest:

    python -m pytest tests/test_sequenced_idx.py
"""

import io

import pandas as pd
import pytest

from evaluation import utils

SCORES_9 = ",".join(["-0.1"] * 9)


def read_output(csv: str) -> pd.DataFrame:
    """Read an output.csv the way evaluate.py does, so an empty cell becomes NaN."""
    return pd.read_csv(io.StringIO("sequence,score,aa_scores,spectrum_id\n" + csv))


def test_missing_aa_scores_marks_the_row_failed_instead_of_crashing(capsys):
    output_data = read_output(
        "PEPTIDEK,0.9,,run_0:0\n"  # valid sequence, empty aa_scores
        f'ACDEFGHIK,0.8,"{SCORES_9}",run_0:1\n'
    )
    assert pd.isna(output_data.loc[0, "aa_scores"])

    sequenced_idx = utils.get_sequenced_idx(output_data)

    assert sequenced_idx.tolist() == [False, True]
    printed = capsys.readouterr().out
    assert "run_0:0" in printed
    assert "number of scores: 0" in printed


def test_empty_string_aa_scores_marks_the_row_failed():
    output_data = read_output(f'PEPTIDEK,0.9,"",run_0:0\nACDEFGHIK,0.8,"{SCORES_9}",run_0:1\n')
    assert utils.get_sequenced_idx(output_data).tolist() == [False, True]


def test_rows_without_a_sequence_are_not_sequenced():
    output_data = read_output(f',,,run_0:0\nACDEFGHIK,0.8,"{SCORES_9}",run_0:1\n')
    assert utils.get_sequenced_idx(output_data).tolist() == [False, True]


def test_wrong_number_of_scores_is_still_reported():
    output_data = read_output('PEPTIDEK,0.9,"-0.1,-0.1",run_0:0\n')
    assert utils.get_sequenced_idx(output_data).tolist() == [False]


@pytest.mark.parametrize("scores, expected", [("-0.1,-0.2,-0.3", 3), ("-0.1", 1), ("", 0), (float("nan"), 0), (None, 0)])
def test_get_n_scores(scores, expected):
    assert utils.get_n_scores(scores) == expected
