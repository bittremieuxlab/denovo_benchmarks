"""Tests for the handling of predictions in evaluation/utils.py and algorithms/base/output_mapper.py.

Run from the repository root with the evaluation dependencies (see evaluation.def) and pytest:

    python -m pytest tests/test_evaluation_utils.py
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "algorithms"))
from base import OutputMapperBase
from evaluation import utils
from evaluation.ground_truth_mapper import format_sequence
from evaluation.metrics import aa_match_batch
from evaluation.token_masses import AA_MASSES


@pytest.mark.parametrize(
    "prediction, label, n_residues, match",
    [
        ("PEPTIDEK", "PEPTIDEK", 8, True),
        ("PEPC[UNIMOD:4]M[UNIMOD:35]K", "PEPC[57.021]M[15.995]K", 6, True),
        ("PEPM[+15.995]K", "PEPM[15.995]K", 5, True),
        ("PEPM[Oxidation]K", "PEPM[15.995]K", 5, True),
        ("[UNIMOD:1]-GATVHGHSVR", "G[42.011]ATVHGHSVR", 10, True),
        ("[UNIMOD:5][UNIMOD:385]-QAATMESLTPLLINLAK", "Q[25.979]AATMESLTPLLINLAK", 17, True),
        ("[UNIMOD:385][UNIMOD:5]-QAATMESLTPLLINLAK", "Q[25.979]AATMESLTPLLINLAK", 17, True),
        ("[UNIMOD:1][UNIMOD:385]-M[UNIMOD:35]PEPK", "M[40.979]PEPK", 5, True),
        ("PEPTIDEK-[UNIMOD:2]", "PEPTIDEK[-0.984]", 8, True),
        ("PEPM[UNIMOD:35][UNIMOD:1]K", "PEPM[58.005]K", 5, True),
        ("[UNIMOD:5][UNIMOD:385]-QAATMESLTPLLINLAK", "QAATMESLTPLLINLAK", 17, False),
    ],
)
def test_prediction_is_matched_against_its_label(prediction, label, n_residues, match):
    """A valid prediction goes through the same steps as in evaluate.py: one score per token,
    conversion to delta mass notation (terminal and stacked modifications added to the residue,
    their scores merged), then matching against the label."""
    scores = ",".join(["0.5"] * utils.get_n_tokens(prediction))
    assert utils.validate_sequence(prediction)
    assert utils.validate_token_scores(scores, prediction)

    sequence, scores = utils.ptms_to_delta_mass(prediction, scores)
    assert len(scores.split(",")) == n_residues
    batch, n_aa_pred, n_aa_true = aa_match_batch([sequence], [format_sequence(label)], AA_MASSES)
    assert n_aa_pred == n_residues
    assert batch[0][1] == match


@pytest.mark.parametrize(
    "sequence, valid",
    [
        ("PEPTIDEK", True),
        ("[UNIMOD:5][UNIMOD:385]-QAATMESLTPLLINLAK", True),
        ("PEPTIDEK-[UNIMOD:2]", True),
        ("PEPM[UNIMOD:35][UNIMOD:1]K", True),
        ("[UNIMOD:5]-[UNIMOD:385]-QAATMESLTPLLINLAK", False),  # two hyphens: not ProForma
        ("PEP)TIDEK", False),
        ("PEPUK", False),  # residue without a mass in AA_MASSES
        ("PEP[not a modification]K", False),  # modification without a mass
        ("[UNIMOD:21]?PEPTIDEK", False),  # unlocalized
        ("{Glycan:Hex}PEPTIDEK", False),  # labile
        ("<[UNIMOD:4]@C>PEPCK", False),  # global fixed
        ("PR(TEIN)[+19.05]K", False),  # on a range
        ("<13C>PEPTIDEK", False),  # isotope label
    ],
)
def test_validate_sequence(sequence, valid):
    """Invalid sequences are counted as no prediction by evaluate.py."""
    assert utils.validate_sequence(sequence) == valid


def test_prediction_without_aa_scores_is_not_counted():
    """A prediction with an empty aa_scores cell is counted as no prediction (#95)."""
    output_data = pd.DataFrame({"sequence": ["PEPTIDEK"], "aa_scores": [np.nan], "spectrum_id": ["run_0:0"]})
    assert utils.get_sequenced_idx(output_data).tolist() == [False]


def test_output_mapper_fills_missing_aa_scores():
    """The output mapper gives a row without per-token scores the peptide score for each token."""
    output_data = pd.DataFrame({"sequence": ["PEPK", "PEPM[UNIMOD:35]K"], "score": [0.9, 0.8], "aa_scores": [np.nan, "0.1,0.2,0.3,0.4,0.5"]})
    output_data = OutputMapperBase().format_output(output_data)
    assert output_data["aa_scores"].tolist() == ["0.9,0.9,0.9,0.9", "0.1,0.2,0.3,0.4,0.5"]
