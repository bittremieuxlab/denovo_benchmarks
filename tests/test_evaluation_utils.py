"""Unit tests for the ProForma handling in evaluation/utils.py and evaluation/metrics.py.

Run from the repository root with the evaluation dependencies installed (see evaluation.def):

    python -m pytest tests/test_evaluation_utils.py

pyteomics resolves Unimod accessions on first use, which needs network access.
"""

import pytest

from evaluation import utils
from evaluation.ground_truth_mapper import format_sequence
from evaluation.metrics import aa_match_batch, get_token_mass
from evaluation.token_masses import AA_MASSES

CARBAMYL = 43.005814
AMMONIA_LOSS = -17.026549
ACETYL = 42.010565
OXIDATION = 15.994915
AMIDATION = -0.984016


def scores_for(sequence):
    return ",".join(str(float(i)) for i in range(utils.get_n_tokens(sequence)))


def delta(token):
    """The modification mass of a delta-mass token such as `Q[+25.979265]`, 0 for a bare residue."""
    return get_token_mass(token, AA_MASSES) - AA_MASSES[token[0]]


def convert(sequence):
    sequence_out, scores_out = utils.ptms_to_delta_mass(sequence, scores_for(sequence))
    tokens = [sequence_out[i] + sequence_out[i + 1 : j] for i, j in _token_bounds(sequence_out)]
    return tokens, [float(s) for s in scores_out.split(",")]


def _token_bounds(sequence):
    starts = [i for i, c in enumerate(sequence) if c.isupper() and (i == 0 or sequence[i - 1] != "[")]
    return list(zip(starts, starts[1:] + [len(sequence)]))


def matches(prediction, label):
    sequence, _ = utils.ptms_to_delta_mass(prediction, scores_for(prediction))
    batch, _, _ = aa_match_batch([sequence], [format_sequence(label)], AA_MASSES)
    return batch[0][1]


# ---- stacked N-terminal groups ------------------------------------------------------------------


def test_stacked_nterm_group_is_one_scored_token():
    # 17 residues plus one token for the whole N-terminal group, however many tags it holds
    assert utils.get_n_tokens("[UNIMOD:5][UNIMOD:385]-QAATMESLTPLLINLAK") == 18
    assert utils.get_n_tokens("[UNIMOD:385]-QAATMESLTPLLINLAK") == 18
    assert utils.get_n_tokens("QAATMESLTPLLINLAK") == 17


def test_stacked_nterm_group_is_folded_into_the_first_residue():
    tokens, scores = convert("[UNIMOD:5][UNIMOD:385]-QAATMESLTPLLINLAK")
    assert len(tokens) == 17
    assert tokens[0].startswith("Q[")
    assert delta(tokens[0]) == pytest.approx(CARBAMYL + AMMONIA_LOSS, abs=1e-6)
    assert all(delta(t) == 0 for t in tokens[1:])
    # the group's score and the first residue's score are merged into one
    assert scores == [0.5] + [float(i) for i in range(2, 18)]


def test_stacked_nterm_group_also_sums_a_first_residue_modification():
    tokens, _ = convert("[UNIMOD:1][UNIMOD:385]-M[UNIMOD:35]PEPK")
    assert delta(tokens[0]) == pytest.approx(ACETYL + AMMONIA_LOSS + OXIDATION, abs=1e-6)


@pytest.mark.parametrize(
    "prediction, label",
    [
        ("[UNIMOD:5][UNIMOD:385]-QAATMESLTPLLINLAK", "Q[25.979]AATMESLTPLLINLAK"),
        ("[UNIMOD:385][UNIMOD:5]-QAATMESLTPLLINLAK", "Q[25.979]AATMESLTPLLINLAK"),
        ("[UNIMOD:5][UNIMOD:385]-DITTLDATVM[UNIMOD:35]AAK", "D[25.979]ITTLDATVM[15.995]AAK"),
    ],
)
def test_correct_stacked_prediction_matches_its_label(prediction, label):
    assert matches(prediction, label)


# ---- behaviour that must not change -------------------------------------------------------------


@pytest.mark.parametrize(
    "prediction, label",
    [
        ("[UNIMOD:385]-QAATMESLTPLLINLAK", "Q[-17.027]AATMESLTPLLINLAK"),
        ("[UNIMOD:1]-GATVHGHSVR", "G[42.011]ATVHGHSVR"),
        ("GATVHGHSVR", "GATVHGHSVR"),
        ("PEPC[UNIMOD:4]M[UNIMOD:35]K", "PEPC[57.021]M[15.995]K"),
        ("PEPM[+15.995]K", "PEPM[15.995]K"),
    ],
)
def test_single_and_unmodified_predictions_still_match(prediction, label):
    assert matches(prediction, label)


def test_single_nterm_modification_conversion():
    tokens, scores = convert("[UNIMOD:1]-GATVHGHSVR")
    assert delta(tokens[0]) == pytest.approx(ACETYL, abs=1e-6)
    assert scores == [0.5] + [float(i) for i in range(2, 11)]


def test_wrong_prediction_does_not_match():
    assert not matches("[UNIMOD:5][UNIMOD:385]-QAATMESLTPLLINLAK", "QAATMESLTPLLINLAK")


# ---- other terminal and residue groups ----------------------------------------------------------


def test_cterm_group_is_folded_into_the_last_residue():
    tokens, scores = convert("PEPTIDEK-[UNIMOD:2]")
    assert len(tokens) == 8
    assert delta(tokens[-1]) == pytest.approx(AMIDATION, abs=1e-6)
    assert scores == [float(i) for i in range(7)] + [7.5]


def test_stacked_residue_modifications_are_summed():
    tokens, scores = convert("PEPM[UNIMOD:35][UNIMOD:1]K")
    assert delta(tokens[3]) == pytest.approx(OXIDATION + ACETYL, abs=1e-6)
    assert len(scores) == 5


# ---- validation ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sequence",
    [
        "PEPTIDEK",
        "[UNIMOD:1]-PEPTIDEK",
        "[UNIMOD:5][UNIMOD:385]-QAATMESLTPLLINLAK",
        "PEPM[UNIMOD:35][UNIMOD:1]K",
        "PEPTIDEK-[UNIMOD:2]",
        "PEPM[+15.995]K",
    ],
)
def test_valid_sequences(sequence):
    assert utils.validate_sequence(sequence)
    assert utils.validate_token_scores(scores_for(sequence), sequence)


@pytest.mark.parametrize(
    "sequence",
    [
        "[UNIMOD:5]-[UNIMOD:385]-QAATMESLTPLLINLAK",  # two hyphens: not ProForma
        "PEP[not a modification]K",  # no mass for the modification
        "PEPUK",  # no mass for the residue in AA_MASSES
        "PEP)TIDEK",
    ],
)
def test_invalid_sequences(sequence):
    assert not utils.validate_sequence(sequence)


# ---- token masses -------------------------------------------------------------------------------


def test_token_mass():
    assert get_token_mass("M", AA_MASSES) == AA_MASSES["M"]
    assert get_token_mass("M[+15.994915]", AA_MASSES) == pytest.approx(AA_MASSES["M"] + OXIDATION)
    assert get_token_mass("Q[-17.026549]", AA_MASSES) == pytest.approx(AA_MASSES["Q"] + AMMONIA_LOSS)


@pytest.mark.parametrize("token", ["[+43.005814][-17.026549]-", "U", "M[oxidation]"])
def test_unreadable_token_raises_instead_of_weighing_nothing(token):
    with pytest.raises(ValueError):
        get_token_mass(token, AA_MASSES)
