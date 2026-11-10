"""Methods to evaluate peptide-spectrum predictions. Borrowed from
Casanovo https://github.com/Noble-Lab/casanovo"""

import re
import numpy as np
from numpy.linalg import norm
from typing import Dict, Iterable, List, Tuple


# Method is borrowed from 'spectrum_utils' package
# (temporary removed 'spectrum_utils' dependence
# due to numba cache issues when running on HPC cluster).
def mass_diff(mz1, mz2, mode_is_da):
    """
    Calculate the mass difference(s).

    Parameters
    ----------
    mz1
        First m/z value(s).
    mz2
        Second m/z value(s).
    mode_is_da : bool
        Mass difference in Dalton (True) or in ppm (False).

    Returns
    -------
        The mass difference(s) between the given m/z values.
    """
    return mz1 - mz2 if mode_is_da else (mz1 - mz2) / mz2 * 10**6


def get_token_mass(
    token: str, aa_dict: Dict[str, float], default: float = 0
) -> float:
    """Mass of a token in delta mass notation, e.g. `M` or `M[+15.994915]`.

    Raises ValueError for a residue missing from `aa_dict` or a modification that is not a number,
    rather than giving it a mass of 0. `default` is unused and kept for the callers' signature.
    """
    match = re.fullmatch(r"([A-Z])(?:\[([^\]]+)\])?", token)
    if match is None or match.group(1) not in aa_dict:
        raise ValueError(f"Cannot compute the mass of token {token!r}")
    return aa_dict[match.group(1)] + (float(match.group(2)) if match.group(2) else 0.0)


def aa_match_prefix(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool, Tuple[np.ndarray]]:
    """
    Find the matching prefix amino acids between two peptide sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    per_seq_aa_matches : Tuple[np.ndarray]
        TODO.
    """
    aa_matches = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    aa_matches_1 = np.zeros(len(peptide1), np.bool_)
    aa_matches_2 = np.zeros(len(peptide2), np.bool_)

    # Find longest mass-matching prefix.
    i1, i2, cum_mass1, cum_mass2 = 0, 0, 0.0, 0.0
    while i1 < len(peptide1) and i2 < len(peptide2):
        aa_mass1 = get_token_mass(peptide1[i1], aa_dict, 0)
        aa_mass2 = get_token_mass(peptide2[i2], aa_dict, 0)

        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            match = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            aa_matches[max(i1, i2)] = match
            aa_matches_1[i1] = match
            aa_matches_2[i2] = match

            i1, i2 = i1 + 1, i2 + 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2

        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 + 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 + 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all(), (aa_matches_1, aa_matches_2)


def aa_match_prefix_suffix(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool, Tuple[np.ndarray]]:
    """
    Find the matching prefix and suffix amino acids between two peptide
    sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    per_seq_aa_matches : Tuple[np.ndarray]
        TODO.
    """
    # Find longest mass-matching prefix.
    aa_matches, pep_match, (aa_matches_1, aa_matches_2) = aa_match_prefix(
        peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
    )
    # No need to evaluate the suffixes if the sequences already fully match.
    if pep_match:
        return aa_matches, pep_match, (aa_matches_1, aa_matches_2)

    # Find longest mass-matching suffix.
    i1, i2 = len(peptide1) - 1, len(peptide2) - 1
    i_stop = np.argwhere(~aa_matches)[0]
    cum_mass1, cum_mass2 = 0.0, 0.0
    while i1 >= i_stop and i2 >= i_stop:
        aa_mass1 = get_token_mass(peptide1[i1], aa_dict, 0)
        aa_mass2 = get_token_mass(peptide2[i2], aa_dict, 0)

        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            match = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            aa_matches[max(i1, i2)] = match
            aa_matches_1[i1] = match
            aa_matches_2[i2] = match

            i1, i2 = i1 - 1, i2 - 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2

        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 - 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 - 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all(), (aa_matches_1, aa_matches_2)


def aa_match(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[np.ndarray, bool, Tuple[np.ndarray]]:
    """
    Find the matching amino acids between two peptide sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    per_seq_aa_matches : Tuple[np.ndarray]
        TODO.
    """
    if mode == "best":
        return aa_match_prefix_suffix(
            peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
        )
    elif mode == "forward":
        return aa_match_prefix(
            peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
        )
    elif mode == "backward":
        aa_matches, pep_match, (aa_matches_1, aa_matches_2) = aa_match_prefix(
            list(reversed(peptide1)),
            list(reversed(peptide2)),
            aa_dict,
            cum_mass_threshold,
            ind_mass_threshold,
        )
        return (
            aa_matches[::-1],
            pep_match,
            (aa_matches_1[::-1], aa_matches_2[::-1]),
        )
    else:
        raise ValueError("Unknown evaluation mode")


def aa_match_batch(
    peptides1: Iterable,
    peptides2: Iterable,
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[List[Tuple[np.ndarray, bool]], int, int]:
    """
    Find the matching amino acids between multiple pairs of peptide sequences.

    Parameters
    ----------
    peptides1 : Iterable
        The first list of peptide sequences to be compared.
    peptides2 : Iterable
        The second list of peptide sequences to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches_batch : List[Tuple[np.ndarray, bool, Tuple[np.ndarray]]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match,
        (iii) TODO.
    n_aa1: int
        Total number of amino acids in the first list of peptide sequences.
    n_aa2: int
        Total number of amino acids in the second list of peptide sequences.
    """
    aa_matches_batch, n_aa1, n_aa2 = [], 0, 0
    for peptide1, peptide2 in zip(peptides1, peptides2):
        # Split peptides into individual AAs if necessary.
        if isinstance(peptide1, str):
            peptide1 = re.split(r"(?<=.)(?=[A-Z])", peptide1) if peptide1 else []
        if isinstance(peptide2, str):
            peptide2 = re.split(r"(?<=.)(?=[A-Z])", peptide2) if peptide2 else []
        n_aa1, n_aa2 = n_aa1 + len(peptide1), n_aa2 + len(peptide2)
        aa_matches_batch.append(
            aa_match(
                peptide1,
                peptide2,
                aa_dict,
                cum_mass_threshold,
                ind_mass_threshold,
                mode,
            )
        )
    return aa_matches_batch, n_aa1, n_aa2


def aa_match_metrics(
    aa_matches_batch: List[Tuple[np.ndarray, bool]],
    n_aa_true: int,
    n_aa_pred: int,
) -> Tuple[float, float, float]:
    """
    Calculate amino acid and peptide-level evaluation metrics.

    Parameters
    ----------
    aa_matches_batch : List[Tuple[np.ndarray, bool]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match.
    n_aa_true: int
        Total number of amino acids in the true peptide sequences.
    n_aa_pred: int
        Total number of amino acids in the predicted peptide sequences.

    Returns
    -------
    aa_precision: float
        The number of correct AA predictions divided by the number of predicted
        AAs.
    aa_recall: float
        The number of correct AA predictions divided by the number of true AAs.
    pep_precision: float
        The number of correct peptide predictions divided by the number of
        peptides.
    """
    n_aa_correct = sum(
        [aa_matches[0].sum() for aa_matches in aa_matches_batch]
    )
    aa_precision = n_aa_correct / (n_aa_pred + 1e-8)
    aa_recall = n_aa_correct / (n_aa_true + 1e-8)
    pep_precision = sum([aa_matches[1] for aa_matches in aa_matches_batch]) / (
        len(aa_matches_batch) + 1e-8
    )
    return float(aa_precision), float(aa_recall), float(pep_precision)


def aa_precision_recall(
    aa_scores_correct: List[float],
    aa_scores_all: List[float],
    n_aa_total: int,
    threshold: float,
) -> Tuple[float, float]:
    """
    Calculate amino acid level precision and recall at a given score threshold.

    Parameters
    ----------
    aa_scores_correct : List[float]
        Amino acids scores for the correct amino acids predictions.
    aa_scores_all : List[float]
        Amino acid scores for all amino acids predictions.
    n_aa_total : int
        The total number of amino acids in the predicted peptide sequences.
    threshold : float
        The amino acid score threshold.

    Returns
    -------
    aa_precision: float
        The number of correct amino acid predictions divided by the number of
        predicted amino acids.
    aa_recall: float
        The number of correct amino acid predictions divided by the total number
        of amino acids.
    """
    n_aa_correct = sum([score > threshold for score in aa_scores_correct])
    n_aa_predicted = sum([score > threshold for score in aa_scores_all])
    return n_aa_correct / n_aa_predicted, n_aa_correct / n_aa_total


# Methods calculate spectral angle between two spectra

def cosine_similarity(a, b):
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return (a @ b) / denom


def spectral_angle(mz_pred, I_pred, mz_true, I_true, tol=0.02, eps=1e-8):
    if I_pred.size == 0 or I_true.size == 0:
        return 0.0
    
    # Sort experimental m/z for binary search
    sort_idx = np.argsort(mz_true)
    mz_true_sorted = mz_true[sort_idx]
    I_true_sorted = I_true[sort_idx]

    # Find matching m/z in both spectra
    matched_pred = []
    matched_true = []
    matched_true_idxs = []
    
    pred_mask = (I_pred >= 0)
    for mz_p, I_p in zip(mz_pred[pred_mask], I_pred[pred_mask]):
        # Find position where this m/z would be inserted
        idx = np.searchsorted(mz_true_sorted, mz_p)
    
        candidates = []
        for offset in [-1, 0, 1]: # a[i] is (always?) closer to b than a[i + 1] -- ? TODO: do we need to check offset = 1 at all?
            i = idx + offset
            if 0 <= i < len(mz_true_sorted):
                delta = abs(mz_p - mz_true_sorted[i])
                if delta <= tol: # Da? 
                    candidates.append((i, delta, mz_true_sorted[i], I_true_sorted[i], offset))
    
        if candidates: # if any matches found
            # Pick the closest match
            best_match = min(candidates, key=lambda x: x[1])
            matched_pred.append(I_p)
            
            match_true_idx = best_match[0]
            match_I = best_match[3]
            matched_true.append(match_I) # add matching true peak intensity
            matched_true_idxs.append(match_true_idx)
            
        else: # add as unmached predicted (mz, I) pair
            matched_pred.append(I_p)
            matched_true.append(0) # add 0 as there is no matching true peak

    unmatched_true_idxs = list(set(list(range(len(I_true_sorted)))) - set(matched_true_idxs))
    unmatched_true = list(I_true_sorted[unmatched_true_idxs])
    unmatched_pred = [0] * len(unmatched_true_idxs)
    cossim = cosine_similarity(np.array(matched_pred + unmatched_pred), np.array(matched_true + unmatched_true))
    # transform cossim to SA
    cossim = np.clip(cossim, -1 + eps, 1 - eps)
    SA = np.acos(cossim)
    SA = 1 - 2 * SA / np.pi
    return SA