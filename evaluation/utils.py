import os
import re
import numpy as np
import pandas as pd
from pyteomics import mgf, proforma
from tqdm import tqdm

from .token_masses import AA_MASSES


def parse_scores(aa_scores: str) -> list[float]:
    """
    TODO.
    * assumes that AA confidence scores always come
    as a string of float numbers separated by a comma.
    """
    if not aa_scores:
        return []
    aa_scores = aa_scores.split(",")
    aa_scores = list(map(float, aa_scores))
    return aa_scores


def format_scores(aa_scores: list[float]) -> str:
    """
    Write a list of float per-token scores
    into a string of float scores separated by ','.
    """
    return ",".join(map(str, aa_scores))


def merge_n_term_score(aa_scores: str) -> str:
    """Merge the N-terminal group's score into the first residue's score, by averaging them."""
    aa_scores = parse_scores(aa_scores)
    aa_scores[0:2] = [np.mean(aa_scores[0:2])]
    return format_scores(aa_scores)


def merge_c_term_score(aa_scores: str) -> str:
    """Merge the C-terminal group's score into the last residue's score, by averaging them."""
    aa_scores = parse_scores(aa_scores)
    aa_scores[-2:] = [np.mean(aa_scores[-2:])]
    return format_scores(aa_scores)


def _group_mass(tags) -> float:
    """Summed mass of the modifications on one position, resolved by pyteomics."""
    return sum(tag.mass for tag in tags or [])


def ptms_to_delta_mass(sequence, aa_scores):
    """Convert a ProForma sequence to delta mass notation, e.g. `[UNIMOD:1]-M[UNIMOD:35]K` -> `M[+58.005479]K`.

    The modifications on a residue are summed, and each terminal group, however many tags it holds,
    is added into the residue at that end, as the ground truth labels write it. A terminal group has
    one score of its own, which is merged with that residue's score.
    """
    residues, properties = proforma.parse(sequence)
    n_term, c_term = properties["n_term"], properties["c_term"]
    last = len(residues) - 1
    tokens = []
    for i, (aa, tags) in enumerate(residues):
        modified = bool(tags) or (i == 0 and n_term) or (i == last and c_term)
        mass = _group_mass(tags)
        if i == 0:
            mass += _group_mass(n_term)
        if i == last:
            mass += _group_mass(c_term)
        tokens.append(f"{aa}[{mass:+}]" if modified else aa)
    if n_term:
        aa_scores = merge_n_term_score(aa_scores)
    if c_term:
        aa_scores = merge_c_term_score(aa_scores)
    return "".join(tokens), aa_scores


def remove_ptms(sequence, ptm_pattern="[^A-Z]"):
    return re.sub(ptm_pattern, "", sequence)


# Auxiliary methods to validate output format of predicted sequences and aa scores

def get_n_tokens(sequence: str) -> int:
    """Number of scored tokens in a ProForma sequence: one per residue, plus one per modified terminus.

    A terminal group is one token however many modifications it holds.
    """
    residues, properties = proforma.parse(sequence)
    return len(residues) + bool(properties["n_term"]) + bool(properties["c_term"])


def get_n_scores(scores: str) -> int:
    return len(scores.split(","))


def validate_spectrum_id(spectrum_id: str) -> bool:
    SPECTRUM_ID_PATTERN = r"[^:]+:\d+"
    return bool(re.fullmatch(SPECTRUM_ID_PATTERN, spectrum_id))


# ProForma features whose mass `ptms_to_delta_mass` cannot place on a residue.
UNSUPPORTED_PROPERTIES = ("unlocalized_modifications", "labile_modifications", "fixed_modifications", "isotopes")


def validate_sequence(sequence: str) -> bool:
    """Whether a sequence is valid ProForma, with every residue and modification of known mass.

    Modifications that are not on a residue or terminus (unlocalized, labile, global fixed, or on a
    range) and isotope labels are rejected, because `ptms_to_delta_mass` would drop their mass.
    """
    try:
        residues, properties = proforma.parse(sequence)
        if any(properties.get(key) for key in UNSUPPORTED_PROPERTIES):
            return False
        if any(interval.tags for interval in properties.get("intervals") or []):
            return False
        for tags in [properties["n_term"], properties["c_term"], *(tags for _, tags in residues)]:
            _group_mass(tags)
    except Exception:
        return False
    return all(aa in AA_MASSES for aa, _ in residues)


def validate_token_scores(scores: str, sequence: str) -> bool:
    # Sanity check: scores is a string (and not NaN)
    if not isinstance(scores, str):
        return False
    n_tokens = get_n_tokens(sequence)
    return get_n_scores(scores) == n_tokens


def load_predictions(output_path, sequences_true):
    """Load de novo predictions and combine them with ground truth sequences."""
    use_cols = ["sequence", "score", "aa_scores", "spectrum_id", "pred_RT", "SA", "proteome_match"]
    
    output_data = pd.read_csv(output_path)
    output_data = pd.merge(
        sequences_true,
        output_data[use_cols],
        on="spectrum_id",
        how="outer",
    )
    output_data = output_data.rename({"seq": "sequence_true"}, axis=1)
    return output_data


def get_sequenced_idx(output_data):
    """Validate predictions and collect indices of correctly formatted predicted sequences."""
    
    sequenced_idx = output_data["sequence"].notnull()
    failed_idx = []
    print("Failed sequences:\n")
    for row_idx, row in output_data[sequenced_idx].iterrows():

        if not validate_sequence(row["sequence"]):
            failed_idx.append(row_idx)
            print(f"Spectrum id: {row['spectrum_id']}")
            print(f"Predicted sequence: {row['sequence']}")
            print(f"FAILED: sequence is not in the ProForma format.\n")

        elif not validate_token_scores(row["aa_scores"], row["sequence"]):
            failed_idx.append(row_idx)
            print(f"{row['spectrum_id']}")
            print(f"Predicted sequence: {row['sequence']}, number of tokens: {get_n_tokens(row['sequence'])}")
            print(f"Predicted scores: {row['aa_scores']}, number of scores: {get_n_scores(row['aa_scores'])}")
            print(f"FAILED: number of per-token scores (','-separated scores in `aa_scores`) does not match number of individual tokens in the predicted sequence.\n")

    sequenced_idx[failed_idx] = False
    return sequenced_idx


# Methods to extract parameters of a real experimental spectrum

def get_charge(spectrum):
    return spectrum["params"]["charge"][0]

def get_RT(spectrum):
    rt = spectrum["params"]["rtinseconds"]
    rt = float(rt)
    return rt

def get_precursor_mass(spectrum):
    return spectrum["params"]["pepmass"][0]

def get_mz_I(spectrum):
    mz = spectrum['m/z array']
    I = spectrum["intensity array"]
    if I.size > 0:
        I = I / I.max()
    return mz, I

def extract_spectra_params(dataset_path):
    "Extract parameters of all the experimental spectra in the dataset."
    
    spectra_params = {
        "filename": [],
        "idx": [],
        "charge": [], 
        "precursor_mass": [],
        "true_RT": [], 
    }
    for mgf_file in os.listdir(dataset_path):
        mgf_path = os.path.join(dataset_path, mgf_file)
        with mgf.IndexedMGF(mgf_path) as spectra:
            filename = os.path.splitext(mgf_file)[0]
            print(f"Extract spectra params: {filename}, ({len(spectra)} spectra)")
            for i, spectrum in tqdm(enumerate(spectra), total=len(spectra)):
                spectra_params["filename"].append(filename)
                spectra_params["idx"].append(i)
                spectra_params["charge"].append(get_charge(spectrum))
                spectra_params["precursor_mass"].append(get_precursor_mass(spectrum))
                spectra_params["true_RT"].append(get_RT(spectrum))
    print()
    
    spectra_params = pd.DataFrame(spectra_params)
    spectra_params["spectrum_id"] = spectra_params["filename"] + ":" + spectra_params["idx"].astype(str)
    spectra_params["run"] = spectra_params["filename"].apply(lambda s: s.rsplit("_", 1)[0])
    return spectra_params


# def remove_precursor_peak(
#     mz,
#     intensity,
#     precursor_mz,
#     precursor_charge,
#     fragment_tol_mass,
#     isotope: int = 0,
# ):
#     msms_spectrum = sus.MsmsSpectrum(
#         identifier="",
#         precursor_mz=precursor_mz,
#         precursor_charge=precursor_charge,
#         mz=mz,
#         intensity=intensity,
#     )
#     msms_spectrum = msms_spectrum.remove_precursor_peak(fragment_tol_mass=fragment_tol_mass, fragment_tol_mode="Da", isotope=isotope)
#     return msms_spectrum.mz, msms_spectrum.intensity

# TODO: check that this function works the same as spectrum_utils version
def remove_precursor_peak(
    mz,
    intensity,
    precursor_mz,
    precursor_charge,
    fragment_tol_mass: float, # in Da
    isotope: int = 0,
):
    # TODO: This assumes [M+H]x charged ions.
    adduct_mass = 1.007825
    neutral_mass = (
        precursor_mz - adduct_mass
    ) * precursor_charge
    c_mass_diff = 1.003355
    remove_mz = [
        (neutral_mass + iso * c_mass_diff) / charge + adduct_mass
        for charge in range(precursor_charge, 0, -1)
        for iso in range(isotope + 1)
    ]
    
    mask = np.full_like(mz, True, np.bool_)
    mz_i = remove_i = 0
    while mz_i < len(mz) and remove_i < len(remove_mz):
        md = mz[mz_i] - remove_mz[remove_i] # tol in Da
        if md < -fragment_tol_mass:
            mz_i += 1
        elif md > fragment_tol_mass:
            remove_i += 1
        else:
            mask[mz_i] = False
            mz_i += 1
    
    mz, intensity = mz[mask], intensity[mask]
    return mz, intensity
