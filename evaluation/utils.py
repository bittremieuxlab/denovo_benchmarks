import os
import re
import numpy as np
import pandas as pd
from pyteomics import mgf, proforma
from tqdm import tqdm


# Instance of Unimod database to look up PTM masses
try:
    from pyteomics.mass.unimod import Unimod
    UNIMOD_DB = Unimod()
except:
    from pyteomics.mass import Unimod
    UNIMOD_DB = Unimod()
# Cache of PTM masses to avoid repeated lookups in the Unimod database
ptm_masses = {}


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
    aa_scores = parse_scores(aa_scores)
    aa_scores[0:2] = [np.mean(aa_scores[0:2])]
    return format_scores(aa_scores)


def _transform_match_ptm(match: re.Match) -> str:
    """
    TODO
    """
    ptm_idx = int(match.group(1))
    
    if ptm_idx not in ptm_masses:
        try:
            ptm_masses[ptm_idx] = UNIMOD_DB.get(ptm_idx).monoisotopic_mass
        except:
            ptm_masses[ptm_idx] = UNIMOD_DB.by_id(ptm_idx)["mono_mass"]
        print(ptm_masses)
    
    ptm_mass = ptm_masses[ptm_idx]
    return f"[{ptm_mass:+}]"


def _transform_match_n_term(match: re.Match) -> str:
    """
    TODO
    """
    n_term_mod = match.group(1)
    first_aa = match.group(2)
    
    if match.group(3) is not None:
        first_aa_ptm = match.group(3)
        first_aa_ptm_mass = float(first_aa_ptm) # convert from str to float
        n_term_mod_mass = float(n_term_mod) # convert n_term_mod_mass from str to float
        first_aa_ptm_mass += n_term_mod_mass # sum up
        first_aa_ptm = f"{first_aa_ptm_mass:+}" # convert back to string
    else:
        first_aa_ptm = n_term_mod
    return f"{first_aa}[{first_aa_ptm}]"


def ptms_to_delta_mass(sequence, aa_scores):
    """Convert PTM representation from Unimod notation to delta mass notation."""
    
    PTM_PATTERN = r"\[UNIMOD:([0-9]+)\]" # find ptms
    N_TERM_PATTERN = r"^\[([0-9+-.]+)\]-([A-Z])(?:\[([0-9+-.]+)\])?" # find n-term modifications in ProForma notation
    
    merged_n_term_mod = False
    
    sequence = re.sub(PTM_PATTERN, _transform_match_ptm, sequence)
    sequence, merged_n_term_mod = re.subn(N_TERM_PATTERN, _transform_match_n_term, sequence)
    
    if merged_n_term_mod == 1:
        aa_scores = merge_n_term_score(aa_scores)
    
    return sequence, aa_scores


def remove_ptms(sequence, ptm_pattern="[^A-Z]"):
    return re.sub(ptm_pattern, "", sequence)


# Auxiliary methods to validate output format of predicted sequences and aa scores

def get_n_tokens(sequence: str) -> int:
    """Calculate number of tokens in a sequence in ProForma notation."""
    seq = proforma.parse(sequence.replace("][", ", "))
    n_tokens = len(seq[0])
    if seq[1]["n_term"]:
        n_tokens += len(seq[1]["n_term"])
    if seq[1]["c_term"]:
        n_tokens += len(seq[1]["c_term"])
    return n_tokens


def get_n_scores(scores: str) -> int:
    return len(scores.split(","))


def validate_spectrum_id(spectrum_id: str) -> bool:
    SPECTRUM_ID_PATTERN = r"[^:]+:\d+"
    return bool(re.fullmatch(SPECTRUM_ID_PATTERN, spectrum_id))


def validate_sequence(sequence: str) -> bool:
    try:
        # Merge subsequent modifications together, because pyteomics.proforma 
        # cannot parse sequence with multiple N-term modifications
        # TODO: maybe only merge _terminal_ modifications?
        seq = proforma.parse(sequence.replace("][", ", "))
    except:
        return False
    return True


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
