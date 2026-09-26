"""Methods to predict spectrum intensity and RT with Koina models"""

import os
import re
import time
import numpy as np
import pandas as pd
from pyteomics import mgf
from tqdm import tqdm

from . import utils
from .metrics import spectral_angle
 
from koinapy import Koina
from sklearn.linear_model import LinearRegression


N_CALIBRATION_PSMS = 2000
WINDOW_SIZE = 2000
FRAGMENT_MASS_TOL = 0.02 # Da
SLEEP = 60 # seconds between Koina requests when chunking

# - deeplc_hela_hf
# Modifications for this model include carbamidomethyl, oxidation of methionine, and N-terminal acetylatio (in train?). 
# Supported are all modification from UNIMOD. 
# Cysteine residues are assumed to be carbamidomethylated (C == C[UNIMOD:4]). # This doesn't seem to be true?

# - ms2pip_HCD2021
# Valid sequences lengths up to 30 AA
# There are no limitations to valid Precursor Charges.
# The model was trained on peptides with oxidation of methionine and fixed or variable carbamidomethylation of cysteine. 
# Supported are all modification from UNIMOD but modifications are only used to shift fragment mz they do not affect intensity.

# - manuscript
# MS2PIP and DeepLC will both be queried through Koina.
# For MS2PIP, specific prediction models based on 
# instrument type, fragmentation mode, and peptide characteristics 
# matching the experimental data will be used. 
# Because MS2PIP does not natively support PTMs, 
# the fragment ion intensity evaluation will be restricted to 
# compatible, unmodified, peptides.


# RT prediction model
# - modifications that the model was trained on: Carbamidomethyl, Oxidation, Acetyl
# - it says to support all the modifications but they must be converted to PSI-MS names
# (for GT: delta_mass -> UNIMOD -> PSI-MS)
# (delta_mass -> UNIMOD we also do for I)
# (UNIMOD -> PSI-MS we anyway do for de novo peptides)
# - [Always assumes C to have [Carbamidomethyl]? So no direct support for immunopeptides?]


def get_RT_model_mods(
    dataset_tags: tuple[str] = (), 
    # TODO: should be of type DatasetTag, but we have a relative import problem!
) -> tuple[str, list]:
    """
    Get name of the RT model and its supported modifications
    according to dataset_tags.
    """
    model_names_rt = {
        # single model for everything
        "default": "Deeplc_hela_hf",
    }
    supported_mods_rt = ["[UNIMOD:4]", "[UNIMOD:35]", "[UNIMOD:1]"] # deeplc_hela_hf

    if "tmt" in dataset_tags:
        supported_mods_rt += ["[UNIMOD:737]"]
        return model_names_rt["default"], supported_mods_rt

    return model_names_rt["default"], supported_mods_rt

# Intensity prediction model
# - modifications that the model supports: [UNIMOD:35] and [UNIMOD:4]
# - for TMT probably also [UNIMOD:737] (one or two) ?
# - immunopeptides (C without [UNIMOD:4]) are ok
# - peptides with other modifications are not supported!
# - do we need to map delta_mass modifications to UNIMOD? 
# probably yes, at least to check whether they are supported 

def get_intensity_model_mods(
    dataset_tags: tuple[str] = (), 
    # TODO: should be of type DatasetTag, but we have a relative import problem!
) -> tuple[str, list]:
    """
    Get name of the intensity model and its supported modifications
    according to dataset_tags.
    """
    model_names_I = {
        # TMT data
        "TMT": "ms2pip_CID_TMT",
        # TimsTOF data
        "TimsTOF": "ms2pip_timsTOF2024",
        # TOF data
        "TOF": "ms2pip_TTOF5600",
        # others (Orbitrap HCD data)
        "default": "ms2pip_HCD2021",
    }
    supported_mods_I = ["[UNIMOD:4]", "[UNIMOD:35]"] # all ms2pip models (train)

    if "tmt" in dataset_tags:
        supported_mods_I += ["[UNIMOD:737]"]
        return model_names_I["TMT"], supported_mods_I

    if "timstof" in dataset_tags:
        return model_names_I["TimsTOF"], supported_mods_I

    if "sciex" in dataset_tags or "agilent" in dataset_tags:
        return model_names_I["TOF"], supported_mods_I

    return model_names_I["default"], supported_mods_I


def map_mods_delta_mass_to_unimod(sequence, supported_mods=None):
    """
    Map modifications in the sequence from delta mass format to UNIMOD format.
    Only modifications from SUPPORTED_MODS are mapped to their UNIMOD accession.
    Other modifications are replaced with generic [UNIMOD:] tag to be later filtered out.
    """

    PTM_PATTERN = r"\[[0-9.+-]+\]" # find AAs with PTMs 
    MASS2UNIMOD = {
        "[+57.022]": "[UNIMOD:4]",
        "[+15.995]": "[UNIMOD:35]",
        "[+42.011]": "[UNIMOD:1]",
        "[+229.163]": "[UNIMOD:737]", # TMT6plex
        "[+458.326]": "[UNIMOD:737][UNIMOD:737]", # 2 x TMT6plex
        "UNK": "[UNIMOD:]",
    }

    mass2unimod = MASS2UNIMOD if supported_mods is None else {
        k: v for k, v in MASS2UNIMOD.items() if v in supported_mods
    }

    def _transform_match_ptm(match: re.Match) -> str:
        """delta mass to UNIMOD"""
        ptm = match.group(0)
        ptm = mass2unimod.get(ptm, mass2unimod["UNK"])
        return ptm

    sequence = re.sub(PTM_PATTERN, _transform_match_ptm, sequence)
    return sequence


def map_mods_unimod_to_psims(sequence, supported_mods=None):
    """
    Map modifications in the sequence from UNIMOD format to PSI-MS names.
    Only modifications from SUPPORTED_MODS are mapped to their PSI-MS names.
    """

    PTM_PATTERN = r"\[UNIMOD:[0-9]+\]" # find AAs with PTMs 
    UNIMOD2PSIMS = {
        "[UNIMOD:4]": "[Carbamidomethyl]",
        "[UNIMOD:35]": "[Oxidation]",
        "[UNIMOD:1]": "[Acetyl]",
        "[UNIMOD:737]": "[TMT6plex]", # TMT6plex # Interim name, no PSI-MS name on Unimod
        # "[UNIMOD:737][UNIMOD:737]": "[TMT6plex][TMT6plex]", # 2 x TMT6plex TODO: no need to repeat? 
        "UNK": "", # TODO: what is the optimal replacement here?
    }

    unimod2psims = UNIMOD2PSIMS if supported_mods is None else {
        k: v for k, v in UNIMOD2PSIMS.items() if k in supported_mods
    }

    def _transform_match_ptm(match: re.Match) -> str:
        """UNIMOD to PSI-MS"""
        ptm = match.group(0)
        ptm = unimod2psims.get(ptm, unimod2psims["UNK"])
        return ptm

    sequence = re.sub(PTM_PATTERN, _transform_match_ptm, sequence)
    return sequence


def check_supported_by_model(
    peptide, charge, supported_mods=None, 
    min_seq_len=6, max_seq_len=30, max_number_mods=None,
):
    if not isinstance(peptide, str):
        return False

    if not (1 <= charge):
        return False

    # If max_number_mods given, check that peptide contains no more than this number of mods
    if max_number_mods is not None:
        n_mods = len(re.findall(r"\[UNIMOD:[0-9]+\]", peptide))
        if n_mods > max_number_mods:
            return False

    # If supported_mods is given, check that peptide contains only these mods
    if supported_mods is not None:
        # Replace mods - only to test for presence of unsupported mods
        mod_replacements = {mod: mod.strip("[]").split(":")[-1] for mod in supported_mods}
        for mod, replacement in mod_replacements.items():
            peptide = peptide.replace(mod, replacement)
        # check for presence of any unsupported mods
        # (that were not mapped to numerical codes at the previous step)
        if "UNIMOD:" in peptide:
            return False

    # Check that peptide contains only valid characters
    if re.search("[^GASPVTCLINDQKEMHFRYW0-9]", peptide):
        return False
    
    seq_len = len(re.sub("[^A-Z]", "", peptide))
    if not (min_seq_len <= seq_len <= max_seq_len):
        return False
    
    return True


def predict_intensities(
    model_name_I: str, data: pd.DataFrame, chunk_size: int = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # initialize model
    model_I = Koina(model_name_I, "koina.wilhelmlab.org:443")

    inputs = data[["sequence", "charge"]]
    inputs.columns = ["peptide_sequences", "precursor_charges"]

    if chunk_size is None:
        predictions = model_I.predict({col: inputs[col].values for col in inputs})
        predictions_I = pd.DataFrame(predictions["intensities"], index=inputs.index)
        predictions_mz = pd.DataFrame(predictions["mz"], index=inputs.index)
        
    else:
        predictions = []
        for i in range(0, len(inputs), chunk_size):
            predictions.append(
                model_I.predict({col: inputs.iloc[i:i + chunk_size][col].values for col in inputs})
            )
            time.sleep(SLEEP)
        
        predictions_I = pd.DataFrame(
            np.concatenate([predictions_i["intensities"] for predictions_i in predictions]), 
            index=inputs.index
        )
        predictions_mz = pd.DataFrame(
            np.concatenate([predictions_i["mz"] for predictions_i in predictions]), 
            index=inputs.index
        )
    return predictions_mz, predictions_I

def predict_RT(
    model_name_rt: str, data: pd.DataFrame, chunk_size: int = None
) -> np.array:
    # initialize model
    model_rt = Koina(model_name_rt, "koina.wilhelmlab.org:443")

    inputs = data[["sequence"]]
    inputs.columns = ["peptide_sequences"]

    # prepare sequences: convert modifications from UNIMOD to PSI-MS names
    inputs["peptide_sequences"] = inputs["peptide_sequences"].apply(map_mods_unimod_to_psims)
    # drop "-" from ProForma N-term notation
    inputs["peptide_sequences"] = inputs["peptide_sequences"].str.replace("-", "", regex=False)

    if chunk_size is None:
        predictions_rt = model_rt.predict({col: inputs[col].values for col in inputs})
    
    else:
        predictions = []
        for i in range(0, len(inputs), chunk_size):
            predictions.append(
                model_rt.predict({col: inputs.iloc[i:i + chunk_size][col].values for col in inputs}, debug=True)
            )
            time.sleep(SLEEP)
        
        predictions_rt = {
            "irt": np.concatenate([predictions_i["irt"] for predictions_i in predictions])
        }
    return predictions_rt["irt"][:, 0]

def get_calibration_model(calib_psms):
    """
    Train RT calibration model to map true_RT to iRT units.
    """
    rt_calib_reg = LinearRegression()
    calib_rt_true = calib_psms["true_RT"].values
    calib_rt_pred = calib_psms["pred_RT"].values
    rt_calib_reg.fit(calib_rt_true[:, None], calib_rt_pred[:, None])
    print("DEBUG RT calibration:")
    print("Train:", rt_calib_reg.score(calib_rt_true[:, None], calib_rt_pred[:, None]))
    return rt_calib_reg

def calculate_SA(spec_idxs, predictions_mz, predictions_I, mgf_path):
    """
    Load mgf file, iterate through experimental spectra, calculate SA.

    Parameters
    ----------
    spec_idxs : pd.Series or dict
        spec_idx: index - psm idx in dataframe, value - 0-based spectrum idx in mgf file
    predictions_mz : pd.DataFrame
        DataFrame with predicted m/z arrays, indexed by psm idx
    predictions_I : pd.DataFrame
        DataFrame with predicted intensity arrays, indexed by psm idx
    mgf_path : str
        Path to the mgf file
    """
    with mgf.IndexedMGF(mgf_path) as spectra:
        SA_metrics = []
        for (psm_idx, spec_idx) in tqdm(spec_idxs.items(), total=len(spec_idxs)):
            spectrum = spectra[spec_idx]
            mz_true, I_true = utils.get_mz_I(spectrum)
            prec_mass = utils.get_precursor_mass(spectrum)
            charge = utils.get_charge(spectrum)

            mz_true, I_true = utils.remove_precursor_peak(
                np.array(mz_true), np.array(I_true), prec_mass, charge, FRAGMENT_MASS_TOL, isotope=3,
            )
            mz_pred, I_pred = predictions_mz.loc[psm_idx].values, predictions_I.loc[psm_idx].values

            SA = spectral_angle(mz_pred, I_pred, mz_true, I_true, FRAGMENT_MASS_TOL)
            SA_metrics.append(SA)
    return np.array(SA_metrics)
