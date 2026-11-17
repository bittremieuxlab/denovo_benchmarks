# python 2.7
"""
Script to convert input .mgf files from the common input format 
to the algorithm expected format.
"""

import argparse
import os
import re
from pyteomics import mgf
from pyteomics.mass.unimod import Unimod
from tqdm import tqdm
from collections import namedtuple
from base import InputMapperBase


KEYS = ["title", "pepmass", "charge", "scans", "rtinseconds", "seq"]
PTM = namedtuple("DeepNovoPTM", ["amino_acid", "ptm_unimod_id", "representation"])

class InputMapper(InputMapperBase):
    
    def format_input(self, spectrum, spectrum_idx, filename):
        """
        Convert the spectrum (annotation sequence and params) to the
        input format expected by the algorithm.

        Parameters
        ----------
        spectrum : dict
            Peptide sequence in the original format.
        filename: int
            Name of .mgf file being processed. Used to ensure a unique
            scan_id for each spectrum.

        Returns
        -------
        transformed_spectrum : dict
            Peptide sequence in the algorithm input format.
        """
        
        # add dummy labels
        spectrum["params"]["seq"] = "PEPTIDE"

        # fix pepmass to be single value
        spectrum["params"]["pepmass"] = spectrum["params"]["pepmass"][0]

        # create "scans" based on filename and spectrum index (0-based)
        spectrum["params"]["scans"] = filename + ":" + str(spectrum_idx)

        # keep only the relevant keys in params (others may cause issues)
        spectrum["params"] = {k: v for k, v in spectrum["params"].items() if k in KEYS}
        return spectrum


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    help="The path to the input .mgf file.",
)
parser.add_argument(
    "--output_path",
    help="The path to write prepared input data in the format expected by the algorithm.",
)
args = parser.parse_args()

# Transform data to the algorithm input format
input_mapper = InputMapper()

filename = os.path.basename(args.input_path).split(".")[0]
spectra = mgf.read(args.input_path)
mapped_spectra = [
    input_mapper.format_input(spectra[i], i, filename)
    for i in tqdm(range(len(spectra)))
]

mgf.write(
    mapped_spectra,
    args.output_path,
    key_order=KEYS,
    file_mode="w",
)
print(
    "{} spectra written to {}.".format(len(mapped_spectra), args.output_path)
)
