"""
Script to convert predictions from the algorithm output format 
to the common output format.
"""
import argparse
import ast

import numpy as np
import pandas as pd
from base import OutputMapperBase


class OutputMapper(OutputMapperBase):
    """Output mapper for InstaNovo predictions.
    
    Note: InstaNovo already outputs sequences in ProForma format with UNIMOD notation,
    so no sequence transformation is needed.
    """

    def format_sequence(self, sequence):
        """
        Convert peptide sequence to the common output data format.
        
        InstaNovo already outputs in ProForma format with UNIMOD notation,
        so this just ensures the sequence is a string.
        """
        if pd.isna(sequence) or sequence == "":
            return ""
        return str(sequence)

    def format_scores(self, scores: str) -> str:
        """
        Convert token_log_probs from InstaNovo format to comma-separated string.
        
        InstaNovo outputs token_log_probs as a Python list string: "[-0.1, -0.2, -0.3]"
        """
        if pd.isna(scores) or scores == "":
            return ""
        
        # Parse the list from string representation
        scores_list = ast.literal_eval(scores)
        return ",".join(map(str, scores_list))

    def format_sequence_and_scores(self, sequence, aa_scores):
        """
        Format sequence and per-token scores for the common output format.
        
        InstaNovo sequences are already in ProForma format.
        """
        sequence = self.format_sequence(sequence)
        aa_scores = self.format_scores(aa_scores)
        return sequence, aa_scores


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path", required=True, help="The path to the algorithm predictions file."
)
args = parser.parse_args()

# Read predictions from output file
output_data = pd.read_csv(args.output_path)

# Convert log probabilities to confidence score
output_data["score"] = output_data["log_probs"].apply(np.exp)

# Rename columns to the expected column names
output_data = output_data.rename(
    columns={
        "predictions": "sequence",
        "token_log_probs": "aa_scores",
    }
)

# Select only the necessary columns
output_data = output_data[["sequence", "score", "aa_scores", "spectrum_id"]]

# Transform data to the common output format
output_mapper = OutputMapper()
output_data = output_mapper.format_output(output_data)

# Save processed predictions to outputs.csv
# (the expected name for the algorithm output file)
output_data.to_csv("outputs.csv", index=False)