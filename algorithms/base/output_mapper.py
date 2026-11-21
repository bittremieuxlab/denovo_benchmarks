"""Base class with methods for OutputMapper."""

from pyteomics import proforma

class OutputMapperBase:
    def _format_scores(self, scores):
        """
        Write a list of float per-token scores
        into a string of float scores separated by ','.
        """
        return ",".join(map(str, scores))

    def format_spectrum_id(self, spectrum_id):
        """
        Represent spectrum spectrum id as {filename}:{index} string, 
        where
        - `filename` - name of the .mgf file in a dataset 
            (without .mgf extension)
        - `index` - index (0-based) of each spectrum in an .mgf file.
        """
        return spectrum_id

    def format_sequence(self, sequence):
        """
        Convert peptide sequence to the common output data format 
        (ProForma with modifications represented with 
        Unimod accession codes, e.g. M[UNIMOD:35]).

        Parameters
        ----------
        sequence : str
            Peptide sequence in the original algorithm output format.

        Returns
        -------
        transformed_sequence : str
            Peptide sequence in the common output data format.  
        """
        return sequence

    def format_sequence_and_scores(self, sequence, aa_scores):
        """
        Convert peptide sequence to the common output data format
        (ProForma with modifications represented with 
        Unimod accession codes, e.g. M[UNIMOD:35])
        and modify per-token scores if needed.

        This method is only needed if per-token scores have to be modified 
        to correspond the transformed sequence in ProForma format.
        Otherwise use `format_sequence` method instead.

        Parameters
        ----------
        sequence : str
            Peptide sequence in the original algorithm output format.
        aa_scores: str
            String of per-token scores for each token in the sequence.

        Returns
        -------
        transformed_sequence : str
            Peptide sequence in the common output data format.
        transformed_aa_scores: str
            String of per-token scores corresponding to each token
            in the transformed sequence.
        """
        sequence = self.format_sequence(sequence)
        return sequence, aa_scores

    def simulate_token_scores(self, pep_score, sequence):
        """
        Define proxy per-token scores from the peptide score
        if per-token scores are not provided by the model.
        Expects the sequence to be already in 
        the ProForma delta mass notation!
        """
        try:
            seq = proforma.parse(sequence.replace("][", ", "))
            n_tokens = len(seq[0])
            if seq[1]["n_term"]:
                n_tokens += len(seq[1]["n_term"])
            if seq[1]["c_term"]:
                n_tokens += len(seq[1]["c_term"])
            scores = [str(pep_score),] * n_tokens
            
        except:
            print("ProForma failed on:", sequence)
            scores = []

        return self._format_scores(scores)

    def format_output(self, output_data):
        """
        Transform ['spectrum_id', 'sequence', 'score', 'aa_scores'] columns 
        of `output_data` dataframe to the common output format.
        Assumes that predicted sequences are provided 
        for all dataframe entries (no NaNs).

        Parameters
        ----------
        output_data : pd.DataFrame
            Dataframe with algorithm outputs. Must contain columns:
            - 'sequence' - predicted peptide sequence;
            - 'score' - confidence score for the predicted sequence;
            - 'aa_scores' - per-amino acid scores, if available. 
                Otherwise, the whole peptide `score` will be used 
                as a score for each amino acid.
            - 'spectrum_id' - `{filename}:{index}` string to match 
                each prediction with its ground truth sequence.

        Returns
        -------
        transformed_output_data : pd.DataFrame
            Dataframe with algorithm predictions
            in the common output data format. 
        """
        
        if "aa_scores" in output_data:
            output_data[["sequence", "aa_scores"]] = output_data.apply(
                lambda row: self.format_sequence_and_scores(row["sequence"], row["aa_scores"]),
                axis=1,
                result_type="expand",
            )
            # rows without per-token scores get the peptide score for each token
            missing = output_data["sequence"].notna() & ~output_data["aa_scores"].apply(lambda s: isinstance(s, str) and s != "")
            if missing.any():
                output_data.loc[missing, "aa_scores"] = output_data[missing].apply(
                    lambda row: self.simulate_token_scores(row["score"], row["sequence"]),
                    axis=1,
                )

        else:
            output_data["sequence"] = output_data["sequence"].apply(
                self.format_sequence,
            )
            output_data["aa_scores"] = output_data.apply(
                lambda row: self.simulate_token_scores(row["score"], row["sequence"]), 
                axis=1,
            )
            
        if "spectrum_id" in output_data:
            output_data["spectrum_id"] = output_data["spectrum_id"].apply(self.format_spectrum_id)

        return output_data
