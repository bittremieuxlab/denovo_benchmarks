"""Methods to convert ground truth labels to the common data format."""

import re


REPLACEMENTS = []
PTM_PATTERN = r"([A-Z])\[([0-9.+-]+)\]" # find AAs with PTMs 
# N_TERM_MOD_PATTERN = r"^n\[([0-9.+-]+)\]" # find N-term modifications
# TODO: check: Notation of ground truth sequences is fixed from n[mod]PEP to [mod]PEP
N_TERM_MOD_PATTERN = r"^\[([0-9.+-]+)\]" # find N-term modifications

def _transform_match_ptm(match: re.Match) -> str:
    """
    Transform representation of amino acids substring matching
    the PTM pattern.
    Expects PTMs in ProForma notation, e.g. 'M[UNIMOD:35]'.

    Parameters
    ----------
    match : re.Match
        Substring matching the PTM pattern.

    Returns
    -------
    transformed_match : str
        Transformed PTM pattern representation.
    """
    aa, ptm = match.group(1), match.group(2)

    if not ptm.startswith("-") and not ptm.startswith("+"):
        ptm = "+" + ptm
    return f"{aa}[{ptm}]"

def _transform_match_n_term_mod(match: re.Match) -> str:
    """
    Transform representation of peptide substring matching
    the N-term modification pattern.
    `n[+n_mod]PEP` -> `[+n_mod]-PEP`
    
    TODO.
    """
    ptm = match.group(1)
    
    if not ptm.startswith("-"):
        ptm = "+" + ptm
    return f"[{ptm}]-"

def format_sequence(sequence: str) -> str:
    """
    Convert peptide sequence to the common output data format.

    Parameters
    ----------
    sequence : str
        Peptide sequence in the original ground truth format.

    Returns
    -------
    transformed_sequence : str
        Peptide sequence in the common output data format.
    """

    # direct (token-to-token) replacements (if any)
    for repl_args in REPLACEMENTS:
        sequence = sequence.replace(*repl_args)

    # transformation of PTM notation:
    # represent in ProForma delta mass notation PE[+ptm]P
    sequence = re.sub(PTM_PATTERN, _transform_match_ptm, sequence)
    
    # transform n-term modification notation
    # represent in ProForma delta mass notation [+n_term_mod]-PEP
    sequence = re.sub(N_TERM_MOD_PATTERN, _transform_match_n_term_mod, sequence)

    return sequence
