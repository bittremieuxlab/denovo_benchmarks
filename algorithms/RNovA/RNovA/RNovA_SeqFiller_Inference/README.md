# RNovA SeqFiller

**RNovA SeqFiller** is a core module of the RNovA framework for high-accuracy *de novo* peptide sequencing. It fills in missing peptide segments by leveraging reinforcement learning to infer the most probable amino acid completions from MS/MS spectra.

## Features

- Sequence completion using a reinforcement learning model
- Supports both standard and modified amino acids
- Compatible with open PTM search scenarios
- Supports arbitrary PTM-modified amino acids without requiring retraining or fine-tuning

## Installation

We recommend using Python 3.10+ and a clean environment. You can install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv venv
uv pip install -r requirements.uv
```

> WARNING: FlashAttention is not included in requirements. It must be installed manually if needed for other components.

Before using the module, compile the Cython components by running:

```bash
python setup.py build_ext --inplace
```

This will generate the necessary compiled modules required for inference.

## Usage

```bash
python Inference_Seqquence.py input1.mgf input2.mgf ... inputN.mgf "A;C|UniMod:4;D;E;F;G;H;K;L;M;N;P;Q;R;S;T;V;W;Y;M|UniMod:35;W|UniMod:351"
```

- **MGF files**: One or more input `.mgf` files containing MS/MS spectra.
- **Candidate amino acid list**: A quoted string of semicolon-separated amino acid entries. Each entry may optionally include modifications using the format:

  ```
  <SingleLetterAminoAcid>           # e.g., A (unmodified)
  <SingleLetterAminoAcid>|UniMod:x  # e.g., C|UniMod:4 for carbamidomethyl-C
  ```

  Example:

  ```
  "A;C|UniMod:4;D;E;F;G;H;K;L;M;N;P;Q;R;S;T;V;W;Y;M|UniMod:35"
  ```

  This specifies canonical amino acids plus modifications on C (carbamidomethylation, UniMod:4) and M (oxidation, UniMod:35).

## Custom PTMs

To support additional PTMs, edit the file:

```
utils/AA_PTM_Mol_Formula
```

Add a new line with the format:

```
<SingleLetterAminoAcid>|<R Group mod>|<N Term mod>|<C Term mod>,<Molecular Formula>
```

For example:

```
K|TestMod:1,C3H8N2O2
```

After adding, you can refer to this modified amino acid in the candidate list string directly.

## Output

For each input `.mgf` file, the output will be saved in the same directory as the input file.
The output filename will be the original filename with `_rnova_denovo_path.csv` appended.

Each output CSV file contains the following columns: with the following columns:

1. **Scan number**: The scan number from the input MGF file
2. **De novo sequence**: The predicted peptide sequence, including any modifications
3. **Per-residue scores**: A comma-separated list of confidence scores (one for each amino acid in the sequence)

Example:

```csv
28247,GTLFPM[UniMod:35]C[UniMod:4]GMNLAFDR,5.2539;5.7227;5.4609;5.1875;5.4023;5.2422;4.9688;5.0664;4.4102;4.7852;4.4492;4.7266;5.0703;4.9805;4.8789
```
## Developer

Zeping Mao

## Citation

If you use this module in your work, please cite:

Not published yet.

## License

This project is licensed under the MIT License.