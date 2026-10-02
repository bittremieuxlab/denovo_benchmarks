# RNovA

[RNovA](https://github.com/zqq66/RNovA) (Zero-shot de novo peptide sequencing with open
posttranslational modification discovery, https://www.nature.com/articles/s41587-026-03116-1)
for the benchmark. Only its sequencing module, **SeqFiller**, is used: PathSearcher, decoy
generation and FDR filtering are not part of a benchmark run.

## Contents

| Path | |
|---|---|
| `RNovA/RNovA_SeqFiller_Inference/` | SeqFiller inference code, vendored from [zqq66/RNovA_SeqFiller_Inference](https://github.com/zqq66/RNovA_SeqFiller_Inference) @ `8196c1a`, with the modifications below |
| `RNovA/LICENSE` | RNovA licence (Apache-2.0); SeqFiller's README states MIT |
| `container.def` | Python 3.12 image: requirements, FlashAttention wheel, Cython knapsack extension, SeqFiller weights from [Zenodo 18352464](https://zenodo.org/records/18352464) (md5-checked), prebuilt knapsack table |
| `make_predictions.sh` | merge the dataset, run SeqFiller, map the output |
| `input_mapper.py` | merge all `.mgf` files into one input, with the benchmark `spectrum_id` written as `SCANS` |
| `output_mapper.py` | ProForma with `UNIMOD` accessions, per-residue scores, best isotope copy per spectrum |
| `prebuild_knapsack.py`, `candidates.txt` | default candidate residues and their prebuilt knapsack table |

## Settings

- Candidate residues: the 20 amino acids with carbamidomethyl C and oxidised M, for every
  dataset (`candidates.txt`).
- Precursor tolerance 10 ppm; 8 refinement iterations; batch size 128.
- Precursor isotope shifts −1,0: every spectrum is also decoded with its precursor one ¹³C
  lighter (a mis-picked monoisotopic peak) and the higher-scoring result is kept, the same
  direction as `isotope_error_range: [0, 1]` in Casanovo and InstaNovo.
- Sequences whose mean residue score is below −2.0 are not reported (fixed).

Optional overrides via the harness `--env-file` are listed at the top of `make_predictions.sh`.

## Changes to SeqFiller

Relative to upstream `8196c1a`:

- Precursor tolerance and refinement iterations are configurable, defaulting to 10 ppm and
  8 (upstream: 5 ppm, 4). The iteration embedding has 40 slots, so 8 iterations load the
  released checkpoint unchanged.
- `Inference_Sequence.py` does not write sequences with mean residue score < −2.0.
- Speed, with output unchanged: the knapsack table is cached on disk (prebuilt in the image)
  instead of rebuilt per file; the knapsack mask is built on the GPU instead of in a
  per-step Python loop; `torch.compile` is removed from the decode step; 8 data-loader
  workers.
- `RNOVA_BATCH_SIZE` overrides the batch size; unknown candidate names raise a clear error.

The wrapper skips spectra SeqFiller cannot process (no peaks, or charge outside 1–9, beyond
the model's charge embedding); they get no prediction.

## Requirements

An NVIDIA GPU of the Ampere generation or newer (FlashAttention); about 16 GB of GPU memory at
batch size 128 (`RNOVA_BATCH_SIZE` lowers it).
