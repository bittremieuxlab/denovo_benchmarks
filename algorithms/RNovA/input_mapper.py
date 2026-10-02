"""
Merge a dataset's .mgf files into a single RNovA SeqFiller input file.

SeqFiller names each result after the spectrum's SCANS value, so every spectrum is
written with SCANS={filename}:{index} - the benchmark spectrum_id, with index the
0-based position in its original file - and results need no lookup afterwards.

Spectra are written ordered by peak count. SeqFiller pads each batch up to the node
bucket of its largest spectrum, so homogeneous batches waste far less compute; the
order is irrelevant to the output because results are keyed by SCANS.

Peak lines are normalised to "mz intensity", which is the only layout SeqFiller's
reader accepts, and peaks with non-positive intensity are dropped because SeqFiller
takes log(intensity). Two kinds of spectra are skipped and simply get no
prediction, because SeqFiller fails on them and takes the whole dataset down:
spectra left without peaks, and spectra whose charge is outside 1..9 (the model's
charge embedding has 10 slots, so charge 10+ is a CUDA device-side assert).

With --isotopes (e.g. "-1,0,1"), every spectrum is written once per isotope shift k,
its precursor m/z moved by k * 1.00335 / z and its SCANS id tagged "|iso<k>";
output_mapper.py keeps the highest-scoring copy. This lets SeqFiller's strict
precursor constraint recover spectra whose monoisotopic peak was mis-picked.
"""

import argparse
import os
import re

CHARGE_PATTERN = re.compile(r"(\d+)\s*([+-]?)")
MAX_CHARGE = 9  # nn.Embedding(10, ...) in model/node_embedding.py
C13_SHIFT = 1.0033548  # 13C - 12C
DEFAULT_CHARGE = 2     # what SeqFiller assumes when CHARGE is missing


def normalise_charge(value):
    """'2+', '2', '3+ and 4+' -> '2+' / '2+' / '3+' (first charge); None if unparsable."""
    m = CHARGE_PATTERN.search(value)
    if not m:
        return None
    return f"{m.group(1)}{m.group(2) or '+'}"


def read_spectra(mgf_path):
    """Yield (header lines, peak lines) for each spectrum in file order."""
    header, peaks, inside = [], [], False
    with open(mgf_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                header, peaks, inside = [], [], True
            elif line == "END IONS":
                if inside:
                    yield header, peaks
                inside = False
            elif inside and line[0].isdigit():
                fields = line.split()
                if len(fields) >= 2 and float(fields[1]) > 0:
                    peaks.append(f"{fields[0]} {fields[1]}")
            elif inside and "=" in line:
                header.append(line)


def charge_of(header):
    """Precursor charge as an int, or None when the spectrum has no usable CHARGE."""
    for line in header:
        key, _, value = line.partition("=")
        if key.upper() == "CHARGE":
            m = CHARGE_PATTERN.search(value)
            return int(m.group(1)) * (-1 if m.group(2) == "-" else 1) if m else None
    return None


def map_spectrum(header, peaks, spectrum_id, isotope=0, charge=None):
    out = ["BEGIN IONS"]
    for line in header:
        key, _, value = line.partition("=")
        key = key.upper()
        if key == "SCANS":
            continue  # replaced by the benchmark spectrum_id below
        if key == "PEPMASS" and isotope:
            mz, *rest = value.split()
            value = " ".join([f"{float(mz) + isotope * C13_SHIFT / abs(charge or DEFAULT_CHARGE):.6f}"] + rest)
        if key == "CHARGE":
            value = normalise_charge(value)
            if value is None:
                continue  # SeqFiller assumes 2+ when CHARGE is absent
        out.append(f"{key}={value}")
    out.append(f"SCANS={spectrum_id}")
    out.extend(peaks)
    out.append("END IONS")
    return "\n".join(out) + "\n"


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="Folder with the dataset .mgf files.")
parser.add_argument("--output_path", required=True, help="Merged .mgf file to write.")
parser.add_argument("--isotopes", default="-1,0", help="Comma-separated precursor isotope shifts; '0' disables.")
args = parser.parse_args()
isotopes = [int(k) for k in args.isotopes.split(",")]

records = []
for mgf_file in sorted(os.listdir(args.input_dir)):
    if not mgf_file.endswith(".mgf"):
        continue
    filename = os.path.splitext(mgf_file)[0]
    n = no_peaks = bad_charge = 0
    for idx, (header, peaks) in enumerate(read_spectra(os.path.join(args.input_dir, mgf_file))):
        n += 1
        if not peaks:
            no_peaks += 1
            continue
        charge = charge_of(header)
        if charge is not None and not 1 <= charge <= MAX_CHARGE:
            bad_charge += 1
            continue
        spectrum_id = f"{filename}:{idx}"
        if isotopes == [0]:
            records.append((len(peaks), map_spectrum(header, peaks, spectrum_id)))
        else:
            for k in isotopes:
                records.append((len(peaks), map_spectrum(header, peaks, f"{spectrum_id}|iso{k}", k, charge)))
    notes = [f"{no_peaks} without peaks"] * bool(no_peaks) + [f"{bad_charge} with charge outside 1..{MAX_CHARGE}"] * bool(bad_charge)
    print(f"{mgf_file}: {n} spectra" + (f", skipped {' and '.join(notes)}" if notes else ""))

records.sort(key=lambda r: r[0])
with open(args.output_path, "w") as f:
    for _, text in records:
        f.write(text)
print(f"{len(records)} spectra written to {args.output_path}.")
