"""Download reference proteomes into PROTEOMES_DIR.

By default, downloads the proteome files named in dataset_tags.tsv from the MassIVE reanalysis RMSV000000721, where the
files used for the public benchmark datasets are published. With --uniprot, downloads the current version of a UniProt
proteome instead, named like the files in dataset_tags.tsv.

    python download_proteomes.py [--datasets 9_species_human mouse_mAb_trypsin] [--out_dir DIR]
    python download_proteomes.py --uniprot UP000005640 --organism human [--isoforms] [--out_dir DIR]
"""

import argparse
import csv
import datetime
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MASSIVE_URL = "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile"
MASSIVE_DIR = "f.RMSV000000721/2026-11-04_marinapoms_a0dce19b/sequence/proteomes"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/stream"


def download(url, path, retries=10):
    """Download url to path. Return False if the file isn't there; wait and retry while the server limits requests."""
    request = urllib.request.Request(url, headers={"User-Agent": "denovo_benchmarks download_proteomes.py"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=600) as response, open(path + ".part", "wb") as f:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    f.write(chunk)
            break
        except urllib.error.HTTPError as e:
            if e.code in (404, 500):  # MassIVE answers 500 for a file it doesn't have
                return False
            if e.code != 429 or attempt == retries - 1:
                raise
            print("  server busy (HTTP 429), retrying in 60 s")
            time.sleep(60)
    with open(path + ".part", "rb") as f:
        is_fasta = f.read(1) == b">"
    if not is_fasta:
        os.remove(path + ".part")
        return False
    os.replace(path + ".part", path)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default=os.environ.get("PROTEOMES_DIR"), help="target folder (default: $PROTEOMES_DIR)")
    parser.add_argument("--datasets", nargs="+", help="only the proteomes of these datasets (default: all in dataset_tags.tsv)")
    parser.add_argument("--uniprot", help="UniProt proteome ID to download instead, e.g. UP000005640")
    parser.add_argument("--organism", help="organism name for the file name, e.g. human (with --uniprot)")
    parser.add_argument("--isoforms", action="store_true", help="include isoforms (with --uniprot)")
    args = parser.parse_args()
    if not args.out_dir:
        sys.exit("Set PROTEOMES_DIR or pass --out_dir.")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.uniprot:
        if not args.organism:
            sys.exit("--uniprot needs --organism.")
        date = datetime.date.today().strftime("%Y_%m_%d")
        kind = "isoforms_uniprotkb" if args.isoforms else "uniprotkb"
        path = os.path.join(args.out_dir, f"{args.organism}_{kind}_proteome_{args.uniprot}_{date}.fasta")
        query = {"query": f"proteome:{args.uniprot}", "format": "fasta", "includeIsoform": str(args.isoforms).lower()}
        if not download(f"{UNIPROT_URL}?{urllib.parse.urlencode(query)}", path):
            sys.exit(f"UniProt returned no FASTA for {args.uniprot}.")
        print(f"Saved {path}")
        return

    tags_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_tags.tsv")
    with open(tags_path) as f:
        tags = {row["dataset"]: row["proteome"] for row in csv.DictReader(f, delimiter="\t")}
    datasets = args.datasets or list(tags)
    unknown = [d for d in datasets if d not in tags]
    if unknown:
        sys.exit(f"Not in dataset_tags.tsv: {' '.join(unknown)}")
    files = sorted({tags[d] for d in datasets if tags[d] != "0"} | {"crap.fasta"})

    missing = []
    for name in files:
        path = os.path.join(args.out_dir, name)
        if os.path.exists(path):
            print(f"{name}: already there")
            continue
        print(f"{name}: downloading")
        query = {"file": f"{MASSIVE_DIR}/{name}", "forceDownload": "true"}
        if not download(f"{MASSIVE_URL}?{urllib.parse.urlencode(query)}", path):
            print(f"{name}: not on MassIVE")
            missing.append(name)
    if missing:
        print(f"\nNot available on MassIVE ({len(missing)}): {' '.join(missing)}")


if __name__ == "__main__":
    main()
