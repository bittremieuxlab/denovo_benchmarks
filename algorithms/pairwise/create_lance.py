import argparse
from pathlib import Path
from depthcharge.data import SpectrumDataset
import pyarrow as pa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Input directory containing .mgf files"
    )
    parser.add_argument("--output", required=True, help="Path to output .lance dataset")
    parser.add_argument(
        "--suffix", default=".mgf", help="File type suffix (default: .mgf)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size (default: 1000)"
    )
    args = parser.parse_args()

    source_dir = Path(args.input)
    output_path = Path(args.output)
    files = list(source_dir.rglob(f"*{args.suffix}"))
    print(f"Found {len(files)} files.")

    # Create the .lance dataset without any annotation fields
    SpectrumDataset(
        spectra=files,
        path=output_path,
        batch_size=args.batch_size,
        min_peaks=1,
        custom_fields=[pa.field("title", pa.string())],
    )

    print(f"Lance dataset created at {output_path}")


if __name__ == "__main__":
    main()
