import argparse
import os
import glob
from spacial_boxcounting.api import boxcount_from_file, fractal_dimension

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def main():
    parser = argparse.ArgumentParser(description="spacial-boxcounting CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands: single, batch")

    # Single file processing
    parser_single = subparsers.add_parser("single", help="Process a single file")
    parser_single.add_argument("--file", required=True, help="Path to the input file")
    parser_single.add_argument("--mode", choices=["spatial", "single"], default="spatial", help="Mode to use")
    parser_single.add_argument("--hilbert", action="store_true", help="Apply Hilbert transform")

    # Batch processing
    parser_batch = subparsers.add_parser("batch", help="Process a folder of files")
    parser_batch.add_argument("--folder", required=True, help="Path to input folder")
    parser_batch.add_argument("--mode", choices=["spatial", "single"], default="spatial", help="Mode to use")
    parser_batch.add_argument("--hilbert", action="store_true", help="Apply Hilbert transform")
    parser_batch.add_argument("--pattern", default="*.*", help="File pattern for matching")

    args = parser.parse_args()

    if args.command == "single":
        result = boxcount_from_file(args.file, mode=args.mode, hilbert=args.hilbert)
        print(f"Result for file {args.file}:")
        print(result)
        fd = fractal_dimension(args.file, hilbert=args.hilbert)
        print(f"Fractal dimension: {fd:.3f}")

    elif args.command == "batch":
        files = glob.glob(os.path.join(args.folder, args.pattern))
        results = {}
        if tqdm:
            iterator = tqdm(files, desc="Processing files")
        else:
            iterator = files
        for file in iterator:
            try:
                res = boxcount_from_file(file, mode=args.mode, hilbert=args.hilbert)
                results[os.path.basename(file)] = res
            except Exception as e:
                results[os.path.basename(file)] = f"Error: {e}"
        print("Batch processing results:")
        for fname, res in results.items():
            print(f"{fname}: {res}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
