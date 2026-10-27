"""Build the advanced model's label-free, complete-read cache."""
from __future__ import annotations
import argparse
import json
from data import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON or gzipped JSON site file")
    parser.add_argument("--output-dir", required=True, help="Separate cache directory")
    args = parser.parse_args()
    audit = prepare(args.input, args.output_dir)
    print(json.dumps({"status": audit["status"], "n_sites": audit["n_sites"],
                      "n_reads_total": audit["n_reads_total"],
                      "output_dir": audit["output_dir"],
                      "measurements": audit["measurements"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
