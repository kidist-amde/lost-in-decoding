#!/usr/bin/env python3
"""Truncate a top_bow docid_to_tokenids.json from m=64 to a smaller m.

Streams through the source file to avoid loading ~3GB into memory at once.
Usage:
    python scripts/truncate_bow.py --m 32
"""
import argparse
import os
import sys

SPLADE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "experiments-splade", "t5-splade-0-12l",
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True, help="Target number of tokens per document")
    parser.add_argument("--src", default=os.path.join(SPLADE_DIR, "top_bow", "docid_to_tokenids.json"))
    parser.add_argument("--dst", default=None, help="Output path (default: top_bow_{m}/docid_to_tokenids.json)")
    args = parser.parse_args()

    if args.dst is None:
        args.dst = os.path.join(SPLADE_DIR, f"top_bow_{args.m}", "docid_to_tokenids.json")

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)

    print(f"Truncating {args.src} -> {args.dst}  (m={args.m})")
    print(f"Source size: {os.path.getsize(args.src) / 1e9:.2f} GB")

    n_docs = 0
    with open(args.src) as fin, open(args.dst, "w") as fout:
        fout.write("{")
        first = True
        # Stream-parse: the file is one big JSON object on a single line.
        # We read it in chunks and parse key-value pairs incrementally.
        # Since ujson.load would need all memory, we use a different approach:
        # Read the whole file but process with ijson or a manual approach.
        # Actually, for a single-line JSON dict, the simplest robust approach
        # is to use ujson with a file object (it streams internally).
        import ujson
        print("Loading source file...")
        data = ujson.load(fin)
        print(f"Loaded {len(data)} documents. Writing truncated output...")
        for docid, tokens in data.items():
            truncated = tokens[:args.m]
            if not first:
                fout.write(",")
            fout.write(f'"{docid}":{ujson.dumps(truncated)}')
            first = False
            n_docs += 1
            if n_docs % 1_000_000 == 0:
                print(f"  {n_docs:,} documents processed...")
        fout.write("}")

    dst_size = os.path.getsize(args.dst)
    print(f"Done. {n_docs:,} documents, output size: {dst_size / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
