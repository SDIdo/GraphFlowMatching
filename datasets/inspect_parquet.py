#!/usr/bin/env python3
"""
Inspect a HuggingFace-style parquet ImageNet dump.

Reports the schema, row counts and -- crucially -- whether the shards carry the
ORIGINAL FILENAMES. That single fact decides whether ImageNet-LT can use its
official split file:

  * filenames present  -> the official ImageNet_LT_train.txt can be matched
                          row-by-row, giving the exact published split.
  * filenames absent   -> only a Pareto reconstruction from the labels is
                          possible: same long-tailed count profile, different
                          images, NOT comparable to published ImageNet-LT numbers.

Usage
-----
    python datasets/inspect_parquet.py --root /path/to/ImageNet/data
    python datasets/inspect_parquet.py --root /path/to/data --shards 3
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Directory holding *.parquet")
    p.add_argument("--pattern", default="*.parquet")
    p.add_argument("--shards", type=int, default=1,
                   help="How many shards to open in detail")
    p.add_argument("--rows", type=int, default=5, help="Sample rows to print")
    args = p.parse_args(argv)

    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    if not files:
        raise SystemExit(f"no files matching {args.pattern} in {args.root}")

    train = [f for f in files if "train" in os.path.basename(f).lower()]
    val = [f for f in files if "val" in os.path.basename(f).lower()]
    print(f"[shards] {len(files)} parquet files "
          f"({len(train)} train, {len(val)} val/other)")
    total_bytes = sum(os.path.getsize(f) for f in files)
    print(f"[size]   {total_bytes / 1e9:.1f} GB total")

    f0 = (train or files)[0]
    pf = pq.ParquetFile(f0)
    print(f"\n[schema] {os.path.basename(f0)}")
    print(f"  rows          : {pf.metadata.num_rows:,}")
    print(f"  row groups    : {pf.metadata.num_row_groups}")
    print(f"  columns       : {pf.schema_arrow.names}")
    print("  arrow schema  :")
    for line in str(pf.schema_arrow).splitlines():
        print(f"    {line}")

    _sch = pf.metadata.schema
    leaf_paths = [_sch.column(i).path for i in range(len(_sch))]
    print(f"  parquet leaves: {leaf_paths}")

    # Which column might hold the original filename?
    name_cols = [c for c in leaf_paths
                 if c.split(".")[-1].lower() in ("path", "filename", "file_name",
                                                 "name", "id", "image_id")]
    print(f"\n[filenames] candidate columns: {name_cols or 'NONE FOUND'}")

    if name_cols:
        col = name_cols[0]
        tbl = pf.read_row_group(0, columns=[col])
        vals = tbl.column(0).to_pylist()[:args.rows]
        print(f"  sample values from '{col}':")
        for v in vals:
            print(f"    {v!r}")
        looks_original = any(
            isinstance(v, str) and v and ("n0" in v or "n1" in v or "JPEG" in v.upper())
            for v in vals)
        print(f"\n  -> {'LOOKS LIKE original ImageNet filenames' if looks_original else 'does NOT look like original filenames'}")
        print("     " + ("ImageNet-LT can use its official split."
                         if looks_original else
                         "ImageNet-LT would need the Pareto reconstruction."))
    else:
        print("  -> no filename column; ImageNet-LT can only be reconstructed "
              "from labels (not the official split).")

    # Labels
    label_cols = [c for c in leaf_paths if "label" in c.lower()]
    if label_cols:
        tbl = pf.read_row_group(0, columns=[label_cols[0]])
        lv = tbl.column(0).to_pylist()
        print(f"\n[labels] column '{label_cols[0]}', "
              f"first values {lv[:args.rows]}, "
              f"range in row-group 0: {min(lv)}..{max(lv)}")

    # Total rows across the train shards (metadata only -- cheap).
    n = 0
    for f in train or files:
        n += pq.ParquetFile(f).metadata.num_rows
    print(f"\n[total] {n:,} rows across {len(train or files)} train shard(s)")
    if n == 1_281_167:
        print("  matches ImageNet-1k train exactly (1,281,167)")

    for f in (train or files)[1:args.shards]:
        pf = pq.ParquetFile(f)
        print(f"  {os.path.basename(f)}: {pf.metadata.num_rows:,} rows, "
              f"{pf.metadata.num_row_groups} row groups")


if __name__ == "__main__":
    main()
