"""
Generic local dataset module for working with parquet files.

This module provides utilities for:
- Listing parquet files in the data directory
- Iterating over parquet files and yielding batches of documents
- Getting statistics about the dataset (shard count, rows, size)
- Validating dataset integrity

The dataset is expected to be generated locally via the scraping/repackaging
pipeline. For details on data preparation, see `repackage_data_reference.py`.
"""

import os
import argparse
import pyarrow.parquet as pq

from common import get_base_dir

# -----------------------------------------------------------------------------
# Data directory setup

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

def dataset_info(data_dir=None):
    """
    Scan the local parquet files and return dataset statistics.

    Returns:
        dict with keys:
            - shard_count: number of parquet files
            - total_row_groups: sum of row groups across all shards
            - estimated_docs: estimated total documents (sum of rows across all row groups)
            - total_size_bytes: total size of all parquet files in bytes
    """
    parquet_paths = list_parquet_files(data_dir)

    shard_count = len(parquet_paths)
    total_row_groups = 0
    estimated_docs = 0
    total_size_bytes = 0

    for filepath in parquet_paths:
        total_size_bytes += os.path.getsize(filepath)
        pf = pq.ParquetFile(filepath)
        total_row_groups += pf.num_row_groups
        for rg_idx in range(pf.num_row_groups):
            estimated_docs += pf.metadata.row_group(rg_idx).num_rows

    return {
        'shard_count': shard_count,
        'total_row_groups': total_row_groups,
        'estimated_docs': estimated_docs,
        'total_size_bytes': total_size_bytes,
    }

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset utilities for local parquet files")
    parser.add_argument("--info", action="store_true", help="Print dataset statistics")
    parser.add_argument("--validate", action="store_true", help="Validate all parquet files are readable and have 'text' column")
    args = parser.parse_args()

    if args.info:
        print("=" * 60)
        print("Dataset Information")
        print("=" * 60)
        print(f"Data directory: {DATA_DIR}")
        print()

        info = dataset_info()
        print(f"{'Shard count':20s}: {info['shard_count']}")
        print(f"{'Total row groups':20s}: {info['total_row_groups']:,}")
        print(f"{'Estimated documents':20s}: {info['estimated_docs']:,}")
        print(f"{'Total size':20s}: {info['total_size_bytes']:,} bytes ({info['total_size_bytes'] / (1024**3):.2f} GB)")
        print("=" * 60)

    if args.validate:
        print("=" * 60)
        print("Dataset Validation")
        print("=" * 60)
        print(f"Data directory: {DATA_DIR}")
        print()

        parquet_paths = list_parquet_files()
        if not parquet_paths:
            print("No parquet files found!")
            exit(1)

        all_valid = True
        for i, filepath in enumerate(parquet_paths, 1):
            filename = os.path.basename(filepath)
            try:
                pf = pq.ParquetFile(filepath)
                schema = pf.schema_arrow

                if 'text' not in schema.names:
                    print(f"[{i}/{len(parquet_paths)}] {filename}: FAILED - missing 'text' column")
                    all_valid = False
                else:
                    num_rows = sum(pf.metadata.row_group(rg_idx).num_rows for rg_idx in range(pf.num_row_groups))
                    print(f"[{i}/{len(parquet_paths)}] {filename}: OK ({pf.num_row_groups} row groups, {num_rows:,} rows)")
            except Exception as e:
                print(f"[{i}/{len(parquet_paths)}] {filename}: FAILED - {e}")
                all_valid = False

        print()
        if all_valid:
            print("All parquet files are valid!")
        else:
            print("Some parquet files failed validation.")
            exit(1)
        print("=" * 60)

    if not args.info and not args.validate:
        parser.print_help()
