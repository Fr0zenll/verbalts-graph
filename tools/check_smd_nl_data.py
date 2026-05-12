import argparse
import json
import random
from pathlib import Path

import numpy as np


SPLITS = ("train", "valid", "test")
REQUIRED_SUFFIXES = ("ts.npy", "text_caps.npy", "attrs_idx.npy")
EXPECTED_TS_LEN = 240
EXPECTED_NUM_VARS = 10
EXPECTED_NUM_CAPS = 3
EXPECTED_NUM_ATTRS = 5
VALID_ATTR_VALUES = {0, 1, 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check SMD natural-language time-series dataset files."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing SMD npy files and meta.json.",
    )
    return parser.parse_args()


def required_files(data_dir):
    files = [data_dir / "meta.json"]
    for split in SPLITS:
        for suffix in REQUIRED_SUFFIXES:
            files.append(data_dir / f"{split}_{suffix}")
    return files


def check_required_files(data_dir):
    missing = [path for path in required_files(data_dir) if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required files:\n{missing_text}")


def load_split(data_dir, split):
    ts = np.load(data_dir / f"{split}_ts.npy")
    text_caps = np.load(data_dir / f"{split}_text_caps.npy", allow_pickle=True)
    attrs_idx = np.load(data_dir / f"{split}_attrs_idx.npy")
    return ts, text_caps, attrs_idx


def check_2d_shape(name, array, expected_second_dim):
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}.")
    if array.shape[1] != expected_second_dim:
        raise ValueError(
            f"{name} second dimension must be {expected_second_dim}, got {array.shape}."
        )


def check_split(split, ts, text_caps, attrs_idx):
    print(f"{split}:")
    print(f"  ts shape: {ts.shape}")
    print(f"  text_caps shape: {text_caps.shape}")
    print(f"  attrs_idx shape: {attrs_idx.shape}")

    if ts.ndim != 3:
        raise ValueError(f"{split}_ts.npy must be 3D, got shape {ts.shape}.")

    n_ts, window_len, num_vars = ts.shape
    n_caps = text_caps.shape[0]
    n_attrs = attrs_idx.shape[0]

    if not (n_ts == n_caps == n_attrs):
        raise ValueError(
            f"{split} sample counts differ: ts={n_ts}, "
            f"text_caps={n_caps}, attrs_idx={n_attrs}."
        )
    if window_len != EXPECTED_TS_LEN:
        raise ValueError(
            f"{split}_ts.npy window length must be {EXPECTED_TS_LEN}, got {window_len}."
        )
    if num_vars != EXPECTED_NUM_VARS:
        raise ValueError(
            f"{split}_ts.npy variable count must be {EXPECTED_NUM_VARS}, got {num_vars}."
        )

    check_2d_shape(f"{split}_text_caps.npy", text_caps, EXPECTED_NUM_CAPS)
    check_2d_shape(f"{split}_attrs_idx.npy", attrs_idx, EXPECTED_NUM_ATTRS)

    unique_attrs = set(np.unique(attrs_idx).tolist())
    invalid_attrs = sorted(unique_attrs - VALID_ATTR_VALUES)
    if invalid_attrs:
        raise ValueError(
            f"{split}_attrs_idx.npy contains invalid values {invalid_attrs}; "
            f"allowed values are {sorted(VALID_ATTR_VALUES)}."
        )


def print_train_examples(train_ts, train_text_caps, train_attrs_idx, n_examples=3):
    n_samples = train_ts.shape[0]
    if n_samples == 0:
        raise ValueError("train split is empty; cannot print sample examples.")

    n_examples = min(n_examples, n_samples)
    sample_indices = random.sample(range(n_samples), k=n_examples)

    print("Random train samples:")
    for idx in sample_indices:
        print(f"  sample index: {idx}")
        print(f"    ts shape: {train_ts[idx].shape}")
        print("    text descriptions:")
        for cap_id, cap in enumerate(train_text_caps[idx]):
            print(f"      {cap_id}: {cap}")
        print(f"    attrs_idx: {train_attrs_idx[idx].tolist()}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"--data_dir is not a directory: {data_dir}")

    check_required_files(data_dir)

    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"Loaded meta.json with keys: {sorted(meta.keys())}")

    loaded = {}
    for split in SPLITS:
        ts, text_caps, attrs_idx = load_split(data_dir, split)
        check_split(split, ts, text_caps, attrs_idx)
        loaded[split] = (ts, text_caps, attrs_idx)

    print_train_examples(*loaded["train"])
    print("SMD data check passed.")


if __name__ == "__main__":
    main()
