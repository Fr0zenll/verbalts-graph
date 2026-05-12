import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an SMD variable dependency graph from train_ts.npy."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing train_ts.npy.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for graph outputs. Defaults to --data_dir.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Keep top-k strongest non-self neighbors per variable. Default: 3.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Keep edges with absolute correlation >= threshold. Overrides top-k.",
    )
    return parser.parse_args()


def load_train_ts(data_dir):
    train_path = data_dir / "train_ts.npy"
    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find train_ts.npy: {train_path}")

    train_ts = np.load(train_path)
    if train_ts.ndim != 3:
        raise ValueError(f"train_ts.npy must be 3D, got shape {train_ts.shape}.")
    if train_ts.shape[1] != 240:
        raise ValueError(f"Expected window length 240, got shape {train_ts.shape}.")
    if train_ts.shape[2] != 10:
        raise ValueError(f"Expected 10 variables, got shape {train_ts.shape}.")
    return train_ts


def compute_abs_corr(train_ts):
    # train_ts: [N, T, V]. Flatten all windows and time points to [N*T, V],
    # so Pearson correlation is computed between variables over all observations.
    num_samples, window_len, num_vars = train_ts.shape
    flattened = train_ts.reshape(num_samples * window_len, num_vars)
    corr = np.corrcoef(flattened, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr, np.abs(corr)


def sparsify_top_k(strength, top_k):
    if top_k < 0:
        raise ValueError(f"--top_k must be non-negative, got {top_k}.")

    num_vars = strength.shape[0]
    adj = np.zeros_like(strength, dtype=np.float64)
    k = min(top_k, max(num_vars - 1, 0))

    for i in range(num_vars):
        row = strength[i].copy()
        row[i] = -np.inf  # Self-loop is added later and does not count in top-k.
        if k > 0:
            neighbor_ids = np.argsort(row)[-k:]
            adj[i, neighbor_ids] = strength[i, neighbor_ids]
    return adj


def sparsify_threshold(strength, threshold):
    if threshold < 0:
        raise ValueError(f"--threshold must be non-negative, got {threshold}.")

    adj = np.where(strength >= threshold, strength, 0.0)
    np.fill_diagonal(adj, 0.0)  # Self-loop is added later.
    return adj


def add_self_loops_and_symmetrize(adj):
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 1.0)
    return adj


def normalize_adj(adj):
    degree = adj.sum(axis=1)
    inv_sqrt_degree = np.zeros_like(degree, dtype=np.float64)
    nonzero = degree > 0
    inv_sqrt_degree[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]


def connected_neighbors(adj_unnormalized):
    neighbors = {}
    for i in range(adj_unnormalized.shape[0]):
        ids = np.where(adj_unnormalized[i] > 0)[0].tolist()
        neighbors[str(i)] = ids
    return neighbors


def count_edges(adj_unnormalized):
    # Count undirected non-self edges once.
    upper = np.triu(adj_unnormalized > 0, k=1)
    return int(upper.sum())


def print_matrix(title, matrix):
    print(title)
    print(np.array2string(matrix, precision=4, suppress_small=True))


def save_outputs(output_dir, corr_raw, adj_unnormalized, adj_norm, info):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "graph_corr_raw.npy", corr_raw)
    np.save(output_dir / "graph_adj.npy", adj_norm)
    np.save(output_dir / "graph_adj_unnormalized.npy", adj_unnormalized)
    with open(output_dir / "graph_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"--data_dir is not a directory: {data_dir}")

    train_ts = load_train_ts(data_dir)
    corr_raw, strength = compute_abs_corr(train_ts)

    if args.threshold is not None:
        print("Both --threshold and --top_k are available; using threshold sparsification.")
        adj = sparsify_threshold(strength, args.threshold)
        sparsify_method = "threshold"
    else:
        adj = sparsify_top_k(strength, args.top_k)
        sparsify_method = "top_k"

    adj_unnormalized = add_self_loops_and_symmetrize(adj)
    adj_norm = normalize_adj(adj_unnormalized)

    num_vars = adj_unnormalized.shape[0]
    edge_count = count_edges(adj_unnormalized)
    max_edges = num_vars * (num_vars - 1) // 2
    sparsity = 1.0 - (edge_count / max_edges if max_edges > 0 else 0.0)
    neighbors = connected_neighbors(adj_unnormalized)

    info = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "train_ts_shape": list(train_ts.shape),
        "reshape_for_corr": [int(train_ts.shape[0] * train_ts.shape[1]), int(train_ts.shape[2])],
        "num_vars": int(num_vars),
        "corr_type": "pearson",
        "use_abs_corr": True,
        "sparsify_method": sparsify_method,
        "top_k": int(args.top_k),
        "threshold": args.threshold,
        "self_loop": True,
        "symmetrize": "max(A, A.T)",
        "normalization": "D^{-1/2} A D^{-1/2}",
        "edge_count_undirected_no_self": edge_count,
        "max_edges_undirected_no_self": max_edges,
        "sparsity_no_self": sparsity,
        "neighbors_with_self_loop": neighbors,
        "output_files": [
            "graph_corr_raw.npy",
            "graph_adj.npy",
            "graph_adj_unnormalized.npy",
            "graph_info.json",
        ],
    }

    print(f"Loaded train_ts.npy shape: {train_ts.shape}")
    print(f"Reshaped for correlation: {info['reshape_for_corr']}")
    print_matrix("Raw Pearson correlation matrix:", corr_raw)
    print_matrix("Sparsified adjacency matrix before normalization:", adj_unnormalized)
    print_matrix("GCN-normalized adjacency matrix:", adj_norm)

    print("Variable connections:")
    for node, ids in neighbors.items():
        print(f"  {node} -> {ids}")
    print(f"Undirected edge count without self-loops: {edge_count}")
    print(f"Sparsity without self-loops: {sparsity:.4f}")

    save_outputs(output_dir, corr_raw, adj_unnormalized, adj_norm, info)
    print(f"Saved graph files to: {output_dir}")


if __name__ == "__main__":
    main()
