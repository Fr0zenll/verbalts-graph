import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERBALTS_ROOT = PROJECT_ROOT / "VerbalTS-main"
DEFAULT_ADJ_CANDIDATES = (
    PROJECT_ROOT / "data" / "smd_nl" / "graph_adj.npy",
    PROJECT_ROOT / "data" / "SMD" / "graph_adj.npy",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test SimpleVariableGCN independently.")
    parser.add_argument(
        "--adj_path",
        default=None,
        help="Optional path to graph_adj.npy for testing.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device for the test. Default: auto.",
    )
    return parser.parse_args()


def import_torch_and_gnn():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the current Python environment. "
            "Activate the VerbalTS environment or create one, for example: "
            "conda create -n verbalts python=3.10 && conda activate verbalts && "
            "pip install torch==2.2.1 numpy"
        ) from exc

    sys.path.insert(0, str(VERBALTS_ROOT))
    try:
        from models.graph_modules import SimpleVariableGCN
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import SimpleVariableGCN from {VERBALTS_ROOT}."
        ) from exc
    return torch, SimpleVariableGCN


def resolve_device(torch, requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_adj(torch, adj):
    degree = adj.sum(dim=1)
    inv_sqrt_degree = torch.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt_degree[nonzero] = 1.0 / torch.sqrt(degree[nonzero])
    return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]


def build_identity_adj(torch, num_variables, device):
    return torch.eye(num_variables, dtype=torch.float32, device=device)


def build_random_symmetric_adj(torch, num_variables, device):
    random_adj = torch.rand(num_variables, num_variables, dtype=torch.float32, device=device)
    random_adj = 0.5 * (random_adj + random_adj.T)
    random_adj.fill_diagonal_(1.0)
    return normalize_adj(torch, random_adj)


def load_adj(torch, adj_path, device):
    path = Path(adj_path)
    if not path.exists():
        raise FileNotFoundError(f"adj_path does not exist: {path}")
    adj = np.load(path)
    if adj.shape != (10, 10):
        raise ValueError(f"Loaded adjacency must have shape (10, 10), got {adj.shape}.")
    return torch.as_tensor(adj, dtype=torch.float32, device=device)


def discover_loaded_adjs(torch, args, device):
    adjs = [
        ("identity", build_identity_adj(torch, 10, device)),
        ("random_symmetric", build_random_symmetric_adj(torch, 10, device)),
    ]

    if args.adj_path:
        adjs.append((f"loaded:{args.adj_path}", load_adj(torch, args.adj_path, device)))
    else:
        for candidate in DEFAULT_ADJ_CANDIDATES:
            if candidate.exists():
                adjs.append((f"loaded:{candidate}", load_adj(torch, candidate, device)))
    return adjs


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def assert_tensor_ok(torch, y, expected_shape, x):
    if tuple(y.shape) != expected_shape:
        raise AssertionError(f"Expected output shape {expected_shape}, got {tuple(y.shape)}.")
    if torch.isnan(y).any():
        raise AssertionError("Output contains NaN.")
    if torch.isinf(y).any():
        raise AssertionError("Output contains inf.")
    if y.dtype != torch.float32:
        raise AssertionError(f"Expected output dtype torch.float32, got {y.dtype}.")
    if y.device != x.device:
        raise AssertionError(f"Output device {y.device} differs from input device {x.device}.")


def run_case(torch, SimpleVariableGCN, adj_name, adj, ndim, device):
    if ndim == 3:
        x_shape = (4, 240, 10)
        expected_shape = x_shape
        model = SimpleVariableGCN(
            num_variables=10,
            input_dim=1,
            hidden_dim=32,
            output_dim=1,
            num_layers=2,
            dropout=0.1,
            residual=True,
            activation="relu",
        ).to(device)
    elif ndim == 4:
        x_shape = (4, 240, 10, 32)
        expected_shape = x_shape
        model = SimpleVariableGCN(
            num_variables=10,
            input_dim=32,
            hidden_dim=32,
            output_dim=32,
            num_layers=2,
            dropout=0.1,
            residual=True,
            activation="relu",
        ).to(device)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")

    x = torch.randn(*x_shape, dtype=torch.float32, device=device, requires_grad=True)
    model.train()
    y = model(x, adj)
    assert_tensor_ok(torch, y, expected_shape, x)

    loss = y.pow(2).mean()
    loss.backward()

    if x.grad is None:
        raise AssertionError("Backward failed: input gradient is None.")
    if torch.isnan(x.grad).any() or torch.isinf(x.grad).any():
        raise AssertionError("Backward failed: input gradient contains NaN or inf.")

    grad_params = [param for param in model.parameters() if param.requires_grad and param.grad is not None]
    if not grad_params:
        raise AssertionError("Backward failed: no trainable parameter received gradients.")

    print(f"[OK] adj={adj_name}, input={x_shape}, output={tuple(y.shape)}")
    print(f"     dtype={y.dtype}, device={y.device}, params={count_parameters(model)}")
    print("     backward=success")


def main():
    args = parse_args()
    torch, SimpleVariableGCN = import_torch_and_gnn()
    device = resolve_device(torch, args.device)
    print(f"Using device: {device}")

    adjs = discover_loaded_adjs(torch, args, device)
    for adj_name, adj in adjs:
        if adj.shape != (10, 10):
            raise AssertionError(f"Adjacency {adj_name} has invalid shape {adj.shape}.")
        if adj.dtype != torch.float32:
            raise AssertionError(f"Adjacency {adj_name} must be float32, got {adj.dtype}.")
        if adj.device != device:
            raise AssertionError(f"Adjacency {adj_name} is on {adj.device}, expected {device}.")
        print(f"\nTesting adjacency: {adj_name}, shape={tuple(adj.shape)}")
        run_case(torch, SimpleVariableGCN, adj_name, adj, ndim=3, device=device)
        run_case(torch, SimpleVariableGCN, adj_name, adj, ndim=4, device=device)

    print("\nGraph module tests passed.")


if __name__ == "__main__":
    main()
