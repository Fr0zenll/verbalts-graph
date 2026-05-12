import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def fail(category, message):
    raise RuntimeError(f"[{category}] {message}")


def resolve_path(path, project_root, verbalts_root):
    candidates = [
        Path(path),
        project_root / path,
        verbalts_root / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def load_yaml_config(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as exc:
        fail("config problem", f"Failed to read YAML config '{config_path}': {exc}")
    if not isinstance(config, dict):
        fail("config problem", f"Config must be a YAML mapping, got {type(config).__name__}.")
    return config


def extract_diff_config(raw_config):
    if "diffusion" in raw_config:
        diff_config = copy.deepcopy(raw_config["diffusion"])
        device = raw_config.get("device", "cpu")
        generator_pretrain_path = raw_config.get("generator_pretrain_path", "")
    elif "model_diff" in raw_config and "diffusion" in raw_config["model_diff"]:
        diff_config = copy.deepcopy(raw_config["model_diff"]["diffusion"])
        device = raw_config["model_diff"].get("device", raw_config.get("device", "cpu"))
        generator_pretrain_path = raw_config["model_diff"].get("generator_pretrain_path", "")
    else:
        fail(
            "config problem",
            "Config must contain a 'diffusion' section, or 'model_diff.diffusion'.",
        )
    return diff_config, device, generator_pretrain_path


def load_batch(data_dir, verbalts_root, batch_size=2):
    sys.path.insert(0, str(verbalts_root))
    try:
        from data import GenerationDataset
    except Exception as exc:
        fail("config problem", f"Failed to import VerbalTS data loader: {exc}")

    try:
        dataset = GenerationDataset({"name": "custom", "folder": str(data_dir)})
        loader = dataset.get_loader("train", batch_size=batch_size, shuffle=False, num_workers=0)
        batch = next(iter(loader))
    except FileNotFoundError as exc:
        fail("data problem", f"Missing SMD-NL dataset file: {exc}")
    except Exception as exc:
        fail("data problem", f"Failed to load one SMD-NL batch: {exc}")
    return dataset, batch


def validate_batch(batch):
    required = ("ts", "tp", "attrs", "cap")
    missing = [key for key in required if key not in batch]
    if missing:
        fail("data problem", f"Batch is missing required keys: {missing}")

    ts = batch["ts"]
    attrs = batch["attrs"]
    if tuple(ts.shape) != (2, 240, 10):
        fail("shape problem", f"Expected batch['ts'] shape (2, 240, 10), got {tuple(ts.shape)}.")
    if attrs.ndim != 2:
        fail("shape problem", f"Expected batch['attrs'] to be 2D, got {tuple(attrs.shape)}.")


def build_generator(diff_config, device, graph_adj_path, use_gnn, generator_pretrain_path, verbalts_root):
    try:
        from models.unconditional_generator import UnConditionalGenerator
    except Exception as exc:
        fail("config problem", f"Failed to import VerbalTS model code: {exc}")

    diff_config = copy.deepcopy(diff_config)
    diff_config["device"] = device
    diff_config["use_gnn"] = use_gnn
    diff_config["debug_shape"] = True

    if use_gnn:
        if not graph_adj_path:
            fail("graph_adj problem", "--use_gnn true requires --graph_adj_path.")
        if not Path(graph_adj_path).exists():
            fail("graph_adj problem", f"graph_adj_path does not exist: {graph_adj_path}")
        diff_config["graph_adj_path"] = str(graph_adj_path)
    else:
        diff_config["graph_adj_path"] = str(graph_adj_path) if graph_adj_path else ""

    configs = {
        "device": device,
        "generator_pretrain_path": generator_pretrain_path or "",
        "diffusion": diff_config,
    }

    try:
        model = UnConditionalGenerator(configs).to(device)
        model.eval()
    except FileNotFoundError as exc:
        fail("graph_adj problem", str(exc))
    except ValueError as exc:
        message = str(exc)
        category = "graph_adj problem" if "graph_adj" in message else "shape problem"
        fail(category, message)
    except Exception as exc:
        fail("model forward parameter problem", f"Failed to construct model: {exc}")
    return model


def print_batch_info(batch):
    print("batch keys:", sorted(batch.keys()))
    print("time series shape:", tuple(batch["ts"].shape))
    cap = batch["cap"]
    if isinstance(cap, (list, tuple)) and len(cap) > 0:
        print("text condition example:", cap[0])
    else:
        print("text condition example:", cap)
    print("attrs_idx shape:", tuple(batch["attrs"].shape))


def print_graph_info(model, graph_adj_path, use_gnn):
    if graph_adj_path:
        try:
            graph_adj = np.load(graph_adj_path)
        except Exception as exc:
            fail("graph_adj problem", f"Failed to load graph_adj.npy: {exc}")
        print("graph_adj shape:", tuple(graph_adj.shape))
        if not np.isfinite(graph_adj).all():
            fail("graph_adj problem", "graph_adj.npy contains NaN or inf.")
    elif use_gnn:
        fail("graph_adj problem", "--use_gnn true requires --graph_adj_path.")
    else:
        print("graph_adj shape: None (GNN disabled)")

    diff_model = model.diff_model
    if use_gnn and not hasattr(diff_model, "graph_adj"):
        fail("graph_adj problem", "Model was built with use_gnn=true but has no graph_adj buffer.")
    if use_gnn:
        print("model graph_adj buffer shape:", tuple(diff_model.graph_adj.shape))


def run_forward_debug(model, batch, device, use_gnn):
    captured = {}
    handle = None
    if use_gnn:
        variable_gnn = getattr(model.diff_model, "variable_gnn", None)
        if variable_gnn is None:
            fail("graph_adj problem", "use_gnn=true but model.diff_model.variable_gnn is None.")

        def capture_gnn_shape(module, inputs, output):
            captured["gnn_input_shape"] = tuple(inputs[0].shape)
            captured["gnn_output_shape"] = tuple(output.shape)

        handle = variable_gnn.register_forward_hook(capture_gnn_shape)

    try:
        # VerbalTS expects x as [B, V, L]. predict_noise then unsqueezes it to
        # [B, 1, V, L] before calling the diffusion network.
        x = batch["ts"].to(device).float().permute(0, 2, 1)
        tp = batch["tp"].to(device).float()
        batch_size = x.shape[0]
        diffusion_step = torch.zeros(batch_size, device=device, dtype=torch.long)
        noise = torch.randn_like(x)
        noisy_x = model.ddpm.forward(x, diffusion_step, noise)

        print("model key input x shape:", tuple(x.shape))
        print("model key input tp shape:", tuple(tp.shape))
        print("model key input diffusion_step shape:", tuple(diffusion_step.shape))
        print("diffusion network input shape:", tuple(noisy_x.unsqueeze(1).shape))

        with torch.no_grad():
            pred_noise, inner_loss_dict = model.predict_noise(noisy_x, tp, None, diffusion_step)
            residual = noise - pred_noise
            noise_loss = (residual ** 2).mean()

        print("model output shape:", tuple(pred_noise.shape))
        if use_gnn:
            if "gnn_input_shape" not in captured:
                fail("shape problem", "GNN hook did not capture input/output shapes.")
            print("GNN input shape:", captured["gnn_input_shape"])
            print("GNN output shape:", captured["gnn_output_shape"])

        if pred_noise.shape != x.shape:
            fail("shape problem", f"Expected model output shape {tuple(x.shape)}, got {tuple(pred_noise.shape)}.")
        if not torch.isfinite(pred_noise).all():
            fail("shape problem", "Model output contains NaN or inf.")

        print("loss noise_loss:", float(noise_loss.detach().cpu()))
        if not torch.isfinite(noise_loss):
            fail("shape problem", "noise_loss is NaN or inf.")

        if inner_loss_dict:
            for name, value in inner_loss_dict.items():
                if torch.is_tensor(value):
                    print(f"loss {name}:", float(value.detach().cpu()))
                    if not torch.isfinite(value).all():
                        fail("shape problem", f"Loss '{name}' contains NaN or inf.")
                else:
                    print(f"loss {name}:", value)
    except RuntimeError:
        raise
    except Exception as exc:
        fail("model forward parameter problem", f"Forward failed: {exc}")
    finally:
        if handle is not None:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description="Minimal Graph-VerbalTS forward debug script.")
    parser.add_argument("--config", required=True, help="Diffusion YAML config path.")
    parser.add_argument("--data_dir", required=True, help="SMD-NL data directory.")
    parser.add_argument("--graph_adj_path", default="", help="Path to graph_adj.npy.")
    parser.add_argument("--use_gnn", required=True, type=str2bool, help="true or false.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    verbalts_root = project_root / "VerbalTS-main"
    config_path = resolve_path(args.config, project_root, verbalts_root)
    data_dir = resolve_path(args.data_dir, project_root, verbalts_root)
    graph_adj_path = (
        resolve_path(args.graph_adj_path, project_root, verbalts_root)
        if args.graph_adj_path
        else Path("")
    )

    if not verbalts_root.exists():
        fail("config problem", f"VerbalTS-main directory not found under {project_root}.")
    if not config_path.exists():
        fail("config problem", f"Config path does not exist: {config_path}")
    if not data_dir.exists():
        fail("data problem", f"Data directory does not exist: {data_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        fail("config problem", f"Requested device '{args.device}', but CUDA is not available.")
    device = torch.device(args.device)

    raw_config = load_yaml_config(config_path)
    diff_config, config_device, generator_pretrain_path = extract_diff_config(raw_config)
    if config_device != args.device:
        print(f"override config device: {config_device} -> {args.device}")

    _, batch = load_batch(data_dir, verbalts_root, batch_size=2)
    validate_batch(batch)
    print_batch_info(batch)

    model = build_generator(
        diff_config=diff_config,
        device=str(device),
        graph_adj_path=str(graph_adj_path) if args.graph_adj_path else "",
        use_gnn=args.use_gnn,
        generator_pretrain_path=generator_pretrain_path,
        verbalts_root=verbalts_root,
    )
    print_graph_info(model, str(graph_adj_path) if args.graph_adj_path else "", args.use_gnn)
    run_forward_debug(model, batch, device, args.use_gnn)

    print("Graph-VerbalTS forward debug passed.")


if __name__ == "__main__":
    main()
