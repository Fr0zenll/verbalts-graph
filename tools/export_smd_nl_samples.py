import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(verbalts_root, data_folder, diff_config_path, cond_config_path, ckpt_path, device):
    sys.path.insert(0, str(verbalts_root))
    from data import GenerationDataset
    from models.conditional_generator import ConditionalGenerator

    diff_configs = load_yaml(diff_config_path)
    cond_configs = load_yaml(cond_config_path)

    diff_configs["device"] = device
    diff_configs["generator_pretrain_path"] = ""
    cond_configs["cond_modal"] = "simple_text"

    dataset = GenerationDataset({"name": "custom", "folder": str(data_folder)})
    if "attrs" in cond_configs:
        cond_configs["attrs"]["num_attr_ops"] = dataset.num_attr_ops.tolist()

    model = ConditionalGenerator(diff_configs, cond_configs).to(device)
    state = torch.load(ckpt_path, map_location=device)
    load_info = model.load_state_dict(state, strict=False)
    if load_info.missing_keys:
        print(f"[warn] Missing keys for {ckpt_path}: {load_info.missing_keys}")
    if load_info.unexpected_keys:
        print(f"[warn] Unexpected keys for {ckpt_path}: {load_info.unexpected_keys}")
    model.eval()
    return model, dataset


def load_test_batch(dataset, sample_start, batch_size, num_workers):
    split = dataset.dataset.get_split("test")
    indices = list(range(sample_start, min(sample_start + batch_size, len(split))))
    if not indices:
        raise ValueError(f"No test indices selected from sample_start={sample_start}.")
    loader = DataLoader(
        Subset(split, indices),
        batch_size=len(indices),
        shuffle=False,
        num_workers=num_workers,
    )
    return indices, next(iter(loader))


@torch.no_grad()
def generate_median(model, batch, n_samples, sampler, sample_seed):
    set_seed(sample_seed)
    multi_preds = model.generate(batch, n_samples, sampler)
    # model.generate returns [S, B, V, L]; paper plots use [B, L, V].
    multi_preds = multi_preds.permute(0, 1, 3, 2).detach().cpu()
    median_pred = multi_preds.median(dim=0).values
    return median_pred.numpy(), multi_preds.numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Export matched SMD-NL real/baseline/Graph-Pearson samples for paper figures."
    )
    parser.add_argument("--verbalts_root", default="VerbalTS-main")
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--cond_config", default="configs/smd_nl/cond/text_msmdiffmv_full.yaml")
    parser.add_argument("--baseline_diff_config", default="configs/smd_nl/diff/model_text2ts_baseline_full.yaml")
    parser.add_argument("--pearson_diff_config", default="configs/smd_nl/diff/model_text2ts_graph_full.yaml")
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--pearson_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    verbalts_root = Path(args.verbalts_root).resolve()
    data_folder = Path(args.data_folder).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")

    set_seed(args.seed)
    baseline_model, dataset = build_model(
        verbalts_root=verbalts_root,
        data_folder=data_folder,
        diff_config_path=verbalts_root / args.baseline_diff_config,
        cond_config_path=verbalts_root / args.cond_config,
        ckpt_path=args.baseline_ckpt,
        device=args.device,
    )
    pearson_model, _ = build_model(
        verbalts_root=verbalts_root,
        data_folder=data_folder,
        diff_config_path=verbalts_root / args.pearson_diff_config,
        cond_config_path=verbalts_root / args.cond_config,
        ckpt_path=args.pearson_ckpt,
        device=args.device,
    )

    indices, batch = load_test_batch(dataset, args.sample_start, args.batch_size, args.num_workers)
    real = batch["ts"].detach().cpu().numpy()
    attrs = batch["attrs"].detach().cpu().numpy()
    captions = list(batch["cap"])

    baseline_pred, baseline_multi = generate_median(
        baseline_model, batch, args.n_samples, args.sampler, args.seed
    )
    pearson_pred, pearson_multi = generate_median(
        pearson_model, batch, args.n_samples, args.sampler, args.seed
    )

    np.savez_compressed(
        out_dir / "smd_nl_paper_samples.npz",
        indices=np.array(indices, dtype=np.int64),
        real=real,
        baseline_pred=baseline_pred,
        pearson_pred=pearson_pred,
        baseline_multi=baseline_multi,
        pearson_multi=pearson_multi,
        attrs=attrs,
        captions=np.array(captions, dtype=object),
    )

    meta = {
        "indices": indices,
        "captions": captions,
        "attrs": attrs.tolist(),
        "real_shape": list(real.shape),
        "baseline_pred_shape": list(baseline_pred.shape),
        "pearson_pred_shape": list(pearson_pred.shape),
        "baseline_multi_shape": list(baseline_multi.shape),
        "pearson_multi_shape": list(pearson_multi.shape),
        "n_samples": args.n_samples,
        "sampler": args.sampler,
        "seed": args.seed,
        "sample_start": args.sample_start,
        "batch_size": args.batch_size,
        "baseline_ckpt": str(args.baseline_ckpt),
        "pearson_ckpt": str(args.pearson_ckpt),
    }
    with open(out_dir / "smd_nl_paper_samples_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_dir / 'smd_nl_paper_samples.npz'}")
    print(f"Saved: {out_dir / 'smd_nl_paper_samples_meta.json'}")
    print("Shapes:")
    print("  real:", real.shape)
    print("  baseline_pred:", baseline_pred.shape)
    print("  pearson_pred:", pearson_pred.shape)


if __name__ == "__main__":
    main()
