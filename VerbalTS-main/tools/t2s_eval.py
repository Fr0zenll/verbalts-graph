#!/usr/bin/env python3
"""Evaluate T2S generations on SMD-NL with the exact VerbalTS protocol.

FID / JFTSD: Frechet distances against the train-split statistics cached by the
VerbalTS evaluator (identical across variants; reuse the baseline cache).
CTTP: mean over test windows of dot(ts_emb, cap_emb) with the fixed first test caption.
Run: /home/kangjiale/miniconda3/envs/verbalts/bin/python tools/t2s_eval.py --gen x_gen.npy
"""
import argparse
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, "/data/kangjiale/verbalts-graph/VerbalTS-main")
sys.path.insert(0, "/data/kangjiale/verbalts-graph/VerbalTS-main/evaluation")

from models.cttp.cttp_model import CTTP  # noqa: E402
from evaluation.base_evaluator import calculate_frechet_distance  # noqa: E402

ROOT = "/data/kangjiale/verbalts-graph/VerbalTS-main"
DATA = "/data/kangjiale/verbalts-graph/data/smd_nl"
CACHE = f"{ROOT}/cache/smd_nl_cttp_baseline_epoch30_lr5e5_5seeds_1"  # train-split reference stats
CLIP_CFG = f"{ROOT}/save/SMD_NL_cttp/model_configs.yaml"
CLIP_CKPT = f"{ROOT}/save/SMD_NL_cttp/clip_model_best.pth"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gen', type=str, required=True)
    args = p.parse_args()

    clip = CTTP(yaml.safe_load(open(CLIP_CFG)))
    clip.load_state_dict(torch.load(CLIP_CKPT, map_location="cpu"))
    clip = clip.to(clip.device).eval()

    gen = np.load(args.gen)                        # (1173, 240, 10) raw scale
    caps = np.load(f"{DATA}/test_text_caps.npy", allow_pickle=True)
    test_texts = np.array([caps[i][0] for i in range(len(caps))])

    def embed_ts(x):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 64):
                ts = torch.tensor(x[i:i + 64], dtype=torch.float32, device=clip.device)
                ts_len = torch.full((ts.shape[0],), ts.shape[1], dtype=torch.int32, device=clip.device)
                out.append(clip.get_ts_coemb(ts, ts_len).float().cpu())
        return torch.cat(out).numpy()

    def embed_text(texts):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 128):
                out.append(clip.get_text_coemb(list(texts[i:i + 128]), None).float().cpu())
        return torch.cat(out).numpy()

    gen_emb = embed_ts(gen)
    real_test_emb = embed_ts(np.load(f"{DATA}/test_ts.npy"))
    cap_emb = embed_text(test_texts)
    joint_gen = np.concatenate([gen_emb, cap_emb], axis=1)

    ts_mean = np.load(f"{CACHE}/fid_mean.npy")
    ts_cov = np.load(f"{CACHE}/fid_cov.npy")
    joint_mean = np.load(f"{CACHE}/jftsd_mean.npy")
    joint_cov = np.load(f"{CACHE}/jftsd_cov.npy")

    fid = calculate_frechet_distance(ts_mean, ts_cov, np.mean(gen_emb, 0), np.cov(gen_emb, rowvar=False))
    jftsd = calculate_frechet_distance(joint_mean, joint_cov, np.mean(joint_gen, 0), np.cov(joint_gen, rowvar=False))
    cttp = float(np.mean(np.sum(gen_emb * cap_emb, axis=1)))

    print(f"RESULTS gen={args.gen}")
    print(f"FID {fid:.4f}")
    print(f"JFTSD {jftsd:.4f}")
    print(f"CTTP {cttp:.4f}")
    print(f"CTTP real_test_ref {float(np.mean(np.sum(real_test_emb * cap_emb, axis=1))):.4f}")


if __name__ == '__main__':
    main()
