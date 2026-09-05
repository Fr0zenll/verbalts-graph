#!/usr/bin/env python3
"""Export SMD-NL into T2S-ready npz files with frozen CTTP text embeddings (512-d).

- train: one caption per window, drawn once with fixed seed from the 3 templates.
- test:  fixed first caption (matches the paper's fixed-description protocol).
Output: /data/kangjiale/T2S/Data/SMD/smd_nl_{train,test}.npz with
  x (N,240,10) float32  text (N,) str  emb (N,512) float32
Run with the `verbalts` env (needs the CTTP model deps).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/data/kangjiale/verbalts-graph/VerbalTS-main")
import yaml  # noqa: E402

from models.cttp.cttp_model import CTTP  # same class the evaluator uses

ROOT = "/data/kangjiale/verbalts-graph/VerbalTS-main"
DATA = "/data/kangjiale/verbalts-graph/data/smd_nl"
OUT = "/data/kangjiale/T2S/Data/SMD"
CLIP_CFG = f"{ROOT}/save/SMD_NL_cttp/model_configs.yaml"
CLIP_CKPT = f"{ROOT}/save/SMD_NL_cttp/clip_model_best.pth"

os.makedirs(OUT, exist_ok=True)
cfg = yaml.safe_load(open(CLIP_CFG))
clip = CTTP(cfg)
clip.load_state_dict(torch.load(CLIP_CKPT, map_location="cpu"))
clip = clip.to(clip.device).eval()


def embed(caps):
    out = []
    with torch.no_grad():
        for i in range(0, len(caps), 128):
            e = clip.get_text_coemb(list(caps[i:i + 128]), None)
            out.append(e.float().cpu())
    return torch.cat(out).numpy()


for split in ["train", "test"]:
    ts = np.load(f"{DATA}/{split}_ts.npy")
    caps = np.load(f"{DATA}/{split}_text_caps.npy", allow_pickle=True)
    n = len(ts)
    if split == "train":
        rng = np.random.RandomState(2026)
        chosen = np.array([caps[i][rng.randint(len(caps[i]))] for i in range(n)])
    else:
        chosen = np.array([caps[i][0] for i in range(n)])
    emb = embed(chosen)
    assert emb.shape == (n, 512), emb.shape
    np.savez(f"{OUT}/smd_nl_{split}.npz", x=ts.astype(np.float32),
             text=chosen.astype(str), emb=emb.astype(np.float32))
    print(split, ts.shape, emb.shape, "->", f"{OUT}/smd_nl_{split}.npz")
print("EXPORT DONE")
