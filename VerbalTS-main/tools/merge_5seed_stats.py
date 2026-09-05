#!/usr/bin/env python3
"""Merge 3+2 seed results into n=5 per variant; paired stats (t + Wilcoxon) vs Graph-Pearson."""
import numpy as np, pandas as pd
from scipy import stats

BASE = "/data/kangjiale/verbalts-graph/VerbalTS-main/save"
V = {
    "VerbalTS":  ["smd_nl_baseline_epoch30_lr5e5_bs16_eval10_3seeds_1",
                  "smd_nl_baseline_epoch30_lr5e5_bs16_eval10_5seeds_1"],
    "Random":    ["smd_nl_graph_random_epoch30_lr5e5_bs16_eval10_3seeds_1",
                  "smd_nl_graph_random_epoch30_lr5e5_bs16_eval10_5seeds_1"],
    "Pearson":   ["smd_nl_graph_pearson_epoch30_lr5e5_bs16_eval10_3seeds_1",
                  "smd_nl_graph_pearson_epoch30_lr5e5_bs16_eval10_5seeds_1"],
    "Identity":  ["smd_nl_graph_identity_epoch30_lr5e5_bs16_eval10_3seeds_1"],
    "Metadata":  ["smd_nl_attr_epoch30_lr5e5_bs16_eval10_3seeds_1",
                  "smd_nl_attr_epoch30_lr5e5_bs16_eval10_5seeds_1"],
}

def load(dirs):
    df = pd.concat([pd.read_csv(f"{BASE}/{d}/results.csv") for d in dirs], ignore_index=True)
    df = df.sort_values("run").reset_index(drop=True)
    assert sorted(df.run) == list(range(len(df))), f"runs missing: {sorted(df.run)}"
    return df

data = {k: load(d) for k, d in V.items()}
rows = []
for k, df in data.items():
    rows.append({**{"Variant": k, "n": len(df)},
                 **{m: df[m].values for m in ["cttp", "fid", "jftsd"]}})
pear = data["Pearson"]
print(f"{'Variant':<10}{'n':>3} {'CTTP':>20} {'FID':>20} {'JFTSD':>20}")
for k, df in data.items():
    print(f"{k:<10}{len(df):>3} "
          f"{df.cttp.mean():>10.4f}±{df.cttp.std(ddof=1):>7.4f} "
          f"{df.fid.mean():>10.4f}±{df.fid.std(ddof=1):>7.4f} "
          f"{df.jftsd.mean():>10.4f}±{df.jftsd.std(ddof=1):>7.4f}")

print("\nPaired tests vs Graph-Pearson (per-seed):")
for k in ["VerbalTS", "Random", "Metadata", "Identity"]:
    df = data[k]; n = len(df)
    print(f"  {k} (n={n}):")
    for m, name in [("cttp", "CTTP"), ("fid", "FID"), ("jftsd", "JFTSD")]:
        d = pear[m].values[:n] - df[m].values[:n]
        t, p_t = stats.ttest_rel(pear[m].values[:n], df[m].values[:n])
        try:
            w, p_w = stats.wilcoxon(pear[m].values[:n], df[m].values[:n])
        except ValueError:
            w, p_w = np.nan, np.nan
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.inf
        print(f"    {name:<6} Δ={d.mean():+.4f}  d_z={dz:+.2f}  p_t={p_t:.4f}  p_w={p_w:.4f}  wins={int((d>0).sum())}/{n}")
print("\nPer-seed FID (run order 1,7,42,100,123):")
for k, df in data.items():
    print(f"  {k:<10}", np.round(df.fid.values, 3))
print("\nPer-seed JFTSD:")
for k, df in data.items():
    print(f"  {k:<10}", np.round(df.jftsd.values, 3))
