# /data 盘故障抢救与恢复手册（2026-09-07）

## 1. 事件概述

- `/dev/sda1`（/data，3.6T ext4）出现区域性 I/O 错误，内核已自动挂载为**只读**。
- 失效分布不均：`verbalts-graph` 部分可读；`T2S` 目录读取已完全瘫痪。
- 计划更换 /data 硬盘。本手册记录抢救状态、损失清单与换盘后的完整恢复流程。

## 2. 资产状态清单（2026-09-07 快照）

### 2.1 已安全 ✅

| 资产 | 位置 |
|---|---|
| verbalts-graph 全仓库（论文终稿 + 全部实验/评测代码 + 5-seed results.csv + 统计脚本 + T2S 的 export/eval 脚本） | GitHub `main @ 22815b5`；本地恢复副本 `/home/kangjiale/data_rescue_20260907/vgr/` |
| 论文投稿材料 | PDF 已编译（9 页，Overleaf 产物），数据全部固化在论文中 |
| /home 分区（幸存，不换） | conda `verbalts` 环境、`~/.ssh`（GitHub 密钥）、`.gitconfig`、Trae 记忆 |

### 2.2 已损失，需重建 ⚠️

T2S 仓库中**未提交**的多变量适配代码（盘面该区域已死，IDE 缓存无副本）：

- `generate_smdnl.py`（DiT 采样 + SMD-NL 输出适配）
- `run_t2s_seed.sh` / `run_t2s_geneval.sh`（每 seed 全流程编排）
- `model/vqvae/vqvae.py` 多变量修改（10 通道 Encoder/Decoder）
- `datafactory/dataloader/dataset.py`（逐变量 MinMax 归一化）
- `model/denoiser/transformer.py`（512→128 文本投影层）

**重建估时 2–4 小时**：上游 https://github.com/WinfredGe/T2S 公开可克隆，设计细节见 §4，eval 与数据导出脚本在本仓库 `VerbalTS-main/tools/`（`t2s_eval.py`、`export_smdnl_for_t2s.py`）。

### 2.3 待抢救（权重/数据集，仍在故障盘上）

| 优先级 | 内容 | 路径 | 大小 | 丢失后果 |
|---|---|---|---|---|
| ① | VerbalTS 系主模型权重 | `verbalts-graph/VerbalTS-main/save/` | ~8.6G | 重训 ~12h/variant |
| ② | T2S VAE/DiT 权重 + 生成样本 | `T2S/results/` | ~数 G | 重训 ~2h/seed + 代码重建 |
| ③ | 数据集 | `VerbalTS-main/data/`、`/data/kangjiale/SMD_cttp_2.tar.gz`、`machine-1-1.csv` | ~2G | SMD 可重新下载 + 管线重建 |
| ④ | CLIP/CTTP cache、t2s conda env | `VerbalTS-main/cache/`、`/data/kangjiale/envs/t2s` | ~0.5G/数G | 自动重算 / pip 重建 |

**抢救命令（目标 NAS，执行前先 `touch /nas_data/kangjiale/.wtest` 验证可写）**：

```bash
mkdir -p /nas_data/kangjiale/rescue_20260907 && cd /data/kangjiale
rsync -avh --partial --timeout=60 --retry=3 \
  verbalts-graph/VerbalTS-main/save/ /nas_data/kangjiale/rescue_20260907/verbalts_save/
rsync -avh --partial --timeout=60 --retry=3 \
  T2S/results/ /nas_data/kangjiale/rescue_20260907/t2s_results/
rsync -avh --partial --timeout=60 --retry=3 \
  verbalts-graph/VerbalTS-main/data/ SMD_cttp_2.tar.gz machine-1-1.csv \
  /nas_data/kangjiale/rescue_20260907/datasets/
rsync -avh --partial --timeout=60 --retry=3 \
  verbalts-graph/VerbalTS-main/cache/ envs/ /nas_data/kangjiale/rescue_20260907/cache_envs/
```

注意：盘在持续恶化，rsync `code 23`（部分失败）属预期，逐项目清点即可；每完成一项立即 `du -sh` 核对。

## 3. T2S 五种子评测结果（已固化，不依赖盘上数据）

| seed | FID | JFTSD | CTTP |
|---|---|---|---|
| 1 | 17.8037 | 19.8317 | 11.8857 |
| 7 | 16.6006 | 18.6260 | 11.4975 |
| 42 | 20.3179 | 22.3414 | 11.7124 |
| 100 | 14.5164 | 16.6251 | 11.7550 |
| 123 | 15.3628 | 17.4756 | 11.7861 |
| **均值±std** | **16.9203±2.2711** | **18.9800±2.2337** | **11.7273±0.1435** |

与论文 Table 4 的 T2S (external) 行一致（paper/main.tex 已在 GitHub）。

## 4. T2S 适配设计细节（重建用）

管线：`冻结 CTTP 文本嵌入(512) → 投影到 128 → DiT adaLN 条件(c = t + text)；LA-VAE 多变量化后潜空间保持 (30,64)，DiT 不动`。

- **VAE**：Encoder `in_channels=10`、`view(B,10,L)`；Decoder `out_channels=10`；逐变量 MinMaxScaler。潜空间 (B,30,64)。重构 MAE ≈ 0.021（归一化尺度），单次预训练 ~4 分钟。
- **DiT**：512 维 CTTP 文本嵌入经 `nn.Linear(512→128)` 投影后与时间嵌入相加；flow matching（上游默认采样器）；500 epochs，终止 loss ≈ 0.09；~90 分钟/seed。
- **数据导出**：SMD-NL 4104 train / 1173 test，`train_ts.npy` + `train_text_caps.npy` → 冻结 CTTP 嵌入（本仓库 `tools/export_smdnl_for_t2s.py`）。
- **生成协议**：每窗口 10 样本取逐点中值，输出 `(N,240,10)` 反归一化，与本仓库所有变体评测协议一致。
- **评测**：`tools/t2s_eval.py`（verbalts 环境运行），计算 FID/JFTSD/CTTP。
- **种子**：1/7/42/100/123，每 seed 完整重训 VAE+DiT（与 VerbalTS 协议对等）。

## 5. 换盘后恢复流程（预计半天）

```bash
# 1. 环境
conda env export -n verbalts   # 换盘前在旧 /home 导出（/home 不换则跳过）
# 新 /data 挂载后：
rsync -avh --partial /nas_data/kangjiale/rescue_20260907/ /data/kangjiale/rescued/
git clone git@github.com:Fr0zenll/verbalts-graph.git /data/kangjiale/verbalts-graph
# t2s 独立环境（原在 /data/kangjiale/envs/t2s，torch 2.3.1+cu121）：
#   pip freeze 清单若已丢失，按 torch 2.3.1 + timm 1.0.11 + 上游 requirements 重建

# 2. 验证
python -c "import torch; print(torch.cuda.is_available())"     # True
chmod 600 ~/.ssh/id_ed25519 && ssh -T git@github.com            # Hi Fr0zenll!
cd /data/kangjiale/verbalts-graph && git fsck                   # 仓库完整
```

- 若 T2S 需要重新产出：按 §4 重建代码后重训（结果应与 §3 在 seed 噪声内一致）。
- 论文投稿（PIC 2026，CMT 截稿 2026-09-30）**不依赖本盘任何内容**。

## 6. 抢救执行记录

- 2026-09-07：确认盘故障（ro + I/O error）；verbalts-graph 经 GitHub 验证完整（22815b5）并克隆恢复至 `/home/kangjiale/data_rescue_20260907/vgr/`；T2S 未提交代码确认不可读；本手册提交入 GitHub。
