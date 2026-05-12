# Graph-VerbalTS Handoff Summary

## 1. Current Goal

Build a minimal runnable Graph-VerbalTS prototype based on the reproduced VerbalTS project.

Current scope:

- Use the SMD-NL dataset with natural-language descriptions.
- Keep original VerbalTS usable.
- Add optional GNN enhancement controlled by `use_gnn`.
- Use a fixed variable dependency graph built from training data.
- Do not handle paper writing, full training experiments, graph loss, or final benchmark results yet.

Workspace:

```text
D:\文档\New project 3
```

VerbalTS project:

```text
D:\文档\New project 3\VerbalTS-main
```

SMD-NL data:

```text
D:\文档\New project 3\data\smd_nl
```

Graph file:

```text
D:\文档\New project 3\data\smd_nl\graph_adj.npy
```

## 2. Completed Work

### Project Analysis

- Read VerbalTS project structure.
- Identified:
  - main entry: `VerbalTS-main/run.py`
  - trainer: `VerbalTS-main/train/trainer.py`
  - dataset: `VerbalTS-main/data/data.py`
  - data registry: `VerbalTS-main/data/__init__.py`
  - conditional generator: `VerbalTS-main/models/conditional_generator.py`
  - unconditional generator: `VerbalTS-main/models/unconditional_generator.py`
  - main diffusion model: `VerbalTS-main/models/diffusion/verbalts.py`
  - samplers: `VerbalTS-main/samplers/ddpm.py`, `VerbalTS-main/samplers/ddim.py`
  - text encoders/projectors: `VerbalTS-main/models/encoders/`

### Shape Analysis

SMD-NL time series shape:

```text
(batch_size, 240, 10)
```

VerbalTS data flow:

```text
Dataset batch["ts"]:       [B, 240, 10]
Generator internal x:      [B, 10, 240]
Denoising network input:   [B, 1, 10, 240]
VerbalTS hidden x_in:      [B, C, V, Lp]
Final pred_noise output:   [B, 10, 240]
```

Chosen GNN insertion point:

```python
x_in = torch.cat(x_list, dim=-1)
side_in = torch.cat(side_list, dim=-1)
x_in = self._apply_variable_gnn(x_in)
```

Location:

```text
VerbalTS-main/models/diffusion/verbalts.py
class VerbalTS.forward()
```

GNN shape transform:

```text
VerbalTS hidden: [B, C, V, Lp]
GNN input:       [B, Lp, V, C]
GNN output:      [B, Lp, V, C]
Back to model:   [B, C, V, Lp]
```

### SMD-NL Dataset Adaptation

No VerbalTS dataset code was modified.

Instead, SMD-NL was adapted by editing `meta.json`:

- Created backup:

```text
D:\文档\New project 3\data\SMD\meta_origin.json
```

- Added to `meta.json`:

```json
"attr_list": [
  "global_state",
  "compute_state",
  "memory_state",
  "disk_state",
  "network_state"
],
"attr_n_ops": [3, 3, 3, 3, 3]
```

This allows original `CustomDataset` to read SMD-NL with:

```yaml
data:
  name: custom
```

### Data Check Script

Created:

```text
D:\文档\New project 3\tools\check_smd_nl_data.py
```

Purpose:

- Check required SMD-NL files.
- Print train/valid/test shapes.
- Validate:
  - `ts`: `(N, 240, 10)`
  - `text_caps`: `(N, 3)`
  - `attrs_idx`: `(N, 5)`
  - `attrs_idx` values only in `{0, 1, 2}`
- Print 3 random train samples.

Run:

```powershell
python tools\check_smd_nl_data.py --data_dir "D:\文档\New project 3\data\smd_nl"
```

### Graph Construction

Created:

```text
D:\文档\New project 3\tools\build_smd_graph.py
```

It:

- Reads `train_ts.npy`.
- Reshapes `(N, 240, 10)` to `(N * 240, 10)`.
- Computes Pearson correlation matrix.
- Uses absolute correlation as dependency strength.
- Supports `--top_k` and `--threshold`.
- Adds self-loops.
- Symmetrizes graph.
- Applies GCN normalization:

```text
A_norm = D^{-1/2} A D^{-1/2}
```

Generated:

```text
D:\文档\New project 3\data\smd_nl\graph_corr_raw.npy
D:\文档\New project 3\data\smd_nl\graph_adj.npy
D:\文档\New project 3\data\smd_nl\graph_adj_unnormalized.npy
D:\文档\New project 3\data\smd_nl\graph_info.json
```

Run:

```powershell
python tools\build_smd_graph.py --data_dir "D:\文档\New project 3\data\smd_nl" --top_k 3
```

### Independent GNN Module

Created:

```text
D:\文档\New project 3\VerbalTS-main\models\graph_modules.py
```

Classes:

- `GraphConvolution`
  - Computes `H = adj @ X @ W + b`.
  - Input shape: `(..., V, C_in)`.
  - Adj shape: `(V, V)`.
  - Output shape: `(..., V, C_out)`.

- `SimpleVariableGCN`
  - Runs message passing along variable dimension.
  - Supports 3D:

```text
input:  [B, T, V]
output: [B, T, V]
```

  - Supports 4D:

```text
input:  [B, T, V, C]
output: [B, T, V, C] when output_dim == C
```

- Alias:

```python
VariableGCN = SimpleVariableGCN
```

### Independent GNN Test

Created:

```text
D:\文档\New project 3\tools\test_graph_module.py
```

Tests:

- 3D input `(4, 240, 10)`.
- 4D input `(4, 240, 10, 32)`.
- identity adjacency.
- random symmetric adjacency.
- existing `graph_adj.npy`.
- output shape.
- no NaN/inf.
- dtype/device.
- backward.

Run:

```powershell
conda run -n verbalts python tools\test_graph_module.py
```

This test has passed.

### GNN Integrated Into VerbalTS

Modified:

```text
D:\文档\New project 3\VerbalTS-main\models\diffusion\verbalts.py
```

Added:

- `SimpleVariableGCN` import.
- `use_gnn` config.
- `graph_adj_path` config.
- `gnn_hidden_dim`.
- `gnn_num_layers`.
- `gnn_dropout`.
- `gnn_residual`.
- `debug_shape`.
- `_init_gnn()`.
- `_apply_variable_gnn()`.
- `_maybe_print_debug_shape()`.

Graph loading:

```python
graph_adj = np.load(graph_adj_path)
graph_adj = torch.from_numpy(graph_adj).float()
self.register_buffer("graph_adj", graph_adj)
```

Important:

- `graph_adj` is a buffer, not a trainable parameter.
- It moves with model `.to(device)`.
- It is included in `state_dict`.

Smoke test passed:

```text
use_gnn=false -> output (2, 10, 240)
use_gnn=true  -> output (2, 10, 240)
```

### Existing Configs Updated With Default Disabled GNN

Modified:

```text
D:\文档\New project 3\VerbalTS-main\configs\Weather\diff\model_text2ts_dep.yaml
D:\文档\New project 3\VerbalTS-main\configs\synth-m\diff\model_text2ts_dep.yaml
```

Added defaults:

```yaml
use_gnn: false
graph_adj_path: ""
gnn_hidden_dim: 64
gnn_num_layers: 1
gnn_dropout: 0.0
gnn_residual: true
debug_shape: false
```

Purpose:

- Original VerbalTS remains usable.
- GNN is opt-in only.

### SMD-NL Debug Configs

Created:

```text
D:\文档\New project 3\VerbalTS-main\configs\smd_nl\train_debug.yaml
D:\文档\New project 3\VerbalTS-main\configs\smd_nl\evaluate_debug.yaml
D:\文档\New project 3\VerbalTS-main\configs\smd_nl\diff\model_text2ts_baseline_debug.yaml
D:\文档\New project 3\VerbalTS-main\configs\smd_nl\diff\model_text2ts_graph_debug.yaml
D:\文档\New project 3\VerbalTS-main\configs\smd_nl\cond\text_msmdiffmv_debug.yaml
```

Baseline debug:

```yaml
use_gnn: false
```

Graph debug:

```yaml
use_gnn: true
graph_adj_path: D:/文档/New project 3/data/smd_nl/graph_adj.npy
gnn_hidden_dim: 32
gnn_num_layers: 2
gnn_dropout: 0.1
gnn_residual: true
debug_shape: true
```

YAML parsing has been verified with `PyYAML 6.0.2`.

Random forward with parsed configs has passed:

```text
baseline (2, 10, 240)
graph    (2, 10, 240)
```

## 3. Environment Notes

Conda environment:

```text
verbalts
```

Confirmed:

```text
Python: D:\IDE\Anaconda\envs\verbalts\python.exe
torch: 2.2.1+cpu
numpy: 1.26.4
PyYAML: 6.0.2
CUDA available: False
```

Use for lightweight tests:

```powershell
conda run -n verbalts python ...
```

Important:

- CPU only.
- Local CPU/GPU performance is limited.
- Use this environment only for shape tests, smoke tests, and small debug runs.
- Do not launch heavy full training unless explicitly requested.

If additional packages are needed, ask the user to install manually. Do not silently install.

Potential missing package for full VerbalTS run:

```powershell
conda activate verbalts
pip install transformers
```

Original project also mentions:

```text
torch==2.2.1
pandas==2.0.3
pyyaml==6.0.2
linear_attention_transformer==0.19.1
tensorboard==2.14.0
scikit-learn==1.3.2
```

## 4. Key Decisions

1. Do not modify VerbalTS Dataset code.
   - SMD-NL is adapted through `meta.json`.

2. Use original `CustomDataset`.
   - SMD-NL file names and shapes already match original loader.

3. GNN insertion point is hidden representation after multi-scale patch embedding.
   - Not raw input.
   - Not final output.

4. GNN runs along variable dimension `V`.

5. `graph_adj.npy` is fixed and registered as buffer.
   - Not trainable.

6. `use_gnn=false` must preserve original behavior as much as possible.

7. Existing Weather and synth-m configs keep `use_gnn: false`.

8. Debug configs are intentionally tiny.
   - CPU.
   - small batch.
   - small channels.
   - few diffusion steps.

9. The current goal is forward/debug integration, not full training quality.

## 5. Not Yet Completed

1. Full `run.py` debug training has not been completed yet.

2. It is unknown whether local tokenizer path exists:

```text
D:\文档\New project 3\VerbalTS-main\save\Longclip
```

3. `transformers` may be missing in `verbalts`.

4. Full evaluator may require CTTP/clip checkpoints and extra dependencies.

5. Formal SMD-NL training configs are not created yet.
   - Only debug configs exist.

6. Pretrained checkpoint compatibility has not been solved.
   - With `use_gnn=true`, model has extra GNN parameters and graph buffer.
   - Loading old checkpoint strictly may fail.

7. No graph loss has been implemented.

8. No full experiment results.

## 6. Suggested Next Steps

### Step A: Check tokenizer path

Run:

```powershell
dir "D:\文档\New project 3\VerbalTS-main\save\Longclip"
```

If missing, either provide tokenizer/model files or adjust:

```yaml
tokenizer_path: ...
pretrain_model_path: ...
```

in:

```text
VerbalTS-main/configs/smd_nl/cond/text_msmdiffmv_debug.yaml
```

### Step B: Check required package `transformers`

Run:

```powershell
conda run -n verbalts python -c "import transformers; print(transformers.__version__)"
```

If missing, ask user to install:

```powershell
conda activate verbalts
pip install transformers
```

### Step C: Run independent GNN test

From workspace root:

```powershell
conda run -n verbalts python tools\test_graph_module.py
```

### Step D: Run baseline debug

From:

```text
D:\文档\New project 3\VerbalTS-main
```

Command:

```powershell
conda run -n verbalts python run.py --cond_modal simple_text --training_stage finetune --save_folder ./save/smd_nl_debug_baseline --model_diff_config_path configs/smd_nl/diff/model_text2ts_baseline_debug.yaml --model_cond_config_path configs/smd_nl/cond/text_msmdiffmv_debug.yaml --train_config_path configs/smd_nl/train_debug.yaml --evaluate_config_path configs/smd_nl/evaluate_debug.yaml --data_folder "D:/文档/New project 3/data/smd_nl" --epochs 1 --batch_size 2 --n_runs 1
```

### Step E: Run graph debug

From:

```text
D:\文档\New project 3\VerbalTS-main
```

Command:

```powershell
conda run -n verbalts python run.py --cond_modal simple_text --training_stage finetune --save_folder ./save/smd_nl_debug_graph --model_diff_config_path configs/smd_nl/diff/model_text2ts_graph_debug.yaml --model_cond_config_path configs/smd_nl/cond/text_msmdiffmv_debug.yaml --train_config_path configs/smd_nl/train_debug.yaml --evaluate_config_path configs/smd_nl/evaluate_debug.yaml --data_folder "D:/文档/New project 3/data/smd_nl" --epochs 1 --batch_size 2 --n_runs 1
```

### Step F: If `run.py` fails because evaluator/CTTP/clip is missing

Do not overfit around evaluator immediately.

Recommended next move:

- Add a forward-only smoke test script for Graph-VerbalTS.
- It should:
  - load config files,
  - load one DataLoader batch,
  - instantiate `ConditionalGenerator`,
  - run one forward loss call,
  - skip evaluator and checkpoint logic.

Suggested file:

```text
D:\文档\New project 3\tools\smoke_graph_verbalts_forward.py
```

## 7. Old Plans To Avoid

Do not return to these older approaches:

1. Do not add `SMDNLDataset`.
   - Current approach is meta.json adaptation + original `CustomDataset`.

2. Do not modify VerbalTS Dataset/DataLoader unless absolutely necessary.

3. Do not insert GNN before denoising network on raw `(B, 10, 240)` input.
   - Chosen insertion is hidden `x_in`.

4. Do not insert GNN as final output post-processing.
   - Lower modeling value and more likely to distort predicted noise.

5. Do not use `torch_geometric` or `dgl`.

6. Do not make `graph_adj` trainable for first version.
   - It should remain a fixed buffer.

7. Do not redesign VerbalTS config system.
   - Keep original multi-YAML style.

8. Do not run heavy training on local machine.
   - Use only lightweight tests unless user explicitly requests more.

9. Do not remove original VerbalTS code.

10. Do not change Weather/synth-m behavior by default.
    - Keep `use_gnn: false`.

## 8. Quick Reference Commands

Check SMD-NL data:

```powershell
python tools\check_smd_nl_data.py --data_dir "D:\文档\New project 3\data\smd_nl"
```

Build graph:

```powershell
python tools\build_smd_graph.py --data_dir "D:\文档\New project 3\data\smd_nl" --top_k 3
```

Test GNN module:

```powershell
conda run -n verbalts python tools\test_graph_module.py
```

Test import versions:

```powershell
conda run -n verbalts python -c "import torch, numpy, yaml; print(torch.__version__, numpy.__version__, yaml.__version__)"
```

