# VerbalTS + GNN Final Summary

Project path:

```text
D:\文档\New project 3
```

This document summarizes the current Graph-VerbalTS prototype work. It is written as a handoff document for ChatGPT or another coding assistant to quickly understand the current state.

## 一、最终实现了什么

当前项目已经在原 VerbalTS 基础上完成了一个最小可运行的 Graph-VerbalTS 原型：

- 支持读取 SMD-NL 数据集，数据形状为 `(N, 240, 10)`。
- 支持从训练集自动构造变量依赖图 `graph_adj.npy`。
- 新增独立 GNN 模块，可单独测试，不依赖 VerbalTS 主流程。
- 将 GNN 接入 VerbalTS diffusion hidden representation。
- 通过 `use_gnn` 控制是否启用 GNN。
- `use_gnn=true` 时加载固定图 `graph_adj.npy`，并注册为 model buffer。
- 新增最小 forward debug 脚本，可分别验证 baseline 和 Graph-VerbalTS forward、shape、loss、NaN。

## 二、修改文件列表

### `tools/check_smd_nl_data.py`

- 新增；
- 检查 SMD-NL 数据文件、shape、attrs 范围和样例；
- 关键函数：数据加载与 shape 校验逻辑。

### `tools/build_smd_graph.py`

- 新增；
- 从 `train_ts.npy` 计算变量相关性图，生成 `graph_adj.npy`；
- 关键逻辑：Pearson correlation、top-k/threshold、self-loop、GCN normalization。

### `tools/test_graph_module.py`

- 新增；
- 独立测试 GNN 模块；
- 关键函数：3D/4D 输入测试、identity/random/真实 graph 测试、backward 测试。

### `tools/debug_graph_verbalts_forward.py`

- 新增；
- 最小 Graph-VerbalTS forward debug；
- 关键函数：`load_batch`、`build_generator`、`run_forward_debug`。

### `VerbalTS-main/models/graph_modules.py`

- 新增；
- 独立变量图 GNN；
- 关键类：`GraphConvolution`、`SimpleVariableGCN`、`VariableGCN`。

### `VerbalTS-main/models/diffusion/verbalts.py`

- 修改；
- 接入 GNN；
- 关键函数：`_init_gnn`、`_apply_variable_gnn`、`_maybe_print_debug_shape`、`forward`。

### `VerbalTS-main/configs/Weather/diff/model_text2ts_dep.yaml`

- 修改；
- 给原 Weather 配置增加 GNN 默认关闭项；
- 关键配置：`use_gnn: false`。

### `VerbalTS-main/configs/synth-m/diff/model_text2ts_dep.yaml`

- 修改；
- 给原 synth-m 配置增加 GNN 默认关闭项；
- 关键配置：`use_gnn: false`。

### `VerbalTS-main/configs/smd_nl/train_debug.yaml`

- 新增；
- SMD-NL debug 训练配置；
- 关键配置：`data.folder`、`batch_size`、`epochs`。

### `VerbalTS-main/configs/smd_nl/evaluate_debug.yaml`

- 新增；
- SMD-NL debug evaluate 配置；
- 关键配置：`data.folder`、`eval.batch_size`。

### `VerbalTS-main/configs/smd_nl/diff/model_text2ts_baseline_debug.yaml`

- 新增；
- SMD-NL baseline diffusion debug 配置；
- 关键配置：`use_gnn: false`。

### `VerbalTS-main/configs/smd_nl/diff/model_text2ts_graph_debug.yaml`

- 新增；
- SMD-NL Graph-VerbalTS diffusion debug 配置；
- 关键配置：`use_gnn: true`、`graph_adj_path`。

### `VerbalTS-main/configs/smd_nl/cond/text_msmdiffmv_debug.yaml`

- 新增；
- SMD-NL 文本条件 debug 配置；
- 关键配置：`cond_modal: simple_text`、`text_projector`。

### `data/smd_nl/graph_adj.npy`

- 新增生成文件；
- GCN-normalized 变量依赖图；
- 供 `use_gnn=true` 加载。

### `data/smd_nl/graph_corr_raw.npy`

- 新增生成文件；
- 原始 Pearson correlation 矩阵。

### `data/smd_nl/graph_adj_unnormalized.npy`

- 新增生成文件；
- 未归一化邻接矩阵。

### `data/smd_nl/graph_info.json`

- 新增生成文件；
- 记录构图参数和输出文件。

### `data/smd_nl/meta.json`

- 修改；
- 添加 `attr_list` 和 `attr_n_ops`，使原 `CustomDataset` 可读取 SMD-NL。

### `data/smd_nl/meta_origin.json`

- 新增；
- 原始 `meta.json` 备份。

## 三、Graph-VerbalTS 的模型流程

### 1. 输入是什么

SMD-NL batch：

```text
batch["ts"]:    [B, 240, 10]
batch["tp"]:    [B, 240]
batch["cap"]:   text condition
batch["attrs"]: [B, 5]
```

### 2. 文本条件如何进入原模型

完整 conditional 模型中，文本条件通过 `TextEncoder` 编码，再由 `TextProjectorMVarMScaleMStep` 投影到 diffusion hidden condition。

当前最小 forward debug 脚本为了绕过 tokenizer、LongCLIP、`transformers` 等外部依赖，直接测试 diffusion / GNN 主链路，没有走完整文本 encoder。

### 3. 时间序列如何进入原模型

时间序列进入模型前会从：

```text
[B, 240, 10]
```

转为：

```text
[B, 10, 240]
```

然后在 `UnConditionalGenerator.predict_noise()` 中 unsqueeze 为：

```text
[B, 1, 10, 240]
```

### 4. `graph_adj.npy` 如何加载

`graph_adj.npy` 在 `VerbalTS._init_gnn()` 中加载：

```python
graph_adj = np.load(graph_adj_path)
graph_adj = torch.from_numpy(graph_adj).float()
self.register_buffer("graph_adj", graph_adj)
```

`graph_adj` 是 buffer，不是 trainable parameter，会随模型 `.to(device)` 自动移动。

### 5. GNN 在哪里插入

GNN 插入位置在 VerbalTS 多尺度 patch embedding 之后、residual diffusion layers 之前：

```python
x_in = torch.cat(x_list, dim=-1)
side_in = torch.cat(side_list, dim=-1)
x_in = self._apply_variable_gnn(x_in)
```

### 6. GNN 输入 shape

VerbalTS hidden：

```text
[B, C, V, Lp]
```

GNN input：

```text
[B, Lp, V, C]
```

SMD-NL debug 中实际为：

```text
[B, Lp, V, C] = [2, 360, 10, 32]
```

### 7. GNN 输出 shape

GNN output：

```text
[B, Lp, V, C]
```

SMD-NL debug 中实际为：

```text
[2, 360, 10, 32]
```

### 8. 如何恢复到原始模型需要的 shape

GNN 输出再 permute 回 VerbalTS 原始 hidden shape：

```text
[B, Lp, V, C] -> [B, C, V, Lp]
```

### 9. 最终输出是什么

最终输出仍是原模型预期的 predicted noise：

```text
[B, V, L] = [B, 10, 240]
```

## 四、配置说明

### `use_gnn`

- 是否启用 GNN；
- `false` 时完全绕过 GNN，不读取图文件；
- `true` 时必须提供 `graph_adj_path`。

### `graph_adj_path`

- `graph_adj.npy` 路径；
- 只有 `use_gnn=true` 时才需要。

### `gnn_hidden_dim`

- GNN 隐藏层维度；
- 多层 GNN 时用于中间层。

### `gnn_num_layers`

- GNN 层数；
- `1` 表示一层 graph convolution。

### `gnn_dropout`

- GNN 中间层 dropout；
- 仅训练模式下生效。

### `gnn_residual`

- 是否启用 GNN 残差连接；
- 当输入输出 shape 一致时执行 `h = h + residual_input`。

### `debug_shape`

- 是否打印 Graph-VerbalTS 关键 shape；
- 每个模型实例只打印一次；
- 正式训练建议设为 `false`。

## 五、运行命令

### 1. 检查数据

```powershell
python tools\check_smd_nl_data.py --data_dir "D:\文档\New project 3\data\smd_nl"
```

### 2. 构造 `graph_adj.npy`

```powershell
python tools\build_smd_graph.py --data_dir "D:\文档\New project 3\data\smd_nl" --top_k 3
```

### 3. 测试 GNN 模块

```powershell
conda run -n verbalts python tools\test_graph_module.py
```

### 4. Baseline forward debug

```powershell
python tools\debug_graph_verbalts_forward.py --config VerbalTS-main\configs\smd_nl\diff\model_text2ts_baseline_debug.yaml --data_dir "D:\文档\New project 3\data\smd_nl" --use_gnn false --device cpu
```

### 5. Graph-VerbalTS forward debug

```powershell
python tools\debug_graph_verbalts_forward.py --config VerbalTS-main\configs\smd_nl\diff\model_text2ts_graph_debug.yaml --data_dir "D:\文档\New project 3\data\smd_nl" --graph_adj_path "D:\文档\New project 3\data\smd_nl\graph_adj.npy" --use_gnn true --device cpu
```

## 六、如何关闭 GNN 回到原始 VerbalTS

在 diffusion config 中设置：

```yaml
use_gnn: false
graph_adj_path: ""
debug_shape: false
```

此时：

- 不构建 GNN；
- 不加载 `graph_adj.npy`；
- 不要求图路径存在；
- forward 中 `_apply_variable_gnn()` 直接返回原 hidden tensor。

Weather 和 synth-m 原配置已经默认关闭 GNN。

## 七、如何更换数据路径

如果使用 debug forward 脚本，直接改命令行：

```powershell
--data_dir "你的数据路径"
--graph_adj_path "你的数据路径\graph_adj.npy"
```

如果使用 VerbalTS 原 `run.py`，需要改或传入：

```powershell
--data_folder "你的数据路径"
```

SMD-NL debug YAML 中也有数据路径：

```yaml
data:
  folder: ../data/smd_nl
```

更换服务器目录时，可改这里，或优先用命令行覆盖。

## 八、如何把代码复制到远程服务器运行

需要一起复制：

- 整个 `VerbalTS-main/`
- 整个 `tools/`
- 整个 `data/smd_nl/`
- 尤其是这些新增/关键文件：

```text
VerbalTS-main/models/graph_modules.py
VerbalTS-main/models/diffusion/verbalts.py
VerbalTS-main/configs/smd_nl/
tools/check_smd_nl_data.py
tools/build_smd_graph.py
tools/test_graph_module.py
tools/debug_graph_verbalts_forward.py
data/smd_nl/graph_adj.npy
data/smd_nl/meta.json
```

如果远程服务器重新构图，也需要复制原始数据文件：

```text
train_ts.npy
valid_ts.npy
test_ts.npy
train_text_caps.npy
valid_text_caps.npy
test_text_caps.npy
train_attrs_idx.npy
valid_attrs_idx.npy
test_attrs_idx.npy
```

如果不重新构图，至少必须保证 `graph_adj.npy` 和数据集一起存在。

## 九、当前尚未完成的内容

当前还没有做：

- 完整训练；
- 效果评估；
- graph loss；
- 论文实验；
- 消融实验；
- Graph-VerbalTS 与 baseline 的正式指标对比；
- 旧 checkpoint 与新增 GNN 参数的兼容加载策略。

## 十、当前最大风险

后续仍需重点检查：

- GNN 插入位置是否最优，目前选择的是 patch embedding 后、residual layers 前。
- diffusion 中间 hidden tensor 是否最适合做变量图建模。
- 不同数据集的变量维度 `V` 是否始终和 `graph_adj.npy` 一致。
- 训练时 GNN 会增加显存和速度开销。
- `graph_adj.npy` 当前来自 Pearson correlation，是否足够表达真实依赖关系需要实验验证。
- 生成质量是否提升还完全未知，需要完整训练、评估和消融实验确认。
