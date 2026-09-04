#!/bin/bash
# Extend seeds 100/123 (runids 3,4) for VerbalTS / Random / Pearson on GPU 1.
# Each variant trains runids 3-4 sequentially (~9h), total ~27h.
set -u
cd /data/kangjiale/verbalts-graph/VerbalTS-main
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
PY=/home/kangjiale/miniconda3/envs/verbalts/bin/python
COMMON="--cond_modal simple_text --training_stage finetune \
  --model_cond_config_path configs/smd_nl/cond/text_msmdiffmv_full.yaml \
  --train_config_path configs/smd_nl/train_full.yaml \
  --evaluate_config_path configs/smd_nl/evaluate_full.yaml \
  --data_folder /data/kangjiale/verbalts-graph/data/smd_nl \
  --clip_folder ./save/SMD_NL_cttp \
  --epochs 30 --batch_size 16 --lr 0.00005 --start_runid 3 --n_runs 5"

for v in baseline graph_random graph_pearson; do
  case $v in
    baseline)      diff_cfg=model_text2ts_baseline_full.yaml ;;
    graph_random)  diff_cfg=model_text2ts_graph_random_full.yaml ;;
    graph_pearson) diff_cfg=model_text2ts_graph_full.yaml ;;
  esac
  echo "[$(date)] === variant $v ($diff_cfg) start ==="
  $PY run.py \
    --save_folder ./save/smd_nl_${v}_epoch30_lr5e5_bs16_eval10_5seeds_1 \
    --model_diff_config_path configs/smd_nl/diff/$diff_cfg \
    --clip_cache_path ./cache/smd_nl_cttp_${v}_epoch30_lr5e5_5seeds_1 \
    $COMMON
  echo "[$(date)] === variant $v done ==="
done
echo "[$(date)] ALL DONE"
