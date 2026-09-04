#!/bin/bash
# Extend metadata (attr) baseline seeds 100/123 (runids 3,4) on GPU 0.
# Configs identical to the 3-seed attr run (baseline diff cfg + cond_modal attr).
set -u
cd /data/kangjiale/verbalts-graph/VerbalTS-main
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
PY=/home/kangjiale/miniconda3/envs/verbalts/bin/python

$PY run.py \
  --cond_modal attr \
  --training_stage finetune \
  --model_cond_config_path configs/smd_nl/cond/attr_full.yaml \
  --model_diff_config_path configs/smd_nl/diff/model_text2ts_baseline_full.yaml \
  --train_config_path configs/smd_nl/train_full.yaml \
  --evaluate_config_path configs/smd_nl/evaluate_full.yaml \
  --data_folder /data/kangjiale/verbalts-graph/data/smd_nl \
  --clip_folder ./save/SMD_NL_cttp \
  --clip_cache_path ./cache/smd_nl_cttp_attr_epoch30_lr5e5_5seeds_1 \
  --save_folder ./save/smd_nl_attr_epoch30_lr5e5_bs16_eval10_5seeds_1 \
  --epochs 30 --batch_size 16 --lr 0.00005 --start_runid 3 --n_runs 5

echo "[$(date)] ATTR 5SEEDS DONE"
