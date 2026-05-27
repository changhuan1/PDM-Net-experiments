#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-Micro_Facial_Expressions}"
MODEL="${MODEL:-models_hf/vit-face-expression}"
EPOCHS_FULL="${EPOCHS_FULL:-20}"
EPOCHS_LOW_SHOT="${EPOCHS_LOW_SHOT:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LOW_SHOT_BATCH_SIZE="${LOW_SHOT_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEEDS="${SEEDS:-0 1 2}"
RUN_LOW_SHOT="${RUN_LOW_SHOT:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"

COMMON_ARGS=(
  --model "$MODEL"
  --lr 1e-5
  --head-lr 1e-4
  --weight-decay 0.01
  --label-smoothing 0.05
  --lambda-consistency 0.05
  --lambda-mask 0.0005
  --temperature 0.1
  --fusion-alpha 0.7
  --mask-scale 5.0
  --dropout 0.2
  --num-workers "$NUM_WORKERS"
)

echo "[1/4] Running PDM-ViT full-data experiments for seeds: $SEEDS"
for seed in $SEEDS; do
  python scripts/train_pdm_vit.py \
    --split "data/splits/${DATASET}/full_seed${seed}" \
    --output-dir "outputs/pdmvit_full_seed${seed}" \
    --epochs "$EPOCHS_FULL" \
    --batch-size "$BATCH_SIZE" \
    --seed "$seed" \
    "${COMMON_ARGS[@]}"
done

if [[ "$RUN_LOW_SHOT" == "1" ]]; then
  echo "[2/4] Running 50-shot PDM-ViT experiments for seeds: $SEEDS"
  for seed in $SEEDS; do
    python scripts/train_pdm_vit.py \
      --split "data/splits/${DATASET}/50shot_seed${seed}" \
      --output-dir "outputs/pdmvit_50shot_seed${seed}" \
      --epochs "$EPOCHS_LOW_SHOT" \
      --batch-size "$LOW_SHOT_BATCH_SIZE" \
      --seed "$seed" \
      "${COMMON_ARGS[@]}"
  done
else
  echo "[2/4] Skipping low-shot experiments because RUN_LOW_SHOT=$RUN_LOW_SHOT"
fi

if [[ "$RUN_ABLATION" == "1" ]]; then
  echo "[3/4] Running ablations on full_seed0"
  python scripts/train_pdm_vit.py \
    --split "data/splits/${DATASET}/full_seed0" \
    --output-dir "outputs/pdmvit_ablate_no_token_mask_full_seed0" \
    --epochs "$EPOCHS_FULL" \
    --batch-size "$BATCH_SIZE" \
    --seed 0 \
    --disable-token-mask \
    "${COMMON_ARGS[@]}"

  python scripts/train_pdm_vit.py \
    --split "data/splits/${DATASET}/full_seed0" \
    --output-dir "outputs/pdmvit_ablate_no_consistency_full_seed0" \
    --epochs "$EPOCHS_FULL" \
    --batch-size "$BATCH_SIZE" \
    --seed 0 \
    --lambda-consistency 0.0 \
    --model "$MODEL" \
    --lr 1e-5 \
    --head-lr 1e-4 \
    --weight-decay 0.01 \
    --label-smoothing 0.05 \
    --lambda-mask 0.0005 \
    --temperature 0.1 \
    --fusion-alpha 0.7 \
    --mask-scale 5.0 \
    --dropout 0.2 \
    --num-workers "$NUM_WORKERS"

  python scripts/train_pdm_vit.py \
    --split "data/splits/${DATASET}/full_seed0" \
    --output-dir "outputs/pdmvit_ablate_no_mask_sparsity_full_seed0" \
    --epochs "$EPOCHS_FULL" \
    --batch-size "$BATCH_SIZE" \
    --seed 0 \
    --lambda-mask 0.0 \
    --model "$MODEL" \
    --lr 1e-5 \
    --head-lr 1e-4 \
    --weight-decay 0.01 \
    --label-smoothing 0.05 \
    --lambda-consistency 0.05 \
    --temperature 0.1 \
    --fusion-alpha 0.7 \
    --mask-scale 5.0 \
    --dropout 0.2 \
    --num-workers "$NUM_WORKERS"

  python scripts/train_pdm_vit.py \
    --split "data/splits/${DATASET}/full_seed0" \
    --output-dir "outputs/pdmvit_ablate_global_only_full_seed0" \
    --epochs "$EPOCHS_FULL" \
    --batch-size "$BATCH_SIZE" \
    --seed 0 \
    --disable-prototype-branch \
    "${COMMON_ARGS[@]}"
else
  echo "[3/4] Skipping ablations because RUN_ABLATION=$RUN_ABLATION"
fi

echo "[4/4] Aggregating metrics"
python scripts/aggregate_results.py --outputs outputs --out outputs/pdmvit_summary.csv

echo "Done. Fill the paper TODO tables with:"
echo "  outputs/pdmvit_summary.csv"
echo "  outputs/pdmvit_summary_grouped.csv"
