# PDM-Net Experiments

This directory contains a runnable experiment scaffold for PDM-Net and the baselines used in the paper draft.

## 1. Put the dataset here

Place your downloaded dataset under:

```text
experiments/data/raw/<dataset_name>/
```

Two layouts are supported:

```text
<dataset_name>/
  angry/*.jpg
  disgust/*.jpg
  ...
```

or:

```text
<dataset_name>/
  train/angry/*.jpg
  val/angry/*.jpg
  test/angry/*.jpg
```

## 2. Create few-shot splits

```bash
python scripts/make_splits.py --source data/raw/<dataset_name> --out data/splits/<dataset_name> --shots 5 10 20 50 --seeds 0 1 2
```

## 3. Train

```bash
python train.py --config configs/pdmnet.yaml --split data/splits/<dataset_name>/5shot_seed0 --model pdmnet
```

Available model names:

```text
resnet18
mobilenetv2
resnet18_se
resnet18_cbam
prototype
pdmnet
```

Ablation configs are available under `configs/ablations/`, for example:

```bash
python train.py --config configs/ablations/class_agnostic_attention.yaml --split data/splits/<dataset_name>/20shot_seed0
```

For stronger numbers, especially in the paper tables, prefer the ResNet50 configs:

```bash
# Low-shot friendly: freezes the ImageNet backbone and trains only task heads.
python train.py --config configs/pdmnet_resnet50_frozen.yaml --split data/splits/<dataset_name>/5shot_seed0

# Stronger 50-shot/full-data fine-tuning.
python train.py --config configs/pdmnet_resnet50_strong.yaml --split data/splits/<dataset_name>/50shot_seed0

# Strong baseline for fair comparison.
python train.py --config configs/resnet50_strong.yaml --split data/splits/<dataset_name>/50shot_seed0
```

## 4. Test and visualize

```bash
python test.py --config configs/pdmnet.yaml --split data/splits/<dataset_name>/5shot_seed0 --checkpoint outputs/pdmnet_5shot_seed0/best.pt
```

The test script writes predictions, metrics, a confusion matrix, and optional PDM-Net mask visualizations to the output directory.

## 5. Aggregate results

```bash
python scripts/aggregate_results.py --outputs outputs --out outputs/summary.csv
```

## 6. Paper TODO

After running all shots and baselines, copy the averaged metrics into `../Paper/pdmnet_en.tex` and `../Paper/pdmnet_zh.tex`, replacing every `TODO` entry.

## PDM-ViT One-click Experiments

The PRCV-ready PDM-ViT paper uses the prototype-guided ViT fine-tuning script:

```bash
cd /root/experiments
bash scripts/run_pdmvit_full_experiment.sh
```

The runner trains PDM-ViT on full-data splits, optionally runs 50-shot experiments, runs ablations for the proposed token mask/prototype/consistency components, and writes:

```text
outputs/pdmvit_summary.csv
outputs/pdmvit_summary_grouped.csv
```

Useful switches:

```bash
SEEDS="0" RUN_LOW_SHOT=0 RUN_ABLATION=0 bash scripts/run_pdmvit_full_experiment.sh
MODEL=models_hf/vit-face-expression BATCH_SIZE=64 bash scripts/run_pdmvit_full_experiment.sh
```

Copy the averaged metrics from `outputs/pdmvit_summary_grouped.csv` into `../Paper/pdmvit_en.tex` and `../Paper/pdmvit_zh.tex`.
