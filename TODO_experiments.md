# Experiment TODO List

## Dataset

- [ ] Download the facial affect image dataset.
- [ ] Put it under `experiments/data/raw/<dataset_name>/`.
- [ ] Confirm class names and remove invalid/corrupted images.
- [ ] Decide whether to use six classes or seven classes with `neutral`.
- [ ] Run `scripts/make_splits.py` for 5-shot, 10-shot, 20-shot, 50-shot, and full-data protocols.

## Main Experiments

- [ ] Train `resnet18` for every shot and seed.
- [ ] Train `mobilenetv2` for every shot and seed.
- [ ] Train `resnet18_se` for every shot and seed.
- [ ] Train `resnet18_cbam` for every shot and seed.
- [ ] Train `prototype` for every shot and seed.
- [ ] Train `pdmnet` for every shot and seed.
- [ ] Average Accuracy, Macro-F1, and Weighted-F1 over seeds.
- [ ] Fill Table 1 in the English and Chinese paper drafts.

## Ablation

- [ ] Run full PDM-Net.
- [ ] Run without prototype mask.
- [ ] Run without prototype classifier.
- [ ] Run without consistency loss.
- [ ] Run without mask sparsity.
- [ ] Run class-agnostic attention variant.
- [ ] Run global branch only.
- [ ] Fill the ablation table in both paper drafts.

## Figures

- [ ] Export low-shot performance curves.
- [ ] Export confusion matrices for PDM-Net and the strongest baseline.
- [ ] Export mask visualization grids: original image, Grad-CAM, class-agnostic attention, prototype-guided mask, prediction.
- [ ] Replace all figure placeholders in `Paper/pdmnet_en.tex` and `Paper/pdmnet_zh.tex`.

## Submission Cleanup

- [ ] Replace author and affiliation placeholders.
- [ ] Remove every visible `TODO` from the PDFs.
- [ ] Verify PRCV page limit and formatting requirements.
- [ ] Compile final camera-ready PDF from the English draft.
