# Lightweight Transfer Learning Across Heterogeneous Medical Imaging Sources

SKIN: https://www.kaggle.com/code/ab0y04/skin-lesion-imagenet

CHEST: https://www.kaggle.com/code/nihalabhay/chest-imagenet

DR: https://www.kaggle.com/code/asivakumarnair/diabetic-retinopathy-imagenet

## What this project tests

The central hypothesis: ImageNet-pretrained lightweight CNNs (EfficientNetB0, MobileNetV2, ResNet50) exhibit something resembling automatic domain adaptation when trained on data pooled from multiple heterogeneous, disjoint clinical sources, purely as a byproduct of having been pretrained on a large, diverse natural image corpus, with no explicit domain adaptation technique applied. A CNN trained entirely from scratch, with no such prior exposure, does not show this property to the same degree, and may even be actively hurt by the heterogeneity that the pretrained models exploit.

This is tested identically across three independent medical imaging tasks, each with its own genuinely disjoint, real-world heterogeneous data sources: skin lesion classification, chest X-ray pathology detection, and diabetic retinopathy grading. If the same pattern holds across all three, that is evidence the effect is a property of pretraining itself, not a coincidence specific to one dataset or one imaging modality.

## Shared methodology across all three tasks

The same four architectures are benchmarked in every task: a custom three-block CNN trained from scratch (Conv2D and BatchNormalization pairs at 32, 64, and 128 filters, GlobalAveragePooling2D, Dense(256), Dropout, softmax output), and three ImageNet-pretrained backbones (EfficientNetB0, MobileNetV2, ResNet50) with `include_top=False`, wrapped identically and trained in two phases, a frozen-backbone phase followed by a full fine-tune phase.

All three tasks share the same 17-stage pipeline shape: environment setup, data cleaning and verification, splitting, class weighting, model definitions, pooled training for all four architectures, locked pooled results, pairwise statistical testing, confusion matrices and Grad-CAM, a single-source baseline (each architecture trained in isolation on each individual source, the load-bearing control that separates genuine domain gap from simple architecture capacity differences), a cross-source evaluation matrix built by reusing the single-source models with no retraining, Maximum Mean Discrepancy analysis in feature space, a fine-tuning depth ablation, a CORAL domain-alignment baseline, five-fold cross-validation, and final results compilation.

The same statistical standard is used throughout: McNemar's test with Holm correction for all pairwise architecture comparisons, and correlated DeLong as the primary AUC comparison test rather than the naive independent-samples version, since all architectures in a given task are evaluated on the same shared test set and their AUC estimates are therefore statistically dependent. Plain DeLong is reported alongside correlated DeLong in every task specifically to expose the discrepancy between the two, since on both skin and chest this discrepancy changed which architecture pairs were found significant.

All three tasks run on Kaggle Notebooks, TensorFlow 2.19.0 with the legacy Keras 2 backend forced via `TF_USE_LEGACY_KERAS=1` (set before TensorFlow's first import, since Kaggle's default environment has been observed to drift to Keras 3 across sessions), Tesla T4 GPUs run single-GPU rather than via a multi-GPU strategy (to avoid altering batch normalization statistics across architectures), and seed 42 throughout.

Every dataset path, column name, and count referenced anywhere in this project was verified live against the actual mounted files before being used, not assumed from documentation. Each task's own README documents the specific contamination risks, leakage bugs, and near misses this live verification caught and fixed.

## The three tasks

**[Skin](./Skin)**: ISIC 2019 and HAM10000 dermoscopic images, six-class lesion classification. Heterogeneity here comes from different clinical institutions and dermoscope equipment. ISIC 2019 was discovered to actually contain HAM10000 as a subset, requiring the two sources to be made genuinely disjoint before any comparison could be valid.

**[Chest](./Chest)**: CheXpert, NIH ChestX-ray14, and VinBigData, binary pathology presence detection. Heterogeneity comes from different scanners, populations, and labeling processes across three sources with incompatible pathology vocabularies, which is why this task uses binary rather than multi-class labels. The three sources also have deliberately different pathology prevalence (90.0, 46.2, and 29.3 percent), treated as a second, real axis of domain gap rather than balanced away.

**[Diabetic Retinopathy](./Diabetic%20Retinopathy)**: APTOS 2019, EyePACS, and Messidor-2 fundus photographs, five-class ordinal grading. Heterogeneity comes from different cameras, countries, and acquisition protocols, directly measurable in a colour and resolution profile across the three sources. EyePACS's dataset download was found to also contain the APTOS competition images under a different folder name, requiring explicit exclusion.

## Status at a glance

| Task | Pooled training (Stages 4-7) | Pooled results locked | Single-source baseline (Stage 11) | Cross-source and MMD (12-13) | Overall |
|---|---|---|---|---|---|
| Skin | Done | Done | Done, 8/8 models | Done | Furthest along. Stage 15 (CORAL) in progress, Stage 16 (5-fold) not started |
| Chest | Done | Done | In progress, 9/12 models | Not started | Mid-pipeline. Stages 0-10 fully locked |
| Diabetic Retinopathy | In progress, 3/4 done (ResNet50 rerunning after a session-restart loss) | Not yet | Not started | Not started | Earliest stage. Stages 0-6 done |

One open decision applies to all three tasks: whether Stage 16 (five-fold cross-validation) runs in full across all four architectures or a reduced scope, given it is the single largest remaining GPU-hour cost in every task. This is being treated as one decision rather than three, since the same tradeoff and the same shared weekly GPU quota apply across all of them.

## Reproducing this work

Every task folder is self-contained: its README documents the exact dataset paths, verified counts, split logic, class weights, environment pins, and stage-by-stage status needed to reproduce that task independently. Start with the task-specific README for implementation detail, this document exists only to orient across all three.
