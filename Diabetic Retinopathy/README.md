# Diabetic Retinopathy: Implicit Domain Adaptation Across Fundus Photography Sources

LINK TO KAGGLE NOTEBOOK: https://www.kaggle.com/code/asivakumarnair/diabetic-retinopathy-imagenet

## Hypothesis

Pretrained CNNs (EfficientNetB0, MobileNetV2, ResNet50) trained on data pooled across multiple heterogeneous fundus photography sources actively exploit that cross source variation, features learned on one source transfer and generalize to others. A custom CNN trained from scratch on the same pooled data shows no equivalent gain, because it lacks the pretrained feature basis needed to exploit heterogeneity. This mirrors the skin and chest tasks in the same project, testing whether the effect is a property of pretraining itself rather than a coincidence of any one dataset.

## Dataset sources and counts, verified live

Three public fundus photography datasets, every path verified live against the actual Kaggle filesystem before use, not assumed from documentation.

**APTOS 2019**
- `/kaggle/input/competitions/aptos2019-blindness-detection/train.csv`
- `/kaggle/input/competitions/aptos2019-blindness-detection/train_images`
- 3,662 rows, columns `id_code`, `diagnosis`
- Grade distribution: 0: 1805, 1: 370, 2: 999, 3: 193, 4: 295
- Path check: 200/200 sampled images present

**EyePACS (2015 subset only, not 2019)**
- `/kaggle/input/datasets/benjaminwarner/resized-2015-2019-blindness-detection-images/labels/trainLabels15.csv`
- `/kaggle/input/datasets/benjaminwarner/resized-2015-2019-blindness-detection-images/resized train 15`
- 35,126 rows, columns `image`, `level`
- Grade distribution (full set): 0: 25810, 1: 2443, 2: 5292, 3: 873, 4: 708
- Patient ID recoverable from filename via `{id}_{left|right}` regex, 0 extraction failures, 17,563 unique patients across 35,126 images
- Path check: 200/200 sampled images present
- **This dataset hosts both a 2015 and 2019 resized folder in the same download.** The 2019 folder (`resized train 19`) is the APTOS competition set under a different name, 3,662 files. Pointing EyePACS loading at that folder instead of `resized train 15` would silently reintroduce the exact APTOS/EyePACS overlap the disjointness gate exists to catch.

**Messidor-2**
- `/kaggle/input/datasets/mariaherrerot/messidor2preprocess/messidor_data.csv`
- `/kaggle/input/datasets/mariaherrerot/messidor2preprocess/messidor-2/messidor-2/preprocess`
- 1,744 rows, columns `id_code`, `diagnosis`, `adjudicated_dme`, `adjudicated_gradable`
- Grade distribution: 0: 1017, 1: 270, 2: 347, 3: 75, 4: 35
- `adjudicated_gradable` is 1 for all 1,744 rows, no filtering needed on gradability
- Path check: 200/200 sampled images present
- Images are already preprocessed, 512x512, single resolution, unlike APTOS (15 distinct resolutions, median 1536x2048) or EyePACS (81 distinct resolutions, median 873x1024)

## Source disjointness, verified not assumed

Image ID sets computed and intersected pairwise across all three sources before any split occurs:
- APTOS 3,662 ids, EyePACS 35,126 ids, Messidor 1,744 ids
- APTOS ∩ EyePACS: 0
- APTOS ∩ Messidor: 0
- EyePACS ∩ Messidor: 0
- Result: PASS, asserted in code, not just printed

## Messidor filename structure, decoded not assumed

Messidor-2's 1,744 filenames follow two different conventions, discovered by inspection rather than documentation:

- **1,057 rows**: date-style convention `{8-digit date}_{session_id}_{side_code}_PP.png`, e.g. `20051020_43808_0100_PP.png`. Session IDs mostly appear twice (both eyes, same session) but this pattern was found to only reliably identify 17 genuine two-eye pairs out of 1,040 unique session IDs when checked directly, most session IDs are effectively unique per image. This subset is treated as image-level with no reliable pairing signal, disclosed as a limitation rather than forced into a pairing scheme that doesn't hold.
- **687 rows**: older `IMxxxxxx.JPG` convention, e.g. `IM000012.JPG`. Sequential numbering was tested directly: sorting by the numeric part and checking the gap between consecutive rows shows a gap of exactly 1 for 332 rows, meaning roughly two thirds of this subset pairs cleanly by consecutive numbering (same patient, both eyes). The remaining rows have larger gaps (6, 7, 8, 13, 14+), meaning they don't have an adjacent same-session partner in this file.

## The pairing agreement number, and why it changed

This required two separate rounds of verification in this session, and the correction matters for anything written about it later.

**First pass** (exploratory, gap-based): restricting to the 332 rows where the gap to the previous row is exactly 1, and comparing grades between each such row and its predecessor, gave 274 matching grades out of 332, **82.5 percent agreement**. Compared against a same-distribution random-pairing baseline of 65.4 percent (computed from the sum of squared grade proportions), this looked like strong evidence for genuine patient pairing.

**Second pass** (the actual pairing rule used in the pipeline): the pipeline does not use gap-based consecutive differencing, it groups the sorted 687 rows into pairs of 2 by `index // 2`, giving 343 full pairs and 1 singleton (687 is odd). Recomputing grade agreement on this actual grouping gives **74.9 percent agreement**, not 82.5 percent. A second candidate offset, `(index + 1) // 2`, was also computed for comparison and gives 73.2 percent, confirming `index // 2` is the better of the two available offsets, not an arbitrary choice.

The 82.5 percent figure was a different construction (332 gap-1 candidate pairs) than what the code actually does (343 groups of 2 across all 687 rows). Both numbers are real and reproducible, verified live twice in this session with matching results both times, but only 74.9 percent describes what the pipeline's `index // 2` pairing actually produces. This is now the locked figure. Any writeup should state 74.9 percent against the 65.4 percent random baseline (a 9.5 percentage point edge), not the earlier 82.5 percent figure, and should disclose that the two pairing constructions diverge rather than presenting only the cleaner number.

**687 being odd is expected, not a bug.** `index // 2` on 687 rows produces 344 groups (343 pairs of size 2, 1 singleton of size 1), matching what the code prints. An earlier draft stated 343 groups without accounting for the singleton; this is corrected.

## Colour and resolution profile across sources

Computed across 200 sampled images per source (not single-image anecdote):

| Source | Mean RGB | Between-image SD | Within-image SD | Unique sizes | Median H x W | Median aspect (W/H) |
|---|---|---|---|---|---|---|
| APTOS | [107.9, 57.4, 19.6] | [30.8, 15.1, 14.5] | [61.8, 34.0, 13.4] | 15 | 1536 x 2048 | 1.322 |
| EyePACS | [106.6, 74.4, 51.3] | [36.4, 28.9, 28.7] | [55.9, 39.4, 27.8] | 81 | 873 x 1024 | 1.173 |
| Messidor | [119.2, 55.2, 19.4] | [26.9, 15.6, 8.4] | [73.0, 35.0, 13.7] | 1 | 512 x 512 | 1.000 |

This is the direct evidence that the three sources are genuinely heterogeneous in acquisition characteristics, not just in label distribution, different cameras, different preprocessing pipelines, different aspect ratios and colour balance. This table is publishable evidence on its own for the heterogeneity claim, independent of any downstream model result.

## EyePACS subsampling, patient-level, distribution-preserving

EyePACS at full size (35,126 images) would dominate the pooled set relative to APTOS's 3,662 and Messidor's 1,744. Subsampled to match APTOS's size at the **patient level** (not image level, to avoid splitting a patient's two eyes across the kept/discarded boundary), stratified by each patient's max grade, seed 42.

Result: 3,662 images kept, 1,831 unique patients. Grade distribution drift between the full EyePACS set and the subsample, per class:

| Grade | Parent | Subsample | Drift |
|---|---|---|---|
| 0 | 0.7348 | 0.7357 | 0.0009 |
| 1 | 0.0695 | 0.0655 | 0.0040 |
| 2 | 0.1507 | 0.1526 | 0.0020 |
| 3 | 0.0249 | 0.0246 | 0.0003 |
| 4 | 0.0202 | 0.0216 | 0.0014 |

All drift under the 3 percentage point gate. Pooled dataset after subsampling: 9,068 images.

Pooled grade x source crosstab:

| Grade | APTOS | EyePACS (subsampled) | Messidor |
|---|---|---|---|
| 0 | 1805 | 2694 | 1017 |
| 1 | 370 | 240 | 270 |
| 2 | 999 | 559 | 347 |
| 3 | 193 | 90 | 75 |
| 4 | 295 | 79 | 35 |

Grade 4 totals 409 across the pooled set, about 4.5 percent, sparse but not degenerate.

## Splitting, per-source rules, patient-leakage checked not assumed

Three different splitting rules per source, reflecting what each source's identifiers actually allow:

- **APTOS**: image-level split (no patient identifier available in this dataset). 70/15/15 via two sequential `train_test_split` calls (30 percent held out, then split 50/50), stratified by grade, seed 42, wrapped in a `safe_split` helper that falls back to unstratified splitting if a class has too few members to stratify.
- **EyePACS (subsampled)**: patient-level split. Grouped by `patient_id`, split at the patient level so no patient's images cross a train/val/test boundary, then images pulled per the patient-level assignment. Leakage check (`set` intersection across train/val/test patient IDs) asserted empty. Result: **PASS**.
- **Messidor**: mixed-mode split, handled in two parts. The 687 IM-style rows are grouped into pair-level pseudo-patient IDs (`messidor_pair_{index//2}`) using the verified `index // 2` pairing, then split at the pair level with the same leakage assertion. Result: **PASS (687 images, 344 groups)**. The 1,057 date-style rows, lacking a reliable pairing signal, are split at the image level, disclosed as such in the split output rather than silently treated as pair-safe.

**One stratification failure occurred and was handled, not hidden.** At the Messidor IM-style val/test split (the second of the two sequential splits, applied to an already-small subset), one grade class dropped to a single pair-group, making stratified splitting mathematically impossible (sklearn's exact error: "The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2."). The `safe_split` wrapper caught this and fell back to unstratified splitting for that one split only, printing a warning with the verbatim sklearn error. This is the correct behaviour, not a bug, and should be disclosed in any methods writeup as a specific, named limitation rather than glossed over as "stratified where possible."

Final split sizes: **Train 6,344 / Val 1,361 / Test 1,363** (roughly 70/15/15 of the pooled 9,068).

Test set grade x source crosstab:

| Grade | APTOS | EyePACS | Messidor |
|---|---|---|---|
| 0 | 271 | 404 | 153 |
| 1 | 56 | 35 | 41 |
| 2 | 150 | 87 | 48 |
| 3 | 29 | 12 | 12 |
| 4 | 44 | 12 | 9 |

## Class weights, pooled and per-source

**Pooled** (used for Stages 4 through 7, training on the combined 6,344-image train set): computed via `compute_class_weight('balanced', ...)`.

| Grade | Weight |
|---|---|
| 0 | 0.329 |
| 1 | 2.063 |
| 2 | 0.952 |
| 3 | 5.116 |
| 4 | 4.436 |

Span (max/min): **15.6x**, just under the 16x threshold that was observed to cause training instability in the Skin task's custom CNN.

**Per-source** (reserved for Stage 11, single-source training, never the pooled weights, per the explicit design decision that Stage 11 must be a clean isolation control):

| Source | Weight span |
|---|---|
| APTOS | 9.4x |
| EyePACS | 33.7x |
| Messidor | 31.0x |

Both EyePACS and Messidor per-source spans are roughly double the 16x collapse threshold. This is expected: pooling smooths grade 4 scarcity across three sources, but training on any single source in isolation exposes that source's true class imbalance directly, and grade 4 is thin in both EyePACS (79 images in the pooled subsample's crosstab) and especially Messidor (35 images total, split three ways). This is flagged as a specific watch item for Stage 11: check macro-AUC alongside accuracy for the custom CNN on these two sources especially, since AUC held up during Stage 4's pooled instability even while accuracy collapsed, and the same signature is the expected failure mode here. The plan is to check AUC first when a run looks unstable, and only switch that specific run's checkpoint monitor to `val_auc` if AUC is healthy while accuracy or QWK is not, rather than preemptively changing the monitor for all twelve Stage 11 runs.

## Environment and configuration, locked

- TensorFlow 2.19.0, installed explicitly via `pip install tensorflow==2.19.0` before any import
- `TF_USE_LEGACY_KERAS=1` set in environment before TensorFlow import (confirms `tf.keras` resolves to `tf_keras.api._v2.keras`, not native Keras 3)
- Seed 42 throughout (`PYTHONHASHSEED`, `random`, `numpy`, `tf.random.set_seed`), plus `TF_DETERMINISTIC_OPS=1`
- Kaggle notebook, T4 x2 accelerator selected but training run single-GPU, no distribution strategy invoked
- Kaggle account for this task: `asivakumarnair`, a third account distinct from the `ab0y04` and `nihalabhay` accounts used for the Skin task. Quota confirmed ample for this experiment before Stage 4 began.
- Image size 224x224, batch size 32
- Augmentation (train only): rotation range 20, width/height shift 0.1, horizontal flip, zoom range 0.1. Vertical flip explicitly off (retinal images have a meaningful up/down orientation).
- Phase 1 (frozen base) learning rate 1e-3, 10 epochs fixed. Phase 2 (full fine-tune) learning rate 1e-5, up to 60 epochs with early stopping, patience 7. Custom CNN learning rate 1e-3, up to 60 epochs, same patience.
- Default checkpoint monitor: `val_accuracy`, with `val_auc` as a per-run fallback if a collapse pattern is observed (accuracy or QWK broken while AUC stays healthy). AUC is compiled as a tracked metric on every model from the start specifically so this fallback is actually usable, an earlier draft of the training template did not compile AUC at all, which would have made the fallback silently unusable.
- Weight file naming: `dr_best_custom_cnn.keras`, `dr_best_efficientnet.keras`, `dr_best_mobilenet.keras`, `dr_best_resnet50.keras` for pooled models. Single-source (Stage 11) models: `ss_{arch}_{source}_dr.keras`.

## Custom CNN architecture, confirmed parameter count

3 convolutional blocks (32, 64, 128 filters, each block: two conv layers, batch norm, max pool, dropout 0.25), global average pooling instead of flatten, single 256-unit dense head with dropout 0.5, softmax output. Confirmed **23,109 parameters** for the 5-class DR head, this is the canonical parameter count to record in any cross-task methods table alongside skin and chest's custom CNN counts (same architecture, differing only in final layer class count).

## Stage-by-stage status

**Stage 0, environment bootstrap: Done.** TF 2.19.0 confirmed, legacy Keras confirmed, seed set, two T4 GPUs visible.

**Stage 1, load and gate all three sources: Done.** All four sub-gates passed: path checks (200/200 each source), disjointness (PASS), pairing verification (0.749, matching the corrected figure), EyePACS subsample drift (all classes under 3pp).

**Stage 2, splits, weights, generators: Done.** Both patient-leakage checks PASS (EyePACS, Messidor IM-style). One documented `safe_split` fallback on Messidor's rarest grade. Pooled class weights computed (15.6x span). Per-source class weights computed for later use in Stage 11 (APTOS 9.4x, EyePACS 33.7x, Messidor 31.0x). Generator factory built and confirmed correct class index mapping (`{'0':0,'1':1,'2':2,'3':3,'4':4}`).

**Stage 3, model definitions: Done.** Custom CNN and pretrained builder functions defined. Custom CNN parameter count confirmed at 23,109.

**Stage 4, train Custom CNN on pooled data: Done.** Trained 13 epochs before early stopping (patience 7 from an effective peak around epoch 4 to 6). Training showed real instability consistent with the pooled 15.6x weight span, `val_accuracy` and `val_auc` both crashed at epoch 1 (cold start, expected) and again more seriously at epoch 12 (`val_accuracy` 0.23, `val_auc` 0.41) after a stable plateau around epochs 4 through 8. `EarlyStopping` with `restore_best_weights=True` correctly recovered the stable-plateau checkpoint rather than the collapsed one. Final test result restored from the good checkpoint: **test accuracy 0.622, test macro-AUC 0.836**. AUC held up better than accuracy throughout the instability (staying in the 0.79 to 0.84 range even while accuracy swung from 0.23 to 0.62), which is the reference pattern to watch for in Stage 11's higher-imbalance single-source runs.

**Stage 5, train EfficientNetB0 on pooled data, two-phase: Done.** Phase 1 (head only, 10 epochs) and Phase 2 (full fine-tune) both checkpointed separately (`dr_phase1_efficientnet.keras` plus its own CSV log, `dr_best_efficientnet.keras` plus its own CSV log). Phase 2 stopped at epoch 8 via early stopping, training was stable throughout, no collapse pattern. Final result: **test accuracy 0.649, test macro-AUC 0.885.**

**Stage 6, train MobileNetV2 on pooled data, two-phase: Done.** Phase 1 was notably rockier than EfficientNetB0's, `val_accuracy` dropped to 0.28 at epoch 1 and only reached 0.59 by epoch 6 before wobbling again. Phase 2 recovered cleanly and trained the full 28 epochs before early stopping triggered, steady improvement throughout with no collapse. Final result: **test accuracy 0.671, test macro-AUC 0.904**, the best result of the three completed pretrained models so far.

**Stage 7, train ResNet50 on pooled data, two-phase: In progress, one full run lost to a session restart.** First attempt: Phase 1 completed (10 epochs, visibly noisier than EfficientNetB0's or MobileNetV2's Phase 1, `val_accuracy` swinging between 0.48 and 0.66 across the 8 printed epochs, though AUC stayed healthy at 0.82 to 0.88 throughout, so this was not a collapse in the Stage 4 sense). Phase 2 began and reached epoch 9 (71/199 steps into that epoch, val loss and accuracy trending consistently with a normal fine-tune) before the Kaggle session restarted and the run was lost, since it had not yet reached a `ModelCheckpoint` save point past the restart. Rebuilt as a standalone single-cell script (full data prep plus ResNet50-only training) and restarted from scratch, same seed 42, so the split and weights are bit-identical to the lost attempt. As of the last message in this session, this rerun had not yet been reported as complete.

**Stage 8, reload and evaluate all four pooled models, confusion matrices: Pending.**

**Stage 9 and 10: Pending**, not yet discussed in this session in DR-specific detail.

**Stage 11, train all 4 architectures on each of 3 sources in isolation (12 models): Pending.** Per-source class weights already computed and ready (Stage 2), explicit design decision locked that this stage must use `SOURCE_CLASS_WEIGHTS`, never the pooled `CLASS_WEIGHT`, since using pooled weights here would silently invalidate the entire point of Stage 11 as a clean single-source control. Expected to be the highest-risk stage for training instability given the 33.7x and 31.0x weight spans on EyePACS and Messidor.

**Stage 12, cross-source evaluation matrix: Pending.** Reads results directly off the 12 Stage 11 models, no retraining required.

**Stage 13, MMD (Maximum Mean Discrepancy), both directions: Pending.** Extended during this session (before Stage 4 even finished) to include a second analysis: in addition to computing MMD between the 12 Stage 11 single-source models, also extract penultimate-layer features from the Stage 4 through 7 pooled models and compute their pairwise MMD as a reference point. This tests whether pooled training genuinely closes the domain gap in feature space, not just in downstream accuracy or AUC, by showing where the pooled model's cross-source feature distance falls relative to the single-source matrix. Output will be `dr_stage13_mmd.csv` with a `model_type` column distinguishing `single_source` from `pooled` rows, so the two analyses are not conflated in the eventual Stage 17 writeup. This means the Stage 4 through 7 pooled model weight files must be kept, not treated as disposable once Stage 8's confusion matrices are done, since Stage 13 needs them again.

**Stage 14, fine-tune depth ablation: Pending.**

**Stage 15 through 17, statistical testing (McNemar, DeLong), Grad-CAM, final writeup: Pending.**

## Results summary so far, pooled training

| Model | Test Accuracy | Test macro-AUC |
|---|---|---|
| Custom CNN | 0.622 | 0.836 |
| EfficientNetB0 | 0.649 | 0.885 |
| MobileNetV2 | 0.671 | 0.904 |
| ResNet50 | pending (retraining after session restart) | pending |

Directional read so far, not yet statistically tested: accuracy and AUC both climb monotonically from Custom CNN through EfficientNetB0 to MobileNetV2, consistent with the project's core hypothesis that pretrained architectures benefit more from pooled heterogeneous data than a from-scratch model does. This is not yet confirmed statistically (McNemar and DeLong are Stage 15 or later work) and ResNet50's result is still pending, so this should be read as an early trend, not a locked finding.

## What's left to do, in order

1. Finish Stage 7 (ResNet50 pooled training), confirm test accuracy and macro-AUC
2. Stage 8, reload all four pooled models, generate confusion matrices
3. Stage 9 to 10, whatever the manual specifies next (not yet covered in this session)
4. Stage 11, train 12 single-source models, watching EyePACS and Messidor custom CNN runs closely for the collapse pattern seen in Stage 4, using AUC as the tell before considering a monitor switch
5. Stage 12, cross-source matrix from the 12 Stage 11 models, no retraining
6. Stage 13, MMD both directions on the 12 single-source models, plus the added pooled-model reference comparison, tagged by `model_type`
7. Stage 14, fine-tune depth ablation
8. Stage 15 to 17, McNemar and DeLong statistical testing, Grad-CAM interpretability, final writeup

## Open item, not yet resolved in this session

D-dr-3 (from the original manual): whether Stage 16 uses full 5-fold cross-validation or a reduced top-2-architecture approach was flagged as needing Dr. Cunha's input this week, given GPU quota math (Stage 16 alone estimated at 40 to 75 GPU hours out of a 75 to 135 hour total pipeline, against roughly 30 hours per Kaggle account per week and a five-week runway to the September target). This is the same decision needed across all three tasks (skin D2, chest D-chest-3, DR D-dr-3) and was recommended to be resolved as one conversation with Dr. Cunha rather than three separate ones. Not yet confirmed resolved as of this session.
