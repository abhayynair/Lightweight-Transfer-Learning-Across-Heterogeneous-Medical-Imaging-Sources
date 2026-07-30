# Chest X-Ray Pathology Detection

This folder contains the chest X-ray arm of a three-task study testing whether lightweight ImageNet-pretrained CNNs (EfficientNetB0, MobileNetV2, ResNet50) exhibit implicit domain adaptation when trained on pooled data from heterogeneous, disjoint sources, while a custom CNN trained from scratch does not. The other two arms are skin lesion classification and diabetic retinopathy grading, in their respective sibling folders. This document covers chest specifically.

The core hypothesis, stated plainly: pretrained models can actively exploit cross-source heterogeneous data during training, extracting something useful from the fact that the data comes from multiple devices and populations, while a custom CNN trained from scratch on the same pooled data cannot, and may even be hurt by it. Chest X-ray is a binary presence-detection task (pathology present or absent), which makes it structurally the cleanest of the three tasks, since it avoids the label-harmonization problems that come with multi-class or ordinal grading across sources that each use their own vocabulary.

## Datasets

Three sources, chosen because each represents a genuinely different population, scanner, and labeling process, which is the actual domain gap this study measures. All three were verified live against the real files on Kaggle before any code was written against them, not assumed from documentation.

**CheXpert.** Kaggle dataset `ashery/chexpert`. 223,414 images, 201,033 labeled pathology and 22,381 labeled no-finding, a 90.0 percent pathology prevalence. 64,540 unique patients, identified by a `patient` ID embedded in the file path. Labels come from the `train.csv` file's `No Finding` column, where a value of 1.0 means no_finding and anything else (including NaN) means pathology. Split at patient level, since a patient identifier exists and reusing a patient's images across train and test would leak information.

**NIH ChestX-ray14.** Kaggle dataset `nih-chest-xrays` (organization dataset, path `organizations/nih-chest-xrays/data/`). 112,120 images, 51,759 pathology and 60,361 no-finding, a 46.2 percent pathology prevalence, close to balanced. 30,805 unique patients, identified via the `Patient ID` column in `Data_Entry_2017.csv`. A row is labeled no_finding only if its `Finding Labels` value is exactly the string "No Finding". Split at patient level.

**VinBigData Chest X-ray Abnormalities Detection.** Kaggle competition dataset, CSV from `vinbigdata-chest-xray-abnormalities-detection`, PNGs from `xhlulu/vinbigdata-chest-xray-png-512px-original-ratio`. This source ships as a detection dataset with multiple radiologist annotation rows per image, so it required collapsing to one label per unique image before it could be used the same way as the other two. An image was collapsed to no_finding only if every one of its rows carried `class_id 14` (confirmed as the "No finding" class), and to pathology if any row carried any other class ID. After collapse, 15,000 unique images remained, 4,394 pathology and 10,606 no-finding, a 29.3 percent pathology prevalence, the lowest of the three sources. No patient identifier exists for VinBigData, images are hash-anonymized at the source with no linkage available, so this source is split at image level rather than patient level. This is disclosed as an unavoidable limitation of the source, not treated as an oversight.

All three sources' path resolution, label rules, and counts above were confirmed by actually running the code against the real files in the Kaggle environment (loading each CSV, applying the label rule, sampling 200 resolved image paths per source and checking all 200 existed on disk) rather than assumed from documentation. One real discrepancy surfaced during this verification and is described in the Bugs and Near Misses section below.

## Key design decisions

**Binary presence-detection, not multi-class.** CheXpert, NIH, and VinBigData each define their own vocabulary of specific pathology classes, and those vocabularies do not agree with each other. Building a shared multi-class scheme across three different clinical labeling conventions would require harmonization decisions a reviewer could reasonably dispute, and any noise from that harmonization would contaminate the actual signal being measured, which is the domain gap itself. Binary classification, "is anything abnormal present," sidesteps this entirely: each source already made that determination as part of its own labeling process, and this study reads that determination rather than re-deriving it.

**Natural prevalence preserved per source, not force-balanced.** The three sources have deliberately different pathology base rates, 90.0, 46.2, and 29.3 percent. This prevalence difference is treated as a second, real axis of domain gap (label-level shift) sitting on top of the image-level shift (different scanners, different populations), not as noise to be balanced away. The direct consequence of this choice is that AUC, which is threshold-independent, is the primary metric for any cross-source comparison, since a prevalence mismatch between a model's training source and its test source should not be able to masquerade as a genuine domain gap the way a fixed-threshold metric like raw accuracy could. Accuracy, sensitivity, and specificity at the standard 0.5 threshold are still reported alongside AUC in every stage, because how a fixed threshold behaves differently across three very different prevalences is itself part of the story, not something to hide.

**Source-balanced subsampling to 15,000 images per source.** VinBigData's 15,000 unique images is the smallest of the three sources after its detection-to-classification collapse, so it sets the floor. CheXpert and NIH are each subsampled down to 15,000 images at the patient level (whole patients selected until the running image count reaches roughly 15,000, never splitting a single patient's images across the boundary), preserving each source's own natural prevalence rather than the pooled prevalence. This produces a pooled training set of approximately 45,000 images, 15,000 from each source, so no single source can dominate the pooled experiment purely by having more data. Every subsample's realized prevalence was checked against its parent source's prevalence and required to be within 3 percentage points, confirmed at CheXpert 0.02 percentage points drift, NIH 0.30 percentage points drift, and VinBigData exactly 0 drift (VinBigData is used in full, so there is nothing to drift).

**Both-direction MMD, planned for Stage 13.** For each pair of sources, Maximum Mean Discrepancy in feature space will be computed twice, once through the model trained on the first source and once through the model trained on the second source, rather than picking one direction arbitrarily. The asymmetry between the two directions is itself informative, and the MMD number is only reproducible if the extraction model is pinned, so both directions are computed and reported.

**Full training-history logging across both phases.** For every two-phase model, `CSVLogger` writes continuously from epoch 1 of Phase 1 (frozen backbone, head only) through the end of Phase 2 (full fine-tune), so a complete loss and metric curve exists for the entire training run, not just the fine-tuning portion.

**Correlated DeLong as the primary statistical test, plain DeLong reported alongside it.** All four architectures are evaluated on the exact same shared test set, which means their AUC estimates are statistically dependent on each other, not independent. Correlated DeLong accounts for this dependence correctly. Plain DeLong, which wrongly assumes independence, is computed and reported alongside it specifically to expose the discrepancy between the two, which is itself evidence the statistical choice was made deliberately rather than by default. On the pooled Stage 9 results, this discrepancy was substantial: plain DeLong lost statistical significance entirely on two pairs, EfficientNetB0 versus ResNet50 (correlated p_holm about 1.37e-7, plain p_holm about 0.0905) and MobileNetV2 versus ResNet50 (correlated p_holm about 1.98e-10, plain p_holm about 0.0571), that correlated DeLong correctly found significant. This exact pattern was independently corroborated by McNemar's test on the same pairs, which also found both differences significant. The maximum absolute difference between correlated and plain p-values across all six architecture pairs was 0.358.

**Binary classification implemented as 2-class softmax, not a single sigmoid output.** This keeps the loss function (`categorical_crossentropy`), the generator's `class_mode='categorical'`, and the overall model output shape identical to the skin task's setup, so architecture is the only variable that differs across tasks, not the classification mechanics. `FINAL_CLASSES = ['no_finding', 'pathology']`, index 0 and 1 respectively, and AUC is computed from the pathology-column probability.

**Horizontal flip retained in augmentation despite cardiac asymmetry.** Standard image augmentation includes horizontal flipping, which is a real concern for chest X-rays since flipping reverses the anatomical left-right orientation of the heart. It was kept anyway because the task here is binary presence-detection, not localization or laterality-sensitive diagnosis, and horizontal flip is standard practice in the chest X-ray classification literature for this reason. This was an explicit, disclosed choice, reversible if a future stage's results suggest it matters.

**Frozen and not to be changed:** TensorFlow and Keras version 2.19.0, binary 2-class output, 224 by 224 input images, batch size 32, per-model preprocessing functions, the four architectures (custom CNN, EfficientNetB0, MobileNetV2, ResNet50), two-phase fine-tuning structure, all hyperparameters, augmentation settings, the class-weighting scheme, and seed 42.

## Environment and reproducibility

All training runs on Kaggle Notebooks using a Tesla T4 by 2 GPU instance, TensorFlow 2.19.0 with legacy Keras 2 forced via the `TF_USE_LEGACY_KERAS=1` environment variable (this must be set before TensorFlow is imported for the first time in a session, and Kaggle's default TensorFlow version has been observed to drift to 2.20.0 with Keras 3 across different session restarts, so this pin is re-applied explicitly at the start of every single session rather than assumed to hold over). Seed 42 is set globally at the very start of every session, before any other import or operation, covering Python's hash seed, NumPy, and TensorFlow's random state, along with `TF_DETERMINISTIC_OPS=1` for GPU operation determinism. A separate subsample seed, also 42, controls the 15,000-per-source draw specifically. Five-fold cross-validation, where it is eventually run, uses the fold seeds 42, 123, 456, 789, and 1024.

Two GPUs are visible in every session by design (T4 by 2 gives more available memory), but training runs on a single GPU only. Multi-GPU strategies such as `MirroredStrategy` are deliberately not used, since data-parallel splitting would alter batch normalization statistics and effective batch size in ways that would become an uncontrolled variable across models.

Trained model weights are stored in a persistent Kaggle dataset (`ab0y04/all-best-models` for the original pooled models, later checkpoints also mirrored into a second persistent dataset under the `nihalabhay` account) with chest-prefixed filenames, since the same person's account also holds skin and DR model weights in the same space. Every checkpoint is downloaded and re-uploaded to a persistent dataset promptly after its training run finishes, because Kaggle's `/kaggle/working/` directory does not survive a session restart, which was learned directly from a near-miss described below.

## Invariants (rules the pipeline depends on and would break silently without)

**Per-model preprocessing is mutually exclusive.** The custom CNN uses `rescale=1./255` with no preprocessing function. Each of the three pretrained architectures uses its own `preprocess_input` function and no rescale. These are never combined. The generator factory function enforces this structurally by branching on whether a preprocessing function was passed in, rather than relying on the caller to remember.

**`load_model` is used for every reload, never a hand-rebuilt architecture with weights poured in via `load_weights`.** A `.keras` file is a complete, self-contained model, and reading it back with `load_model` reconstructs the true saved architecture rather than requiring it to be reconstructed by hand and hoping the weights line up.

**`shuffle=False` on validation and test generators, always.** Shuffling an evaluation generator scrambles the correspondence between predictions and true labels silently, with no error raised, so this is treated as non-negotiable rather than a preference.

**Recompile after changing a model's `trainable` attribute.** Phase 2 of every two-phase model unfreezes the previously frozen backbone by setting `trainable=True`, and the model must be recompiled after that change or the unfreeze silently does not take effect during training.

**Splitting rule differs by source and is chosen explicitly, never assumed.** CheXpert and NIH both have a genuine patient identifier and are split at patient level, with an explicit assertion after every split confirming that no patient ID appears in more than one of train, validation, or test. VinBigData has no patient identifier and is split at image level instead, a disclosed limitation of that specific source rather than a shortcut taken for convenience.

**Stratified splits use a fallback wrapper, never a bare stratified `train_test_split` call.** A wrapper function attempts a stratified split first and falls back to an unstratified split only if stratification fails on a rare class, printing a warning identifying exactly which split step triggered the fallback. Under this task's binary labels with tens of thousands of images per source, this fallback was not observed to trigger during any split actually run.

**Save discipline: download and persist every checkpoint within roughly two minutes of a training run finishing.** `ModelCheckpoint` writes to `/kaggle/working/`, which does not survive a session restart. This is not a theoretical risk, it caused an actual delay described in the section below.

**Kaggle quota and account are confirmed live before every GPU stage, not assumed from a prior check.** Chest is the heaviest of the three tasks by a wide margin, since it trains on three sources instead of one or two, so GPU hour budgeting was treated as a first-class concern throughout rather than an afterthought.

## Bugs and near misses actually caught

**CheXpert's on-disk folder structure did not match the CSV's path convention.** The `train.csv` file's `Path` column contains paths that begin with `CheXpert-v1.0-small/train/...`, but the specific Kaggle mirror used here (`ashery/chexpert`) strips that top-level `CheXpert-v1.0-small/` folder from the actual on-disk structure, so `train/` and `valid/` sit directly under the dataset root. Joining the raw CSV path onto the expected base directory therefore produced 0 out of 200 sampled paths resolving to real files on the first attempt. This was caught by the mandatory 200-sample existence check the pipeline runs before any split, not discovered later. The fix was to strip the `CheXpert-v1.0-small/` prefix from the CSV path before joining it to the base directory, confirmed afterward with a clean 200 out of 200 resolution, and disclosed explicitly as a deviation from the manual's original assumption rather than silently patched.

**A Kaggle-default TensorFlow and Keras version drift caused a silent, non-obvious failure mid-training.** Several sessions into training, a `class_weight` argument combined with categorical (one-hot) labels under Keras 3 produced a `ValueError: dtype='string' is not a valid dtype for Keras type promotion` on the very first training batch. The root cause was that Keras 3, which TensorFlow 2.19 and 2.20 both bundle by default depending on which exact version Kaggle's environment happened to load that session, changed how `class_weight` is applied internally when targets are one-hot encoded, in a way that Keras 2 did not have a problem with. The fix that was ultimately adopted, after an initial workaround using a manually computed `sample_weight` column proved unnecessary, was to force TensorFlow to use the legacy Keras 2 implementation via the `TF_USE_LEGACY_KERAS=1` environment variable, set before TensorFlow's first import in every session. This fix was actually discovered first in the parallel skin task's session and carried over to chest once confirmed.

**CustomCNN's checkpoint selection collapsed to a majority-class shortcut on CheXpert-only training, caught by specificity being exactly zero.** During Stage 11 (single-source baseline), training the custom CNN on CheXpert alone, which is 90.0 percent pathology, using `val_accuracy` as the checkpoint monitor resulted in a final model with specificity of exactly 0.0000, meaning it never once correctly identified a no-finding image across all 240 no-finding cases in that source's test set, despite achieving 89.20 percent overall accuracy. The mechanism: an "always predict pathology" model scores roughly 90 percent accuracy for free on a 90 percent pathology source, and `val_accuracy`-based checkpoint selection was locking onto exactly that shortcut whenever it appeared during training, rather than a genuinely discriminative epoch. The model's own AUC on that same run, 0.7072, showed there was real learnable signal being thrown away by the selection criterion. The fix, decided as a general policy going forward rather than a one-off patch, was to switch the checkpoint monitor and the early stopping criterion from `val_accuracy` to `val_auc`, since AUC cannot be gamed the same way by a constant-prediction shortcut. Rerunning CheXpert-only with `val_auc` monitoring produced AUC 0.7980 and specificity 0.1375, a genuine, if imperfect, improvement, and this AUC-based monitoring was subsequently applied to every remaining Stage 11 run and every pretrained model's Phase 1 and Phase 2 training from that point forward, as a precaution, not only where a problem had already been observed.

**A model checkpoint was nearly lost to a session restart before it was persisted.** After MobileNetV2's Phase 1 training completed and was saved to `/kaggle/working/`, the Kaggle session was restarted before that Phase 1 checkpoint had been downloaded and uploaded to a persistent dataset. Since `/kaggle/working/` does not survive a restart, this checkpoint was at risk of being lost entirely, which would have required rerunning roughly two hours of Phase 1 training. It was recovered because the checkpoint had, in that specific instance, already been uploaded to a persistent dataset under a different account shortly before the restart. This incident is the direct reason the save-discipline invariant above (download and persist within roughly two minutes of any training run finishing, before starting anything else) is now treated as strict rather than a suggestion.

**Grad-CAM gradient computation fails on GPU under enforced determinism.** Attempting to compute Grad-CAM heatmaps, which requires backpropagating gradients through a model's batch normalization layers with training disabled, produced `UnimplementedError: A deterministic GPU implementation of fused batch-norm backprop, when training is disabled, is not currently available`. This is a genuine limitation of the `TF_DETERMINISTIC_OPS=1` setting on GPU, not a bug in the Grad-CAM implementation itself. The fix was to run the Grad-CAM computation specifically inside a `tf.device('/CPU:0')` context, since the restriction is GPU-specific, without disabling determinism globally, which would have compromised the reproducibility of every actual training result. Grad-CAM is a visualization step over a small number of exemplar images, so the CPU speed cost is negligible.

## Stage map

The pipeline runs as 17 stages, structured so that expensive training happens exactly once per model and every downstream analysis stage reuses already-trained model checkpoints through inference only, rather than retraining.

| Stage | What it does | Compute | Trains new models |
|---|---|---|---|
| 0 | Session bootstrap: TensorFlow version pin, legacy Keras enforcement, seed, imports | CPU | No |
| 1 | Clean data build for all three sources, binary label derivation, image path resolution and verification, subsampling to 15,000 per source | CPU | No |
| 2 | Train/validation/test split (patient-level for CheXpert and NIH, image-level for VinBigData), class weight computation, generator factory | CPU | No |
| 3 | Model architecture definitions for all four architectures | CPU | No |
| 4 | Train custom CNN from scratch on the pooled 45,000-image training set | GPU | Yes |
| 5 | Train EfficientNetB0 (two-phase) on the pooled training set | GPU | Yes |
| 6 | Train MobileNetV2 (two-phase) on the pooled training set | GPU | Yes |
| 7 | Train ResNet50 (two-phase) on the pooled training set | GPU | Yes |
| 8 | Evaluate all four pooled models on the shared test set: AUC, accuracy, macro-F1, sensitivity, specificity, plus a per-source breakdown | GPU (inference only) | No |
| 9 | All-pairs statistical testing across the four pooled models: McNemar's test with Holm correction, correlated and plain DeLong AUC comparisons with 95 percent confidence intervals | CPU / GPU (inference only) | No |
| 10 | Confusion matrices for all four pooled models, plus systematically selected Grad-CAM exemplars | GPU (inference only) | No |
| 11 | Single-source baseline: each of the four architectures trained separately on CheXpert-only, NIH-only, and VinBigData-only, 12 models total | GPU | Yes, 12 runs |
| 12 | Cross-source 3-by-3 evaluation matrix, each of the 12 Stage 11 models evaluated on every source's held-out test set | GPU (inference only) | No, reuses Stage 11 |
| 13 | Maximum Mean Discrepancy on cross-source features, both directions per source pair, plus (by explicit later decision) the same MMD computation run through the four pooled Stage 4 to 7 models as a reference point, to test whether pooled training closes the domain gap in feature space and not only in downstream accuracy | GPU (inference only) | No |
| 14 | Fine-tuning depth ablation on EfficientNetB0 | GPU | Yes, 2 to 3 runs |
| 15 | CORAL domain-alignment baseline on EfficientNetB0 | GPU | Yes, 1 run |
| 16 | Five-fold cross-validation | GPU | Yes, heavy |
| 17 | Final results compilation | CPU | No |

Stage 11's 12 models are the load-bearing control for the entire study. Without a genuine single-source baseline, a critic could reasonably argue that the custom CNN simply has less capacity than the pretrained architectures in general, independent of any domain gap. Stage 11 isolates architecture from domain gap by training every architecture on one clean source at a time. Stages 12 and 13 then read results directly off those same 12 persisted checkpoints rather than training anything new, which is valid specifically because the three sources are fully disjoint and each source's own held-out test set is safe (patient-level for CheXpert and NIH, image-level, disclosed, for VinBigData).

Class weights for every Stage 11 single-source run are computed against that specific source's own prevalence, not the pooled prevalence, since the point of Stage 11 is to isolate how each architecture behaves on one source in true isolation, as if the other two sources did not exist. A direct consequence of this choice, recorded explicitly, is that Stage 12's cross-source evaluations cross both an image-domain gap and a label-prevalence gap simultaneously, since a model trained and weighted for one source's prevalence is then evaluated against a different source's prevalence. This is intentional, not an oversight, and is exactly why AUC, which does not depend on the training prevalence, is the primary metric for interpreting Stage 12's results.

## Current status

**Fully complete and locked:** Stage 0 through Stage 10.

Stage 8's locked pooled results, evaluated on the shared 6,505-image pooled test set:

| Model | AUC | Accuracy | Macro-F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Custom CNN | 0.8921 | 0.7852 | 0.7783 | 0.8897 | 0.6621 |
| EfficientNetB0 | 0.9190 | 0.8327 | 0.8324 | 0.8113 | 0.8580 |
| MobileNetV2 | 0.9175 | 0.8350 | 0.8343 | 0.8357 | 0.8342 |
| ResNet50 | 0.9280 | 0.8453 | 0.8446 | 0.8448 | 0.8459 |

ResNet50 is the top performer on every metric in this task, unlike the skin task where EfficientNetB0 and ResNet50 diverged (one clinically optimal by macro-AUC, the other accuracy-optimal). This divergence pattern from skin did not reproduce on chest, which is itself a disclosed, genuine finding rather than something to force into matching the other task's narrative.

Stage 8's per-source breakdown surfaced two important, disclosed findings. First, CheXpert specificity collapsed to near zero across all four pooled models (0.0 for the custom CNN, 0.075 to 0.175 for the three pretrained models), meaning the pooled models almost never correctly identify a no-finding CheXpert image, plausibly because a val_accuracy-selected checkpoint on a roughly balanced pooled set can still under-represent a 90-percent-prevalence source's minority class. Second, NIH was the hardest source for every one of the four models (AUC 0.67 to 0.74), and VinBigData was the easiest by a wide margin for every model (AUC 0.92 to 0.98), a pattern that has since been independently corroborated by the Stage 11 single-source results for both the custom CNN and EfficientNetB0.

Stage 9's all-pairs statistics: every pretrained model beats the custom CNN with McNemar's p_holm below 3.6e-19. EfficientNetB0 versus MobileNetV2 is the one architecture pair with no significant difference (McNemar p_holm 0.514, correlated DeLong p_holm 0.373). ResNet50 significantly beats both EfficientNetB0 (McNemar p_holm 0.0003, correlated DeLong p_holm about 1.37e-7) and MobileNetV2 (McNemar p_holm 0.0012, correlated DeLong p_holm about 1.98e-10). The correlated-versus-plain DeLong discrepancy described in the design decisions section above is one of the more citable methodological findings produced so far.

Stage 10's confusion matrices and Grad-CAM exemplars are complete and saved. The Grad-CAM exemplars were selected systematically rather than by eye, defined as test images where the custom CNN's prediction was wrong and all three pretrained models' predictions were correct. This selection surfaced 567 qualifying candidates, 501 of them no-finding images and only 66 pathology images, itself a visual confirmation of the CheXpert specificity problem described above, since a systematic search for the custom CNN's failures overwhelmingly found no-finding misclassifications.

**In progress:** Stage 11, the 12-model single-source baseline. Nine of twelve runs are complete.

Custom CNN single-source results, all three complete:

| Source | AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| CheXpert | 0.7980 | 0.8934 | 0.9849 | 0.1375 |
| NIH | 0.6860 | 0.6363 | 0.6100 | 0.6563 |
| VinBigData | 0.9239 | 0.8591 | 0.6434 | 0.9485 |

EfficientNetB0 single-source results, all three complete:

| Source | AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| CheXpert | 0.7980 | 0.8052 | 0.8280 | 0.6167 |
| NIH | 0.7228 | 0.6786 | 0.5507 | 0.7758 |
| VinBigData | 0.9731 | 0.9293 | 0.8452 | 0.9642 |

MobileNetV2 single-source runs across all three sources are the currently active run at the time of this document, structured as one unattended cell training and evaluating CheXpert, then NIH, then VinBigData in sequence, both training phases each, with AUC-based checkpoint monitoring throughout.

**Not yet started:** ResNet50's three single-source runs (the remaining three of the twelve), and Stages 12 through 17 in full.

**Open, non-blocking decision:** the scope of Stage 16's five-fold cross-validation is the single largest remaining GPU-hour lever in the entire task (estimated at over 90 hours on its own, against a total task estimate of roughly 185 to 235 GPU hours across all 17 stages). A descoped default (for example, three-fold cross-validation on the top two architectures rather than the full five-fold sweep across all four) has been identified as the fallback if compute runway becomes tight, but this decision has not been finalized and does not block any work currently in progress, since Stages 11 through 15 all come before Stage 16 regardless of how it is eventually scoped.

## Notes for the paper

The heterogeneity across the three sources bundles several distinct things together: different scanner or device characteristics, different patient populations, different labeling processes (CheXpert and NIH use different finding vocabularies and different degrees of automation in label derivation, VinBigData uses radiologist bounding-box annotations collapsed to a presence label), and different image file formats and resolutions as delivered by each Kaggle mirror (CheXpert's images are downsampled JPEGs, NIH's are 1024 pixel PNGs, VinBigData's are 512 pixel PNGs derived from original DICOM). Some portion of what looks like domain gap is therefore partly a format or provenance artifact rather than a purely clinical or device-level difference, and this should be named explicitly as a limitation rather than left for a reviewer to point out. The claim throughout should be framed as heterogeneous multi-source domain gap, with the format heterogeneity disclosed as one contributing, imperfectly separable factor within it.
