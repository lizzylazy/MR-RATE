# Glossary

Terms that appear across the codebase with a specific MR-RATE meaning.

## Clinical / imaging

**Patient → Study → Series.** A *patient* has one or more *studies*
(examinations); a study contains several *series* (individual acquisitions such
as T1w, T2w, FLAIR, SWI, DWI, MRA). MR-RATE pairs one report with one study, and
a study contributes a variable number of series — typically 2–12+.

**`study_uid` / `patient_uid`.** De-identified identifiers assigned during
upload. `study_uid` is the join key across every repository, CSV and cache in the
project. Read `patient_uid` as a string; leading zeros are significant.

**Series ID.** Standardized modality identifier such as `t1w-raw-axi` —
modality, derivation, and acquisition plane. Constructed in step 4 of the imaging
pipeline. Filenames are `<study_uid>_<series_id>.nii.gz`.

**Center modality.** The one T1w series per study elected as the spatial
reference. Everything else in the study co-registers to it, and it is the series
that maps to the atlas. Marked `is_center_modality` in the metadata CSV.

**Space.** Which coordinate frame a volume lives in:

| Space | Meaning |
|---|---|
| `native_space` | as acquired (defaced), each series in its own geometry |
| `coreg_space` | every series resampled into the center modality's frame |
| `atlas_space` | every series in MNI152 (ICBM 2009c Nonlinear Symmetric) |

**Defacing.** Removal of identifiable facial features. HD-BET predicts a binary
brain mask, Quickshear removes the face. The dataset ships the defaced image plus
both masks. Because defacing biases the anterior part of the head, the loader
applies a 15 mm **posterior shift** to the crop centre.

**Batch (`batchXX`).** A dataset partition unit for upload/download, not a
training batch. Directory level in every repository.

## Modeling

**VL-CABS.** The contrastive objective. Each *sentence* attends over a study's
visual tokens, the attention-pooled vector is compared to that sentence, and the
loss is symmetric with sibling sentences of the same study excluded from the
negatives. The image representation is therefore sentence-specific rather than a
single global embedding.

**Fusion mode.** How a variable number of series becomes one representation:
`early` (channel stacking), `mid_cnn` (merge after the depth CNN), `late` (masked
mean over per-series tokens), `late_attn` (learned pooling over per-series
tokens). Anything MIL- or report-related requires `late`, because every other
mode collapses series before per-series tokens exist.

**Pooling strategy.** For `late_attn` only: `simple_attn` (learned query),
`cross_attn` (text-guided multi-head), `gated` (text-conditioned gate MLP).

**Temporal CNN.** `ResidualTemporalDownsample` — compresses the depth axis 4×
(256 slices → 64 frames) before the ViT. "Temporal" because the backbone is a
video transformer; the axis is anatomical depth.

**Sliding / tiling encoder.** The alternative to the temporal CNN: depth is split
into `chunk_size` non-overlapping tiles, each encoded fully, then mean-pooled.

**`MAX_TOKENS`.** 18432. Every fusion path zero-pads its token set to this length
with a boolean mask, so ranks holding studies with different series counts can
still all-gather.

**LoRA.** Low-rank adapters on the frozen ViT (`r=32`, `alpha=64` in contrastive
training; `r=16`, `alpha=32` for the Gemma writer). The backbone stays frozen.

**MIL / bag / instance.** Multiple-instance learning. A study is a *bag*; its
projected per-series visual tokens are the *instances*. `ClassifyThenAggregate`
is NeuroVFM's class-specific gated-attention head: each token is classified, then
attention-weighted within its bag and class. Softmax normalizes independently per
bag and per class.

**Ragged / `cu_seqlens` / offsets.** Bags have different sizes, so tokens are
stored flat with an offsets array marking boundaries rather than padded into a
rectangle.

**Exact cache.** A token cache built with `max_tokens_per_study=0` — no
subsampling, every token retained. Report-generation training requires it;
capped or pooled caches are refused.

**Linear probe.** Freeze the encoder, cache one pooled vector per study, train
`nn.Linear(dim, n_classes)`. Measures how linearly separable the representation
is, independent of prompt quality.

**Zero-shot.** No training on labels at all: score a pathology as
`sim("There is X") − sim("There is no X")`.

**Query resampler.** Cross-attention module turning a variable-length visual
token bag into a fixed set (512) of language-prefix tokens for the LLM.

**MIL conditioning.** What the writer is conditioned on. `all_classes` prepends
74 class tokens (label-name embedding + projected probability and
probability−threshold) built from the frozen MIL head; `none` is the ablation
with visual tokens only.

**Rebalancing strategies.** Per-subject sampling weights from inverse prevalence:
`inverse_freq` (sum over positive classes — rewards co-occurring rare findings),
`sqrt_inverse_freq` (sum with diminishing returns), `max_inverse_freq` (the
single rarest positive class sets the weight). Unlabeled and all-negative
subjects get `--rebalance_base_weight`.

## Self-supervised (MR-DINO)

**Anchor.** The sequence a training sample is built around. Every sequence of
every retained study is an anchor exactly once per local epoch.

**Global / local crops.** DINO's multi-crop: two large views (with ≥75% overlap
on every axis) and several small ones constrained to the globals' intersection.
Stage-dependent: `64×192×192` / `32×96×96` ×8 for `pretrain` and `gram`,
`64×384×384` / `32×192×192` ×4 for `highres`.

**Cross-sequence probability.** 0.75 — the chance that global view 2 comes from a
*different* atlas-aligned sequence than the anchor. This is what forces the model
to learn contrast-invariant anatomy. iBOT patch targets never cross sequences.

**iBOT.** Masked-patch self-distillation. Teacher and student always see the
identical sequence, crop and patch order for a patch target.

**KoLeo.** A regularizer spreading embeddings out to prevent collapse; computed
across ranks.

**Gram anchoring.** Optional later-stage loss matching the student's token Gram
matrix to a frozen teacher anchor.

**EMA teacher.** The teacher is an exponential moving average of the student, not
independently trained.

**Physical 3-D RoPE.** Rotary position embedding over *millimetre* coordinates
rather than voxel indices, so position is resolution-aware.

**Sequence weighting.** A sequence gets weight `1/sequences_in_study`, rescaled
to global mean 1, so many-series studies do not dominate the objective.

**Stage.** `pretrain` → `gram` → `highres`, each resuming from the previous.

## Infrastructure

**Accelerate.** Hugging Face's distributed launcher used by contrastive training
(multi-GPU, bf16, SyncBatchNorm).

**FSDP2.** Fully-sharded data parallel, the MR-DINO production path. `train_ddp.py`
is the smaller debug alternative.

**FP8.** Optional H200 8-bit linear path (`mr_dino/fp8.py`).

**`COMPLETE` marker.** A file written after a distributed checkpoint is fully
flushed. `RESUME=latest` only loads checkpoints that have it, so a job killed
mid-write is never resumed from a half-written directory.

**Preflight.** `mrrate_report_training.preflight` — the pre-training validation
gate: checkpoint SHA-256, architecture, label order, cache shape and fingerprint.

**Provenance.** Recorded metadata (hashes, config, label order) that proves an
encoder, a MIL head, and a token cache came from one origin.

**Xet.** Hugging Face's high-performance transfer backend, enabled with
`--xet-high-perf`.
