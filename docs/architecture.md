# Architecture

MR-RATE is four independent codebases that share one dataset contract. Nothing in
the repository is a Python package that imports across submodule boundaries at
install time — the coupling is through *files*: NIfTI trees, CSVs, a JSONL, and
checkpoints. Two deliberate exceptions exist and are documented below.

## 1. System overview

```
                    ┌──────────────────────────────────────────────┐
   PACS / DICOM ───▶│  data-preprocessing/                         │
   Turkish reports  │  imaging pipeline + report NLP + labels      │
                    └───────────────┬──────────────────────────────┘
                                    │  publishes 4 Hugging Face repos
                                    ▼
        ┌───────────────────────────────────────────────────────────┐
        │  MR-RATE / MR-RATE-coreg / MR-RATE-atlas / -nvseg-ctmr    │
        │  + reports CSV + splits.csv + pathology label CSVs        │
        └──────┬──────────────────────────┬─────────────────────────┘
               │                          │
               ▼                          ▼
  ┌──────────────────────────┐  ┌────────────────────────────────┐
  │ contrastive-pretraining/ │  │ dino-pretraining/              │
  │ VL-CABS image↔text       │  │ self-supervised 3-D DINOv3     │
  │ (uses reports)           │  │ (atlas space, no reports)      │
  └──────────┬───────────────┘  └────────────────────────────────┘
             │ frozen encoder checkpoint + MIL head
             ▼
  ┌────────────────────────────────────────────────────────────────┐
  │ report-generation-training/                                    │
  │ frozen encoder → query resampler → Gemma LoRA writer           │
  │ → generated findings → clinical + NLG evaluation               │
  └────────────────────────────────────────────────────────────────┘
```

Two directions of travel exist through the same dataset:

- **Language-supervised** (`contrastive-pretraining` → `report-generation-training`) uses radiology reports as supervision.
- **Self-supervised** (`dino-pretraining`) uses no text at all; it learns from cross-sequence anatomical correspondence in atlas space.

### Cross-submodule imports

| Importer | Imports | Why |
|---|---|---|
| `dino-pretraining/mr_dino/data.py` | `contrastive-pretraining/scripts/data.py` (`discover_subjects`, `preprocess_nii`, normalizers) | So the DINO volume transform is *bit-for-bit* the contrastive/MIL transform rather than a reimplementation. Verified by the synthetic test suite. |
| `report-generation-training/src/mrrate_report_training/mil.py` | the `ClassifyThenAggregate` class from the upstream contrastive checkout (`upstream_root` in the config) | So the frozen MIL head is the exact class its checkpoint was trained with. |

Both are path-based (`PYTHONPATH` / `upstream_root`), not packaging dependencies.

## 2. Data preprocessing

Three pipelines, each producing a distinct artifact family.

### Imaging pipeline (7 steps)

`run/run_mri_preprocessing.py` drives steps 1–5 from one YAML config;
`run/run_mri_upload.py` drives 6–7.

| # | Step | Module | Produces |
|---|---|---|---|
| 1 | DICOM → NIfTI | `dcm2nii.py` (wraps `dcm2niix`) | per-accession `.nii.gz` + JSON sidecars |
| 2 | PACS metadata filtering | `pacs_metadata_filtering.py` | cleaned metadata CSV |
| 3 | Series classification | `series_classification.py` | modality label per series via a 5-level rule hierarchy (diffusion tags → vendor sequence IDs → scanning parameters → description keywords → numeric fallback) |
| 4 | Modality filtering | `modality_filtering.py` | accepted-series list, standardized series IDs (`t1w-raw-sag`), **center modality** election |
| 5 | Brain segmentation + defacing | `brain_segmentation_and_defacing.py` (HD-BET + Quickshear) | defaced image, brain mask, defacing mask |
| 6 | Zip & upload MRI | `zip_and_upload.py` | per-study zips on Hugging Face |
| 7 | Metadata upload | `prepare_metadata.py` | anonymized metadata CSV on Hugging Face |

Registration runs **separately**, on downloaded data, because it is done on a
different machine: `registration/registration.py` co-registers every moving
series to the T1w center modality with ANTs, then maps the center modality to
MNI152 (ICBM 2009c Nonlinear Symmetric) and propagates the transform to all
series. That yields the `coreg` and `atlas` repositories.

### Report pipeline (6 stages + a 7th label schema)

Turkish free text → structured English, each stage a SLURM-parallel script with
a **run → automated QC → retry → manual review** loop:

`01_anonymization` → `02_translation` → `03_translation_qc` → `04_structuring`
(`clinical_information` / `technique` / `findings` / `impression`) →
`05_structure_qc` → `06_pathology_classification`.

`07_neurovfm_diagnosis_extraction` is a *different* label schema (74 NeuroVFM
diagnoses) built by the same machinery; it is the ground truth used by the
report-generation evaluation.

### Label schemas — three of them

Because three coexist in the repo, be explicit about which one you are using:

| Schema | Classes | Where | Used by |
|---|---|---|---|
| SNOMED-grounded pathologies | **37** | `06_pathology_classification/data/pathologies.json` | the original classification output CSV |
| Shipped prompt set | **32** | `contrastive-pretraining/data/pathologies.json` | zero-shot inference; 5 of the 37 dropped for poor auto-label agreement (`Ventriculomegaly`, `Cerebral infarction`, `Mastoiditis`, `Cerebral hemorrhage`, `Lipoma of brain`) |
| Merged clinical groups | **14** | `contrastive-pretraining/scripts/eval_labels/splits_merged_majority/` | linear probe; 3-model majority vote collapsed into 8 `PP_*` pathophysiology + 6 `BP_*` imaging-phenotype groups |
| NeuroVFM diagnoses | **74** | `07_neurovfm_diagnosis_extraction/data/neurovfm_mri_diagnoses.json` | MIL head, report-writer conditioning, clinical evaluation |

Prompt index ↔ label column ↔ output index are positionally aligned within a
schema; they are **not** interchangeable across schemas. The tooling enforces
this (`extract_labels` refuses the legacy 37-pathology JSON on the vLLM backend).

## 3. Contrastive pretraining

### The model (`mr_rate/mr_rate/mr_rate.py`)

`MRRATE` is a dual encoder with a non-standard, token-level contrastive loss.

**Text side.** BiomedVLP-CXR-BERT-specialized. The `[CLS]` hidden state is
projected to `dim_latent` (512) and L2-normalized. Supervision is *per sentence*,
not per report: each study contributes up to `max_sentences_per_image` (34)
findings sentences, padded with a validity mask.

**Vision side.** A VJEPA2/2.1 ViT-G with a frozen backbone and LoRA adapters
(`r=32`, `alpha=64`, on `attention.{key,query,value,proj}` and `mlp.fc{1,2}`).
Each study has a *variable* number of series, so the batch size is fixed at 1
per GPU and the collate function simply unwraps the single item.

**Depth handling — two families:**

- *Temporal CNN* (`vjepa2`, `vjepa21`): `ResidualTemporalDownsample` — two stride-2 `Conv3d` layers on the depth axis plus a stride-4 skip — compresses 256 slices to 64 frames before the transformer sees anything. Grayscale is repeated to 3 channels; inputs with `D < 4` are repeated so the stride-4 stack still emits one frame.
- *Sliding* (`vjepa2_sliding`, `vjepa21_sliding`): no CNN. Depth is split into `chunk_size` (default 64, must be even) non-overlapping tiles, each is pushed through the full transformer, and the token outputs are mean-pooled. Training processes chunks sequentially with gradient checkpointing and a running mean; inference batches all chunks.

**Fusion — how variable series become one token set:**

| Mode | Mechanism |
|---|---|
| `early` | series stacked into channels before the encoder |
| `mid_cnn` | per-series CNN, masked mean over series, then one transformer pass |
| `late` | per-series full encoding, masked mean over series at token level |
| `late_attn` | per-series full encoding, then a learned pool: `simple_attn` (learned query), `cross_attn` (text-guided multi-head), `gated` (text-conditioned gate MLP) |

All fusion paths end in the same shape contract: tokens are zero-padded to
`MAX_TOKENS = 18432` with a boolean `token_mask`, so ranks with different series
counts can still `all_gather`.

**VL-CABS loss.** For each (sentence, study) pair, the sentence latent attends
over that study's visual tokens (masked softmax, padded tokens forced to `-1e4`),
the attention-weighted pooled vector is L2-normalized, and its cosine similarity
with the sentence is scaled by a learned temperature (initialized `1/0.07`,
clamped to `exp([0,4])`). So the "image embedding" is *sentence-specific* — this
is what distinguishes it from plain CLIP.

The final loss is symmetric:

- text→image: masked cross-entropy over studies,
- image→text: a log-sum-exp formulation where negatives exclude all sentences belonging to the same study (`image_membership`), so sibling sentences of the same report are never negatives of each other.

Numerical hardening is explicit throughout: float32 logit computation, `nan_to_num`
on inputs, logits clamped to ±100, `torch.where` instead of multiplication for
masking (to avoid `0 * NaN`).

**Distributed detail.** Text latents are gathered across ranks with a
differentiable all-gather; each rank computes similarities for its *own* images
against *all* texts using its **local** token mask, then the logit columns are
gathered. This keeps the per-rank memory bounded while giving a global-batch
contrastive denominator.

### Inference and probes

- **Zero-shot** (`inference.py`): per pathology, `score = sim("There is X") − sim("There is no X")` using the prompt pairs in `data/pathologies.json`. Positive means predicted present. AUROC comes from `eval.py`, which also offers bootstrap CIs and F1-optimal thresholds.
- **Linear probe** (`extract_features.py` → `linear_probe.py`): freeze everything, cache one pooled 512-d vector per study, train `nn.Linear(512, n_classes)` with `BCEWithLogitsLoss` and per-class `pos_weight = #neg/#pos` capped at 100. Class count is read from the labels CSV header — nothing is hardcoded. `relabel_features.py` swaps the label set on cached features without re-encoding.
- **MIL probe** (`mil_probe.py`, `mil_probe_online.py`): keeps per-series tokens instead of pooling them, and trains a `ClassifyThenAggregate` gated-attention head (NeuroVFM's class-specific MIL). Requires `--fusion_mode late`, because every other mode aggregates series before the projected tokens exist — `MRRATE._encode_visual_instances` raises if you try. Cached mode writes ragged bags once; online mode recomputes them each epoch.

## 4. DINOv3 pretraining

`dino-pretraining/` adapts FORA's CT DINOv3 recipe to atlas-registered
multi-sequence MRI. No reports, no labels.

**Sampling contract.** Every sequence of every retained study is an anchor
exactly once per local epoch. Studies are assigned to ranks by *sequence count*
(source bytes break ties), and each sequence gets weight `1/sequences_in_study`,
rescaled to global mean 1 — so a 12-series study does not outweigh a 2-series
one. Anchors stay grouped by study so a worker holds one study stack in memory.

**View construction.** Global view 1 is the anchor sequence. Global view 2 is a
*different* atlas-aligned sequence with probability 0.75, and the two crops
overlap ≥75% on every axis. Local crops live inside the global intersection and
may come from any aligned sequence. iBOT's teacher/student patch targets always
use the identical sequence, crop, and patch order — cross-sequence pairs are used
for DINO and KoLeo but never as patch-level targets. There are no flips or axis
reversals (anatomical laterality matters); student-only augmentation is bounded
gain/bias, noise, and mild 3-D blur, and the teacher sees clean input.

**Backbone.** `DinoVisionTransformer3D` with a 3-D conv patch embed (`(2,16,16)`
kernel) and physical 3-D RoPE — positions are in millimetres, not voxel indices,
so the embedding is resolution-aware. Sizes: `vit_large_3d`, `vit_hplus_3d`,
`vit_7b_3d`.

**Objective.** DINO global/local self-distillation with Sinkhorn-Knopp centering,
iBOT block masking, distributed KoLeo, optional Gram anchoring, EMA teacher.

**Stages** (`stage_crop_spec`): `pretrain`/`gram` use `64×192×192` globals with
`32×96×96` locals ×8; `highres` uses `64×384×384` globals with `32×192×192`
locals ×4.

**Scale.** `train_fsdp.py` is the production path (FSDP2, BF16, optional H200
FP8 via a custom autograd linear, `torch.compile` with node-local caches,
activation and selective checkpointing, atomic distributed checkpoints with exact
sampler and RNG resume, SLURM deadline signal → checkpoint → requeue).
`train_ddp.py` is the small/debug path.

## 5. Report generation

The writer never sees pixels directly. The chain is:

```
frozen MR-RATE encoder ──▶ projected per-series visual token bag  [T, 512]
        │                              │
        │                              ├─▶ QueryResampler (cross-attn, 512 learned queries)
        │                              │        └─▶ 512 language-prefix tokens
        │                              │
        └──────────────────────────────┴─▶ frozen 74-class ClassifyThenAggregate MIL head
                                                 └─▶ 74 class tokens
                                                     (label-name embedding +
                                                      projected [p, p − threshold])

LLM prefix = <image_start> · visual queries · <image_end> · [74 MIL tokens] · report prompt
             ──────────────────────────────────────────────────────────────────────────────▶
                                                       Gemma-3-12B-it + LoRA (trainable)
```

Only the query resampler/connector and the report LoRA are trained. Supervision
is the **natural `findings` text** from `all_reports.csv` — not the standardized
`findings_sentences.jsonl`, which stays an encoder/MIL data dependency used only
to reproduce the cohort. The MIL probabilities are soft context, never targets.

**Two execution modes, deliberately equivalent.** `online` runs the frozen
encoder during training; `cached` reads the identical complete bag from a ragged
memmap cache built by `build_cache.py`. The cache must be *exact*
(`max_tokens_per_study=0`, `feature_level=projected_per_series_visual_tokens`) —
pooled or token-capped caches are rejected.

**Provenance enforcement.** `preflight.py` verifies the encoder checkpoint's
SHA-256 against what the MIL head recorded, plus architecture, label order, and
cache fingerprint. Mixing an arbitrary MIL head with an arbitrary encoder fails
before training starts, not after.

**Ablation.** `writer.mil_conditioning: none` drops the 74 MIL tokens entirely
(visual conditioning only); no MIL head is loaded. Checkpoints carry a
`mil_conditioning` stamp and cross-mode loading fails loudly in both directions,
so an ablation checkpoint can never silently be evaluated as a full one.

**Evaluation.** Two independent axes:

- *Clinical*: re-extract the 74 NeuroVFM diagnoses from the generated text with the same prompt used for ground truth, then score per-pathology TP/FP/TN/FN, sensitivity, specificity, precision, NPV, F1, accuracy, balanced accuracy, prevalence — aggregated macro / micro / positive-support-weighted, plus subset and Hamming accuracy. Undefined ratios are null and excluded from macro means.
- *NLG*: corpus BLEU-1..4, ROUGE-1/2/L F1, METEOR (when NLTK wordnet data is available).

## 6. Design principles you will keep meeting

1. **One transform, one source of truth.** The per-volume pipeline lives in `contrastive-pretraining/scripts/data.py::preprocess_nii`. The offline `.npz` cache, the live loader, and MR-DINO all call it; the equivalence is asserted in tests.
2. **Manifests over trust.** Every derived cache carries a manifest of the config that produced it, and consumers refuse mismatches unless explicitly overridden.
3. **Fail loudly on silent-corruption paths.** Checkpoint loading hashes parameters before and after to catch a `strict=False` no-op; MIL and writer modes refuse cross-mode checkpoints; label CSVs must cover every cached subject or extraction errors out rather than misalign rows.
4. **Labels are never baked into features.** Encoder features are label-agnostic, so a new label schema costs a cheap relabel step, never a re-encode.
