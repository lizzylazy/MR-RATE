# Data formats and artifacts

Every file MR-RATE reads or writes, what is inside it, and which code owns it.

- [Volume axis convention](#volume-axis-convention)
- [Dataset artifacts](#dataset-artifacts)
- [Directory layouts](#directory-layouts)
- [Derived caches](#derived-caches)
- [Checkpoints](#checkpoints)
- [Evaluation outputs](#evaluation-outputs)

---

## Volume axis convention

Everything downstream of the loader assumes **canonical RAS**. `nibabel.as_closest_canonical`
is applied on load, so acquisition plane no longer matters at the tensor level.

| Symbol | Axis | RAS meaning | Default spacing | Default size |
|---|---|---|---|---|
| `D` | 0 | S — superior/inferior (slice) | 1.0 mm | 256 |
| `H` | 1 | R — left/right | 0.5 mm | 384 |
| `W` | 2 | A — anterior/posterior | 0.5 mm | 384 |

Note the transpose: NIfTI arrives as `(X, Y, Z)` and is transposed to
`(Z, X, Y)`, with the header zooms reordered to match.

The pipeline, in order (`scripts/data.py::preprocess_nii` — the single source of
truth):

1. load → `as_closest_canonical` → `nan_to_num` → transpose to `(D, H, W)`
2. resample to `target_spacing`
3. normalize
4. crop/pad to `target_shape`, with a **posterior shift** on `W`

The posterior shift compensates for defacing: the crop centre moves toward lower
`W` by `round(posterior_shift_mm / target_spacing[2])` voxels (15 mm ÷ 0.5 mm =
30 voxels by default), clamped so it never runs off the edge. Padding is 0.

### Normalizers

| `--normalizer` | Behaviour |
|---|---|
| `zscore` | z-score over nonzero voxels, clip to [-5, 5], rescale to [-1, 1] |
| `percentile` | clip to the [0.5, 99.5] percentile range, rescale to [-1, 1] |
| `minmax` | plain min-max to [-1, 1] |

`--normalizer_kwargs` takes JSON, e.g. `'{"lower_percentile": 1.0, "upper_percentile": 99.0}'`.

### Batch tensor shapes

| Stage | Shape |
|---|---|
| One subject from the dataset | `[N, 1, D, H, W]` bf16, `N` = series count (variable) |
| After `collate_fn` (batch size 1) | `[1, N, 1, D, H, W]` |
| Model signature | `[B, R, C, D, H, W]` |
| Projected visual tokens (contrastive) | `[B, MAX_TOKENS=18432, dim_latent]` + boolean `token_mask` |
| Projected visual tokens (MIL, `return_visual_tokens=True`) | `[B, R·T, dim_latent]` + `[B, R·T]` mask |
| Pooled study embedding (inference) | `[B, dim_latent]`, L2-normalized |

---

## Dataset artifacts

### `metadata/batchXX_metadata.csv`

One row per **series**. Carries `study_uid`, `patient_uid`, the standardized
series id, shape/spacing, and the `is_center_modality` flag electing the T1w
series that registration and segmentation are anchored on. Read `patient_uid` as
a string — leading zeros matter:

```python
meta = pd.read_csv("data/MR-RATE/metadata/batch00_metadata.csv",
                   dtype={"patient_uid": str}, low_memory=False)
```

### `reports/batchXX_reports.csv`

Structured English reports, one row per study: `clinical_information`,
`technique`, `findings`, `impression`. The **natural `findings` field is the
report-generation supervision target**.

### `splits.csv`

Patient-level assignment materialized per study. Columns `batch_id`,
`patient_uid`, `study_uid`, `split` — training scripts only need `study_uid`
and `split`. Patient-level means no patient appears in two splits.

### `data/findings_sentences.jsonl`

Stored via Git LFS. One JSON object per line:

```json
{"volume_name": "SUBJECT_001", "valid_json": true,
 "extracted_sentences": ["Normal brain MRI.", "No acute infarction."]}
```

This is the *standardized* sentence list, used as contrastive supervision and to
reproduce the encoder/MIL cohort. It is explicitly **not** report-generation
supervision.

At load time each study contributes up to `max_sentences_per_image` (34)
sentences: more are randomly subsampled, fewer are padded with `""` and a
validity mask that zeroes them out of the loss.

### Pathology labels CSV

`study_uid` plus one binary (0/1) column per class. Column *names* must match the
pathology names in the prompt/diagnosis JSON, and column *order* is the output
order — index alignment is positional across `inference.py`,
`extract_features.py`, `linear_probe.py`, and the MIL head.

```csv
study_uid,Cerebral infarction,Glioma,...,Pituitary adenoma
STUDY001,0,1,...,1
```

Class count is always derived from the header at runtime; no script hardcodes it.
The four schemas in the repo (37 / 32 / 14 / 74) are listed in
[Architecture § label schemas](architecture.md#label-schemas--three-of-them).

### `data/pathologies.json`

Two accepted shapes:

```json
{"pathologies": {"Glioma": {"positive": "There is glioma",
                            "negative": "There is no glioma"}}}
```

or the legacy list form `["Glioma", "..."]`, which falls back to generic
`"There is {name}"` prompts. The structured form is what ships, because prompts
are tuned to radiological phrasing ("There is infarct", not "There is cerebral
infarction").

### `all_reports.csv`

Report-generation supervision. Requires `study_uid` and `findings`. Whitespace is
normalized within lines while line boundaries are preserved; rows with empty
findings are skipped, and duplicate `study_uid`s are a hard error.

---

## Directory layouts

Both are auto-detected by `discover_subjects(data_folder, space)` — layout 1 is
assumed when the first entry contains a `<space>` subfolder.

**Layout 1 — space-based:**

```
data_folder/<study_uid>/<space>/img/*.nii.gz
```

**Layout 2 — batch-based (the Hugging Face shape):**

```
data_folder/batchXX/<study_uid>/<img_subdir>/*.nii.gz
```

`--space` selects `<img_subdir>`:

| `--space` | subfolder | repository |
|---|---|---|
| `native_space` *(default)* | `img/` | `Forithmus/MR-RATE` |
| `coreg_space` | `coreg_img/` | `Forithmus/MR-RATE-coreg` |
| `atlas_space` | `atlas_img/` | `Forithmus/MR-RATE-atlas` |

Discovery is report- and split-agnostic: any subject directory with at least one
`.nii.gz` in its image subfolder is returned, sorted. Split filtering happens
afterwards.

### Full per-study contents after downloading everything and merging

```
<study_uid>/
├── img/                <study_uid>_<series_id>.nii.gz                     defaced native (uint16|float32)
├── seg/                <study_uid>_<series_id>_brain-mask.nii.gz          (uint8)
│                       <study_uid>_<series_id>_defacing-mask.nii.gz       (uint8)
│                       <study_uid>_<center_id>_nvseg-ctmr-brain.nii.gz    NV-Segment MR_BRAIN
│                       <study_uid>_<series_id>_nvseg-ctmr-wb.nii.gz       NV-Segment MR_BODY
├── coreg_img/          center series (copy) + <study_uid>_coreg_<series_id>.nii.gz  (float32)
├── coreg_seg/          center-series masks (copies)
├── atlas_img/          <study_uid>_atlas_<series_id>.nii.gz               (float32)
├── atlas_seg/          <study_uid>_atlas_<center_id>_*-mask.nii.gz        (uint8)
└── transform/          M_coreg_<series_id>.mat  (moving→center)
                        M_atlas_<center_id>.mat  (center→MNI152)
```

---

## Derived caches

### `.npz` volume cache — `preprocess_volumes.py`

```
<out_dir>/<space>/_manifest.json
<out_dir>/<space>/<study_uid>.npz      # key "volumes": [N, D, H, W]
```

Manifest:

```json
{"version": 1, "layout": "per_subject_stack", "space": "coreg_space",
 "target_spacing": [1.0, 0.5, 0.5], "target_shape": [256, 384, 384],
 "posterior_shift_mm": 15.0, "normalizer": "zscore",
 "normalizer_kwargs": {}, "dtype": "float16"}
```

The loader compares `space`, `target_spacing`, `target_shape`,
`posterior_shift_mm`, `normalizer`, `normalizer_kwargs` and **refuses** a
mismatch unless `--cache_allow_mismatch` is passed. `dtype` is a cache-only knob
(`float16` default, ~2× smaller); the loader casts to bf16 and adds the channel
dim, giving a tensor byte-for-byte identical to the live path.

Cache-only orchestration flags: `--compress` (smaller, slower reads — off by
default), `--overwrite`, `--num_workers`, `--num_shards`/`--shard_index`,
`--limit`.

Consumed by: `run_train.py`, `inference.py`, `extract_features.py`,
`mil_probe_online.py`, and MR-DINO (`--preprocessed-dir`) — all via
`--use_preprocessed`.

### Pooled feature cache — `extract_features.py --feature_level pooled`

| File | Shape | Content |
|---|---|---|
| `features_<split>.npy` | `[N, dim_latent]` float32 | frozen encoder output, L2-normalized masked-mean pool |
| `labels_<split>.npy` | `[N, num_classes]` float32 | 0/1 ground truth |
| `subject_ids_<split>.txt` | `N` lines | `study_uid` per row, same order as the `.npy` |
| `label_names.json` | — | CSV header, written once on the first split |

### Ragged token cache — `extract_features.py --feature_level tokens`

Requires `--fusion_mode late`. A raw memmap plus index arrays, described by a
manifest published atomically **last**:

```
token_features_<split>.json          manifest
tokens_<split>_<cache_id>.npy        [total_tokens, dim] float16|float32 memmap
token_offsets_<split>_<cache_id>.npy [N+1] int64   bag boundaries
labels_<split>_<cache_id>.npy        [N, num_classes]
subject_ids_<split>_<cache_id>.txt   N lines
full_token_counts_<split>_...npy     [N] int64     pre-subsampling counts
series_counts_<split>_...npy         [N] int32
```

Manifest keys include `format: "raw_numpy_memmap"`, `format_version`,
`feature_level: "projected_per_series_visual_tokens"`, `dim`, `dtype`,
`num_studies`, `num_tokens`, `max_tokens_per_study`, `cache_fingerprint`, and a
`provenance` block.

Report-generation's cached mode requires `max_tokens_per_study == 0` (an *exact*
cache) — a capped or pooled cache is rejected at construction.

> Pooled and token caches are not interchangeable. The linear probe consumes
> pooled; MIL and report generation consume tokens.

---

## Checkpoints

### Contrastive

| File | Content | Use |
|---|---|---|
| `MrRate.{step}.pt` | model state dict only | inference, probes, report generation |
| `MrRate.full.{step}.pt` | model + optimizer + scheduler + step | `--resume` (overwritten each save to bound disk) |

`MRRATE.load()` strips a `module.` prefix and loads with `strict=False`. Because
`strict=False` hides a wrong architecture, `extract_features.py` hashes a sample
of parameters before and after loading and **aborts if zero changed** — this is
what catches a mismatched `--fusion_mode`/`--encoder`/`--dim_latent`. Missing and
unexpected keys are printed; `--strict_missing` promotes missing keys to errors.

### MR-DINO

Distributed (DCP) checkpoints under `checkpoints/step_XXXXXXXX/`, each with a
`COMPLETE` marker. A checkpoint without the marker is never resumed. State
includes per-rank sampler position and RNG, so a requeued job continues the exact
data stream. `--keep-checkpoints` bounds retention.

### Report generation

`checkpoint-XXXXXXXX.pt` / `last.pt` hold only the *trainable* parts: query
resampler/connector, report LoRA, optimizer, scheduler, next data position, and
per-rank RNG. They carry a `mil_conditioning` stamp; loading an `all_classes`
checkpoint into a `none` model or vice versa fails loudly. Ablation checkpoints
contain no `label_embeddings`/`mil_*` tensors at all.

---

## Evaluation outputs

### Zero-shot inference

| File | Content |
|---|---|
| `predicted_scores.npz` | raw per-pathology scores (positive − negative similarity) |
| `labels.npz` | ground truth, when labels were supplied |
| `subject_ids.txt` | processed subjects, in row order |
| `scores.json` | per-subject scores, human-readable |
| `aurocs.xlsx` | per-pathology AUROC |

`eval.py` additionally offers bootstrap CIs (`*_bootstrap.xlsx` for AUROC, F1,
accuracy, precision) and F1-optimal operating points. Single-class columns (all 0
or all 1 in the split) are reported as `NaN` and excluded from macro means.

### Linear probe

`linear_head.pt`, `history.json` (per-epoch train loss and val mean AUROC),
`per_class_test_auroc.json`, `test_aurocs.csv`, and
`test_logits.npy` / `test_labels.npy` / `test_subject_ids.txt`. The best epoch by
validation mean AUROC is restored before test evaluation.

### Report generation

| File | Content |
|---|---|
| `generated_<split>.csv` | `study_uid, findings_gt, findings_pred` |
| `generated_<split>...shardNNofMM.csv` | one per shard; the evaluator accepts several |
| `*.meta.json` | provenance sidecar |
| `metrics.json` | BLEU-1..4, ROUGE-1/2/L F1, METEOR, clinical summary |
| `per_pathology_metrics.csv` | TP/FP/TN/FN, sensitivity, specificity, precision, NPV, F1, accuracy, balanced accuracy, prevalence |
| `nlg_per_sample.csv` | per-study NLG scores |

The clinical summary aggregates macro / micro / positive-support-weighted, plus
subset accuracy and Hamming accuracy. Undefined ratios — e.g. specificity for a
pathology with no negative studies — are null and excluded from macro averages.
