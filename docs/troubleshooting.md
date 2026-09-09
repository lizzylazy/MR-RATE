# Troubleshooting

Failures you are likely to hit, ordered by where they occur. Most of these are
deliberate guards — the code refuses to proceed rather than silently producing
wrong numbers, so the fix is usually to correct the configuration, not to
override the check.

---

## Setup

### `findings_sentences.jsonl` has no sentences / looks like a few lines of text

It is a **Git LFS** object. You cloned without LFS and got the pointer file.

```bash
git lfs install && git lfs pull
```

### `403` / `Repository not found` when downloading

All four Hugging Face repositories are **gated**. Request access on the dataset
page, then authenticate:

```bash
hf auth login          # or export HF_TOKEN=<token>
```

Verify what is actually visible without downloading:

```bash
python scripts/hf/download.py --batches all --no-mri --no-metadata --no-reports
```

### `ModuleNotFoundError: dinov3`

MR-DINO needs the official DINOv3 checkout at the pinned commit on `PYTHONPATH`:

```bash
git clone --filter=blob:none https://github.com/facebookresearch/dinov3.git /path/to/dinov3
git -C /path/to/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
export PYTHONPATH=/path/to/dinov3:/path/to/MR-RATE/dino-pretraining:$PYTHONPATH
```

### VJEPA 2.1 fails to initialize

`vjepa21` and `vjepa21_sliding` need both the torch.hub repo (fetched
automatically) **and** a local `.pt` checkpoint passed via
`--vjepa21_checkpoint`. `vjepa2` needs neither — it pulls from Hugging Face.

---

## Data loading

### `[MRReportDataset] Found 0 subjects`

Walk the causes in order:

1. **Wrong `--space` for the tree you downloaded.** `--space coreg_space` looks for `coreg_img/`; if you only downloaded native data there is nothing there.
2. **Wrong `--data_folder` level.** It must point at the directory *containing* `batchXX/` (or containing study directories in layout 1) — typically `.../MR-RATE-coreg/mri`.
3. **Zips not extracted.** Discovery looks for `*.nii.gz`, not `*.zip`. Re-run the download with `--unzip`.
4. **Split filter removed everything.** Check that the `study_uid` values in `splits.csv` match the directory names and that `--split` is spelled `train`/`val`/`test`.

`discover_subjects` returns any subject directory with ≥1 NIfTI in the image
subfolder, so an empty result means the path or the space is wrong, not the data.

### Cache manifest mismatch error

The `.npz` cache records the config that produced it and the loader compares
`space`, `target_spacing`, `target_shape`, `posterior_shift_mm`, `normalizer`,
`normalizer_kwargs`. A mismatch means the cached tensors are **not** what your
run asked for.

Correct fix: rebuild the cache with the settings you actually want, or change
your run's flags to match the cache. `--cache_allow_mismatch` downgrades the
error to a warning and is only appropriate when you have consciously decided the
difference is irrelevant — it will not make the tensors correct.

### `--use_preprocessed requires --preprocessed_dir`

Both flags travel together. With them, `--data_folder` becomes optional.

### Training is GPU-starved / very low utilization

Expected on raw coreg or atlas NIfTI (volumes reach ~2 GB). Build the `.npz`
cache once and train with `--use_preprocessed`. See
[End-to-end workflow § 2.2](end-to-end-workflow.md#22-strongly-recommended-build-the-npz-volume-cache).

---

## Training

### `chunk_size must be even (tubelet_size=2)`

Sliding encoders split depth into tiles that must align with the backbone's
tubelet size. Use an even `--chunk_size` (default 64).

### CUDA out of memory during contrastive training

The available levers, roughly in order of cost:

- Use a sliding encoder with a smaller `--chunk_size` — chunks are processed sequentially with gradient checkpointing and a running mean.
- Prefer `late` over `late_attn` (no pooling module parameters or activations).
- Reduce `--num_processes` per node so each rank gets more memory.
- Gradient checkpointing is already on by default at both volume and chunk level; there is no further switch.

Batch size is not a lever: it is fixed at 1 per GPU because studies have variable
series counts.

### Loss becomes NaN

The loss path is already hardened (float32 logits, `nan_to_num`, clamping to
±100, `torch.where` masking). If NaNs still appear, suspect the input instead:

- a volume that is entirely zero or entirely constant after normalization,
- a study whose sentences are all padding (empty `extracted_sentences`),
- an out-of-range `--normalizer_kwargs` percentile pair.

### Very noisy loss after enabling rebalancing

`inverse_freq` can oversample the same handful of extremely rare subjects.
Switch to `sqrt_inverse_freq` (diminishing returns) or `max_inverse_freq` (a
subject's weight is set by its single rarest positive class, capping
combinatorial blow-up).

### Resume did not resume

`--resume` looks for `MrRate.full.{step}.pt` in `--results_folder`. A model-only
`MrRate.{step}.pt` cannot resume — it has no optimizer or scheduler state. Check
that `--results_folder` points at the same directory as the original run.

For MR-DINO, `RESUME=latest` only loads a checkpoint directory that has a
`COMPLETE` marker. A job killed mid-write leaves an incomplete directory that is
correctly skipped.

---

## Inference and probing

### The checkpoint loaded but the model produces garbage / zero parameters changed

`MRRATE.load()` uses `strict=False`, which happily loads nothing when the
architecture differs. `extract_features.py` detects this: it hashes a sample of
parameters before and after loading and aborts if none changed.

The cause is always an architecture flag that does not match the checkpoint:
`--encoder`, `--fusion_mode`, `--pooling_strategy`, `--dim_latent`, or a missing
`--extra_latent_projection`. Fix the flags to match training. Use
`--strict_missing` to also fail on missing keys.

### `Series-preserving MIL extraction requires fusion_mode='late'`

Every other fusion mode aggregates series *before* the projected per-series
tokens exist, so there is nothing for MIL to build a bag from. This applies to
`--feature_level tokens`, both MIL probes, and report generation. If your
checkpoint was trained with a different fusion mode, MIL is not available for it.

### Linear probe reports `NaN` for some classes

A class that is all-0 or all-1 in the test split has no defined AUROC. It is
reported as `NaN` and excluded from the macro mean by design. If many classes are
`NaN`, your test split is too small or too imbalanced for that label schema.

### AUROC differs between zero-shot and probe on the same checkpoint

Expected — they measure different things. Zero-shot also measures prompt quality;
the linear probe measures linear separability of the frozen representation. They
are complementary, not redundant.

### I need a different label set — do I re-extract features?

No. Labels are never baked into encoder features. Use `relabel_features.py`,
which symlinks the features and rebuilds only the labels in the exact cached
subject order. If a cached subject is missing from the new CSV it errors out
rather than shifting rows. `--out_dir` must differ from `--features_dir`.

---

## Report generation

### Preflight fails on encoder/MIL provenance

The MIL head records the SHA-256 of the encoder checkpoint it was trained
against, along with architecture and label order. A mismatch means you paired an
arbitrary MIL head with an arbitrary encoder. Use the matched pair — there is no
override, because a mismatched pair produces plausible-looking but meaningless
conditioning.

### `Cached training needs the raw_numpy_memmap token cache` / `Pooled features cannot be used for report attention` / `Exact training requires max_tokens_per_study=0`

Three variants of the same problem: the cache you pointed at is not an *exact
token* cache. Rebuild with:

```bash
python scripts/extract_features.py ... --feature_level tokens --max_tokens_per_study 0
```

or use `scripts/build_exact_cache.sh`, which always sets it to 0. Pooled
linear-probe features and token-capped MIL caches are both rejected.

### Checkpoint refuses to load into the model

Ablation and full-conditioning checkpoints carry a `mil_conditioning` stamp and
refuse to load into the other mode — in both directions. Pre-ablation checkpoints
default to `all_classes`. Load the checkpoint into a config with the matching
`writer.mil_conditioning`.

### Generation refuses to write

An existing non-empty output CSV is refused so a completed run is not silently
destroyed. Use `--overwrite` to replace it, or `--resume` to append only the
studies still missing after a preemption.

### Label extraction rejects my diagnoses JSON

The vLLM backend refuses the legacy 37-pathology schema — the ground truth was
built with the 74-diagnosis NeuroVFM schema, and mixing them would silently
compare different label spaces. Use the bundled
`neurovfm_mri_diagnoses.json` (the default) and only set `DIAGNOSES_JSON` for a
deliberately different schema in NeuroVFM format.

Also note: the `keyword` backend of `extract_labels` is a deterministic
name/synonym matcher intended for tests. Never report numbers produced with it.

### METEOR is missing from `metrics.json`

METEOR requires NLTK wordnet data. Install it, or accept BLEU and ROUGE only —
the metric is skipped rather than failing the run.

---

## Cluster

### The SLURM job died at wall time and lost progress

Both production launchers are built for this: MR-DINO catches the wall-time
signal, writes a complete distributed checkpoint with per-rank sampler and RNG
state, and requeues itself. If you lost progress, check that `--save-every` is
not larger than your wall-clock budget and that the output directory is on shared
storage.

### Distributed hang during the contrastive forward pass

Encoding is per-series and logged (`[MRRATE rank=0] encoding volume i/r...`). A
hang at differing series counts across ranks usually means one rank is still
decoding a large NIfTI while others wait in the all-gather. Build the `.npz`
cache; it removes the variance. `debug=True` on the model forward prints the
gather progress per rank.

### Multi-node MR-DINO run behaves differently from the smoke test

The README is explicit: do not start production before
`scripts/smoke_dummy.sbatch` passes **in the same container and with the same
DINOv3 checkout**. Version drift in the DINOv3 source is the usual cause.
