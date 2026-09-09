# End-to-end workflow

The complete chain, in dependency order. Every stage names what it consumes,
what it produces, and where its authoritative documentation lives.

Most users start at **stage 2** — the preprocessed dataset is published, so
stage 1 only matters if you are building a dataset of your own from raw PACS
exports.

```
1. build the dataset ──▶ 2. contrastive encoder ──▶ 3. probes ──▶ 4. report writer ──▶ 5. evaluate
                     └──▶ 2b. self-supervised DINO (parallel branch, no reports)
```

---

## Stage 1 — Build the dataset (optional)

Run in `data-preprocessing/`, env `mr-rate-preprocessing`.

**1a. Imaging, steps 1–5.** One YAML drives everything; copy the template.

```bash
cp run/configs/mri_batch00.yaml run/configs/my_batch.yaml
# fill in input/output paths, HF repo id, processing parameters
python run/run_mri_preprocessing.py --config run/configs/my_batch.yaml
```

Produces `data/processed/batchXX/<study_id>/{img,seg}/` — defaced native images,
brain masks, defacing masks — plus the interim metadata CSVs.

**1b. Publish, steps 6–7.**

```bash
python run/run_mri_upload.py --config run/configs/my_batch.yaml
```

**1c. Registration** (separate machine, on downloaded data):

```bash
python src/mr_rate_preprocessing/registration/registration.py \
    --input-dir    data/MR-RATE/mri/batchXX \
    --metadata-csv data/MR-RATE/metadata/batchXX_metadata.csv \
    --output-dir   data/MR-RATE-reg \
    --num-processes 4 --threads-per-process 4 --verbose
```

Use `--total-partitions` / `--partition-index` to spread a batch across SLURM
array jobs. Upload with `registration/upload.py` (once with `--zip-suffix _coreg`,
once with `_atlas`).

**1d. Reports**, env `mr-rate-reports`. Six sequential stages, each with its own
QC loop; the full command set is in the
[report preprocessing README](../data-preprocessing/src/mr_rate_preprocessing/reports_preprocessing/README.md).
Merge any stage's shards with `utils/merge_shards.py`.

---

## Stage 2 — Contrastive pretraining

Run in `contrastive-pretraining/`.

### 2.1 Inputs you need

| Input | Source |
|---|---|
| MRI tree | `download.py` output, e.g. `data/MR-RATE-coreg/mri` |
| `findings_sentences.jsonl` | shipped at `data/findings_sentences.jsonl` (Git LFS) |
| `splits.csv` | Hugging Face `Forithmus/MR-RATE` |
| labels CSV *(only for rebalancing)* | `pathology_labels/mrrate_labels.csv` |

### 2.2 Strongly recommended: build the `.npz` volume cache

Coreg and atlas volumes reach ~2 GB; decoding, RAS-reorienting and resampling
them per step starves the GPUs. Bake the exact per-volume pipeline once:

```bash
python scripts/preprocess_volumes.py \
    --data_folder /path/to/data/MR-RATE-coreg/mri \
    --out_dir     /path/to/preprocessed \
    --space       coreg_space \
    --normalizer  zscore \
    --num_workers 8
# shard across array jobs: --num_shards N --shard_index $SLURM_ARRAY_TASK_ID
```

This is report- and split-agnostic: it discovers every subject under
`--data_folder` and keys the output only by the preprocessing config, which is
written to `<out_dir>/<space>/_manifest.json`. The loader validates that manifest
against the run's own settings and refuses a mismatch unless you pass
`--cache_allow_mismatch`.

The cached path is byte-for-byte identical to the live path — asserted in
`tests/test_preprocess_cache.py`.

### 2.3 Train

```bash
accelerate launch --multi_gpu --num_processes 4 scripts/run_train.py \
    --encoder vjepa2 --fusion_mode late \
    --space coreg_space --normalizer zscore \
    --jsonl_file data/findings_sentences.jsonl \
    --splits_csv /path/to/splits.csv --split train \
    --use_preprocessed --preprocessed_dir /path/to/preprocessed \
    --results_folder ./mr_rate_results \
    --num_train_steps 100001 --lr 1e-5 \
    --wandb --wandb_project mr-rate
```

Choices that must stay consistent downstream — write them down now, every later
stage has to repeat them: `--encoder`, `--fusion_mode`, `--pooling_strategy`,
`--space`, `--normalizer`, `dim_latent`.

**Rare-pathology rebalancing.** Prevalence is heavily skewed (~43% of studies
all-negative; rarest pathology ~0.23%). Add:

```bash
    --pathology_labels_csv /path/to/mrrate_labels.csv \
    --rebalance_strategy inverse_freq
```

This swaps uniform shuffling for a `WeightedRandomSampler`. Start with
`inverse_freq`; switch to `sqrt_inverse_freq` if the loss gets noisy from
oversampling the same handful of subjects, or `max_inverse_freq` to cap
combinatorial blow-up on multi-finding studies. Total compute is unchanged — only
the draw budget is redistributed.

**Resume.** `--resume` finds the latest `MrRate.full.{step}.pt` and restores
model, optimizer, scheduler and step; the W&B run id is persisted so a requeued
SLURM job continues the same run.

On a cluster: `sbatch scripts/submit_train.sh`, with overrides through
environment variables (`FUSION_MODE=late_attn POOLING_STRATEGY=cross_attn
NUM_TRAIN_STEPS=50000 sbatch scripts/submit_train.sh`).

---

## Stage 2b — Self-supervised DINO (parallel branch)

Run in `dino-pretraining/`. Requires **atlas space** and uses no reports.

Gate first — production must not start before this passes in the same container
and DINOv3 checkout:

```bash
sbatch scripts/smoke_dummy.sbatch     # succeeds only once step_00000002/COMPLETE exists
```

Then:

```bash
DATA_FOLDER=/path/to/data/MR-RATE-atlas/mri \
SPLITS_CSV=/path/to/splits.csv \
OUTPUT=/path/to/mrdino3d_7b/pretrain \
sbatch scripts/train_32n_7b.sbatch
```

Useful overrides: `STEPS`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `WORKERS`,
`LOCAL_CROPS`, `CROSS_SEQUENCE_PROBABILITY`, `WARMUP_STEPS`, `RESUME`, `FP8`,
`COMPILE`. `RESUME=latest` only loads a checkpoint whose `COMPLETE` marker
exists. Later stages chain off a finished one:

```bash
STAGE=gram RESUME=/path/to/pretrain/checkpoints/step_00100000 \
DATA_FOLDER=/path/to/data/MR-RATE-atlas/mri \
OUTPUT=/path/to/mrdino3d_7b/gram \
sbatch scripts/train_32n_7b.sbatch
```

`scripts/train_32n_hplus.sbatch` is the smaller DDP fallback. The launcher
catches the SLURM wall-time signal, writes a complete distributed checkpoint with
per-rank sampler and RNG state, and requeues itself.

---

## Stage 3 — Evaluate the encoder

Three complementary evaluations, all in `contrastive-pretraining/`. Every one of
them must be given the *same* `--encoder` / `--fusion_mode` / `--pooling_strategy`
/ `--dim_latent` the checkpoint was trained with.

### 3.1 Zero-shot classification

```bash
python scripts/inference.py \
    --encoder vjepa2 --fusion_mode late --pooling_strategy simple_attn \
    --weights_path ./mr_rate_results/MrRate.5000.pt \
    --data_folder /path/to/data --jsonl_file data/findings_sentences.jsonl \
    --pathologies_file data/pathologies.json \
    --labels_file /path/to/mrrate_labels.csv \
    --splits_csv /path/to/splits.csv --split test \
    --normalizer zscore --results_folder ./inference_results
```

Outputs `predicted_scores.npz`, `labels.npz`, `subject_ids.txt`, `scores.json`,
and `aurocs.xlsx`.

### 3.2 Linear probe

Encode once per split, then train the head as often as you like:

```bash
for SPLIT in train val test; do
  python scripts/extract_features.py \
      --encoder vjepa2 --fusion_mode late --pooling_strategy simple_attn \
      --weights_path ./mr_rate_results/MrRate.5000.pt \
      --data_folder /path/to/data --jsonl_file data/findings_sentences.jsonl \
      --labels_file scripts/eval_labels/splits_merged_majority/mrrate_merged_labels.csv \
      --splits_csv  scripts/eval_labels/splits_merged_majority/splits.csv \
      --split $SPLIT --normalizer zscore --out_dir ./lp_features
done

python scripts/linear_probe.py \
    --features_dir ./lp_features --results_dir ./lp_results \
    --epochs 50 --batch_size 256 --lr 1e-3 --weight_decay 1e-4
```

To probe a **different label schema**, never re-extract — relabel:

```bash
python scripts/relabel_features.py \
    --features_dir ./lp_features --labels_file /path/to/other_labels.csv \
    --out_dir ./lp_features_other
python scripts/linear_probe.py --features_dir ./lp_features_other --results_dir ./lp_results_other
```

`relabel_features.py` symlinks the features (use `--copy` to move machines) and
rebuilds only the labels in the exact cached subject order; a subject missing
from the new CSV is an error, not a silent row shift.

### 3.3 MIL probe

Same frozen encoder, but per-series tokens instead of one pooled vector.
**Requires `--fusion_mode late`.**

```bash
for SPLIT in train val test; do
  python scripts/extract_features.py \
    --weights_path "$WEIGHTS" --encoder vjepa2 --fusion_mode late \
    --pooling_strategy simple_attn --dim_latent 512 \
    --data_folder "$DATA" --jsonl_file "$REPORTS" \
    --labels_file "$LABELS" --splits_csv "$SPLITS" --split "$SPLIT" \
    --normalizer zscore --feature_level tokens --cache_dtype float16 \
    --out_dir "$FEATURES"
done

python scripts/mil_probe.py --features_dir "$FEATURES" \
    --results_dir /path/to/mil_results --epochs 50 --batch_size 4
```

`--max_tokens_per_study 2048` bounds the cache; `0` keeps everything.
`scripts/mil_probe_online.py` skips the token cache entirely and re-encodes each
epoch — cheaper on storage, more expensive in compute.

> The token cache produced here with `--max_tokens_per_study 0` is exactly what
> stage 4's cached mode consumes. Pooled linear-probe features are **not**
> reusable for MIL or for report generation.

Both MIL modes select the checkpoint and per-class thresholds on validation data
only, then evaluate the fixed model and thresholds on test.

---

## Stage 4 — Report generation training

Run in `report-generation-training/`.

### 4.1 Configure

```bash
cp configs/base.yaml configs/my_run.yaml
```

The shipped `encoder_checkpoint` and `mil_checkpoint` are placeholders. Set at
minimum:

```yaml
encoder_checkpoint: /abs/path/to/mr-rate-checkpoint.pt
mil_checkpoint:     /abs/path/to/matching/mil_head.pt
llm_path:           /abs/path/to/gemma-3-12b-it
upstream_root:      /abs/path/to/MR-RATE/contrastive-pretraining
data:
  data_folder:       /abs/path/to/mri
  reports_csv:       /abs/path/to/all_reports.csv
  jsonl_file:        /abs/path/to/findings_sentences.jsonl
  labels_file:       /abs/path/to/mrrate_neurovfm74_labels.csv
  splits_csv:        /abs/path/to/splits.csv
  cached_tokens_dir: /abs/path/to/exact_tokens
```

The encoder and the MIL head must be a matched pair — see 4.2.

### 4.2 Preflight

```bash
export PYTHONPATH=/path/to/MR-RATE/report-generation-training/src
python -m mrrate_report_training.preflight --config configs/my_run.yaml --mode cached
```

Rejects pooled features, token-capped caches, mismatched dimensions or class
counts, missing findings, and mismatched MIL/encoder provenance (the recorded
encoder SHA-256 is compared against the actual file).

### 4.3 Build the exact token cache (cached mode only)

```bash
export MRRATE_REPORT_CONFIG="$PWD/configs/my_run.yaml"
bash scripts/build_exact_cache.sh train
bash scripts/build_exact_cache.sh val
```

A GPU-node, frozen-encoder pass, always with `max_tokens_per_study=0`. Skip it if
a verified exact cache already exists.

### 4.4 Smoke, then train

Always smoke into a *separate* `output_dir` so it cannot overwrite production
checkpoints:

```bash
GPUS_PER_NODE=1 bash scripts/train_cached.sh configs/my_smoke.yaml --max-studies 8 --max-updates 2
GPUS_PER_NODE=1 bash scripts/train_online.sh configs/my_smoke.yaml --max-studies 2 --max-updates 1
```

Then submit:

```bash
export MODE=cached
export CONFIG="$PWD/configs/my_run.yaml"
export LLM_PATH=/abs/path/to/gemma-3-12b-it
export ENCODER_CHECKPOINT=/abs/path/to/mr-rate-checkpoint.pt
export MIL_CHECKPOINT=/abs/path/to/mil_head.pt
export GPUS_PER_NODE=4
sbatch --nodes=2 scripts/slurm_train.sh
```

Production policy is one epoch, batch size 1, no gradient accumulation, lr 1e-4 —
every real study is seen exactly once and no study is duplicated to fill a
distributed batch. Resume by setting `RESUME=/abs/path/to/checkpoint-00000500.pt`
and keeping `MODE`, `CONFIG`, node count and artifact variables identical.

### 4.5 Generate

```bash
bash scripts/generate_reports.sh cached test configs/my_run.yaml \
     runs/mrrate_single_writer_v1/last.pt
```

Writes `runs/generated_test.csv` (`study_uid, findings_gt, findings_pred`) plus a
`.meta.json` provenance sidecar. `--num-shards`/`--shard-index` splits a large
set across independent GPU jobs, each writing its own `...shardNNofMM.csv`;
`--resume` appends only missing studies after a preemption. An existing non-empty
output is refused unless you pass `--overwrite`.

---

## Stage 5 — Evaluation

### 5.1 Clinical accuracy

Re-extract the 74 NeuroVFM diagnoses from the *generated* text with the same
prompt that built the ground truth, on a GPU node with vLLM and Gemma:

```bash
export GENERATED_CSV="runs/generated_test.csv"
export OUTPUT_CSV="runs/pred_labels_test.csv"
export WORK_DIR="runs/label_extraction_test"
sbatch scripts/slurm_extract_labels.sh
```

The bundled `neurovfm_mri_diagnoses.json` is the default; set `DIAGNOSES_JSON`
only for a deliberately different NeuroVFM-format schema. Empty generated reports
get all-zero labels. The `keyword` backend is a deterministic matcher for tests
only — never for reported numbers.

### 5.2 Metrics (CPU)

```bash
python -m mrrate_report_training.evaluate_reports \
  --generated-csv runs/generated_test.csv \
  --gt-labels     /path/to/mrrate_neurovfm74_labels.csv \
  --pred-labels   runs/pred_labels_test.csv \
  --output-dir    runs/eval_test
```

Produces `metrics.json` (BLEU-1..4, ROUGE-1/2/L F1, METEOR when NLTK wordnet data
is present, plus the clinical summary), `per_pathology_metrics.csv`, and
`nlg_per_sample.csv`. Omit `--pred-labels` to score NLG only.

### 5.3 The conditioning ablation

Measures what the 74 MIL class tokens actually contribute. Make two configs that
differ **only** in `writer.mil_conditioning` and `output_dir`, train both,
generate on the same split, and evaluate both:

```bash
cp configs/my_run.yaml configs/my_run_ablation.yaml
# set writer.mil_conditioning: none  and a distinct output_dir

GPUS_PER_NODE=4 bash scripts/train_cached.sh configs/my_run.yaml
GPUS_PER_NODE=4 bash scripts/train_cached.sh configs/my_run_ablation.yaml
```

The metric deltas are the measured contribution of class conditioning. Cross-mode
checkpoint loading fails loudly in both directions, so the two runs cannot be
confused.

---

## Consistency checklist

Carry these forward unchanged from stage 2 through stage 5. Most confusing
failures downstream are one of these silently drifting.

- [ ] `--encoder` (and `--vjepa21_checkpoint` where applicable)
- [ ] `--fusion_mode` — must be `late` for anything MIL or report-generation
- [ ] `--pooling_strategy`
- [ ] `--dim_latent` (512)
- [ ] `--extra_latent_projection` if it was used in pretraining
- [ ] `--space` and `--normalizer` (and the spacing/shape/posterior-shift defaults)
- [ ] the label schema: 32 vs 14 vs 74 classes, and the matching splits CSV
- [ ] encoder checkpoint ↔ MIL head must be the pair whose provenance was recorded together
