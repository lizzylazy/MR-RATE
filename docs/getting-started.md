# Getting started

This page gets you from a fresh clone to a first working run. It does not repeat
every flag — the submodule READMEs do that — it tells you what to install, in
what order, and what to run first to prove the install works.

## 0. Prerequisites

| | Requirement |
|---|---|
| Python | 3.10–3.12 (CI tests 3.10, 3.11, 3.12) |
| GPU | Required for any training or encoding. Contrastive pretraining assumes bf16-capable GPUs; MR-DINO's production path assumes a multi-node SLURM cluster with FSDP2 |
| Git LFS | Required — `contrastive-pretraining/data/findings_sentences.jsonl` is an LFS object |
| Hugging Face account | Required for the dataset. All four repositories are **gated**; request access first |
| External binaries | Only for the imaging preprocessing pipeline: `dcm2niix`, ANTs |

```bash
git lfs install
git clone https://github.com/forithmus/MR-RATE.git
cd MR-RATE
```

If you cloned before installing Git LFS, run `git lfs pull` — otherwise
`findings_sentences.jsonl` is a pointer file and every loader will report zero
sentences.

## 1. Pick the environments you actually need

The four submodules have **separate, mutually incompatible** environments. Do not
try to merge them into one.

| Submodule | Environment | Install |
|---|---|---|
| `data-preprocessing` (imaging + registration) | conda `mr-rate-preprocessing`, Python 3.10, CUDA 12.1 wheels | `conda env create -f data-preprocessing/environment.yml && conda activate mr-rate-preprocessing && pip install -e data-preprocessing` |
| `data-preprocessing` (reports/LLM) | conda `mr-rate-reports`, Python 3.11, CUDA 12.6 wheels, vLLM | `conda env create -f data-preprocessing/environment_reports.yml && conda activate mr-rate-reports && pip install -e data-preprocessing` |
| `contrastive-pretraining` | any Python 3.11 env | see below |
| `dino-pretraining` | Python 3.12 + a DINOv3 checkout on `PYTHONPATH` | see below |
| `report-generation-training` | Python ≥3.10, needs Gemma weights + (for evaluation) vLLM | `pip install -e report-generation-training` |

### Contrastive pretraining

```bash
conda create -n mrrate python=3.11 -y
conda activate mrrate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r contrastive-pretraining/requirements.txt
pip install -e contrastive-pretraining/mr_rate/ -e contrastive-pretraining/vision_encoder/
```

`vjepa2` pulls its ViT-G from Hugging Face on first use. `vjepa21*` pulls the
`facebookresearch/vjepa2` torch.hub repo automatically but needs a local `.pt`
checkpoint passed via `--vjepa21_checkpoint`.

### DINO pretraining

MR-DINO builds on the official DINOv3 source at a pinned commit:

```bash
pip install -r dino-pretraining/requirements.txt pytest
git clone --filter=blob:none https://github.com/facebookresearch/dinov3.git /path/to/dinov3
git -C /path/to/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
export DINOV3_ROOT=/path/to/dinov3
export PYTHONPATH="$DINOV3_ROOT:/path/to/MR-RATE/dino-pretraining:$PYTHONPATH"
```

## 2. Verify the install before touching data

Both suites are synthetic — they need no dataset and no GPU.

```bash
cd contrastive-pretraining && python -m pytest          # ~106 tests, with coverage
cd ../dino-pretraining     && PYTHONPATH=$DINOV3_ROOT pytest -q
cd ../report-generation-training && python -m pytest
```

`python -m pytest --no-cov` is noticeably faster in `contrastive-pretraining` if
you do not need the coverage report.

The report-generation submodule also ships a CPU-runnable full-chain trial on a
fabricated 74-diagnosis dataset — training, checkpointing, generation, label
extraction, clinical evaluation:

```bash
cd report-generation-training && python tests/e2e_dummy.py
```

## 3. Get the data

All downloads go through `data-preprocessing/scripts/hf/download.py`. Authenticate
first:

```bash
hf auth login          # or: export HF_TOKEN=<your_token>
```

Check what is available without downloading anything:

```bash
cd data-preprocessing
python scripts/hf/download.py --batches all --no-mri --no-metadata --no-reports
```

Then pull one batch to work with. Native space plus metadata and reports:

```bash
python scripts/hf/download.py --batches 00 --unzip --delete-zips --xet-high-perf
```

Co-registered space (what contrastive training is usually run on) and atlas space
(what MR-DINO requires):

```bash
python scripts/hf/download.py --batches 00 --no-native --coreg --atlas \
    --unzip --delete-zips --xet-high-perf
```

Optionally consolidate the derivative repos into the native tree so that
`img/`, `coreg_img/`, and `atlas_img/` sit side by side under each study and
`--space` alone selects between them:

```bash
python scripts/hf/merge_downloaded_repos.py --coreg --atlas --batches 00
```

You end up with, per repository, `mri/batchXX/<study_uid>/…` plus
`metadata/batchXX_metadata.csv`, `reports/batchXX_reports.csv`, and `splits.csv`.
See [Data formats](data-formats.md) for what each file contains.

## 4. First real run

### Smallest useful contrastive run

```bash
cd contrastive-pretraining
accelerate launch --multi_gpu --num_processes 4 scripts/run_train.py \
    --encoder vjepa2 \
    --fusion_mode late \
    --space coreg_space \
    --normalizer zscore \
    --data_folder /path/to/data/MR-RATE-coreg/mri \
    --jsonl_file  data/findings_sentences.jsonl \
    --splits_csv  /path/to/splits.csv --split train \
    --num_train_steps 1000 \
    --results_folder ./mr_rate_results
```

Batch size is fixed at 1 per GPU (studies have variable series counts), so
throughput scales with `--num_processes`. Checkpoints land in `--results_folder`
every 500 steps as a model-only `MrRate.{step}.pt` and a resume-capable
`MrRate.full.{step}.pt`.

> Before any long run on coreg or atlas data, build the `.npz` cache — see
> [End-to-end workflow](end-to-end-workflow.md#22-strongly-recommended-build-the-npz-volume-cache).
> Decoding multi-gigabyte NIfTIs on the fly starves the GPUs.

### Smallest useful DINO run

```bash
cd dino-pretraining
torchrun --standalone --nproc-per-node=1 -m mr_dino.train_ddp \
  --data-folder /path/to/data/MR-RATE-atlas/mri \
  --output-dir /tmp/mrdino_debug \
  --arch tiny --steps 10 --workers 0 --local-crops 2 \
  --prototypes 64 --head-hidden-dim 128 \
  --warmup-steps 1 --teacher-warmup-steps 1 \
  --save-every 5 --log-every 1 --resume none
```

This is the correctness/debug path. Production is FSDP2 via
`sbatch scripts/train_32n_7b.sbatch`, and the README is explicit that
`scripts/smoke_dummy.sbatch` must pass in the same container and DINOv3 checkout
first.

## 5. Where to go next

- The full chain, in order, with real commands: [End-to-end workflow](end-to-end-workflow.md)
- What every produced file means: [Data formats](data-formats.md)
- When something breaks: [Troubleshooting](troubleshooting.md)
