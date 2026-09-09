# Development

Repository layout, test suites, CI, and the conventions worth knowing before
changing anything.

## Repository layout

```
MR-RATE/
├── docs/                        # this documentation set
├── data-preprocessing/          # dataset construction (Apache-2.0 code)
│   ├── run/                     # YAML-driven orchestrators (steps 1-5, 6-7)
│   ├── scripts/hf/              # standalone download / merge tooling
│   └── src/mr_rate_preprocessing/
│       ├── mri_preprocessing/   # steps 1-7
│       ├── registration/        # ANTs coreg + MNI152 atlas
│       ├── multi_label_seg/     # NV-Segment-CTMR (coming soon)
│       └── reports_preprocessing/  # 01..07 LLM stages
├── contrastive-pretraining/
│   ├── mr_rate/                 # installable: MRRATE model, pooling, VL-CABS
│   ├── vision_encoder/          # installable: VJEPA2/2.1 (+ sliding), optimizer
│   ├── scripts/                 # train / infer / probe / cache entry points
│   ├── data/                    # findings_sentences.jsonl (LFS), pathologies.json
│   └── tests/                   # 106 synthetic tests
├── dino-pretraining/
│   ├── mr_dino/                 # data, model, objective, train_ddp, train_fsdp, fp8
│   ├── scripts/                 # SLURM launchers + smoke
│   └── tests/                   # synthetic dataset generators + integration tests
├── report-generation-training/
│   ├── src/mrrate_report_training/
│   ├── configs/base.yaml
│   ├── scripts/                 # cache / train / generate / evaluate launchers
│   └── tests/                   # CPU tests + GPU probes
└── .github/workflows/tests.yml
```

Four submodules, four environments, no shared package. The only cross-module
coupling is path-based and deliberate — see
[Architecture § cross-submodule imports](architecture.md#cross-submodule-imports).

## Tests

### `contrastive-pretraining` (106 tests)

```bash
cd contrastive-pretraining
python -m pytest                      # with coverage (configured in pyproject.toml)
python -m pytest --no-cov             # faster
python -m pytest tests/test_fusion_modes.py -v
```

`pyproject.toml` puts `mr_rate`, `vision_encoder` and `scripts` on the path and
measures coverage over the two packages.

| File | What it guards |
|---|---|
| `test_imports.py` | dependencies and package imports resolve |
| `test_data.py` | the three normalizers, `collate_fn` |
| `test_preprocess_cache.py` | `discover_subjects`, `preprocess_volumes.py`, **live ↔ cache equivalence**, manifest rejection |
| `test_pooling.py` | the three pooling modules: shapes, masking, gradients |
| `test_mr_rate_model.py` | model init, forward, loss, serialization |
| `test_fusion_modes.py` | all 4 fusion modes × all pooling strategies, end to end |
| `test_vision_encoder.py` | `ResidualTemporalDownsample`, VJEPA2 preprocessing, optimizer weight-decay grouping |

### `dino-pretraining`

```bash
cd dino-pretraining
PYTHONPATH=/path/to/dinov3 pytest -q
```

Synthetic atlas NIfTIs (and optionally a dummy cache) are generated, then the
suite asserts: the raw transform is bit-for-bit the MIL transform, manifest
rejection works, cross-sequence alignment is deterministic, sequence indexing is
complete, local crops are contained in the global intersection, 3-D masks are
well formed, a real DINO+iBOT forward/backward optimizer step runs, EMA updates,
and a checkpoint reloads.

The four-GPU FSDP2 path has its own gate: `sbatch scripts/smoke_dummy.sbatch`,
which only succeeds once `step_00000002/COMPLETE` exists.

### `report-generation-training`

```bash
cd report-generation-training
python -m pytest              # CPU unit tests
python tests/e2e_dummy.py     # CPU full chain on a fabricated 74-diagnosis dataset
```

CPU tests cover the cache contract, clinical and NLG metrics, label extraction,
generation, the MIL ablation (prefix composition, mode-mixup errors, cross-mode
checkpoint refusal in both directions), targets, the report writer, and the
training policy. GPU gates live alongside them: `gpu_e2e.py`,
`gpu_training_e2e.py`, `gpu_full_stack_e2e.py`, `gpu_overfit_probe.py`,
`gpu_ablation_probe.py`, plus the `*.sbatch` launchers.

## Continuous integration

`.github/workflows/tests.yml` runs on pushes and PRs to `main`:

- **`test`** — matrix over Python 3.10/3.11/3.12: installs
  `contrastive-pretraining/requirements.txt` plus the two editable packages, runs
  pytest with coverage, uploads `coverage.xml` to Codecov on 3.12 only.
- **`dino-test`** — Python 3.12: installs the MR-DINO requirements, clones DINOv3
  at the pinned commit `6876159a11b4df116f30f667f8c9888617df0751`, runs the
  synthetic suite with `PYTHONPATH` pointed at it.

CI is CPU-only and synthetic — it never touches the dataset. Anything GPU- or
data-dependent has to be run manually.

## Conventions

**Single source of truth for transforms.** Per-volume preprocessing lives in
`contrastive-pretraining/scripts/data.py::preprocess_nii`. If you change it, you
change the live loader, the `.npz` cache builder, and MR-DINO simultaneously —
which is the point. Existing caches become invalid; bump the manifest rather than
letting a mismatch pass.

**Manifests, not assumptions.** Anything derived writes a manifest of the config
that produced it, and consumers validate. Adding a new setting that changes
tensor content means adding it to `build_cache_manifest` *and* `CACHE_CONFIG_KEYS`.

**Fail loudly where corruption would be silent.** The established patterns:
before/after parameter hashing on checkpoint load, cross-mode checkpoint stamps,
erroring on a labels CSV that does not cover every cached subject, refusing a
non-empty generation output without `--overwrite`.

**Atomic publishes.** Caches write payload files first and the manifest last, via
atomic replace, so an interrupted run leaves the previous cache internally
consistent rather than a mixed state.

**Numerical hardening in the loss path.** float32 for logits, `nan_to_num` on
inputs, clamped logits, `torch.where` instead of multiplying by a mask. Keep it
that way — the masked-softmax-over-thousands-of-tokens path is genuinely fragile
in bf16.

**Argument names propagate.** `--encoder`, `--fusion_mode`, `--pooling_strategy`,
`--dim_latent`, `--space`, `--normalizer` must mean the same thing in every
script that touches a checkpoint. If you add an architectural flag, add it to
`extract_features.py`, `inference.py`, and both MIL probes too, or checkpoints
silently load wrong.

## Licensing

- Code (this repository, including preprocessing) — **Apache License 2.0**.
- Datasets and model weights — **CC BY-NC-SA 4.0**: non-commercial research use with attribution and share-alike.

Commercial use of data or weights: `contact@forithmus.com`.

Anything you add under `data-preprocessing/` or the training submodules inherits
Apache-2.0. Do not commit dataset content, PHI, or model weights to the
repository — the dataset is distributed through the gated Hugging Face
repositories.
