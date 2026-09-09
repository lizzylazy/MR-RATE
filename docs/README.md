# MR-RATE Documentation

This directory is the cross-cutting documentation for the MR-RATE repository: how
the four submodules fit together, what every artifact on disk means, and how to
get from raw clinical data to an evaluated report-generation model.

Each submodule keeps its own README as the authoritative reference for *its own*
flags and scripts. These documents sit above them and explain the system.

## Where to start

| If you want to… | Read |
|---|---|
| Understand what the project is and how the pieces connect | [Architecture](architecture.md) |
| Install everything and run something for the first time | [Getting started](getting-started.md) |
| Run the complete chain: data → encoder → probes → reports | [End-to-end workflow](end-to-end-workflow.md) |
| Know exactly what a file on disk contains | [Data formats and artifacts](data-formats.md) |
| Look up a term (`study_uid`, center modality, bag, space) | [Glossary](glossary.md) |
| Work on the code: tests, CI, layout, conventions | [Development](development.md) |
| Fix an error you just hit | [Troubleshooting](troubleshooting.md) |

## Submodule READMEs

| Submodule | README | Scope |
|---|---|---|
| `data-preprocessing/` | [README](../data-preprocessing/README.md) · [Dataset guide](../data-preprocessing/docs/dataset_guide.md) | DICOM → NIfTI, defacing, registration, report NLP, labels, Hugging Face up/download |
| `contrastive-pretraining/` | [README](../contrastive-pretraining/README.md) | VL-CABS contrastive training, zero-shot inference, linear/MIL probes |
| `dino-pretraining/` | [README](../dino-pretraining/README.md) | Self-supervised 3-D DINOv3 on atlas-space MRI |
| `report-generation-training/` | [README](../report-generation-training/README.md) | Frozen encoder + MIL + Gemma LoRA report writer, generation, evaluation |

Additional focused references inside the tree:

- [Report preprocessing pipeline](../data-preprocessing/src/mr_rate_preprocessing/reports_preprocessing/README.md) — the six LLM stages and their QC loops.
- [NeuroVFM diagnosis extraction](../data-preprocessing/src/mr_rate_preprocessing/reports_preprocessing/07_neurovfm_diagnosis_extraction/README.md) — the 74-diagnosis label schema.
- [Merged evaluation labels](../contrastive-pretraining/scripts/eval_labels/splits_merged_majority/README.md) — the 14-group probe label set.

## Conventions used in these documents

- Commands are written to be run from the submodule directory named in the prompt or the surrounding text.
- Placeholder paths are `/path/to/...`; anything else is a real repository path.
- Tensor shapes are written as `[B, R, C, D, H, W]` with the axis meanings defined in [Data formats](data-formats.md#volume-axis-convention).
