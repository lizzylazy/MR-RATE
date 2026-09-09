# **MR-RATE: A Vision-Language Foundation Model and Dataset for Magnetic Resonance Imaging**

Welcome to the official repository for MR-RATE, a pioneering vision-language model and 3D medical imaging dataset for Magnetic Resonance Imaging.

**Resources:**

- Paper: *coming soon*
- Model Weights: *coming soon*
- Dataset: **[MR-RATE on Hugging Face](https://huggingface.co/datasets/Forithmus/MR-RATE)**
- Related Foundation Models: **[NV-Generate-MR-Brain](https://huggingface.co/nvidia/NV-Generate-MR-Brain)** by NVIDIA

## Overview

**MR-RATE** is a unified framework for **vision-language modeling in brain and spine MRI**, comprising a large-scale 3D medical imaging dataset that uniquely pairs textual data with brain and spine MRI volumes and a contrastive pretraining pipeline that aligns multi-sequence MRI volumes with radiology reports using VL-CABS loss.

## Repository Structure

```
MR-RATE/
│
├── data-preprocessing/         # Data preprocessing pipeline and dataset download scripts
│
├── contrastive-pretraining/    # Contrastive pretraining code for vision-language modeling
│
├── dino-pretraining/           # Self-supervised native 3-D DINOv3 on coregistered MRI
│
├── report-generation-training/ # Frozen-encoder report generation training, inference, evaluation
│
├── docs/                       # Cross-cutting documentation (architecture, workflow, formats)
│
└── README.md
```

Each folder includes its own `README.md` detailing configuration, dependencies, and usage.

## Documentation

Start with the [documentation index](docs/README.md), which maps the whole project:

| Document | Covers |
| --- | --- |
| [Architecture](docs/architecture.md) | How the four submodules fit together, the models, the losses, the label schemas |
| [Getting started](docs/getting-started.md) | Environments, install, verification, first run |
| [End-to-end workflow](docs/end-to-end-workflow.md) | The full chain: dataset → encoder → probes → report writer → evaluation |
| [Data formats](docs/data-formats.md) | Every file, tensor shape, cache, manifest, and checkpoint |
| [Glossary](docs/glossary.md) | Terminology (`study_uid`, center modality, space, bag, VL-CABS, …) |
| [Development](docs/development.md) | Layout, tests, CI, conventions, licensing |
| [Troubleshooting](docs/troubleshooting.md) | Common errors and what they actually mean |

## Components


| Component                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Preprocessing**      | End-to-end pipeline converting raw DICOM exports into clean, anonymized, and spatially standardized NIfTI volumes. Covers DICOM-to-NIfTI conversion, PACS metadata filtering, modality classification, binary brain segmentation & defacing, co-registration to a shared T1w reference, atlas registration to MNI152 space, and multi-label anatomical segmentations. Also includes the radiology report preprocessing pipeline (anonymization, translation, structuring, QC) and LLM-based pathology classification producing binary labels for 37 SNOMED CT-grounded pathologies. Includes standalone scripts for downloading and merging all MR-RATE Hugging Face repositories. |
| **Contrastive Pretraining** | Contrastive vision-language model that aligns multi-sequence MRI volumes and radiology reports using VL-CABS loss. Uses a VJEPA2 (ViT-G) image encoder with LoRA fine-tuning and a BiomedVLP-CXR-BERT text encoder. Supports four multi-volume fusion modes (`early`, `mid_cnn`, `late`, `late_attn`) and three image spaces (`native_space`, `coreg_space`, `atlas_space`). Enables zero-shot brain MRI pathology classification at inference time. |
| **3-D DINOv3 Pretraining** | Self-supervised DINO+iBOT+KoLeo training on coregistered multi-sequence MRI, with physical 3-D RoPE, exact per-sequence coverage, study-balanced weighting, FSDP2/FP8 production launchers, and synthetic end-to-end tests. |
| **Report Generation Training** | Single-writer radiology report generation on top of a frozen MR-RATE encoder. Projected visual tokens feed a trainable query resampler (512 language-prefix tokens) and a frozen 74-class Classify-Then-Aggregate MIL head whose class probabilities condition a Gemma LoRA writer trained on the natural `findings` text. Supports equivalent `online` and `cached` execution modes, provenance-checked artifacts, sharded/resumable generation, a no-classification-conditioning ablation, and evaluation covering both clinical accuracy (74-diagnosis re-extraction) and NLG metrics (BLEU, ROUGE, METEOR). |

## Workflow Summary

1. **Download preprocessed data** directly from [MR-RATE on Hugging Face](https://huggingface.co/datasets/Forithmus/MR-RATE) (see `data-preprocessing/` for preprocessing details)
2. **Train the contrastive model** on (multi-sequence MRI, radiology report) pairs using `contrastive-pretraining/` — or, on the label-free branch, **pretrain self-supervised** on atlas-space MRI using `dino-pretraining/`
3. **Run zero-shot inference** for brain MRI pathology classification using trained model weights and `contrastive-pretraining/`, and/or evaluate the frozen representation with the linear and MIL probes
4. **Train and evaluate a report generator** on top of the frozen encoder using `report-generation-training/`

The full command-level walkthrough is in [End-to-end workflow](docs/end-to-end-workflow.md).

## Related Foundation Models

The MR-RATE dataset was used by NVIDIA to train [NV-Generate-MR-Brain](https://huggingface.co/nvidia/NV-Generate-MR-Brain), a 3D latent diffusion model capable of generating high-quality, high-resolution synthetic brain MRI volumes for T1, Fluid Attenuated Inversion Recovery (FLAIR), T2, and Susceptibility Weighted Imaging (SWI). This model is available under the permissive NVIDIA Open Model License, enabling researchers and developers to adapt it for their own downstream medical AI applications.

## Citation

If you use this repository, the dataset, or any of its components, please cite:

```
Coming soon
```

## License

We are committed to fostering innovation and collaboration in the research community.

- **Data preprocessing code and the MR-RATE codebase** are released under the **Apache License 2.0**, allowing broad use, modification, and distribution in both academic and commercial settings, subject to the terms of the license.
- **All datasets and model weights** are released under the **[Creative Commons Attribution–NonCommercial–ShareAlike (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)** license.

This means that the data and model weights may be freely used, modified, and shared for **non-commercial research purposes**, provided that appropriate attribution is given and any derivative works are distributed under the same license.

For commercial use of the datasets or model weights, please contact: **contact@forithmus.com**

## Acknowledgements

This project is conducted by Forithmus and the University of Zurich, in collaboration with NVIDIA and Istanbul Medipol University. 

We are grateful to NVIDIA for their support, which made this work possible. We also sincerely thank Istanbul Medipol University for their support and for providing the data used in this project. High-performance computing resources were provided by NVIDIA and the University of Zurich ScienceCluster. We would also like to thank the following individuals from NVIDIA for their contributions to the development of MR-RATE: Marc Edgar, Daguang Xu, Dong Yang, Yucheng Tang, Can Zhao, Andriy Myronenko, and Pengfei Guo. 

This collaboration represents an important step toward the long-term mission to make high-quality medical intelligence accessible worldwide.
