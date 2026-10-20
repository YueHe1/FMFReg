# FMFReg

This repository provides the FMFReg implementation for the paper **Non-rigid Medical Image Registration Method Driven by Structure-aware Features from Foundation Models**, published in **Acta Automatica Sinica (Chinese Edition)**.

The code supports two workflows for 3D medical image registration: ACDC cardiac MRI and AbdomenReg abdominal CT.

DINOv3 serves as a pretrained feature extractor. Registration training reads precomputed `.npy` features; it does not extract features again or update DINOv3.

## Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Downloading Official DINOv3 Weights](#downloading-official-dinov3-weights)
- [Step 1: Feature Extraction](#step-1-feature-extraction)
- [Step 2: Training and Validation During Training](#step-2-training-and-validation-during-training)
- [Step 3: Standalone Validation](#step-3-standalone-validation)
- [Step 4: Test Set Evaluation](#step-4-test-set-evaluation)
- [Parameters and Paths](#parameters-and-paths)
- [Acknowledgments](#acknowledgments)

## Installation

The tested environment uses Python 3.11, PyTorch 2.6.0, torchvision 0.21.0, and Transformers 4.57.1.

```bash
conda create -n fmfreg python=3.11 -y
conda activate fmfreg
python -m pip install -r requirements.txt
```

## Data Preparation

Use organized and preprocessed 3D NIfTI images with their corresponding segmentation labels. The directory layouts below describe the input format expected by the code; raw download directories cannot be used directly without preparation. Obtain the data under the respective dataset licenses. The upstream [EOIR repository](https://github.com/XiangChen1994/EOIR#datasets) provides information about its dataset sources.

### ACDC

Images and labels have a spatial size of `(128, 128, 16)`. ED and ES images and labels from the same subject should use consistent spatial coordinates and label definitions. Class indices range from `0` to `3`, with `0` denoting background.

```text
data/acdcreg/
├── train/
│   ├── patient001_ed_img.nii.gz
│   ├── patient001_ed_seg.nii.gz
│   ├── patient001_es_img.nii.gz
│   ├── patient001_es_seg.nii.gz
│   └── ...
├── test/
│   ├── patient101_ed_img.nii.gz
│   ├── patient101_ed_seg.nii.gz
│   ├── patient101_es_img.nii.gz
│   ├── patient101_es_seg.nii.gz
│   └── ...
└── features/                      # Generated in the next step; may be stored elsewhere
    ├── patient001_ed_img.npy
    ├── patient001_es_img.npy
    └── ...
```

### AbdomenReg

The input images and labels used in the current experiments have a size of `(192, 160, 256)`. The DataLoader downsamples them to `(96, 80, 128)`. Label indices range from `0` to `13`, with `0` denoting background.

```text
data/abdomenreg/
├── img/
│   ├── img0001.nii.gz
│   └── ...
├── label/
│   ├── label0001.nii.gz
│   └── ...
├── features/                      # Generated in the next step; may be stored elsewhere
│   ├── img0001.npy
│   └── ...
└── cache/                         # Created automatically when the DataLoader first reads the data
```

## Downloading Official DINOv3 Weights

This project uses **DINOv3 ViT-B/16 pretrained on LVD-1689M** and can also be adapted to other foundation models. Obtain the weights from the official sources and follow the upstream requirements for access and license acceptance:

- [Official DINOv3 repository](https://github.com/facebookresearch/dinov3)
- [Meta's official weight access and download page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
- [Meta's official Hugging Face model: facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)

**The current feature extraction script loads the model through Transformers' `AutoModel.from_pretrained()` and requires a local model directory in Hugging Face format.**

The local directory should contain at least the model configuration and weights, for example:

```text
pretrained/dinov3-vitb16-pretrain-lvd1689m/
├── config.json
├── model.safetensors
└── preprocessor_config.json
```

If the model has already been downloaded elsewhere, set `--model` to that local directory; there is no need to download it again.

## Step 1: Feature Extraction

Set the paths for your machine from the project root. The examples below use portable relative paths. If the data or weights are stored outside the project, replace the variable values with the corresponding absolute paths. Run the subsequent commands in the same terminal.

```bash
DATA_ROOT=./data
DINO_MODEL=./pretrained/dinov3-vitb16-pretrain-lvd1689m
ACDC_FEATURES="$DATA_ROOT/acdcreg/features"
ABDOMEN_FEATURES="$DATA_ROOT/abdomenreg/features"
```

`DATA_ROOT` is the parent directory containing both `acdcreg/` and `abdomenreg/`. Features may be stored on a separate disk from the original images.

### Extract All ACDC Features

```bash
python extract_dinov3_features.py \
  --dataset acdc \
  --input-dir "$DATA_ROOT/acdcreg" \
  --output-dir "$ACDC_FEATURES" \
  --model "$DINO_MODEL" \
  --split all \
  --layer-index 5 \
  --image-size 768 \
  --local-files-only
```

### Extract All AbdomenReg Features

```bash
python extract_dinov3_features.py \
  --dataset abdomen \
  --input-dir "$DATA_ROOT/abdomenreg" \
  --output-dir "$ABDOMEN_FEATURES" \
  --model "$DINO_MODEL" \
  --split all \
  --layer-index 5 \
  --image-size 768 \
  --local-files-only
```

Features are saved as `float32` `.npy` files. With the current input sizes and settings, the output shapes are:

| Dataset | Example Files | Feature Shape |
| --- | --- | --- |
| ACDC | `patient101_ed_img.npy`, `patient101_es_img.npy` | `(1, 768, 48, 48, 16)` |
| AbdomenReg | `img0024.npy` | `(1, 768, 48, 40, 256)` |

## Step 2: Training and Validation During Training

After preparing the features, run the corresponding training entry point. The commands below run **full training** for 301 epochs with a batch size of 1, a learning rate of `1e-4`, and 32 starting channels.

### ACDC

```bash
python train_registration_ACDC.py \
  --model dinov3IR_acdc \
  --datasets_path "$DATA_ROOT" \
  --features_path "$ACDC_FEATURES" \
  --epochs 301 \
  --batch_size 1 \
  --lr 1e-4 \
  --start_channel 32 \
  --num_workers 0
```

### AbdomenReg

```bash
python train_registration_abdomen.py \
  --model dinov3IR_abdomen \
  --datasets_path "$DATA_ROOT" \
  --features_path "$ABDOMEN_FEATURES" \
  --epochs 301 \
  --batch_size 1 \
  --lr 1e-4 \
  --start_channel 32 \
  --num_workers 0
```

Both training scripts **automatically evaluate the validation set at the end of every epoch**, reporting `init DSC`, `eval DSC`, and the best validation score.

Checkpoints are saved to the following directories by default:

```text
logs/acdcreg/dinov3IR_acdc/
logs/abdomenreg/dinov3IR_abdomen/
```

For a new experiment, use a name such as `--model dinov3IR_acdc_run1` or `--model dinov3IR_abdomen_run1`. Keep the model name prefix and add a suffix to distinguish the output directory. Use the same `--model` value when subsequently loading the checkpoint with `--load_ckpt best`.

The abdomen DataLoader caches preprocessed images, labels, and features under `<DATA_ROOT>/abdomenreg/cache/`. The cache path is managed in `loaders/abdomenreg_loader.py`. When changing the features or preprocessing settings, clear the corresponding old cache or configure a new cache directory in that DataLoader to ensure that the new features are used.

## Step 3: Standalone Validation

The `eval DSC` reported during training is used to select checkpoints. To compute additional validation metrics, such as HD95 and deformation field metrics, and save a CSV file for a selected checkpoint, run the evaluation entry point with `--field_split val`.

### ACDC Validation Set

```bash
python test_cardiacreg.py \
  --model dinov3IR_acdc \
  --datasets_path "$DATA_ROOT" \
  --features_path "$ACDC_FEATURES" \
  --load_ckpt best \
  --field_split val \
  --start_channel 32 \
  --num_workers 0 \
  --is_save 0
```

### AbdomenReg Validation Set

```bash
python test_abdomenreg.py \
  --model dinov3IR_abdomen \
  --datasets_path "$DATA_ROOT" \
  --features_path "$ABDOMEN_FEATURES" \
  --load_ckpt best \
  --field_split val \
  --start_channel 32 \
  --num_workers 0 \
  --save 0
```

## Step 4: Test Set Evaluation

After selecting a checkpoint using the validation set, evaluate it on the test set. The commands below load the checkpoint with the highest validation score from the corresponding experiment directory. The test set is not used for model selection.

### ACDC

```bash
python test_cardiacreg.py \
  --model dinov3IR_acdc \
  --datasets_path "$DATA_ROOT" \
  --features_path "$ACDC_FEATURES" \
  --load_ckpt best \
  --field_split test \
  --start_channel 32 \
  --num_workers 0 \
  --is_save 0
```

For ACDC, registration is performed in both directions between ED and ES, and the results are aggregated for each subject.

### Abdomen CT

```bash
python test_abdomenreg.py \
  --model dinov3IR_abdomen \
  --datasets_path "$DATA_ROOT" \
  --features_path "$ABDOMEN_FEATURES" \
  --load_ckpt best \
  --field_split test \
  --start_channel 32 \
  --num_workers 0 \
  --save 0
```

Results are saved to:

```text
logs/acdcreg/results_dinov3IR_acdc.csv
logs/abdomenreg/results_dinov3IR_abdomen.csv
```

`--is_save 0` and `--save 0` disable additional image or deformation field outputs for ACDC and abdomen evaluation, respectively, without affecting the result CSV files. Set the corresponding option to `1` if these additional outputs are needed.

## Parameters and Paths

| Parameter or Setting | Purpose |
| --- | --- |
| `--model` for feature extraction | Local DINOv3 model directory or official model ID |
| `--model` for registration training or evaluation | Selects the registration model and names the experiment output directory; keep the `dinov3IR_acdc` or `dinov3IR_abdomen` prefix |
| `--datasets_path` | Parent directory containing `acdcreg/` and `abdomenreg/`, rather than an individual `img/` subdirectory |
| `--features_path` | Directory containing the `.npy` features for the corresponding dataset |
| `--load_ckpt best` | Finds the checkpoint with the highest validation score under `logs/<dataset>/<model>/` |
| `--load_ckpt /absolute/path/to/checkpoint.pth` | Loads the specified registration checkpoint directly, without constructing another checkpoint path from `--model` |
| `logs_path` | Configured in the entry point's `opt` dictionary; defaults to `./logs` |

`--model` determines the model architecture, while `--load_ckpt` determines which checkpoint to load. When an absolute checkpoint path is specified, its parent directory does not need to match `--model`, but its parameter structure must be compatible with the selected model and constructor arguments.

## Acknowledgments

This project builds on the following open-source code, pretrained models, and tools. We thank their authors and maintainers for their contributions:

- [EOIR — Encoder-Only Image Registration](https://github.com/XiangChen1994/EOIR): a key foundation for our registration code.
- [DINOv3](https://github.com/facebookresearch/dinov3): pretrained vision models and the basis for feature extraction.
- [VoxelMorph](https://github.com/voxelmorph/voxelmorph): spatial transformation and deformation field utilities.
- [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration): reference implementations for registration and evaluation.
- [surface-distance](https://github.com/google-deepmind/surface-distance): surface distance and HD95 evaluation utilities.
- [PyTorch](https://pytorch.org/) and [Hugging Face Transformers](https://github.com/huggingface/transformers): tools for training, inference, and loading DINOv3 models.

We also thank the contributors of the ACDC and abdominal CT datasets and the work involved in preparing them. Please follow the upstream projects' instructions when citing or using these resources.

