# FMFReg

本仓库提供论文**《大模型结构感知特征驱动的非刚性医学图像配准方法》**的 FMFReg 实现，论文发表于**《自动化学报》（中文版）**。

**英文题目：** Non-rigid Medical Image Registration Method Driven by Structure-aware Features from Foundation Models

代码面向三维医学图像配准，支持 ACDC 心脏 MRI 和 AbdomenReg 腹部 CT 两套实验流程。

DINOv3 在本流程中作为预训练特征提取器使用。配准训练读取提前保存的 `.npy` 特征，不在训练过程中重新提取或更新 DINOv3。数据集、DINOv3 权重和训练好的配准权重不随本仓库分发。

## 目录

- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [下载 DINOv3 官方权重](#下载-dinov3-官方权重)
- [第一步：提取特征](#第一步提取特征)
- [第二步：训练与训练内验证](#第二步训练与训练内验证)
- [第三步：独立验证集评估](#第三步独立验证集评估)
- [第四步：测试集评估](#第四步测试集评估)
- [参数与路径说明](#参数与路径说明)
- [已完成的运行检查](#已完成的运行检查)
- [论文引用](#论文引用)
- [致谢](#致谢)
- [许可证](#许可证)

## 环境安装

已验证环境为 Python 3.11、PyTorch 2.6.0、torchvision 0.21.0、Transformers 4.57.1。

```bash
conda create -n fmfreg python=3.11 -y
conda activate fmfreg
python -m pip install -r requirements.txt
```

## 数据准备

使用已经整理和预处理好的三维 NIfTI 图像及对应分割标签。以下目录是代码期望的输入格式，不能直接用未经整理的原始下载目录替代。原始数据需按数据集许可自行获取；上游 [EOIR](https://github.com/XiangChen1994/EOIR#datasets) 提供了其数据集来源说明。

### ACDC

图像与标签的空间尺寸为 `(128, 128, 16)`。同一受试者的 ED、ES 图像及标签应具有一致的空间坐标和标签定义；类别编号为 `0–3`，其中 `0` 为背景。

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
└── features/                      # 下一步生成，可放在其他目录
    ├── patient001_ed_img.npy
    ├── patient001_es_img.npy
    └── ...
```

### AbdomenReg

当前实验输入图像与标签尺寸为 `(192, 160, 256)`。DataLoader 会将图像和标签下采样到 `(96, 80, 128)`。标签编号为 `0–13`，其中 `0` 为背景。

```text
data/abdomenreg/
├── img/
│   ├── img0001.nii.gz
│   └── ...
├── label/
│   ├── label0001.nii.gz
│   └── ...
├── features/                      # 下一步生成，可放在其他目录
│   ├── img0001.npy
│   └── ...
└── cache/                         # DataLoader 首次读取时自动生成
```

## 下载 DINOv3 官方权重

本项目使用 **DINOv3 ViT-B/16，LVD-1689M 预训练权重**，也可以适配其他大模型。请从官方入口获取，并按上游要求申请访问权限、接受许可：

- [DINOv3 官方项目](https://github.com/facebookresearch/dinov3)
- [Meta 官网权重申请与下载入口](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
- [Meta 官方 Hugging Face 模型：facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)

**当前特征提取脚本通过 Transformers 的 `AutoModel.from_pretrained()` 加载模型，需要 Hugging Face 格式的本地模型目录。

本地目录应至少包含模型配置和权重，例如：

```text
pretrained/dinov3-vitb16-pretrain-lvd1689m/
├── config.json
├── model.safetensors
└── preprocessor_config.json
```

已经下载到其他位置时，无需重复下载，后续将 `--model` 指向该本地目录即可。

## 第一步：提取特征

先在项目根目录设置本机路径。以下采用可迁移的相对路径示例；数据和权重放在项目外时，将变量值改为相应绝对路径即可。后续命令在同一终端执行。

```bash
DATA_ROOT=./data
DINO_MODEL=./pretrained/dinov3-vitb16-pretrain-lvd1689m
ACDC_FEATURES="$DATA_ROOT/acdcreg/features"
ABDOMEN_FEATURES="$DATA_ROOT/abdomenreg/features"
```

`DATA_ROOT` 是同时包含 `acdcreg/` 和 `abdomenreg/` 的父目录。特征可以放在独立磁盘，不必和原图放在一起。

### ACDC 全量提取

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

### AbdomenReg 全量提取

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

输出为 `float32` 的 `.npy` 文件。当前输入尺寸与设置下，形状如下：

| 数据集 | 文件示例 | 特征形状 |
| --- | --- | --- |
| ACDC | `patient101_ed_img.npy`、`patient101_es_img.npy` | `(1, 768, 48, 48, 16)` |
| AbdomenReg | `img0024.npy` | `(1, 768, 48, 40, 256)` |

## 第二步：训练与训练内验证

特征准备完成后，运行对应训练入口。以下均为**完整训练**，使用 301 个 epoch、batch size 1、学习率 `1e-4` 和 32 个起始通道。

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

两个训练脚本都会在**每个 epoch 结束后自动评估验证集**，输出 `init DSC`、`eval DSC` 和最佳验证分数。

权重默认写入：

```text
logs/acdcreg/dinov3IR_acdc/
logs/abdomenreg/dinov3IR_abdomen/
```

新实验可使用 `--model dinov3IR_acdc_run1` 或 `--model dinov3IR_abdomen_run1`，保留模型名前缀并用后缀区分输出目录；后续使用 `--load_ckpt best` 时需保持同一个 `--model`。

腹部 DataLoader 会将预处理后的图像、标签及特征写入 `<DATA_ROOT>/abdomenreg/cache/`。缓存路径在 `loaders/abdomenreg_loader.py` 中管理。切换特征或预处理设置后，应清理对应旧缓存或在该 DataLoader 中改用新的缓存目录，确保读取新特征。

## 第三步：独立验证集评估

训练内的 `eval DSC` 用于选择权重。若希望对选好的权重进一步计算验证集上的 HD95、形变场指标和 CSV，可使用测试入口并指定 `--field_split val`。

### ACDC 验证集

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

### AbdomenReg 验证集

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


## 第四步：测试集评估

根据验证集选择权重后，在测试集进行最终评估。以下命令加载相应实验目录中验证分数最高的权重，不根据测试集选择模型。

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

ACDC 会执行 ED、ES 之间的两个配准方向，并汇总每位受试者的结果。

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

结果文件为：

```text
logs/acdcreg/results_dinov3IR_acdc.csv
logs/abdomenreg/results_dinov3IR_abdomen.csv
```

`--is_save 0` 与 `--save 0` 分别关闭 ACDC、腹部的额外图像或形变场保存，不影响结果 CSV。需要额外输出时可改为 `1`。

## 参数与路径说明

| 参数或配置 | 作用 |
| --- | --- |
| 特征提取的 `--model` | DINOv3 本地模型目录或官方模型 ID |
| 训练 / 测试的 `--model` | 选择配准模型，并命名实验输出目录；保留 `dinov3IR_acdc` 或 `dinov3IR_abdomen` 前缀 |
| `--datasets_path` | 包含 `acdcreg/`、`abdomenreg/` 的父目录，不是某个 `img/` 子目录 |
| `--features_path` | 对应数据集的 `.npy` 特征目录 |
| `--load_ckpt best` | 在 `logs/<dataset>/<model>/` 中查找验证分数最高的权重 |
| `--load_ckpt /absolute/path/to/checkpoint.pth` | 直接加载指定配准权重文件，不再按 `--model` 重新拼接文件路径 |
| `logs_path` | 在入口主函数的 `opt` 中配置，默认 `./logs` |


`--model` 决定模型结构，`--load_ckpt` 决定加载哪个权重。指定完整文件路径时，权重文件夹不必与 `--model` 同名，但权重结构必须与选中的模型及构造参数一致。
## 致谢

本项目在以下开源代码、预训练模型及工具的基础上完成。我们感谢所有上游作者和维护者的贡献：

- [EOIR — Encoder-Only Image Registration](https://github.com/XiangChen1994/EOIR)：本项目配准代码的重要基础。
- [DINOv3](https://github.com/facebookresearch/dinov3)：预训练视觉模型及特征提取基础。
- [VoxelMorph](https://github.com/voxelmorph/voxelmorph)：空间变换与形变场相关基础模块。
- [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)：配准相关实现与评估代码参考。
- [surface-distance](https://github.com/google-deepmind/surface-distance)：表面距离与 HD95 相关评估工具。
- [PyTorch](https://pytorch.org/) 和 [Hugging Face Transformers](https://github.com/huggingface/transformers)：训练、推理与 DINOv3 模型加载工具。

同时感谢 ACDC、腹部 CT 数据集及其整理工作的贡献者。引用和使用这些项目时，请遵循相应上游仓库的说明。

