#!/usr/bin/env python3
"""Extract slice-wise DINOv3 features for FMFReg ACDC/AbdomenReg datasets."""

from __future__ import annotations

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tempfile
import time
from pathlib import Path
from typing import Iterable


ACDC_SPLITS = {
    "train": list(range(1, 18)) + list(range(21, 38)) + list(range(41, 58))
    + list(range(61, 78)) + list(range(81, 98)),
    "val": list(range(18, 21)) + list(range(38, 41)) + list(range(58, 61))
    + list(range(78, 81)) + list(range(98, 101)),
    "test": list(range(101, 151)),
}
ABDOMEN_SPLITS = {
    "train": list(range(1, 21)),
    "val": list(range(21, 24)),
    "test": list(range(24, 31)),
}
DEFAULT_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_LAYERS = {"acdc": 5, "abdomen": 5}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 ViT patch features consumed by FMFReg."
    )
    parser.add_argument("--dataset", required=True, choices=("acdc", "abdomen"))
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Dataset root: acdcreg/ or abdomenreg/ (a direct abdomen img/ folder is also accepted).",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model ID or a local model directory.",
    )
    parser.add_argument("--split", default="all", choices=("train", "val", "test", "all"))
    parser.add_argument(
        "--subjects",
        help="Optional comma-separated IDs/ranges, for example 24 or 24-30,101.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=None,
        help="Zero-based transformer block index. Defaults: ACDC=5, abdomen=5.",
    )
    parser.add_argument("--image-size", type=int, default=768, help="Target slice height.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Disable model downloads; useful with an existing local model directory/cache.",
    )
    return parser


def parse_subjects(spec: str) -> list[int]:
    result: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending subject range: {item}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(item))
    if not result:
        raise ValueError("--subjects did not contain any IDs")
    return sorted(set(result))


def selected_subjects(dataset: str, split: str, spec: str | None) -> list[int]:
    if spec:
        return parse_subjects(spec)
    split_map = ACDC_SPLITS if dataset == "acdc" else ABDOMEN_SPLITS
    if split == "all":
        return sorted({subject for values in split_map.values() for subject in values})
    return split_map[split]



def input_output_pairs(
    dataset: str, input_dir: Path, output_dir: Path, split: str, subjects: Iterable[int]
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    if dataset == "abdomen":
        image_dir = input_dir if input_dir.name == "img" else input_dir / "img"
        for subject in subjects:
            stem = f"img{subject:04d}"
            pairs.append((image_dir / f"{stem}.nii.gz", output_dir / f"{stem}.npy"))
        return pairs

    for subject in subjects:
        data_split = "test" if subject >= 101 else "train"
        if split == "test":
            data_split = "test"
        base = input_dir / data_split
        stem = f"patient{subject:03d}"
        pairs.extend(
            (
                (base / f"{stem}_ed_img.nii.gz", output_dir / f"{stem}_ed_img.npy"),
                (base / f"{stem}_es_img.nii.gz", output_dir / f"{stem}_es_img.npy"),
            )
        )
    return pairs


def resize_slice(image, image_size: int, patch_size: int, to_pil, tf):
    pil_image = to_pil(image)
    width, height = pil_image.size
    height_patches = image_size // patch_size
    width_patches = int((width * image_size) / (height * patch_size))
    if height_patches < 1 or width_patches < 1:
        raise ValueError(f"Invalid resized patch grid {height_patches}x{width_patches}")
    from PIL import Image

    resized = pil_image.resize(
        (width_patches * patch_size, height_patches * patch_size),
        Image.Resampling.LANCZOS,
    )
    return tf.to_tensor(resized)


def normalize_intermediate(model, hidden_state, torch):
    registers = int(getattr(model.config, "num_register_tokens", 0))
    if (
        getattr(model.config, "untie_cls_and_patch_norms", False)
        and getattr(model, "cls_norm", None) is not None
        and getattr(model, "norm", None) is not None
    ):
        cls_reg = model.cls_norm(hidden_state[:, : registers + 1])
        patches = model.norm(hidden_state[:, registers + 1 :])
        return torch.cat((cls_reg, patches), dim=1)
    norm = getattr(model, "norm", None)
    return norm(hidden_state) if norm is not None else hidden_state


def extract_volume(volume, model, layer_index: int, image_size: int, torch, tf, to_pil):
    patch_size = model.config.patch_size
    if isinstance(patch_size, (tuple, list)):
        if patch_size[0] != patch_size[1]:
            raise ValueError(f"Non-square patch size is unsupported: {patch_size}")
        patch_size = patch_size[0]
    patch_size = int(patch_size)
    num_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if not 0 <= layer_index < num_layers:
        raise ValueError(f"Layer {layer_index} is outside model range 0..{num_layers - 1}")

    slice_features = []
    with torch.inference_mode():
        for depth in range(volume.shape[2]):
            image = volume[:, :, depth].unsqueeze(0).repeat(3, 1, 1)
            image = resize_slice(image, image_size, patch_size, to_pil, tf)
            image = tf.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            image = image.unsqueeze(0).cuda(non_blocking=False)
            outputs = model(
                pixel_values=image,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = normalize_intermediate(model, outputs.hidden_states[layer_index + 1], torch)
            registers = int(getattr(model.config, "num_register_tokens", 0))
            patches = hidden[:, 1 + registers :, :]
            grid_h = image.shape[-2] // patch_size
            grid_w = image.shape[-1] // patch_size
            if patches.shape[1] != grid_h * grid_w:
                raise RuntimeError(
                    f"Patch-token mismatch: got {patches.shape[1]}, expected {grid_h * grid_w}"
                )
            feature = patches.permute(0, 2, 1).reshape(
                patches.shape[0], patches.shape[2], grid_h, grid_w
            )
            slice_features.append(feature.cpu())
    return torch.stack(slice_features, dim=-1).numpy().astype("float32", copy=False)


def load_volume(path: Path, dataset: str, nib, np, torch):
    data = np.asarray(nib.load(path).get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got {data.shape}: {path}")
    if dataset == "abdomen":
        data = np.clip(data, -500.0, 800.0)
        data = (data + 500.0) / 1300.0
    else:
        minimum, maximum = float(data.min()), float(data.max())
        if maximum <= minimum:
            raise ValueError(f"Cannot min-max normalize a constant volume: {path}")
        data = (data - minimum) / (maximum - minimum)
    return torch.from_numpy(np.ascontiguousarray(data))


def atomic_save(array, destination: Path, np) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, destination)


def main() -> int:
    args = build_parser().parse_args()
    import nibabel as nib
    import numpy as np
    import torch
    import torchvision.transforms.functional as tf
    from torchvision.transforms import ToPILImage
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. DINOv3 extraction requires an NVIDIA GPU.")

    subjects = selected_subjects(args.dataset, args.split, args.subjects)
    pairs = input_output_pairs(args.dataset, args.input_dir, args.output_dir, args.split, subjects)
    missing = [str(source) for source, _ in pairs if not source.is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        suffix = f"\n... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(f"Missing {len(missing)} input volume(s):\n{preview}{suffix}")

    pending = [(source, output) for source, output in pairs if args.overwrite or not output.exists()]
    if not pending:
        print(f"All {len(pairs)} feature files already exist; nothing to do.")
        return 0

    layer_index = DEFAULT_LAYERS[args.dataset] if args.layer_index is None else args.layer_index
    print(
        f"dataset={args.dataset} gpu=0 layer={layer_index} "
        f"inputs={len(pairs)} pending={len(pending)} model={args.model}"
    )
    model = AutoModel.from_pretrained(args.model, local_files_only=args.local_files_only)
    model.eval().cuda()
    to_pil = ToPILImage()

    for index, (source, output) in enumerate(pending, start=1):
        started = time.monotonic()
        volume = load_volume(source, args.dataset, nib, np, torch)
        features = extract_volume(
            volume, model, layer_index, args.image_size, torch, tf, to_pil
        )
        if not np.isfinite(features).all():
            raise FloatingPointError(f"Non-finite feature values produced for {source}")
        atomic_save(features, output, np)
        print(
            f"[{index}/{len(pending)}] {source.name} -> {output.name} "
            f"shape={features.shape} time={time.monotonic() - started:.1f}s"
        )
    print(f"Completed {len(pending)} extraction(s) in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
