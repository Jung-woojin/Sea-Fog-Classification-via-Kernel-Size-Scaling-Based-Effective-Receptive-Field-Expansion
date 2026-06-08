# -*- coding: utf-8 -*-
"""
erf_analysis_true.py
True gradient-based ERF measurement for ERF/kernal-expansion experiments.

Core idea:
    Old saliency-style analysis:
        class logit -> input gradient
        = class-specific saliency / sensitivity

    True ERF-style analysis:
        random noise input
        -> last spatial feature map
        -> center spatial activation
        -> input gradient
        = feature-level gradient-based ERF

Usage:
    python erf_analysis_true.py --step erf --port daesan --backbone convnext --n_samples 30
    python erf_analysis_true.py --step delta --port daesan --n_samples 50
    python erf_analysis_true.py --step all --port daesan --n_samples 50

Multiple ports:
    python erf_analysis_true.py --step all --port daesan --port yeosu --port haeundae --n_samples 50
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from models_erf import load_pretrained_for_finetune


# =====================================================================
# Config
# =====================================================================

CLASS_NAMES = ["normal", "lowvis", "seafog"]

PRETRAIN_ROOT = "/data1/wj/seafog/pretrain_ckpt"
RESULT_ROOT = "/data1/wj/seafog/results/erf"
DATA_CSV = "/data1/wj/seafog/data/splits.csv"
OUTPUT_ROOT = "/data1/wj/seafog/results/erf_analysis_true"

IMG_SIZE = 512

ANALYSIS_PAIRS = {
    "convnext": ("typeA_3", "base"),
    "efficientnet": ("base", "typeA_7"),
    "mobilenet": ("base", "typeA_11"),
    "xception": ("base", "typeB_7"),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =====================================================================
# Utilities
# =====================================================================

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def parse_label_name(x) -> str:
    """
    Accept both string labels and numeric labels.
    """
    s = str(x).strip().lower()

    if s in ["0", "0.0", "normal", "clear", "보통시정"]:
        return "normal"
    if s in ["1", "1.0", "lowvis", "low_visibility", "reduced", "저시정"]:
        return "lowvis"
    if s in ["2", "2.0", "seafog", "sea_fog", "fog", "해무"]:
        return "seafog"

    raise ValueError(f"Unknown class_label: {x}")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove 'module.' prefix if checkpoint was saved from DataParallel/DDP.
    """
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def load_state_dict_flexible(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    sd = strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing:
        print(f"  [WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"  [WARN] unexpected keys: {len(unexpected)}")


# =====================================================================
# Model loading
# =====================================================================

def load_model(backbone: str, mode: str, port: str, device: torch.device) -> torch.nn.Module:
    pretrain_ckpt = f"{PRETRAIN_ROOT}/{backbone}_{mode}/best.pth"
    finetune_ckpt = f"{RESULT_ROOT}/{backbone}_{mode}/{port}/best.pth"

    if not Path(pretrain_ckpt).exists():
        raise FileNotFoundError(f"pretrain checkpoint not found: {pretrain_ckpt}")
    if not Path(finetune_ckpt).exists():
        raise FileNotFoundError(f"finetune checkpoint not found: {finetune_ckpt}")

    model = load_pretrained_for_finetune(
        backbone=backbone,
        mode=mode,
        pretrain_ckpt=pretrain_ckpt,
        num_classes=3,
    )

    load_state_dict_flexible(model, finetune_ckpt)

    model = model.to(device)
    model.eval()
    return model


# =====================================================================
# Dataset helper for prediction accuracy only
# =====================================================================

def get_test_images(port: str, num_per_class: int = 50, seed: int = 42) -> List[Tuple[Path, str]]:
    """
    This is used only to calculate prediction accuracy of the loaded model.
    True ERF itself is measured from random noise, not from these real images.
    """
    rng = random.Random(seed)

    samples: List[Tuple[Path, str]] = []

    with open(DATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("port") != port:
                continue
            if row.get("split") != "test":
                continue

            path = row.get("filepath") or row.get("path") or row.get("image_path")
            if path is None:
                raise ValueError(f"No filepath/path/image_path column found in CSV row: {row}")

            label_raw = row.get("class_label") or row.get("label") or row.get("class")
            if label_raw is None:
                raise ValueError(f"No class_label/label/class column found in CSV row: {row}")

            label_name = parse_label_name(label_raw)
            samples.append((Path(path), label_name))

    result: List[Tuple[Path, str]] = []
    for cls in CLASS_NAMES:
        cls_samples = [(p, l) for p, l in samples if l == cls]
        rng.shuffle(cls_samples)
        result.extend(cls_samples[:num_per_class])

    rng.shuffle(result)
    return result


def measure_pred(model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device) -> int:
    model.eval()
    with torch.no_grad():
        out = model(img_tensor.to(device))
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = out.argmax(dim=1).item()
    return pred


# =====================================================================
# True ERF core
# =====================================================================

def _register_last_spatial_feature_hook(model: torch.nn.Module):
    """
    Register forward hooks on leaf modules and capture the last 4D spatial feature map.

    Returns:
        last: mutable dict with keys {"feat", "name", "shape"}
        hooks: registered hook handles
    """
    last = {"feat": None, "name": None, "shape": None}
    hooks = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            t = output
            if isinstance(t, (tuple, list)):
                if len(t) == 0:
                    return
                t = t[0]

            if torch.is_tensor(t) and t.ndim == 4:
                # Use only real spatial maps, not [B,C,1,1].
                if t.shape[-2] > 1 and t.shape[-1] > 1:
                    last["feat"] = t
                    last["name"] = name
                    last["shape"] = tuple(t.shape)
        return hook

    for name, module in model.named_modules():
        # Hook leaf modules only.
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name)))

    return last, hooks


def _weighted_spread(grad_map: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate spatial spread of the ERF map using gradient mass as weights.

    Returns:
        sigma: sqrt(E[(x-mu_x)^2 + (y-mu_y)^2])
        r50: radius containing 50% gradient mass around weighted centroid
        r90: radius containing 90% gradient mass around weighted centroid
        mu_x, mu_y: weighted centroid
        center_offset: distance between weighted centroid and image center
    """
    g = grad_map.astype(np.float64)
    total = g.sum()

    H, W = g.shape

    if total < 1e-12:
        return 0.0, 0.0, 0.0, W / 2.0, H / 2.0, 0.0

    w = g / (total + 1e-12)

    ys, xs = np.mgrid[0:H, 0:W]

    mu_y = float((w * ys).sum())
    mu_x = float((w * xs).sum())

    dist = np.sqrt((ys - mu_y) ** 2 + (xs - mu_x) ** 2)
    sigma = float(np.sqrt((w * (dist ** 2)).sum()))

    flat_d = dist.reshape(-1)
    flat_w = w.reshape(-1)
    order = np.argsort(flat_d)
    sorted_d = flat_d[order]
    sorted_w = flat_w[order]
    cdf = np.cumsum(sorted_w)

    r50_idx = min(np.searchsorted(cdf, 0.50), len(cdf) - 1)
    r90_idx = min(np.searchsorted(cdf, 0.90), len(cdf) - 1)

    r50 = float(sorted_d[r50_idx])
    r90 = float(sorted_d[r90_idx])

    cy, cx = H / 2.0, W / 2.0
    center_offset = float(np.sqrt((mu_y - cy) ** 2 + (mu_x - cx) ** 2))

    return sigma, r50, r90, mu_x, mu_y, center_offset


def measure_erf_true(
    model: torch.nn.Module,
    device: torch.device,
    n_samples: int = 50,
    input_size: int = IMG_SIZE,
) -> Tuple[np.ndarray, float, float, float, float, float, float, str, Tuple[int, ...]]:
    """
    True gradient-based ERF measurement.

    Protocol:
        1. Random noise input in normalized image space.
        2. Forward model.
        3. Capture the last spatial feature map F [B,C,H,W].
        4. Select the center spatial unit F[0,:,H//2,W//2].
        5. Average over channels to obtain a scalar target.
        6. Backpropagate target to input.
        7. Average absolute input gradients over n_samples.

    Returns:
        erf_map_vis: normalized ERF map [H,W] for visualization
        sigma, r50, r90: ERF spread metrics
        mu_x, mu_y, center_offset
        feature_name: hooked layer/module name
        feature_shape: hooked feature shape
    """
    model.eval()

    acc_map = np.zeros((input_size, input_size), dtype=np.float64)

    last, hooks = _register_last_spatial_feature_hook(model)

    feature_name = None
    feature_shape = None

    try:
        for i in range(n_samples):
            last["feat"] = None
            last["name"] = None
            last["shape"] = None

            inp = torch.randn(
                1, 3, input_size, input_size,
                device=device,
                requires_grad=True,
            )

            model.zero_grad(set_to_none=True)

            out = model(inp)
            if isinstance(out, (tuple, list)):
                out = out[0]

            feat = last["feat"]
            if feat is None:
                raise RuntimeError("Could not find the last spatial feature map.")

            feature_name = last["name"]
            feature_shape = last["shape"]

            _, C, Hf, Wf = feat.shape
            cy, cx = Hf // 2, Wf // 2

            # Channel average is more stable than selecting a single channel.
            target = feat[0, :, cy, cx].mean()

            target.backward()

            grad = inp.grad.detach()[0].abs()        # [3,H,W]
            grad = grad.mean(dim=0).cpu().numpy()    # [H,W]

            acc_map += grad

            del inp, out, feat, target, grad

    finally:
        for h in hooks:
            h.remove()

    avg_map = acc_map / max(n_samples, 1)

    sigma, r50, r90, mu_x, mu_y, center_offset = _weighted_spread(avg_map)

    mn, mx = avg_map.min(), avg_map.max()
    erf_map_vis = (avg_map - mn) / (mx - mn + 1e-8)

    return (
        erf_map_vis,
        sigma,
        r50,
        r90,
        mu_x,
        mu_y,
        center_offset,
        feature_name or "unknown",
        feature_shape or tuple(),
    )


# =====================================================================
# Visualization
# =====================================================================

def save_erf_map(
    erf_map: np.ndarray,
    sigma: float,
    r50: float,
    r90: float,
    mu_x: float,
    mu_y: float,
    center_offset: float,
    feature_name: str,
    feature_shape: Tuple[int, ...],
    backbone: str,
    mode: str,
    port: str,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    im = axes[0].imshow(erf_map, cmap="hot", vmin=0, vmax=1)
    axes[0].scatter([mu_x], [mu_y], s=15, c="cyan", marker="x")
    axes[0].set_title(
        f"True ERF Map\n{backbone}_{mode} | {port}\n"
        f"sigma={sigma:.1f}, r90={r90:.1f}, offset={center_offset:.1f}",
        fontsize=9,
    )
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    H, W = erf_map.shape
    cy, cx = H / 2.0, W / 2.0
    ys, xs = np.mgrid[0:H, 0:W]
    dist_map = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).reshape(-1)
    erf_flat = erf_map.reshape(-1)

    max_r = int(min(H, W) / 2)
    bins = np.arange(0, max_r + 2, 2)
    means = []
    for i in range(len(bins) - 1):
        mask = (dist_map >= bins[i]) & (dist_map < bins[i + 1])
        means.append(float(erf_flat[mask].mean()) if mask.sum() > 0 else 0.0)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    axes[1].plot(bin_centers, means, color="red")
    axes[1].axvline(sigma, color="blue", linestyle="--", label=f"sigma={sigma:.1f}")
    axes[1].axvline(r50, color="green", linestyle=":", label=f"r50={r50:.1f}")
    axes[1].axvline(r90, color="orange", linestyle="-.", label=f"r90={r90:.1f}")
    axes[1].set_xlabel("Distance from input center (px)")
    axes[1].set_ylabel("Mean normalized gradient")
    axes[1].set_title("Radial ERF Profile", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"Hooked feature: {feature_name} | shape={feature_shape}",
        fontsize=9,
    )
    fig.tight_layout()

    path = output_dir / f"erf_true_{backbone}_{mode}_{port}.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)

    print(f"  ERF map saved: {path}")


def save_delta_figure(
    erf_base: np.ndarray,
    erf_ext: np.ndarray,
    sigma_base: float,
    sigma_ext: float,
    r90_base: float,
    r90_ext: float,
    backbone: str,
    mode_base: str,
    mode_ext: str,
    port: str,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(erf_base, cmap="hot", vmin=0, vmax=1)
    axes[0].set_title(f"{mode_base}\nsigma={sigma_base:.1f}, r90={r90_base:.1f}", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(erf_ext, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(f"{mode_ext}\nsigma={sigma_ext:.1f}, r90={r90_ext:.1f}", fontsize=9)
    axes[1].axis("off")

    fig.suptitle(f"True ERF Delta: {backbone} | {port}", fontsize=11)

    path = output_dir / f"erf_delta_{backbone}_{port}.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)

    print(f"  Delta figure saved: {path}")


# =====================================================================
# Step 1: ERF measurement
# =====================================================================

def run_erf_measurement(args, port: str, device: torch.device) -> None:
    transform = get_transform(IMG_SIZE)

    output_dir = Path(OUTPUT_ROOT) / "step1_erf_measurement"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for backbone, (mode_base, mode_ext) in ANALYSIS_PAIRS.items():
        if args.backbone and args.backbone != backbone:
            continue

        for mode in [mode_base, mode_ext]:
            print("\n" + "=" * 70)
            print(f"True ERF measurement: {backbone}_{mode} | port={port}")

            try:
                model = load_model(backbone, mode, port, device)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
                continue

            print(f"  Measuring true ERF... n_samples={args.n_samples}")
            (
                erf_map,
                sigma,
                r50,
                r90,
                mu_x,
                mu_y,
                center_offset,
                feature_name,
                feature_shape,
            ) = measure_erf_true(
                model=model,
                device=device,
                n_samples=args.n_samples,
                input_size=args.img_size,
            )

            print(
                f"  ERF: sigma={sigma:.2f}px | r50={r50:.2f}px | r90={r90:.2f}px | "
                f"centroid=({mu_x:.1f},{mu_y:.1f}) | offset={center_offset:.2f}px"
            )
            print(f"  Hooked feature: {feature_name} | shape={feature_shape}")

            # Prediction accuracy on real test images, separated from ERF measurement.
            samples = get_test_images(port, num_per_class=args.num_per_class, seed=args.seed)

            correct = 0
            total = 0
            class_total = {cls: 0 for cls in CLASS_NAMES}
            class_correct = {cls: 0 for cls in CLASS_NAMES}

            for img_path, gt_label in samples:
                if not img_path.exists():
                    print(f"  [WARN] image not found: {img_path}")
                    continue

                img_pil = Image.open(img_path).convert("RGB")
                img_tensor = transform(img_pil).unsqueeze(0)

                pred = measure_pred(model, img_tensor, device)
                gt_idx = CLASS_NAMES.index(gt_label)

                total += 1
                class_total[gt_label] += 1

                if pred == gt_idx:
                    correct += 1
                    class_correct[gt_label] += 1

            acc = correct / total if total > 0 else np.nan

            print(f"  Accuracy on sampled test images: {acc:.4f} ({correct}/{total})")
            for cls in CLASS_NAMES:
                if class_total[cls] > 0:
                    cls_acc = class_correct[cls] / class_total[cls]
                    print(f"    {cls}: acc={cls_acc:.4f} ({class_correct[cls]}/{class_total[cls]})")

            save_erf_map(
                erf_map=erf_map,
                sigma=sigma,
                r50=r50,
                r90=r90,
                mu_x=mu_x,
                mu_y=mu_y,
                center_offset=center_offset,
                feature_name=feature_name,
                feature_shape=feature_shape,
                backbone=backbone,
                mode=mode,
                port=port,
                output_dir=output_dir,
            )

            results.append({
                "port": port,
                "backbone": backbone,
                "mode": mode,
                "sigma": round(sigma, 4),
                "r50": round(r50, 4),
                "r90": round(r90, 4),
                "mu_x": round(mu_x, 4),
                "mu_y": round(mu_y, 4),
                "center_offset": round(center_offset, 4),
                "feature_name": feature_name,
                "feature_shape": str(feature_shape),
                "sample_accuracy": round(acc, 6) if not np.isnan(acc) else np.nan,
                "n_samples_erf": args.n_samples,
                "n_samples_accuracy": total,
            })

            del model
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    out_csv = output_dir / f"erf_true_{port}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {out_csv}")

    if not df.empty:
        print(df[[
            "backbone", "mode", "sigma", "r50", "r90",
            "center_offset", "sample_accuracy"
        ]].to_string(index=False))

    print("\n=== Base vs expanded ERF summary ===")
    for backbone, (mode_base, mode_ext) in ANALYSIS_PAIRS.items():
        base = df[(df["backbone"] == backbone) & (df["mode"] == mode_base)]
        ext = df[(df["backbone"] == backbone) & (df["mode"] == mode_ext)]

        if base.empty or ext.empty:
            continue

        b = float(base.iloc[0]["sigma"])
        e = float(ext.iloc[0]["sigma"])
        rb = float(base.iloc[0]["r90"])
        re = float(ext.iloc[0]["r90"])

        print(
            f"  {backbone:12s}: "
            f"sigma {b:.1f} -> {e:.1f} (delta={e-b:+.1f}) | "
            f"r90 {rb:.1f} -> {re:.1f} (delta={re-rb:+.1f})"
        )


# =====================================================================
# Step 2: backbone comparison
# =====================================================================

def run_backbone_comparison(args, port: str, device: torch.device) -> None:
    output_dir = Path(OUTPUT_ROOT) / "step2_backbone_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    erf_maps = {}
    rows = []

    for backbone, (mode_base, _) in ANALYSIS_PAIRS.items():
        if args.backbone and args.backbone != backbone:
            continue

        print("\n" + "=" * 70)
        print(f"Backbone comparison: {backbone}_{mode_base} | port={port}")

        try:
            model = load_model(backbone, mode_base, port, device)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        (
            erf_map,
            sigma,
            r50,
            r90,
            mu_x,
            mu_y,
            center_offset,
            feature_name,
            feature_shape,
        ) = measure_erf_true(
            model=model,
            device=device,
            n_samples=args.n_samples,
            input_size=args.img_size,
        )

        erf_maps[backbone] = erf_map

        rows.append({
            "port": port,
            "backbone": backbone,
            "mode": mode_base,
            "sigma": round(sigma, 4),
            "r50": round(r50, 4),
            "r90": round(r90, 4),
            "center_offset": round(center_offset, 4),
            "feature_name": feature_name,
            "feature_shape": str(feature_shape),
            "n_samples_erf": args.n_samples,
        })

        print(
            f"  sigma={sigma:.2f}px | r50={r50:.2f}px | r90={r90:.2f}px | "
            f"offset={center_offset:.2f}px"
        )
        print(f"  Hooked feature: {feature_name} | shape={feature_shape}")

        del model
        torch.cuda.empty_cache()

    if erf_maps:
        n = len(erf_maps)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (bb, emap) in zip(axes, erf_maps.items()):
            row = next(r for r in rows if r["backbone"] == bb)
            ax.imshow(emap, cmap="hot", vmin=0, vmax=1)
            ax.set_title(
                f"{bb}\nsigma={row['sigma']:.1f}, r90={row['r90']:.1f}",
                fontsize=9,
            )
            ax.axis("off")

        fig.suptitle(f"True ERF Comparison: base models | {port}", fontsize=12)
        path = output_dir / f"erf_backbone_comparison_{port}.png"
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"Saved figure: {path}")

    df = pd.DataFrame(rows)
    out_csv = output_dir / f"backbone_comparison_{port}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_csv}")
    if not df.empty:
        print(df.to_string(index=False))


# =====================================================================
# Step 3: base vs expanded ERF delta
# =====================================================================

def run_erf_delta(args, port: str, device: torch.device) -> None:
    output_dir = Path(OUTPUT_ROOT) / "step3_erf_delta"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for backbone, (mode_base, mode_ext) in ANALYSIS_PAIRS.items():
        if args.backbone and args.backbone != backbone:
            continue

        print("\n" + "=" * 70)
        print(f"True ERF delta: {backbone} | {mode_base} vs {mode_ext} | port={port}")

        try:
            model_base = load_model(backbone, mode_base, port, device)
            model_ext = load_model(backbone, mode_ext, port, device)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        (
            erf_base,
            sigma_base,
            r50_base,
            r90_base,
            mu_x_base,
            mu_y_base,
            offset_base,
            feature_name_base,
            feature_shape_base,
        ) = measure_erf_true(
            model=model_base,
            device=device,
            n_samples=args.n_samples,
            input_size=args.img_size,
        )

        (
            erf_ext,
            sigma_ext,
            r50_ext,
            r90_ext,
            mu_x_ext,
            mu_y_ext,
            offset_ext,
            feature_name_ext,
            feature_shape_ext,
        ) = measure_erf_true(
            model=model_ext,
            device=device,
            n_samples=args.n_samples,
            input_size=args.img_size,
        )

        delta_sigma = sigma_ext - sigma_base
        delta_r50 = r50_ext - r50_base
        delta_r90 = r90_ext - r90_base

        print(f"  sigma: {sigma_base:.2f} -> {sigma_ext:.2f} (delta={delta_sigma:+.2f})")
        print(f"  r50:   {r50_base:.2f} -> {r50_ext:.2f} (delta={delta_r50:+.2f})")
        print(f"  r90:   {r90_base:.2f} -> {r90_ext:.2f} (delta={delta_r90:+.2f})")

        save_delta_figure(
            erf_base=erf_base,
            erf_ext=erf_ext,
            sigma_base=sigma_base,
            sigma_ext=sigma_ext,
            r90_base=r90_base,
            r90_ext=r90_ext,
            backbone=backbone,
            mode_base=mode_base,
            mode_ext=mode_ext,
            port=port,
            output_dir=output_dir,
        )

        rows.append({
            "port": port,
            "backbone": backbone,
            "mode_base": mode_base,
            "mode_ext": mode_ext,
            "sigma_base": round(sigma_base, 4),
            "sigma_ext": round(sigma_ext, 4),
            "delta_sigma": round(delta_sigma, 4),
            "r50_base": round(r50_base, 4),
            "r50_ext": round(r50_ext, 4),
            "delta_r50": round(delta_r50, 4),
            "r90_base": round(r90_base, 4),
            "r90_ext": round(r90_ext, 4),
            "delta_r90": round(delta_r90, 4),
            "center_offset_base": round(offset_base, 4),
            "center_offset_ext": round(offset_ext, 4),
            "feature_name_base": feature_name_base,
            "feature_name_ext": feature_name_ext,
            "feature_shape_base": str(feature_shape_base),
            "feature_shape_ext": str(feature_shape_ext),
            "n_samples_erf": args.n_samples,
        })

        del model_base, model_ext
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_csv = output_dir / f"erf_delta_{port}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {out_csv}")
    if not df.empty:
        print(df[[
            "backbone", "mode_base", "mode_ext",
            "sigma_base", "sigma_ext", "delta_sigma",
            "r50_base", "r50_ext", "delta_r50",
            "r90_base", "r90_ext", "delta_r90",
        ]].to_string(index=False))


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="True gradient-based ERF analysis using random noise and center feature activation."
    )

    p.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["erf", "compare", "delta", "all"],
        help="erf=measure all models, compare=base backbone comparison, delta=base vs expanded, all=all steps",
    )

    p.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["convnext", "efficientnet", "mobilenet", "xception"],
        help="Optional: run only one backbone.",
    )

    p.add_argument(
        "--port",
        action="append",
        default=None,
        help="Port name. Can be repeated: --port daesan --port yeosu",
    )

    p.add_argument(
        "--num_per_class",
        type=int,
        default=50,
        help="Number of real test images per class for sampled accuracy check.",
    )

    p.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of random noise samples for true ERF averaging.",
    )

    p.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Input image size for ERF measurement.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args()


def main():
    args = parse_args()

    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] img_size={args.img_size}, n_samples={args.n_samples}")

    ports = args.port if args.port is not None else ["daesan"]

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    for port in ports:
        print("\n" + "#" * 80)
        print(f"# PORT: {port}")
        print("#" * 80)

        if args.step in ["erf", "all"]:
            run_erf_measurement(args, port, device)

        if args.step in ["compare", "all"]:
            run_backbone_comparison(args, port, device)

        if args.step in ["delta", "all"]:
            run_erf_delta(args, port, device)

    print("\nAll done.")


if __name__ == "__main__":
    main()
