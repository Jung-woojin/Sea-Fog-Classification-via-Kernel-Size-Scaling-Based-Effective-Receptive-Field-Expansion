import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models_erf import load_pretrained_for_finetune


# ============================================================
# Config
# ============================================================

CLASS_NAMES = ["normal", "lowvis", "seafog"]

PRETRAIN_ROOT = "/data1/wj/seafog/pretrain_ckpt"
RESULT_ROOT = "/data1/wj/seafog/results/erf"
DATA_CSV = "/data1/wj/seafog/data/splits.csv"
OUTPUT_ROOT = "/data1/wj/seafog/results/gradcam_erf_models"

IMG_SIZE = 512

# ConvNeXt는 base 자체가 large-kernel / 확장 성격이므로 typeA_3 -> base 비교.
ANALYSIS_PAIRS = {
    "convnext": ("typeA_3", "base"),
    "efficientnet": ("base", "typeA_7"),
    "mobilenet": ("base", "typeA_11"),
    "xception": ("base", "typeB_7"),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# Utilities
# ============================================================

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s[:180]


def parse_label_name(x) -> str:
    s = str(x).strip().lower()

    if s in ["0", "0.0", "normal", "clear", "보통시정"]:
        return "normal"
    if s in ["1", "1.0", "lowvis", "low_visibility", "reduced", "저시정"]:
        return "lowvis"
    if s in ["2", "2.0", "seafog", "sea_fog", "fog", "해무"]:
        return "seafog"

    raise ValueError(f"Unknown class_label: {x}")


def get_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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


# ============================================================
# Model loading
# ============================================================

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


# ============================================================
# Step 1: sample manifest
# ============================================================

def read_test_rows(port: str) -> List[Dict]:
    rows = []

    with open(DATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("port") != port:
                continue
            if row.get("split") != "test":
                continue

            path = row.get("filepath") or row.get("path") or row.get("image_path")
            if path is None:
                raise ValueError(f"No filepath/path/image_path column found: {row}")

            label_raw = row.get("class_label") or row.get("label") or row.get("class")
            if label_raw is None:
                raise ValueError(f"No class_label/label/class column found: {row}")

            label_name = parse_label_name(label_raw)

            rows.append({
                "port": port,
                "filepath": str(path),
                "class_label": label_name,
                "class_idx": CLASS_NAMES.index(label_name),
            })

    return rows


def make_manifest_for_port(port: str, num_per_class: int, seed: int, out_dir: Path) -> Path:
    rng = random.Random(seed)
    rows = read_test_rows(port)

    selected = []

    for cls in CLASS_NAMES:
        cls_rows = [r for r in rows if r["class_label"] == cls]
        rng.shuffle(cls_rows)
        chosen = cls_rows[:num_per_class]

        if len(chosen) < num_per_class:
            print(f"[WARN] port={port}, class={cls}: requested {num_per_class}, got {len(chosen)}")

        for i, r in enumerate(chosen):
            rr = dict(r)
            rr["class_order"] = i
            selected.append(rr)

    selected = sorted(selected, key=lambda r: (r["class_idx"], r["class_order"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gradcam_manifest_{port}.csv"

    pd.DataFrame(selected).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote manifest: {out_path} rows={len(selected)}")

    return out_path


def run_step1(args) -> None:
    out_dir = Path(OUTPUT_ROOT) / "step1_manifest"
    out_dir.mkdir(parents=True, exist_ok=True)

    for port in args.port:
        make_manifest_for_port(
            port=port,
            num_per_class=args.num_per_class,
            seed=args.seed,
            out_dir=out_dir,
        )


def get_manifest(port: str, args) -> pd.DataFrame:
    path = Path(OUTPUT_ROOT) / "step1_manifest" / f"gradcam_manifest_{port}.csv"

    if not path.exists() or args.rebuild_manifest:
        print(f"[INFO] creating manifest: {path}")
        make_manifest_for_port(
            port=port,
            num_per_class=args.num_per_class,
            seed=args.seed,
            out_dir=path.parent,
        )

    return pd.read_csv(path)


def select_viz_subset(df: pd.DataFrame, viz_per_class: int) -> pd.DataFrame:
    selected = []
    for cls in CLASS_NAMES:
        sub = df[df["class_label"] == cls].copy()
        sub = sub.sort_values("class_order").head(viz_per_class)
        selected.append(sub)
    return pd.concat(selected, ignore_index=True)


# ============================================================
# Grad-CAM core
# ============================================================

class GradCAM:
    """
    Safe generic Grad-CAM.

    기존 오류 원인:
        여러 leaf module에 hook을 걸면서 activation과 gradient가
        서로 다른 layer에서 잡혀 channel mismatch가 발생할 수 있음.

    수정 방식:
        1. forward 중 모든 4D spatial feature를 candidates에 저장
        2. forward 완료 후 마지막 spatial feature 하나만 선택
        3. 그 tensor에 retain_grad() 적용
        4. backward 후 선택된 tensor의 .grad만 사용
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.candidates = []
        self.feature_name = None
        self.feature_shape = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def make_forward_hook(name):
            def forward_hook(module, inputs, output):
                t = output
                if isinstance(t, (tuple, list)):
                    if len(t) == 0:
                        return
                    t = t[0]

                if torch.is_tensor(t) and t.ndim == 4:
                    if t.shape[-2] > 1 and t.shape[-1] > 1 and t.requires_grad:
                        self.candidates.append((name, t, tuple(t.shape)))

            return forward_hook

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                self.handles.append(module.register_forward_hook(make_forward_hook(name)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray, str, Tuple[int, ...]]:
        self.model.zero_grad(set_to_none=True)
        self.candidates = []
        self.feature_name = None
        self.feature_shape = None

        logits = self.model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if len(self.candidates) == 0:
            raise RuntimeError("Grad-CAM hook failed: no 4D spatial feature map was captured.")

        # 마지막 spatial feature map 하나만 선택
        self.feature_name, acts, self.feature_shape = self.candidates[-1]
        acts.retain_grad()

        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred = int(logits.argmax(dim=1).item())

        if target_class is None:
            target_class = pred

        score = logits[0, target_class]
        score.backward()

        grads = acts.grad
        if grads is None:
            raise RuntimeError(
                f"Grad-CAM failed: selected activation has no grad. "
                f"feature={self.feature_name}, shape={self.feature_shape}"
            )

        # Grad-CAM
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1,C,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=True) # [1,1,H,W]
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(x.shape[-2], x.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        cam_np = cam[0, 0].detach().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        cam_np = (cam_np - cam_min) / (cam_max - cam_min + 1e-8)

        return cam_np, pred, probs, self.feature_name or "unknown", self.feature_shape or tuple()


def overlay_cam_on_image(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    img_rgb: uint8 RGB [H,W,3]
    cam: float [H,W], 0~1
    return: uint8 RGB overlay
    """
    cam_uint8 = np.uint8(255 * np.clip(cam, 0, 1))
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = (1 - alpha) * img_rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def load_image_for_cam(path: str, img_size: int, transform) -> Tuple[np.ndarray, torch.Tensor]:
    img_pil = Image.open(path).convert("RGB")
    img_pil = img_pil.resize((img_size, img_size))
    img_rgb = np.array(img_pil).astype(np.uint8)
    x = transform(img_pil).unsqueeze(0)
    return img_rgb, x


# ============================================================
# Step 2: run Grad-CAM
# ============================================================

def run_gradcam_for_model(
    args,
    port: str,
    backbone: str,
    mode: str,
    df_manifest: pd.DataFrame,
    device: torch.device,
) -> Path:
    print("\n" + "=" * 80)
    print(f"Grad-CAM: port={port} | {backbone}_{mode} | target={args.target}")

    model = load_model(backbone, mode, port, device)
    cam_engine = GradCAM(model)
    transform = get_transform(args.img_size)

    out_root = Path(OUTPUT_ROOT) / "step2_cams" / port / backbone / mode
    out_root.mkdir(parents=True, exist_ok=True)

    records = []

    for idx, row in df_manifest.iterrows():
        filepath = str(row["filepath"])
        gt_label = str(row["class_label"])
        gt_idx = int(row["class_idx"])
        class_order = int(row["class_order"])

        img_path = Path(filepath)
        if not img_path.exists():
            print(f"  [WARN] image not found: {img_path}")
            continue

        img_rgb, x = load_image_for_cam(filepath, args.img_size, transform)
        x = x.to(device)

        target_class = None
        if args.target == "gt":
            target_class = gt_idx

        try:
            cam, pred_idx, probs, feature_name, feature_shape = cam_engine(x, target_class=target_class)
        except RuntimeError as e:
            print(f"  [ERROR] Grad-CAM failed: {filepath} | {e}")
            continue

        pred_label = CLASS_NAMES[pred_idx]
        correct = int(pred_idx == gt_idx)

        target_idx = pred_idx if args.target == "pred" else gt_idx
        target_label = CLASS_NAMES[target_idx]

        overlay = overlay_cam_on_image(img_rgb, cam, alpha=args.alpha)

        cls_dir = out_root / gt_label
        cls_dir.mkdir(parents=True, exist_ok=True)

        stem = safe_name(img_path.stem)
        out_name = f"{gt_label}_{class_order:03d}_{stem}_target-{target_label}_pred-{pred_label}.png"
        overlay_path = cls_dir / out_name

        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # 필요하면 heatmap numpy도 저장 가능
        heatmap_path = ""
        if args.save_heatmap_npy:
            heatmap_path = str(overlay_path.with_suffix(".npy"))
            np.save(heatmap_path, cam.astype(np.float32))

        rec = {
            "port": port,
            "backbone": backbone,
            "mode": mode,
            "filepath": filepath,
            "class_label": gt_label,
            "class_idx": gt_idx,
            "class_order": class_order,
            "target_type": args.target,
            "target_idx": target_idx,
            "target_label": target_label,
            "pred_idx": pred_idx,
            "pred_label": pred_label,
            "correct": correct,
            "prob_normal": float(probs[0]),
            "prob_lowvis": float(probs[1]),
            "prob_seafog": float(probs[2]),
            "feature_name": feature_name,
            "feature_shape": str(feature_shape),
            "cam_path": str(overlay_path),
            "heatmap_path": heatmap_path,
        }
        records.append(rec)

        if (len(records) % 50) == 0:
            print(f"  processed {len(records)} / {len(df_manifest)}")

        del x

    cam_engine.remove()
    del model
    torch.cuda.empty_cache()

    df_out = pd.DataFrame(records)
    csv_path = out_root / f"gradcam_records_{port}_{backbone}_{mode}.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote records: {csv_path} rows={len(df_out)}")
    return csv_path


def run_step2(args, device: torch.device) -> None:
    for port in args.port:
        df_manifest = get_manifest(port, args)

        for backbone, (mode_base, mode_ext) in ANALYSIS_PAIRS.items():
            if args.backbone and args.backbone != backbone:
                continue

            for mode in [mode_base, mode_ext]:
                try:
                    run_gradcam_for_model(
                        args=args,
                        port=port,
                        backbone=backbone,
                        mode=mode,
                        df_manifest=df_manifest,
                        device=device,
                    )
                except FileNotFoundError as e:
                    print(f"[SKIP] {port} {backbone}_{mode}: {e}")


# ============================================================
# Step 3: comparison grids
# ============================================================

def read_records(port: str, backbone: str, mode: str) -> Optional[pd.DataFrame]:
    path = Path(OUTPUT_ROOT) / "step2_cams" / port / backbone / mode / f"gradcam_records_{port}_{backbone}_{mode}.csv"
    if not path.exists():
        print(f"  [WARN] records not found: {path}")
        return None
    return pd.read_csv(path)


def make_class_grid(
    port: str,
    backbone: str,
    mode_base: str,
    mode_ext: str,
    cls: str,
    args,
    out_dir: Path,
) -> None:
    df_base = read_records(port, backbone, mode_base)
    df_ext = read_records(port, backbone, mode_ext)

    if df_base is None or df_ext is None:
        return

    base_cls = df_base[df_base["class_label"] == cls].sort_values("class_order").head(args.viz_per_class)
    ext_cls = df_ext[df_ext["class_label"] == cls].sort_values("class_order").head(args.viz_per_class)

    if base_cls.empty or ext_cls.empty:
        print(f"  [WARN] empty records: port={port}, backbone={backbone}, cls={cls}")
        return

    # class_order 기준으로 매칭
    merged = pd.merge(
        base_cls,
        ext_cls,
        on=["port", "filepath", "class_label", "class_idx", "class_order"],
        suffixes=(f"_{mode_base}", f"_{mode_ext}"),
    )

    if merged.empty:
        print(f"  [WARN] merge failed: port={port}, backbone={backbone}, cls={cls}")
        return

    n = min(args.viz_per_class, len(merged))
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        row = merged.iloc[i]

        img_path = row["filepath"]
        img_pil = Image.open(img_path).convert("RGB").resize((args.img_size, args.img_size))
        img_rgb = np.array(img_pil)

        base_cam_path = row[f"cam_path_{mode_base}"]
        ext_cam_path = row[f"cam_path_{mode_ext}"]

        base_overlay = cv2.cvtColor(cv2.imread(base_cam_path), cv2.COLOR_BGR2RGB)
        ext_overlay = cv2.cvtColor(cv2.imread(ext_cam_path), cv2.COLOR_BGR2RGB)

        gt = row["class_label"]
        pred_base = row[f"pred_label_{mode_base}"]
        pred_ext = row[f"pred_label_{mode_ext}"]

        correct_base = int(row[f"correct_{mode_base}"])
        correct_ext = int(row[f"correct_{mode_ext}"])

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"Original\nGT={gt}", fontsize=9)
        axes[i, 0].axis("off")

        color_base = "green" if correct_base else "red"
        axes[i, 1].imshow(base_overlay)
        axes[i, 1].set_title(f"{mode_base}\npred={pred_base}", fontsize=9, color=color_base)
        axes[i, 1].axis("off")

        color_ext = "green" if correct_ext else "red"
        axes[i, 2].imshow(ext_overlay)
        axes[i, 2].set_title(f"{mode_ext}\npred={pred_ext}", fontsize=9, color=color_ext)
        axes[i, 2].axis("off")

    fig.suptitle(
        f"Grad-CAM comparison | {port} | {backbone} | {mode_base} vs {mode_ext} | class={cls}",
        fontsize=12,
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gradcam_grid_{port}_{backbone}_{mode_base}_vs_{mode_ext}_{cls}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)

    print(f"[OK] wrote grid: {out_path}")


def run_step3(args) -> None:
    out_dir = Path(OUTPUT_ROOT) / "step3_comparison_grids"
    out_dir.mkdir(parents=True, exist_ok=True)

    for port in args.port:
        for backbone, (mode_base, mode_ext) in ANALYSIS_PAIRS.items():
            if args.backbone and args.backbone != backbone:
                continue

            print("\n" + "=" * 80)
            print(f"Making grids: port={port} | {backbone} | {mode_base} vs {mode_ext}")

            for cls in CLASS_NAMES:
                make_class_grid(
                    port=port,
                    backbone=backbone,
                    mode_base=mode_base,
                    mode_ext=mode_ext,
                    cls=cls,
                    args=args,
                    out_dir=out_dir,
                )


# ============================================================
# Optional summary
# ============================================================

def run_summary(args) -> None:
    rows = []

    for port in args.port:
        for backbone, (mode_base, mode_ext) in ANALYSIS_PAIRS.items():
            if args.backbone and args.backbone != backbone:
                continue

            for mode in [mode_base, mode_ext]:
                df = read_records(port, backbone, mode)
                if df is None or df.empty:
                    continue

                total = len(df)
                acc = df["correct"].mean()

                rec = {
                    "port": port,
                    "backbone": backbone,
                    "mode": mode,
                    "n": total,
                    "accuracy": acc,
                }

                for cls in CLASS_NAMES:
                    sub = df[df["class_label"] == cls]
                    rec[f"acc_{cls}"] = sub["correct"].mean() if len(sub) > 0 else np.nan
                    rec[f"n_{cls}"] = len(sub)

                rows.append(rec)

    out_dir = Path(OUTPUT_ROOT) / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_sum = pd.DataFrame(rows)
    out_path = out_dir / "gradcam_prediction_summary.csv"
    df_sum.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote summary: {out_path}")
    if not df_sum.empty:
        print(df_sum.to_string(index=False))


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Grad-CAM analysis for ERF experiment models.")

    p.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["step1", "step2", "step3", "summary", "all"],
        help="step1=manifest, step2=Grad-CAM, step3=comparison grids, summary=prediction summary, all=all steps",
    )

    p.add_argument(
        "--port",
        action="append",
        default=None,
        help="Port name. Can be repeated: --port daesan --port yeosu --port haeundae",
    )

    p.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["convnext", "efficientnet", "mobilenet", "xception"],
        help="Optional: run only one backbone.",
    )

    p.add_argument(
        "--num_per_class",
        type=int,
        default=50,
        help="Number of test images per class for Grad-CAM.",
    )

    p.add_argument(
        "--viz_per_class",
        type=int,
        default=5,
        help="Number of images per class for comparison grid.",
    )

    p.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Input image size.",
    )

    p.add_argument(
        "--target",
        type=str,
        default="pred",
        choices=["pred", "gt"],
        help="Grad-CAM target: pred=predicted class, gt=ground-truth class.",
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Heatmap overlay alpha.",
    )

    p.add_argument(
        "--save_heatmap_npy",
        action="store_true",
        help="Also save raw heatmap as .npy.",
    )

    p.add_argument(
        "--rebuild_manifest",
        action="store_true",
        help="Recreate step1 manifests even if they already exist.",
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

    if args.port is None:
        args.port = ["daesan", "yeosu", "haeundae"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] ports={args.port}")
    print(f"[INFO] img_size={args.img_size}, num_per_class={args.num_per_class}, viz_per_class={args.viz_per_class}")
    print(f"[INFO] target={args.target}")

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    if args.step in ["step1", "all"]:
        run_step1(args)

    if args.step in ["step2", "all"]:
        run_step2(args, device)

    if args.step in ["step3", "all"]:
        run_step3(args)

    if args.step in ["summary", "all"]:
        run_summary(args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
