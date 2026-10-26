r"""
Tyre-tread depth inference for the DINOv3 ViT-B / 336 px models (5-fold ensemble, no hand-crafted features).

Input can be
  * one video file                        python infer_tbr_dino_336.py --input D:\videos\tyre.mp4
  * a folder of videos (.mp4 .mov .avi)   python infer_tbr_dino_336.py --input D:\videos
  * a CSV with a 'filename' or 'video_path' column (and optionally 'depth_mm' -> accuracy report is printed)

For every video it prints/saves the ENSEMBLE depth (mean of the fold models), every fold's own prediction, the spread
between folds (a cheap confidence signal) and flags:
  LOW_CONFIDENCE   folds disagree by more than SPREAD_WARN_MM (std)
  OUT_OF_RANGE     the prediction falls outside the 4-24 mm range the models were trained on -> do not trust it
  NO_FRAMES        the video could not be read

Preprocessing is IDENTICAL to training: rotate landscape frames to portrait, keep the 16 sharpest of ~48 evenly spaced
frames, resize to 336x336, ImageNet normalisation, plus the per-frame quality score computed from the edge/groove map
(the temporal-attention pooling takes it as an input, so it must be computed the same way).

Needs:  pip install torch timm opencv-python numpy pandas      (+ matplotlib only for --attention-dir)
Weights are loaded from the checkpoints; nothing is downloaded.
"""

import argparse
import glob
import os
import re
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# DEFAULTS (override on the command line)
# =========================
# folder with dinov3_vitb16_depth_fold{1..5}_best.pt -- by default a "checkpoints_best" folder next to this script
DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_best")
DEFAULT_OUTPUT_CSV = "tbr_predictions.csv"

# --- must match training ---
DINOV3_TIMM_NAME = "vit_base_patch16_dinov3.lvd1689m"
IMG_SIZE = 336
FRAMES_PER_VIDEO = 16
FRAME_OVERSAMPLE_MULTIPLIER = 3
ATTN_MAP_SIZE = 7
ROTATE_FRAMES = True
USE_AMP_ON_CUDA = True               # training used fp16 autocast on the backbone

# --- reporting ---
TRAINED_RANGE_MM = (4.0, 24.0)       # depths the models were trained on
SPREAD_WARN_MM = 1.0                 # std across folds above this -> LOW_CONFIDENCE
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =========================
# Frame sampling + quality score (identical to training)
# =========================

def _sharpness(frame):
    return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def _ensure_vertical(frame):
    if not ROTATE_FRAMES:
        return frame
    h, w = frame.shape[:2]
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) if w > h else frame


def sample_best_frames(video_path, n_frames=FRAMES_PER_VIDEO, oversample=FRAME_OVERSAMPLE_MULTIPLIER):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:                                   # container without a frame count: read everything
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(_ensure_vertical(frame))
        cap.release()
        scored = sorted(enumerate(frames), key=lambda x: _sharpness(x[1]), reverse=True)[:n_frames]
        return [f for _, f in sorted(scored, key=lambda x: x[0])]

    n_candidates = min(total, max(n_frames, n_frames * oversample))
    idxs = np.linspace(0, total - 1, n_candidates).astype(int)
    candidates = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            candidates.append((int(idx), _ensure_vertical(frame)))
    cap.release()
    scored = sorted(candidates, key=lambda x: _sharpness(x[1]), reverse=True)[:n_frames]
    return [f for _, f in sorted(scored, key=lambda x: x[0])]


def _attention_target_map(frame):
    gray_eq = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    edges = cv2.Canny(gray_eq, 50, 150).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray_eq, (31, 31), 0)
    groove = np.abs(blur.astype(np.float32) - gray_eq.astype(np.float32))
    return 0.5 * edges + 0.5 * groove / (groove.max() + 1e-6)


def frame_quality_scores(frames_bgr):
    """Per-frame informativeness, min-max normalised over the video (same as build_attention_targets in training)."""
    small = np.stack([
        cv2.resize(
            (cv2.resize(_attention_target_map(f), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA) * 255)
            .clip(0, 255).astype(np.uint8).astype(np.float32),
            (ATTN_MAP_SIZE, ATTN_MAP_SIZE), interpolation=cv2.INTER_AREA)
        for f in frames_bgr
    ])
    raw = np.clip(small, 0, None).reshape(len(frames_bgr), -1).sum(axis=1).astype(np.float32)
    lo, hi = raw.min(), raw.max()
    return np.zeros(len(frames_bgr), np.float32) if hi - lo < 1e-6 else (raw - lo) / (hi - lo)


def prepare_video(video_path):
    """Returns (frames_rgb_full list, tensor (1,16,3,H,W), frame_quality (1,16)) or None if unreadable."""
    frames = sample_best_frames(video_path)
    if len(frames) == 0:
        return None
    fq = frame_quality_scores(frames)
    rgb_full = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    resized = np.stack([cv2.resize(f, (IMG_SIZE, IMG_SIZE)) for f in rgb_full])
    if len(frames) < FRAMES_PER_VIDEO:               # pad like training
        k = FRAMES_PER_VIDEO - len(frames)
        resized = np.concatenate([resized, np.repeat(resized[-1:], k, axis=0)])
        fq = np.concatenate([fq, np.repeat(fq[-1:], k)])
        rgb_full += [rgb_full[-1]] * k
    x = ((resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(0, 3, 1, 2)
    return rgb_full, torch.from_numpy(x).float().unsqueeze(0), torch.from_numpy(fq).float().unsqueeze(0)


# =========================
# Model (same as the training notebook, hand features removed)
# =========================

class SpatialAttentionPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        logits = self.attn_conv(x)
        N, _, H, W = logits.shape
        w = torch.softmax(logits.view(N, -1), dim=-1).view(N, 1, H, W)
        return (x * w).sum(dim=[2, 3]), w


class TemporalAttentionPool(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(feature_dim + 1, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, x, frame_quality):
        scores = self.attn(torch.cat([x, frame_quality.unsqueeze(-1)], dim=-1)).squeeze(-1)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1), w


class DINOv3Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model(DINOV3_TIMM_NAME, pretrained=False, num_classes=0, global_pool="",
                                          img_size=IMG_SIZE)
        self.feature_dim = self.backbone.num_features
        self.num_prefix = self.backbone.num_prefix_tokens
        ps = self.backbone.patch_embed.patch_size
        ps = ps[0] if isinstance(ps, (tuple, list)) else ps
        self.grid = IMG_SIZE // ps
        self.pool_factor = self.grid // ATTN_MAP_SIZE
        self.spatial_attn_pool = SpatialAttentionPool(self.feature_dim)
        self.temporal_attn_pool = TemporalAttentionPool(self.feature_dim)
        self.head = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, frames, frame_quality, use_amp=False):
        B, T, C, H, W = frames.shape
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            tokens = self.backbone(frames.view(B * T, C, H, W))
        x = tokens[:, self.num_prefix:, :].float().transpose(1, 2).reshape(B * T, self.feature_dim, self.grid, self.grid)
        if self.pool_factor > 1:
            x = F.avg_pool2d(x, self.pool_factor)
        pooled_spatial, spatial_attn = self.spatial_attn_pool(x)
        pooled_temporal, temporal_attn = self.temporal_attn_pool(pooled_spatial.view(B, T, self.feature_dim), frame_quality)
        out = self.head(pooled_temporal).squeeze(1)
        spatial_attn = spatial_attn.view(B, T, 1, spatial_attn.shape[-2], spatial_attn.shape[-1])
        return out, spatial_attn, temporal_attn


def load_models(checkpoint_dir, folds, device):
    paths = {}
    for p in glob.glob(os.path.join(checkpoint_dir, "dinov3_vitb16_depth_fold*_best.pt")):
        m = re.search(r"fold(\d+)_best", p)
        if m:
            paths[int(m.group(1))] = p
    if not paths:
        sys.exit(f"No dinov3_vitb16_depth_fold*_best.pt found in {checkpoint_dir}")
    use = sorted(paths) if folds is None else [f for f in folds if f in paths]
    missing = set(folds or []) - set(paths)
    if missing:
        sys.exit(f"Requested fold(s) {sorted(missing)} not found. Available: {sorted(paths)}")
    models = {}
    for f in use:
        sd = torch.load(paths[f], map_location="cpu", weights_only=True)
        head_in = sd["head.0.weight"].shape[1]
        dim = sd["backbone.norm.weight"].shape[0]
        if head_in != dim:
            sys.exit(f"fold {f}: head expects {head_in} inputs but the embedding is {dim}. These look like the OLD checkpoints "
                     f"trained with 5 hand-crafted features; this script is for the no-hand-feature 336 px models.")
        m = DINOv3Regressor().to(device)
        m.load_state_dict(sd)                        # strict: fails loudly on any mismatch
        models[f] = m.eval()
    return models


# =========================
# Optional attention picture
# =========================

def save_attention_image(frames_rgb, spatial_mean, temporal_mean, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(frames_rgb)
    cols, rows = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.5))
    axes = np.array(axes).reshape(-1)
    for i in range(n):
        h, w = frames_rgb[i].shape[:2]
        a = np.clip(cv2.resize(spatial_mean[i], (w, h), interpolation=cv2.INTER_CUBIC), 0, None)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        heat = cv2.cvtColor(cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        axes[i].imshow((frames_rgb[i] * 0.55 + heat * 0.45).astype(np.uint8))
        axes[i].set_title(f"Frame {i} | temporal wt {temporal_mean[i]:.3f}", fontsize=9)
        axes[i].axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# =========================
# Input handling
# =========================

def collect_inputs(inp):
    """Returns a list of (video_path, actual_depth_or_None)."""
    if os.path.isdir(inp):
        files = sorted(f for f in os.listdir(inp) if f.lower().endswith(VIDEO_EXTENSIONS))
        return [(os.path.join(inp, f), None) for f in files]
    if inp.lower().endswith(".csv"):
        df = pd.read_csv(inp)
        base = os.path.dirname(os.path.abspath(inp))
        col = "video_path" if "video_path" in df.columns else "filename" if "filename" in df.columns else None
        if col is None:
            sys.exit("CSV needs a 'video_path' or 'filename' column.")
        out = []
        for _, r in df.iterrows():
            p = str(r[col]).strip()
            if not os.path.isabs(p):
                p = os.path.join(base, p)
            act = float(r["depth_mm"]) if "depth_mm" in df.columns and pd.notna(r["depth_mm"]) else None
            out.append((p, act))
        return out
    if os.path.isfile(inp):
        return [(inp, None)]
    sys.exit(f"Input not found: {inp}")


def print_accuracy(df):
    d = df.dropna(subset=["actual_mm", "pred_mm"])
    if d.empty:
        return
    y, p = d["actual_mm"].values, d["pred_mm"].values
    e = p - y
    print("\n=== ACCURACY vs the depth_mm column ===")
    print(f"n={len(d)} | MAE {np.abs(e).mean():.2f} | RMSE {np.sqrt((e ** 2).mean()):.2f} | bias {e.mean():+.2f} | "
          f"within 1mm {100 * (np.abs(e) <= 1).mean():.0f}% | 2mm {100 * (np.abs(e) <= 2).mean():.0f}%")
    edges = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 100]
    print(f"{'bin (mm)':<10}{'n':>5}{'MAE':>8}{'bias':>8}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y >= lo) & (y < hi)
        if m.any():
            print(f"{lo}-{hi:<7}{m.sum():>5}{np.abs(e[m]).mean():>8.2f}{e[m].mean():>+8.2f}")


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser(description="TBR tread-depth inference (DINOv3 ViT-B, 336 px)")
    ap.add_argument("--input", required=True, help="video file, folder of videos, or CSV (filename/video_path [+ depth_mm])")
    ap.add_argument("--checkpoints", default=DEFAULT_CHECKPOINT_DIR)
    ap.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="predictions CSV to write")
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="subset of folds to use, e.g. --folds 1 2 (default: all found)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--tta-flip", action="store_true", help="also average predictions on horizontally flipped frames (2x slower)")
    ap.add_argument("--attention-dir", default=None, help="if set, save an attention picture per video here")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = USE_AMP_ON_CUDA and device.type == "cuda"
    items = collect_inputs(args.input)
    if not items:
        sys.exit("No videos found.")
    print(f"Device: {device} | videos: {len(items)}")
    models = load_models(args.checkpoints, args.folds, device)
    fold_ids = list(models)
    print(f"Loaded folds {fold_ids} from {args.checkpoints}")
    if device.type == "cpu":
        print("NOTE: CPU inference is slow (~5-8 s per fold model per video at 336 px).")
    if args.attention_dir:
        os.makedirs(args.attention_dir, exist_ok=True)

    rows = []
    t_start = time.time()
    for n, (path, actual) in enumerate(items, start=1):
        name = os.path.basename(path)
        row = {"video": name, "path": path, "actual_mm": actual}
        prep = prepare_video(path) if os.path.exists(path) else None
        if prep is None:
            row["flag"] = "NO_FRAMES"
            rows.append(row)
            print(f"[{n}/{len(items)}] {name}: could not read frames -- skipped")
            continue
        frames_rgb, x, fq = prep
        x, fq = x.to(device), fq.to(device)

        preds, spatial_all, temporal_all = {}, [], []
        with torch.no_grad():
            for f, model in models.items():
                out, sp, tp = model(x, fq, use_amp=use_amp)
                p = float(out.item())
                if args.tta_flip:
                    out2, _, _ = model(torch.flip(x, dims=[4]), fq, use_amp=use_amp)
                    p = 0.5 * (p + float(out2.item()))
                preds[f] = p
                spatial_all.append(sp.squeeze(0).squeeze(1).cpu().numpy())
                temporal_all.append(tp.squeeze(0).squeeze(-1).cpu().numpy())

        vals = np.array(list(preds.values()))
        ens, spread = float(vals.mean()), float(vals.std())
        flags = []
        if spread > SPREAD_WARN_MM:
            flags.append("LOW_CONFIDENCE")
        if not (TRAINED_RANGE_MM[0] <= ens < TRAINED_RANGE_MM[1]):
            flags.append("OUT_OF_RANGE")
        row.update({"pred_mm": round(ens, 3), "fold_std_mm": round(spread, 3), "fold_min_mm": round(vals.min(), 3),
                    "fold_max_mm": round(vals.max(), 3), "flag": "|".join(flags),
                    **{f"fold{f}_mm": round(v, 3) for f, v in preds.items()}})
        rows.append(row)

        err = f" | actual {actual:.1f} (err {ens - actual:+.2f})" if actual is not None else ""
        print(f"[{n}/{len(items)}] {name}: {ens:6.2f} mm  (folds {vals.min():.2f}-{vals.max():.2f}, std {spread:.2f}){err}"
              f"{'  <-- ' + ','.join(flags) if flags else ''}")

        if args.attention_dir:
            save_attention_image(frames_rgb, np.mean(spatial_all, axis=0), np.mean(temporal_all, axis=0),
                                 f"{name} | predicted {ens:.2f} mm (std {spread:.2f} over {len(fold_ids)} folds)",
                                 os.path.join(args.attention_dir, os.path.splitext(name)[0] + ".png"))

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Predictions saved to {os.path.abspath(args.output)}")
    flagged = df["flag"].fillna("").astype(str)
    if (flagged != "").any():
        print(f"{(flagged != '').sum()} video(s) flagged -- see the 'flag' column.")
    if df["actual_mm"].notna().any():
        print_accuracy(df)


if __name__ == "__main__":
    main()
