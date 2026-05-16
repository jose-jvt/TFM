"""Dataset quality report for segmentation splits.

Generates a self-contained HTML report with:
  - Split overview (counts, class balance, tarp coverage)
  - Image quality distributions and outliers
    (brightness, darkness, sharpness/blur, contrast, saturation)
  - Data leakage: tile-coordinate overlap + perceptual-hash near-duplicates
  - Distribution shift between splits (KS test)
  - Spatial tile scatter plot
  - Temporal and project-type breakdown
  - Resolution distribution
  - Mask coverage heatmap per split

Usage:
    python -m scripts.dataset_report \\
        --train  dataset/train.csv \\
        --val    dataset/val.csv \\
        --test   dataset/test.csv \\
        --output outputs/dataset_report.html \\
        [--max-samples 500]
"""
from __future__ import annotations

import argparse
import base64
import io
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Per-image feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _avg_hash(gray: np.ndarray) -> np.ndarray:
    """64-element uint8 bit array (average hash, 8×8)."""
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    avg = small.mean()
    return (small.flatten() > avg).astype(np.uint8)


def extract_stats(image_path: str, mask_path: str) -> dict:
    img = cv2.imread(str(image_path))
    record: dict = {"image_path": image_path, "mask_path": mask_path, "error": False}

    if img is None:
        record["error"] = True
        return record

    h, w = img.shape[:2]
    record["height"] = h
    record["width"] = w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    record["brightness"] = float(gray.mean())
    record["darkness"] = float(255 - gray.mean())
    record["contrast"] = float(gray.std())

    lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_32F)
    record["sharpness"] = float(lap.var())

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    record["saturation"] = float(hsv[:, :, 1].mean())
    record["hue_mean"] = float(hsv[:, :, 0].mean())

    # Per-channel statistics (helps detect sensor/color-shift issues)
    b, g, r = cv2.split(img)
    record["mean_r"] = float(r.mean())
    record["mean_g"] = float(g.mean())
    record["mean_b"] = float(b.mean())

    record["phash"] = _avg_hash(gray)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        record["tarp_ratio"] = 0.0
        record["has_tarp"] = False
    else:
        tarp_px = int((mask > 0).sum())
        record["tarp_ratio"] = tarp_px / (h * w)
        record["has_tarp"] = tarp_px > 0

    return record


def load_split(csv_path: str, split: str, max_samples: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split:5s}", leave=False):
        r = extract_stats(row["image_path"], row["mask_path"])
        r["split"] = split
        for col in ("tipo_proyecto", "año", "zoom", "tile_x", "tile_y"):
            if col in df.columns:
                r.setdefault(col, row[col])
        records.append(r)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Leakage detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_tile_leakage(df: pd.DataFrame) -> pd.DataFrame:
    if not {"tile_x", "tile_y"}.issubset(df.columns):
        return pd.DataFrame()
    tile_splits = df.groupby(["tile_x", "tile_y"])["split"].apply(set)
    leaked = tile_splits[tile_splits.apply(len) > 1].reset_index()
    leaked["splits_found"] = leaked["split"].apply(lambda s: ", ".join(sorted(s)))
    return leaked[["tile_x", "tile_y", "splits_found"]]


def detect_near_duplicates(df: pd.DataFrame, threshold: int = 2) -> pd.DataFrame:
    """Find images from different splits with perceptual Hamming distance ≤ threshold."""
    valid = df[df["error"] == False]
    if "phash" not in valid.columns:
        return pd.DataFrame()

    all_splits = valid["split"].unique()
    if len(all_splits) < 2:
        return pd.DataFrame()

    split_data: dict[str, tuple[np.ndarray, list[str]]] = {}
    for s in all_splits:
        sub = valid[valid["split"] == s]
        paths = sub["image_path"].tolist()
        bits = np.vstack(sub["phash"].values)      # (N, 64) uint8
        split_data[s] = (bits, paths)

    results = []
    pairs = [(a, b) for i, a in enumerate(all_splits) for b in all_splits[i + 1:]]
    for s_a, s_b in pairs:
        bits_a, paths_a = split_data[s_a]
        bits_b, paths_b = split_data[s_b]
        # Vectorised Hamming: (N_a, N_b)
        dists = np.count_nonzero(bits_a[:, None, :] != bits_b[None, :, :], axis=2)
        ii, jj = np.where(dists <= threshold)
        for i, j in zip(ii, jj):
            results.append({
                "split_a": s_a,
                "split_b": s_b,
                "image_a": Path(paths_a[i]).name,
                "image_b": Path(paths_b[j]).name,
                "hamming": int(dists[i, j]),
            })

    return pd.DataFrame(results) if results else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Outlier detection
# ─────────────────────────────────────────────────────────────────────────────

QUALITY_METRICS: list[dict] = [
    {"col": "brightness", "low_pct": 1,    "high_pct": 99,   "label": "Brightness"},
    {"col": "darkness",   "low_pct": None, "high_pct": 99,   "label": "Darkness (high = very dark image)"},
    {"col": "sharpness",  "low_pct": 2,    "high_pct": None, "label": "Sharpness – Laplacian variance (low = blurry)"},
    {"col": "contrast",   "low_pct": 2,    "high_pct": None, "label": "Contrast – pixel std (low = flat/uniform)"},
    {"col": "saturation", "low_pct": 1,    "high_pct": 99,   "label": "Saturation"},
]


def find_outliers(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid = df[df["error"] == False].copy()
    out: dict[str, pd.DataFrame] = {}

    for m in QUALITY_METRICS:
        col = m["col"]
        if col not in valid.columns:
            continue
        lo = valid[col].quantile(m["low_pct"] / 100) if m["low_pct"] else -np.inf
        hi = valid[col].quantile(m["high_pct"] / 100) if m["high_pct"] else np.inf
        mask = (valid[col] < lo) | (valid[col] > hi)
        sub = valid[mask][["split", "image_path", col]].copy()
        sub["flag"] = sub[col].apply(
            lambda v: f"too low ({v:.2f} < {lo:.2f})" if v < lo else f"too high ({v:.2f} > {hi:.2f})"
        )
        out[col] = sub.sort_values(col).reset_index(drop=True)

    # Images with zero tarp coverage
    if "tarp_ratio" in valid.columns:
        empty = valid[valid["tarp_ratio"] == 0][["split", "image_path", "tarp_ratio"]].copy()
        empty["flag"] = "no tarp pixels"
        out["empty_mask"] = empty.reset_index(drop=True)

    # Images where tarp covers > 50 % (unusual, worth reviewing)
    if "tarp_ratio" in valid.columns:
        heavy = valid[valid["tarp_ratio"] > 0.5][["split", "image_path", "tarp_ratio"]].copy()
        heavy["flag"] = heavy["tarp_ratio"].apply(lambda v: f"{v*100:.1f}% tarp coverage")
        out["heavy_tarp"] = heavy.sort_values("tarp_ratio", ascending=False).reset_index(drop=True)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Distribution shift (KS test train vs val / train vs test)
# ─────────────────────────────────────────────────────────────────────────────

SHIFT_METRICS = ["brightness", "contrast", "sharpness", "saturation", "tarp_ratio"]


def distribution_shift(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["error"] == False]
    train = valid[valid["split"] == "train"]
    rows = []
    for other in ("val", "test"):
        other_df = valid[valid["split"] == other]
        if len(other_df) == 0:
            continue
        for col in SHIFT_METRICS:
            if col not in valid.columns:
                continue
            stat, pval = stats.ks_2samp(
                train[col].dropna().values,
                other_df[col].dropna().values,
            )
            rows.append({
                "comparison": f"train vs {other}",
                "metric": col,
                "KS statistic": f"{stat:.4f}",
                "p-value": f"{pval:.4f}",
                "shift": "⚠️ significant" if pval < 0.05 else "✅ ok",
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
SPLITS_ORDER = ["train", "val", "test"]


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return encoded


def chart_tarp_ratio(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Violin per split
    ax = axes[0]
    data = [df[df["split"] == s]["tarp_ratio"].dropna().values for s in SPLITS_ORDER]
    present = [(d, s) for d, s in zip(data, SPLITS_ORDER) if len(d) > 0]
    if present:
        vp = ax.violinplot([d for d, _ in present], showmedians=True)
        for pc, (_, s) in zip(vp["bodies"], present):
            pc.set_facecolor(SPLIT_COLORS[s])
            pc.set_alpha(0.75)
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels([s for _, s in present])
    ax.set_title("Tarp pixel ratio – distribution")
    ax.set_ylabel("tarp_ratio")

    # CDF comparison
    ax2 = axes[1]
    for s in SPLITS_ORDER:
        vals = np.sort(df[df["split"] == s]["tarp_ratio"].dropna().values)
        if len(vals) == 0:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax2.plot(vals, cdf, label=s, color=SPLIT_COLORS[s])
    ax2.set_title("Tarp pixel ratio – CDF")
    ax2.set_xlabel("tarp_ratio")
    ax2.set_ylabel("Cumulative fraction")
    ax2.legend()

    fig.suptitle("Class balance (tarp coverage)", fontweight="bold")
    return _b64(fig)


def chart_quality_grids(df: pd.DataFrame) -> str:
    metrics = [
        ("brightness", "Brightness"),
        ("contrast", "Contrast"),
        ("sharpness", "Sharpness (Laplacian var)"),
        ("saturation", "Saturation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (col, label) in zip(axes, metrics):
        for s in SPLITS_ORDER:
            vals = df[df["split"] == s][col].dropna().values
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=50, alpha=0.55, density=True, label=s, color=SPLIT_COLORS[s])
        ax.set_title(label)
        ax.legend(fontsize=8)
    fig.suptitle("Image quality metric distributions", fontweight="bold")
    fig.tight_layout()
    return _b64(fig)


def chart_tile_scatter(df: pd.DataFrame) -> str | None:
    if not {"tile_x", "tile_y"}.issubset(df.columns):
        return None
    fig, ax = plt.subplots(figsize=(9, 7))
    for s in SPLITS_ORDER:
        sub = df[df["split"] == s]
        ax.scatter(sub["tile_x"], sub["tile_y"],
                   c=SPLIT_COLORS[s], alpha=0.35, s=8, label=s)
    ax.set_title("Spatial tile distribution per split")
    ax.set_xlabel("tile_x")
    ax.set_ylabel("tile_y")
    ax.legend(markerscale=3)
    return _b64(fig)


def chart_resolution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, dim in zip(axes, ["width", "height"]):
        for s in SPLITS_ORDER:
            vals = df[df["split"] == s][dim].dropna().values
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=20, alpha=0.6, label=s, color=SPLIT_COLORS[s])
        ax.set_title(f"Image {dim}")
        ax.set_xlabel("pixels")
        ax.legend()
    fig.suptitle("Resolution distribution", fontweight="bold")
    return _b64(fig)


def chart_year(df: pd.DataFrame) -> str | None:
    if "año" not in df.columns:
        return None
    counts = df.groupby(["año", "split"]).size().unstack(fill_value=0)
    if counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    counts.plot(kind="bar", ax=ax,
                color=[SPLIT_COLORS.get(c, "#888") for c in counts.columns])
    ax.set_title("Images per year and split")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend()
    return _b64(fig)


def chart_proyecto(df: pd.DataFrame) -> str | None:
    if "tipo_proyecto" not in df.columns:
        return None
    counts = df.groupby(["tipo_proyecto", "split"]).size().unstack(fill_value=0)
    if counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    counts.plot(kind="bar", ax=ax,
                color=[SPLIT_COLORS.get(c, "#888") for c in counts.columns])
    ax.set_title("Images per project type and split")
    ax.set_xlabel("tipo_proyecto")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend()
    return _b64(fig)


def chart_zoom(df: pd.DataFrame) -> str | None:
    if "zoom" not in df.columns:
        return None
    counts = df.groupby(["zoom", "split"]).size().unstack(fill_value=0)
    if counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax,
                color=[SPLIT_COLORS.get(c, "#888") for c in counts.columns])
    ax.set_title("Images per zoom level and split")
    ax.set_xlabel("Zoom level")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend()
    return _b64(fig)


def chart_channel_balance(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    channels = ["mean_r", "mean_g", "mean_b"]
    ch_labels = ["Red", "Green", "Blue"]
    x = np.arange(len(SPLITS_ORDER))
    width = 0.25
    for i, (ch, label) in enumerate(zip(channels, ch_labels)):
        means = [df[df["split"] == s][ch].mean() for s in SPLITS_ORDER]
        ax.bar(x + i * width, means, width, label=label,
               color=["#e74c3c", "#2ecc71", "#3498db"][i], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(SPLITS_ORDER)
    ax.set_title("Mean per-channel value per split (colour shift indicator)")
    ax.set_ylabel("Mean pixel value (0–255)")
    ax.legend()
    return _b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  max-width: 1150px; margin: 40px auto; padding: 0 20px; color: #1a1a2e; background: #fafafa;
}
h1 { color: #0f3460; font-size: 2em; margin-bottom: 4px; }
.subtitle { color: #666; font-size: 0.95em; margin-bottom: 32px; }
h2 { color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: 8px; margin-top: 48px; }
h3 { color: #444; margin-top: 24px; }
table {
  border-collapse: collapse; width: 100%; font-size: 12.5px;
  margin: 10px 0; background: white; border-radius: 6px; overflow: hidden;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
th { background: #0f3460; color: white; padding: 9px 12px; text-align: left; }
td { padding: 7px 12px; border-bottom: 1px solid #eee; }
tr:last-child td { border-bottom: none; }
tr:nth-child(even) td { background: #f7f9fc; }
.ok      { background:#d4edda; border-left:4px solid #28a745; padding:10px 16px; border-radius:4px; margin:8px 0; }
.warning { background:#fff3cd; border-left:4px solid #ffc107; padding:10px 16px; border-radius:4px; margin:8px 0; }
.danger  { background:#f8d7da; border-left:4px solid #dc3545; padding:10px 16px; border-radius:4px; margin:8px 0; }
.info    { background:#cce5ff; border-left:4px solid #004085; padding:10px 16px; border-radius:4px; margin:8px 0; }
img.chart { max-width:100%; border:1px solid #ddd; border-radius:6px; margin:10px 0; display:block; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.section-note { color:#555; font-size:0.9em; margin-bottom:8px; }
"""


def _img(b64: str) -> str:
    return f'<img class="chart" src="data:image/png;base64,{b64}" />'


def _df_html(df: pd.DataFrame, max_rows: int = 60) -> str:
    if len(df) == 0:
        return "<p><em>None found.</em></p>"
    return df.head(max_rows).to_html(index=False, border=0)


def _alert(cls: str, text: str) -> str:
    return f'<div class="{cls}">{text}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Report assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    df: pd.DataFrame,
    outliers: dict[str, pd.DataFrame],
    tile_leakage: pd.DataFrame,
    near_dups: pd.DataFrame,
    shift_df: pd.DataFrame,
) -> str:

    valid = df[df["error"] == False]
    errors = df[df["error"] == True]

    # ── 1. Summary table ───────────────────────────────────────────────────────
    rows = []
    for s in SPLITS_ORDER:
        sub = valid[valid["split"] == s]
        if sub.empty:
            continue
        rows.append({
            "Split": s,
            "Images": len(sub),
            "% of total": f"{100*len(sub)/len(valid):.1f}%",
            "With tarp": f"{sub['has_tarp'].sum()} ({100*sub['has_tarp'].mean():.1f}%)",
            "Avg tarp ratio": f"{sub['tarp_ratio'].mean():.4f}",
            "Median tarp ratio": f"{sub['tarp_ratio'].median():.4f}",
            "Avg brightness": f"{sub['brightness'].mean():.1f}",
            "Avg sharpness": f"{sub['sharpness'].mean():.0f}",
            "Resolutions": ", ".join(
                sorted({f"{int(r.width)}×{int(r.height)}" for r in sub.itertuples()})
            ),
        })
    summary_html = pd.DataFrame(rows).to_html(index=False, border=0)

    # ── Counts for leakage section ─────────────────────────────────────────────
    tile_ok = len(tile_leakage) == 0
    dup_ok = len(near_dups) == 0

    # ── Global warnings ─────────────────────────────────────────────────────────
    global_warnings = ""
    if len(errors) > 0:
        global_warnings += _alert("warning", f"⚠️ {len(errors)} image(s) could not be read and were excluded from analysis.")
    if not tile_ok:
        global_warnings += _alert("danger",  f"🚨 Tile-coordinate leakage detected: {len(tile_leakage)} overlapping tiles across splits.")
    if not dup_ok:
        global_warnings += _alert("warning", f"⚠️ {len(near_dups)} near-duplicate image pair(s) found across splits.")
    sig_shifts = shift_df[shift_df["shift"].str.contains("significant")] if len(shift_df) > 0 else pd.DataFrame()
    if len(sig_shifts) > 0:
        global_warnings += _alert("warning", f"⚠️ Statistically significant distribution shift detected for: {', '.join(sig_shifts['metric'].unique())}.")
    if not global_warnings:
        global_warnings = _alert("ok", "✅ No critical issues detected.")

    # ── Charts ─────────────────────────────────────────────────────────────────
    c_tarp     = chart_tarp_ratio(valid)
    c_quality  = chart_quality_grids(valid)
    c_channel  = chart_channel_balance(valid)
    c_res      = chart_resolution(valid)
    c_scatter  = chart_tile_scatter(valid)
    c_year     = chart_year(valid)
    c_proyecto = chart_proyecto(valid)
    c_zoom     = chart_zoom(valid)

    # ── Outlier section ─────────────────────────────────────────────────────────
    outlier_html = ""
    outlier_defs = [
        ("brightness", "Brightness outliers",
         "Images at the extremes of the brightness distribution (top 1% / bottom 1%). "
         "Very bright tiles may be overexposed; very dark tiles may lack useful signal."),
        ("darkness", "Darkness outliers",
         "Images where the darkness score exceeds the 99th percentile, i.e. tiles with very low mean pixel value."),
        ("sharpness", "Blur / low-sharpness outliers",
         "Images in the bottom 2% by Laplacian variance. These are likely blurry, "
         "cloud-covered, or otherwise degraded tiles that could harm model training."),
        ("contrast", "Low-contrast outliers",
         "Images with very low pixel standard deviation (bottom 2%). "
         "Uniform/flat images often carry no structural information."),
        ("saturation", "Saturation outliers",
         "Images at the extremes of colour saturation: could indicate sensor artefacts "
         "or missing spectral bands."),
        ("empty_mask", "Empty masks (no tarp pixels)",
         "Images whose mask has zero tarp pixels. These are valid background-only tiles "
         "but an unusually high count could indicate annotation gaps."),
        ("heavy_tarp", "Heavy tarp coverage (> 50 % of tile)",
         "Unusual tiles where tarps dominate the image. Worth reviewing for annotation errors."),
    ]
    for key, title, note in outlier_defs:
        df_out = outliers.get(key, pd.DataFrame())
        count = len(df_out)
        cls = "warning" if count > 0 else "ok"
        icon = "⚠️" if count > 0 else "✅"
        display_col = key if key not in ("empty_mask", "heavy_tarp") else "tarp_ratio"
        cols = ["split", "image_path"]
        if display_col in (df_out.columns if len(df_out) > 0 else []):
            cols.append(display_col)
        cols += [c for c in ["flag"] if c in (df_out.columns if len(df_out) > 0 else [])]
        outlier_html += f"""
<h3>{title}</h3>
<p class="section-note">{note}</p>
{_alert(cls, f"{icon} <strong>{count}</strong> outlier(s) found.")}
{_df_html(df_out[[c for c in cols if c in df_out.columns]] if len(df_out) > 0 else df_out)}
"""

    # ── Leakage section ────────────────────────────────────────────────────────
    leakage_html = ""
    if tile_ok:
        leakage_html += _alert("ok", "✅ No tile-coordinate overlap between splits. Anti-leakage constraint respected.")
    else:
        leakage_html += _alert("danger",
            f"🚨 <strong>{len(tile_leakage)}</strong> tile(s) appear in more than one split. "
            "This violates the anti-leakage guarantee — re-split the dataset.")
        leakage_html += _df_html(tile_leakage)

    leakage_html += "<h3>Near-duplicate images (perceptual hash, Hamming ≤ 8)</h3>"
    leakage_html += "<p class='section-note'>Images from different splits with very similar pixel content. Hamming distance 0 = identical hash.</p>"
    if dup_ok:
        leakage_html += _alert("ok", "✅ No near-duplicate images found across splits.")
    else:
        leakage_html += _alert("warning",
            f"⚠️ <strong>{len(near_dups)}</strong> near-duplicate pair(s) detected. "
            "Review these tiles — they may represent the same geographic area at different times.")
        leakage_html += _df_html(near_dups.sort_values("hamming"))

    # ── Distribution shift table ───────────────────────────────────────────────
    shift_html = ""
    if shift_df.empty:
        shift_html = "<p><em>Not computed.</em></p>"
    else:
        shift_html = (
            "<p class='section-note'>Kolmogorov–Smirnov two-sample test. p &lt; 0.05 indicates "
            "a statistically significant difference between train and the comparison split.</p>"
        )
        shift_html += _df_html(shift_df)

    # ── Final HTML ─────────────────────────────────────────────────────────────
    scatter_block = _img(c_scatter) if c_scatter else "<p><em>tile_x / tile_y columns not available.</em></p>"
    year_block    = _img(c_year)    if c_year    else ""
    proy_block    = _img(c_proyecto) if c_proyecto else ""
    zoom_block    = _img(c_zoom)    if c_zoom    else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset Quality Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>📊 Dataset Quality Report</h1>
<p class="subtitle">Tarp segmentation splits · train / val / test</p>

<h2>⚡ Global Alerts</h2>
{global_warnings}

<h2>1 · Split Overview</h2>
{summary_html}

<h2>2 · Class Balance – Tarp Coverage</h2>
<p class="section-note">
  Distribution of the tarp pixel ratio (tarp pixels / total pixels) per split.
  Splits should have similar distributions; a skewed violin in one split may indicate
  that the stratification objective was not met.
</p>
{_img(c_tarp)}

<h2>3 · Image Quality Distributions</h2>
<p class="section-note">
  Histograms of key image quality metrics across splits.
  Large differences between histograms suggest domain shift that may hurt generalisation.
</p>
{_img(c_quality)}

<h3>Per-channel mean values (colour balance)</h3>
<p class="section-note">
  Systematic differences in R/G/B channel means between splits can indicate
  different acquisition conditions or sensor settings.
</p>
{_img(c_channel)}

<h2>4 · Quality Outliers</h2>
{outlier_html}

<h2>5 · Data Leakage Analysis</h2>
<h3>Tile-coordinate overlap</h3>
<p class="section-note">
  Each unique <code>(tile_x, tile_y)</code> must appear in exactly one split.
  Any overlap means geographically adjacent tiles could appear in both training and evaluation.
</p>
{leakage_html}

<h2>6 · Distribution Shift (KS test)</h2>
{shift_html}

<h2>7 · Spatial Distribution</h2>
<p class="section-note">
  Scatter plot of all tile coordinates, coloured by split.
  Good splits cover the geographic area proportionally; isolated clusters in one split
  may introduce spatial bias.
</p>
{scatter_block}

<h2>8 · Temporal & Project Breakdown</h2>
{year_block}
{proy_block}
{zoom_block}

<h2>9 · Resolution Distribution</h2>
<p class="section-note">
  Image width and height histograms per split.
  Mixed resolutions are handled by <code>ResolutionBatchSampler</code>, but an imbalanced
  zoom-level distribution between splits could affect model generalisation.
</p>
{_img(c_res)}

</body>
</html>"""

    return html


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a dataset quality HTML report from split CSVs."
    )
    parser.add_argument("--train",       default="dataset/train.csv")
    parser.add_argument("--val",         default="dataset/val.csv")
    parser.add_argument("--test",        default="dataset/test.csv")
    parser.add_argument("--output",      default="outputs/dataset_report.html")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap images per split (for faster iteration, default: all)")
    args = parser.parse_args()

    print("Loading splits and extracting image features...")
    dfs = []
    for split, csv in [("train", args.train), ("val", args.val), ("test", args.test)]:
        dfs.append(load_split(csv, split, args.max_samples))
    df_all = pd.concat(dfs, ignore_index=True)

    valid = df_all[df_all["error"] == False]
    print(f"  {len(valid)} images analysed ({len(df_all) - len(valid)} read errors)")

    print("Finding outliers...")
    outliers = find_outliers(valid)

    print("Checking tile-coordinate leakage...")
    tile_leakage = detect_tile_leakage(df_all)

    print("Checking near-duplicate images across splits...")
    near_dups = detect_near_duplicates(valid)

    print("Running distribution shift tests (KS)...")
    shift_df = distribution_shift(valid)

    print("Rendering charts and building report...")
    html = build_report(df_all, outliers, tile_leakage, near_dups, shift_df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"\n✅  Report saved → {out_path}")


if __name__ == "__main__":
    main()
