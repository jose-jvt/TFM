"""CLI: per-footprint tarp damage scoring + region analysis from a GeoPackage.

Reads ``footprints`` and ``tarps`` layers from a GPKG, computes:

  * Per-footprint damage score  = intersection(footprint, tarps) / footprint_area
  * Region-level aggregation    – by configurable grid

Outputs (all written to --output-dir):
  footprints_scored.gpkg   GeoPackage with scored footprints  (open in QGIS)
  regions_grid.gpkg        Region stats on a regular grid
  footprint_scores.csv     Flat CSV (no geometry)
  report.html              Self-contained HTML summary report

Usage
-----
    python -m building_analysis.analyze_gpkg \\
        --gpkg       geodata/my_area.gpkg \\
        --output-dir geodata/results/my_area \\
        [--footprints-layer footprints] \\
        [--tarps-layer      tarps] \\
        [--min-area 5.0] \\
        [--grid-size 500]
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from building_analysis.gpkg_scoring import (
    load_gpkg_layers,
    score_footprints,
    footprint_summary,
    print_summary,
    DAMAGE_BINS,
    DAMAGE_LABELS,
)
from area_analysis.region_analysis import (
    aggregate_by_grid,
    print_region_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    s = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return s


def _img(b64: str) -> str:
    return f'<img style="max-width:100%;border:1px solid #ddd;border-radius:6px;margin:8px 0" src="data:image/png;base64,{b64}" />'


def chart_score_distribution(scored: gpd.GeoDataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram (affected only, to avoid zero-inflation)
    affected = scored[scored["is_affected"]]["damage_score"]
    axes[0].hist(affected, bins=50, color="#e74c3c", edgecolor="white", alpha=0.85)
    axes[0].set_title("Damage score – affected buildings only")
    axes[0].set_xlabel("damage_score")
    axes[0].set_ylabel("Count")

    # Damage category bar chart
    cat_df = footprint_summary(scored)
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    axes[1].bar(cat_df["damage_category"], cat_df["n_buildings"],
                color=colors[:len(cat_df)], edgecolor="white")
    axes[1].set_title("Buildings by damage category")
    axes[1].set_xlabel("Coverage category")
    axes[1].set_ylabel("Number of buildings")
    for i, (n, pct) in enumerate(zip(cat_df["n_buildings"], cat_df["pct_buildings"])):
        axes[1].text(i, n + 10, f"{pct}%", ha="center", fontsize=9)

    fig.tight_layout()
    return _b64(fig)


def chart_spatial_map(
    scored_m: gpd.GeoDataFrame,
    enclosures_m: gpd.GeoDataFrame,
    title: str = "Damage score map",
    sample: int = 20_000,
) -> str:
    """Scatter map: footprint centroids coloured by damage score."""
    fp = scored_m.copy()
    if len(fp) > sample:
        fp = fp.sample(sample, random_state=42)

    centroids = fp.geometry.centroid
    xs = centroids.x.values
    ys = centroids.y.values
    scores = fp["damage_score"].values

    fig, ax = plt.subplots(figsize=(10, 8))

    # All footprints (grey)
    ax.scatter(xs[scores == 0], ys[scores == 0],
               c="#cccccc", s=1, alpha=0.3, rasterized=True, label="Not affected")

    # Affected footprints (coloured)
    mask = scores > 0
    sc = ax.scatter(xs[mask], ys[mask], c=scores[mask],
                    cmap="YlOrRd", s=4, alpha=0.8, vmin=0, vmax=1,
                    rasterized=True, label="Affected")
    plt.colorbar(sc, ax=ax, label="damage_score", fraction=0.03, pad=0.02)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Easting (m, EPSG:3857)")
    ax.set_ylabel("Northing (m, EPSG:3857)")
    ax.legend(markerscale=4, fontsize=9)
    fig.tight_layout()
    return _b64(fig)


def chart_region_grid(
    grid_regions: gpd.GeoDataFrame,
    metric: str = "mean_score",
    title: str = "Mean damage score per grid cell",
) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    grid_regions.plot(
        column=metric,
        ax=ax,
        cmap="YlOrRd",
        legend=True,
        legend_kwds={"label": metric, "fraction": 0.03, "pad": 0.02},
        vmin=0, vmax=1,
        edgecolor="none",
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    return _b64(fig)



# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
       max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #1a1a2e; background: #fafafa; }
h1 { color: #0f3460; font-size: 1.9em; }
.subtitle { color: #666; font-size: 0.9em; margin-bottom: 32px; }
h2 { color: #16213e; border-bottom: 3px solid #0f3460;
     padding-bottom: 6px; margin-top: 44px; }
h3 { color: #333; margin-top: 24px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; margin: 10px 0;
        background: white; border-radius: 6px; overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
th  { background: #0f3460; color: white; padding: 9px 12px; text-align: left; }
td  { padding: 7px 12px; border-bottom: 1px solid #eee; }
tr:last-child td { border-bottom: none; }
tr:nth-child(even) td { background: #f7f9fc; }
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }
.stat-card { background: white; border-radius: 8px; padding: 16px;
             box-shadow: 0 1px 4px rgba(0,0,0,0.1); text-align: center; }
.stat-card .value { font-size: 1.8em; font-weight: bold; color: #e74c3c; }
.stat-card .label { font-size: 0.8em; color: #666; margin-top: 4px; }
.note { color: #555; font-size: 0.88em; margin-bottom: 8px; }
"""


def _stat_card(value: str, label: str) -> str:
    return f'<div class="stat-card"><div class="value">{value}</div><div class="label">{label}</div></div>'


def build_html_report(
    scored_m: gpd.GeoDataFrame,
    grid_regions: gpd.GeoDataFrame,
    gpkg_name: str,
    grid_size_m: float,
) -> str:

    total    = len(scored_m)
    affected = int(scored_m["is_affected"].sum())
    mean_s   = scored_m["damage_score"].mean()

    cat_df = footprint_summary(scored_m)

    c_dist = chart_score_distribution(scored_m)
    c_map  = chart_spatial_map(scored_m, gpd.GeoDataFrame())
    c_grid = chart_region_grid(grid_regions)

    top_grid = (
        grid_regions[grid_regions["n_footprints"] > 0]
        .sort_values("mean_score", ascending=False)
        [["n_footprints", "n_affected", "pct_affected",
          "mean_score", "max_score", "area_damage_ratio"]]
        .head(20)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Building Tarp Damage Report – {gpkg_name}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>🏠 Building Tarp Damage Report</h1>
<p class="subtitle">Source: <code>{gpkg_name}</code></p>

<h2>Summary</h2>
<div class="stat-grid">
  {_stat_card(f"{total:,}", "Total footprints analysed")}
  {_stat_card(f"{affected:,}", "Buildings with tarps")}
  {_stat_card(f"{100*affected/total:.1f} %", "% with tarps")}
  {_stat_card(f"{mean_s:.4f}", "Mean damage score")}
</div>

<h2>1 · Footprint-level Analysis</h2>
<p class="note">
  <b>damage_score</b> = tarp area intersecting the footprint ÷ footprint area.
  Value in [0, 1]. Buildings with score = 0 have no tarp overlap.
</p>
{_img(c_dist)}

<h3>Tarp coverage category breakdown</h3>
{cat_df.to_html(index=False, border=0)}

<h2>2 · Spatial Map</h2>
<p class="note">
  Footprint centroids coloured by damage score (up to 20,000 sampled for rendering speed).
  Grey = no tarp detected. Yellow → Red = low → high tarp coverage.
</p>
{_img(c_map)}

<h2>3 · Region Analysis – Grid ({int(grid_size_m)} m cells)</h2>
<p class="note">
  Each cell aggregates all building footprints whose centroid falls inside it.
</p>
{_img(c_grid)}
<h3>Top 20 most affected grid cells</h3>
{top_grid.to_html(index=False, border=0)}

</body>
</html>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(
    gpkg_path: str | Path,
    output_dir: str | Path,
    footprints_layer: str = "footprints",
    tarps_layer: str = "tarps",
    min_area_m2: float = 5.0,
    grid_size_m: float = 500.0,
    metric_crs: str = "EPSG:3857",
) -> None:
    gpkg_path  = Path(gpkg_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    fp_m, tarps_m = load_gpkg_layers(
        gpkg_path, footprints_layer, tarps_layer, metric_crs
    )

    # ── 2. Footprint scoring ──────────────────────────────────────────────────
    print("\n[1/3] Scoring footprints …")
    scored_m = score_footprints(fp_m, tarps_m, min_footprint_area_m2=min_area_m2)
    print_summary(scored_m)

    # ── 3. Region analysis (grid) ─────────────────────────────────────────────
    print("[2/3] Aggregating by grid …")
    grid_regions = aggregate_by_grid(scored_m, cell_size_m=grid_size_m)
    print_region_summary(grid_regions, label=f"Grid ({int(grid_size_m)} m)")

    # ── 4. Save outputs ───────────────────────────────────────────────────────
    print("[3/3] Writing outputs …")

    # Scored footprints – convert back to WGS-84 for QGIS compatibility
    scored_wgs = scored_m.to_crs("EPSG:4326")
    # Remove numpy/object columns that can't be serialised to GPKG
    drop_cols = [c for c in scored_wgs.columns if scored_wgs[c].dtype == object
                 and c != "geometry"]
    fp_out_path = output_dir / "footprints_scored.gpkg"
    scored_wgs.drop(columns=drop_cols).to_file(fp_out_path, driver="GPKG", layer="footprints_scored")
    print(f"  footprints_scored.gpkg  ({len(scored_wgs):,} rows)")

    grid_wgs = grid_regions.to_crs("EPSG:4326")
    grid_path = output_dir / "regions_grid.gpkg"
    grid_wgs.to_file(grid_path, driver="GPKG", layer="regions_grid")
    print(f"  regions_grid.gpkg       ({len(grid_wgs):,} rows)")

    # CSV (no geometry)
    csv_cols = [c for c in scored_m.columns if c != "geometry"]
    scored_m[csv_cols].to_csv(output_dir / "footprint_scores.csv", index=False)
    print(f"  footprint_scores.csv    ({len(scored_m):,} rows)")

    # HTML report
    html = build_html_report(
        scored_m, grid_regions,
        gpkg_name=gpkg_path.name,
        grid_size_m=grid_size_m,
    )
    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"  report.html")

    print(f"\n✅  All outputs written to: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-footprint damage scoring + region analysis from a GPKG."
    )
    parser.add_argument(
        "--gpkg", default="geodata/us-fl-englewood-2023_03202.gpkg",
        help="Path to the GeoPackage file."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: geodata/results/<gpkg-stem>)."
    )
    parser.add_argument("--footprints-layer", default="footprints")
    parser.add_argument("--tarps-layer",      default="tarps")
    parser.add_argument(
        "--min-area", type=float, default=5.0,
        help="Minimum footprint area in m² to include (default: 5)."
    )
    parser.add_argument(
        "--grid-size", type=float, default=500.0,
        help="Grid cell size in metres for region analysis (default: 500)."
    )
    args = parser.parse_args()

    gpkg_path  = Path(args.gpkg)
    output_dir = Path(args.output_dir) if args.output_dir else \
                 Path("geodata/results") / gpkg_path.stem

    run(
        gpkg_path=gpkg_path,
        output_dir=output_dir,
        footprints_layer=args.footprints_layer,
        tarps_layer=args.tarps_layer,
        min_area_m2=args.min_area,
        grid_size_m=args.grid_size,
    )


if __name__ == "__main__":
    main()
