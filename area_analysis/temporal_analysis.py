"""Aggregate damage scores by year, region, or custom zone polygon.

Entry point for area-level impact analysis after building scores have been
computed by building_analysis/report.py.

Usage:
    python -m area_analysis.temporal_analysis \
        --scores outputs/building_scores/building_scores.csv \
        --images data/images \
        --zone "POLYGON((lon1 lat1, lon2 lat2, ...))" \
        --output outputs/area_analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from area_analysis.tile_grouping import (
    build_tile_geodataframe,
    filter_tiles_by_polygon,
    aggregate_zone,
)
from area_analysis.visualization import (
    plot_temporal_trends,
    plot_region_comparison,
    plot_score_heatmap,
)


def load_scores_with_metadata(scores_csv: str | Path, filename_pattern: str) -> pd.DataFrame:
    """Load building scores and add year/region/tile columns from filename."""
    import re

    df = pd.read_csv(scores_csv)
    pattern = re.compile(filename_pattern, re.IGNORECASE)

    def _parse(fname):
        stem = Path(fname).stem
        m = pattern.search(stem)
        if m:
            g = m.groupdict()
            return pd.Series(
                {
                    "year": int(g["year"]) if g.get("year") else None,
                    "region": g.get("region", "unknown"),
                    "tile": g.get("tile", stem),
                }
            )
        return pd.Series({"year": None, "region": "unknown", "tile": stem})

    meta = df["image_filename"].apply(_parse)
    return pd.concat([df, meta], axis=1)


def aggregate_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted mean score per year (weighted by number of buildings)."""
    import numpy as np

    def _wavg(g):
        weights = g["building_area_px"] if "building_area_px" in g.columns else pd.Series(np.ones(len(g)))
        return np.average(g["score"], weights=weights.clip(lower=1))

    yearly = (
        df.groupby("year")
        .apply(
            lambda g: pd.Series(
                {
                    "num_buildings": len(g),
                    "num_tiles": g["image_filename"].nunique(),
                    "mean_score": g["score"].mean(),
                    "median_score": g["score"].median(),
                    "p90_score": g["score"].quantile(0.9),
                    "buildings_30pct": (g["score"] > 0.3).sum(),
                    "buildings_60pct": (g["score"] > 0.6).sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return yearly.dropna(subset=["year"])


def aggregate_by_region(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("region")
        .apply(
            lambda g: pd.Series(
                {
                    "num_buildings": len(g),
                    "num_tiles": g["image_filename"].nunique(),
                    "mean_score": g["score"].mean(),
                    "median_score": g["score"].median(),
                    "p90_score": g["score"].quantile(0.9),
                    "years": sorted(g["year"].dropna().unique().tolist()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def run_analysis(
    scores_csv: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    filename_pattern: str,
    zone_wkt: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_scores_with_metadata(scores_csv, filename_pattern)

    if zone_wkt:
        print("Filtering tiles by zone polygon...")
        tile_gdf = build_tile_geodataframe(images_dir, scores_csv)
        filtered = filter_tiles_by_polygon(tile_gdf, zone_wkt)
        zone_filenames = set(filtered["image_filename"])
        df = df[df["image_filename"].isin(zone_filenames)]
        zone_stats = aggregate_zone(filtered)
        with open(output_dir / "zone_summary.json", "w") as f:
            json.dump(zone_stats, f, indent=2)
        print(f"Zone stats: {zone_stats}")

    # Yearly analysis
    yearly = aggregate_by_year(df)
    yearly.to_csv(output_dir / "yearly_aggregation.csv", index=False)
    if not yearly.empty:
        plot_temporal_trends(yearly, output_dir / "temporal_trends.png")

    # Regional analysis
    regional = aggregate_by_region(df)
    regional.to_csv(output_dir / "regional_aggregation.csv", index=False)
    if not regional.empty:
        plot_region_comparison(regional, output_dir / "region_comparison.png")

    # Heatmap if geo data available
    try:
        tile_gdf = build_tile_geodataframe(images_dir, scores_csv)
        if zone_wkt:
            tile_gdf = filter_tiles_by_polygon(tile_gdf, zone_wkt)
        if not tile_gdf.empty:
            plot_score_heatmap(tile_gdf, output_dir / "score_heatmap.png")
    except Exception as e:
        print(f"Skipping heatmap (metadata not available): {e}")

    print(f"\nArea analysis saved to {output_dir}")
    return {"yearly": yearly, "regional": regional}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="Path to building_scores.csv")
    parser.add_argument("--images", required=True, help="Path to images dir (for metadata JSON)")
    parser.add_argument("--zone", default=None, help="WKT polygon to filter tiles (EPSG:4326)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_analysis(
        scores_csv=args.scores,
        images_dir=args.images,
        output_dir=args.output,
        filename_pattern=cfg["data"]["filename_pattern"],
        zone_wkt=args.zone,
    )


if __name__ == "__main__":
    main()
