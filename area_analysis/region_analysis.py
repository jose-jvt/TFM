"""Region-level aggregation of per-footprint tarp damage scores.

Aggregation strategy:

**Grid regions** – a regular rectangular grid (configurable cell size in metres)
overlaid on the study area. Each cell aggregates the buildings whose centroid
falls inside it. Useful when no administrative boundaries are available.

Each region row contains:

    n_footprints      – total buildings in region
    n_affected        – buildings with damage_score > 0
    pct_affected      – n_affected / n_footprints × 100
    mean_score        – mean damage_score
    max_score         – max damage_score
    total_fp_area_m2  – sum of footprint areas (m²)
    total_tarp_area_m2 – sum of tarp area inside footprints (m²)
    area_damage_ratio – total_tarp_area / total_fp_area
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

_AGG_COLS = {
    "damage_score":       ["mean", "max", "std"],
    "footprint_area_m2":  "sum",
    "tarp_area_m2":  "sum",
    "is_affected":        ["sum", "count"],
}


def _compute_region_stats(grouped: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    agg = grouped.agg(
        n_footprints      =("damage_score",     "count"),
        n_affected        =("is_affected",       "sum"),
        mean_score        =("damage_score",       "mean"),
        max_score         =("damage_score",        "max"),
        std_score         =("damage_score",        "std"),
        total_fp_area_m2  =("footprint_area_m2",  "sum"),
        total_tarp_area_m2=("tarp_area_m2",       "sum"),
    ).reset_index()

    agg["pct_affected"]     = (agg["n_affected"] / agg["n_footprints"] * 100).round(1)
    agg["area_damage_ratio"] = (
        agg["total_tarp_area_m2"] / agg["total_fp_area_m2"].replace(0, np.nan)
    ).fillna(0.0).clip(0, 1).round(4)
    agg["mean_score"]       = agg["mean_score"].round(4)
    agg["max_score"]        = agg["max_score"].round(4)
    agg["std_score"]        = agg["std_score"].fillna(0).round(4)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 – Enclosure regions
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_by_enclosures(
    scored_m: gpd.GeoDataFrame,
    enclosures_m: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """One region per enclosure polygon.

    Footprints whose centroid falls inside an enclosure (or that intersect it)
    are attributed to that enclosure.

    Parameters
    ----------
    scored_m :
        Output of :func:`building_analysis.gpkg_scoring.score_footprints`
        in metric CRS.
    enclosures_m :
        Enclosures GeoDataFrame in the same metric CRS.

    Returns
    -------
    GeoDataFrame with enclosure geometry + aggregated stats columns.
    """
    enc = enclosures_m.copy().reset_index(drop=True)
    enc["_enc_region_id"] = enc.index

    # Use centroid join: attribute each footprint to the enclosure that
    # contains its centroid (faster and avoids double-counting).
    fp_centroids = scored_m.copy()
    fp_centroids["geometry"] = scored_m.geometry.centroid

    joined = gpd.sjoin(
        fp_centroids[["geometry", "damage_score", "footprint_area_m2",
                       "tarp_area_m2", "is_affected"]],
        enc[["_enc_region_id", "geometry", "score"]],
        how="inner",
        predicate="within",
    )

    if joined.empty:
        # Fallback: use 'intersects' if no centroid lands inside any enclosure
        joined = gpd.sjoin(
            fp_centroids[["geometry", "damage_score", "footprint_area_m2",
                           "tarp_area_m2", "is_affected"]],
            enc[["_enc_region_id", "geometry", "score"]],
            how="inner",
            predicate="intersects",
        )

    stats = _compute_region_stats(joined.groupby("_enc_region_id"))

    # Re-attach enclosure geometry
    result = enc.merge(stats, on="_enc_region_id", how="left")
    result["n_footprints"]      = result["n_footprints"].fillna(0).astype(int)
    result["n_affected"]        = result["n_affected"].fillna(0).astype(int)
    result["pct_affected"]      = result["pct_affected"].fillna(0.0)
    result["mean_score"]        = result["mean_score"].fillna(0.0)
    result["max_score"]         = result["max_score"].fillna(0.0)
    result["area_damage_ratio"] = result["area_damage_ratio"].fillna(0.0)

    return result.drop(columns=["_enc_region_id"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 – Regular grid regions
# ─────────────────────────────────────────────────────────────────────────────

def _make_grid(
    bounds: tuple[float, float, float, float],
    cell_size: float,
) -> gpd.GeoDataFrame:
    """Build a regular rectangular grid over *bounds* (in degrees or metres).

    Parameters
    ----------
    bounds :
        (minx, miny, maxx, maxy)
    cell_size :
        Cell side length in the same units as *bounds*.
    """
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)

    cells = [
        box(x, y, x + cell_size, y + cell_size)
        for x in xs
        for y in ys
    ]
    gdf = gpd.GeoDataFrame(geometry=cells)
    gdf["_grid_id"] = gdf.index
    return gdf


def aggregate_by_grid(
    scored_m: gpd.GeoDataFrame,
    cell_size_m: float = 500.0,
) -> gpd.GeoDataFrame:
    """One region per grid cell (cell_size_m × cell_size_m metres).

    Parameters
    ----------
    scored_m :
        Scored footprints in metric CRS.
    cell_size_m :
        Grid cell side length in metres (default 500 m).

    Returns
    -------
    GeoDataFrame with grid cell geometry + aggregated stats.
    Only cells containing at least one footprint are returned.
    """
    crs = scored_m.crs
    bounds = scored_m.total_bounds   # (minx, miny, maxx, maxy)

    grid = _make_grid(bounds, cell_size_m)
    grid.crs = crs

    fp_centroids = scored_m.copy()
    fp_centroids["geometry"] = scored_m.geometry.centroid

    joined = gpd.sjoin(
        fp_centroids[["geometry", "damage_score", "footprint_area_m2",
                       "tarp_area_m2", "is_affected"]],
        grid[["_grid_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    stats = _compute_region_stats(joined.groupby("_grid_id"))
    result = grid.merge(stats, on="_grid_id", how="inner")
    return result.drop(columns=["_grid_id"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Print region summaries
# ─────────────────────────────────────────────────────────────────────────────

def print_region_summary(regions: gpd.GeoDataFrame, label: str = "Region") -> None:
    top = (
        regions[regions["n_footprints"] > 0]
        .sort_values("mean_score", ascending=False)
        .head(10)
    )
    print(f"\n{'─'*60}")
    print(f"  {label} analysis — top 10 most affected regions")
    print(f"{'─'*60}")
    cols = ["n_footprints", "n_affected", "pct_affected",
            "mean_score", "max_score", "area_damage_ratio"]
    cols = [c for c in cols if c in top.columns]
    print(top[cols].to_string(index=False))
    print(f"{'─'*60}\n")
