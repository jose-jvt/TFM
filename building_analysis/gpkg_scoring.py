"""Vector-based per-footprint tarp damage scoring from a GeoPackage.

Reads the ``footprints`` and ``tarps`` layers of a GPKG produced by the
segmentation pipeline and computes, for every building footprint:

    damage_score  = intersection_area(footprint ∩ tarps) / footprint_area

Both geometries are reprojected to a metric CRS (default EPSG:3857) before any
area calculation so results are in m².

The algorithm uses a two-step spatial approach for efficiency:
  1. ``gpd.sjoin`` with an STR-tree to find candidate (footprint, tarp) pairs.
  2. Vectorised ``shapely.intersection`` on the candidate pairs only.

For a GPKG with ~100 k footprints and ~26 k tarp polygons this typically runs
in under 60 s on a modern laptop CPU.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_gpkg_layers(
    gpkg_path: str | Path,
    footprints_layer: str = "footprints",
    tarps_layer: str = "tarps",
    metric_crs: str = "EPSG:3857",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load footprints and tarps layers, reproject to *metric_crs*.

    Returns
    -------
    footprints_m, tarps_m : GeoDataFrames in metric CRS
    """
    gpkg_path = Path(gpkg_path)
    print(f"Loading '{footprints_layer}' …")
    fp = gpd.read_file(gpkg_path, layer=footprints_layer)
    print(f"  {len(fp):,} footprints  (native CRS: {fp.crs})")

    print(f"Loading '{tarps_layer}' …")
    tarps = gpd.read_file(gpkg_path, layer=tarps_layer)
    print(f"  {len(tarps):,} tarp polygons  (native CRS: {tarps.crs})")

    # Ensure geometries are valid
    fp    = fp[~fp.geometry.isna()].copy()
    tarps = tarps[~tarps.geometry.isna()].copy()
    fp.geometry    = fp.geometry.buffer(0)
    tarps.geometry = tarps.geometry.buffer(0)

    fp_m    = fp.to_crs(metric_crs)
    tarps_m = tarps.to_crs(metric_crs)

    return fp_m, tarps_m


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_footprints(
    footprints_m: gpd.GeoDataFrame,
    tarps_m: gpd.GeoDataFrame,
    min_footprint_area_m2: float = 5.0,
    chunk_size: int = 5_000,
) -> gpd.GeoDataFrame:
    """Compute per-footprint tarp damage scores.

    Parameters
    ----------
    footprints_m :
        Footprint GeoDataFrame in a metric CRS.
    tarps_m :
        Tarp polygons GeoDataFrame in the same metric CRS.
    min_footprint_area_m2 :
        Footprints smaller than this (m²) are excluded (noise filter).
    chunk_size :
        Number of footprints processed per batch (controls memory peak).

    Returns
    -------
    GeoDataFrame identical to *footprints_m* with extra columns:

    ``footprint_area_m2`` – footprint area in m²
    ``tarp_area_m2``      – tarp area inside the footprint (m²)
    ``damage_score``      – tarp_area / footprint_area  ∈ [0, 1]
    ``is_affected``       – True when damage_score > 0
    ``n_tarps``           – number of distinct tarp polygons touching the footprint
    """
    fp = footprints_m.copy().reset_index(drop=True)
    fp["_fp_idx"] = fp.index
    fp["footprint_area_m2"] = fp.geometry.area

    # Drop tiny footprints (GPS noise, slivers)
    fp = fp[fp["footprint_area_m2"] >= min_footprint_area_m2].copy()
    print(f"  {len(fp):,} footprints after area filter (≥ {min_footprint_area_m2} m²)")

    tarps = tarps_m.reset_index(drop=True).copy()
    tarps["_tarp_idx"] = tarps.index

    # ── Step 1: candidate pairs via spatial join ──────────────────────────────
    print("  Spatial join (candidate pairs) …")
    fp_geom    = fp[["_fp_idx", "geometry"]].copy()
    tarps_geom = tarps[["_tarp_idx", "geometry"]].copy()

    joined = gpd.sjoin(fp_geom, tarps_geom, how="inner", predicate="intersects")
    print(f"  {len(joined):,} candidate pairs")

    if joined.empty:
        fp["tarp_area_m2"] = 0.0
        fp["damage_score"] = 0.0
        fp["is_affected"]  = False
        fp["n_tarps"]      = 0
        return fp.drop(columns=["_fp_idx"])

    # ── Step 2: vectorised intersection on candidate pairs ────────────────────
    print("  Computing intersections (vectorised shapely 2.x) …")
    fp_geoms_idx   = joined["_fp_idx"].values
    tarp_geoms_idx = joined["index_right"].values

    fp_geoms_arr   = fp.loc[fp_geoms_idx,    "geometry"].values
    tarp_geoms_arr = tarps.loc[tarp_geoms_idx, "geometry"].values

    inter_geoms = shapely.intersection(fp_geoms_arr, tarp_geoms_arr)
    inter_areas = shapely.area(inter_geoms)

    pair_df = pd.DataFrame({
        "_fp_idx":   fp_geoms_idx,
        "_tarp_idx": tarp_geoms_idx,
        "inter_area": inter_areas,
    })
    # Drop pairs with zero actual intersection (bbox touched but no real overlap)
    pair_df = pair_df[pair_df["inter_area"] > 0]

    # ── Step 3: aggregate per footprint ──────────────────────────────────────
    agg = pair_df.groupby("_fp_idx").agg(
        tarp_area_m2=("inter_area",  "sum"),
        n_tarps     =("_tarp_idx",   "nunique"),
    )

    fp = fp.join(agg, on="_fp_idx")
    fp["tarp_area_m2"] = fp["tarp_area_m2"].fillna(0.0)
    fp["n_tarps"]      = fp["n_tarps"].fillna(0).astype(int)
    fp["damage_score"] = (
        fp["tarp_area_m2"] / fp["footprint_area_m2"]
    ).clip(0.0, 1.0)
    fp["is_affected"]  = fp["damage_score"] > 0.0

    return fp.drop(columns=["_fp_idx"])


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

DAMAGE_BINS   = [0.0, 0.10, 0.30, 0.60, 1.001]
DAMAGE_LABELS = ["<10 %", "10–30 %", "30–60 %", ">60 %"]


def footprint_summary(scored: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return a summary DataFrame of tarp damage categories."""
    cat = pd.cut(scored["damage_score"], bins=DAMAGE_BINS, labels=DAMAGE_LABELS,
                 right=False, include_lowest=True)
    counts = cat.value_counts().sort_index()
    total  = len(scored)
    summary = pd.DataFrame({
        "damage_category": counts.index,
        "n_buildings":     counts.values,
        "pct_buildings":   (counts.values / total * 100).round(1),
    })
    return summary


def print_summary(scored: gpd.GeoDataFrame) -> None:
    total    = len(scored)
    affected = scored["is_affected"].sum()
    mean_s   = scored["damage_score"].mean()
    median_s = scored["damage_score"].median()
    print(f"\n{'─'*50}")
    print(f"  Footprints analysed : {total:,}")
    print(f"  Affected (score > 0): {affected:,}  ({100*affected/total:.1f} %)")
    print(f"  Mean damage score   : {mean_s:.4f}")
    print(f"  Median damage score : {median_s:.4f}")
    print(footprint_summary(scored).to_string(index=False))
    print(f"{'─'*50}\n")
