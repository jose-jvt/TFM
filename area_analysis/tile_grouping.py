"""Group tiles by a geographic polygon and aggregate building scores.

Tiles are matched to a polygon by reading the geographic bounds from the
image metadata file (JSON sidecar next to each image).

Metadata JSON expected schema (at minimum):
    {
        "bbox": [min_lon, min_lat, max_lon, max_lat],   # EPSG:4326
        ...
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely import wkt as shapely_wkt


def load_tile_bounds(metadata_file: str | Path) -> tuple[float, float, float, float]:
    """Read tile bounding box from a JSON metadata sidecar file.

    Returns (min_lon, min_lat, max_lon, max_lat).
    """
    with open(metadata_file) as f:
        meta = json.load(f)

    if "bbox" in meta:
        return tuple(meta["bbox"])

    # Fallback: look for common alternative keys
    for key in ("bounds", "extent", "bounding_box"):
        if key in meta:
            b = meta[key]
            if isinstance(b, dict):
                return b["min_lon"], b["min_lat"], b["max_lon"], b["max_lat"]
            return tuple(b)

    raise KeyError(f"No bounding box found in metadata file: {metadata_file}")


def build_tile_geodataframe(
    images_dir: str | Path,
    building_scores_csv: str | Path,
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame with one row per image tile, merged with mean score.

    The metadata JSON must live alongside the image with the same stem:
        images_dir/tile_0001.png  →  images_dir/tile_0001.json
    """
    images_dir = Path(images_dir)
    scores_df = pd.read_csv(building_scores_csv)
    image_summary = scores_df.groupby("image_filename")["score"].agg(
        ["mean", "max", "count"]
    ).reset_index()
    image_summary.columns = ["image_filename", "mean_score", "max_score", "num_buildings"]

    records = []
    for _, row in image_summary.iterrows():
        fname = row["image_filename"]
        meta_path = images_dir / (Path(fname).stem + ".json")
        if not meta_path.exists():
            continue
        try:
            bounds = load_tile_bounds(meta_path)
        except (KeyError, json.JSONDecodeError):
            continue
        geometry = box(*bounds)
        records.append(
            {
                "image_filename": fname,
                "mean_score": row["mean_score"],
                "max_score": row["max_score"],
                "num_buildings": row["num_buildings"],
                "geometry": geometry,
            }
        )

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


def filter_tiles_by_polygon(
    tile_gdf: gpd.GeoDataFrame,
    zone_polygon_wkt: str,
    zone_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Return tiles whose centroid or geometry intersects the zone polygon.

    Args:
        tile_gdf:          GeoDataFrame of tiles (EPSG:4326).
        zone_polygon_wkt:  WKT string defining the zone of interest.
        zone_crs:          CRS of the zone WKT (default EPSG:4326).
    """
    zone = shapely_wkt.loads(zone_polygon_wkt)
    zone_gdf = gpd.GeoDataFrame(geometry=[zone], crs=zone_crs)

    if tile_gdf.crs != zone_gdf.crs:
        tile_gdf = tile_gdf.to_crs(zone_gdf.crs)

    within = gpd.sjoin(tile_gdf, zone_gdf, how="inner", predicate="intersects")
    return within.drop(columns=["index_right"], errors="ignore")


def aggregate_zone(filtered_tiles: gpd.GeoDataFrame) -> dict:
    """Compute summary statistics for a set of tiles."""
    if filtered_tiles.empty:
        return {}
    return {
        "num_tiles": len(filtered_tiles),
        "num_buildings": int(filtered_tiles["num_buildings"].sum()),
        "mean_score": float(filtered_tiles["mean_score"].mean()),
        "max_score": float(filtered_tiles["max_score"].max()),
        "weighted_mean_score": float(
            np.average(
                filtered_tiles["mean_score"],
                weights=filtered_tiles["num_buildings"].clip(lower=1),
            )
        ),
    }
