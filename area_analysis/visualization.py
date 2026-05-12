"""Visualization utilities for area-level damage analysis."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns


_DAMAGE_CMAP = "RdYlGn_r"  # green=low, red=high damage


def plot_temporal_trends(yearly: pd.DataFrame, output_path: str | Path):
    """Line chart: mean damage score over years, with ±p90 band."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(yearly["year"], yearly["mean_score"], marker="o", linewidth=2, label="Mean score")
    ax.fill_between(yearly["year"], 0, yearly["p90_score"], alpha=0.15, label="P90 band")
    ax.set_title("Mean Tarp Coverage Score by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_ylim(0, 1)

    ax = axes[1]
    bars = ax.bar(yearly["year"], yearly["num_buildings"], color="steelblue", edgecolor="white")
    ax2 = ax.twinx()
    ax2.plot(yearly["year"], yearly["buildings_30pct"] / yearly["num_buildings"].clip(lower=1),
             color="orange", marker="s", linewidth=2, label="% buildings >30%")
    ax2.set_ylabel("Fraction of buildings with >30% coverage")
    ax.set_title("Buildings Analyzed per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of buildings")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved temporal trends to {output_path}")


def plot_region_comparison(regional: pd.DataFrame, output_path: str | Path):
    """Horizontal bar chart comparing mean score per region."""
    regional_sorted = regional.sort_values("mean_score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(regional_sorted) * 0.5)))
    colors = plt.cm.RdYlGn_r(regional_sorted["mean_score"].values)
    bars = ax.barh(regional_sorted["region"], regional_sorted["mean_score"], color=colors, edgecolor="white")

    for bar, n in zip(bars, regional_sorted["num_buildings"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"n={int(n)}", va="center", fontsize=9)

    ax.set_xlabel("Mean Tarp Coverage Score")
    ax.set_title("Damage Score by Region")
    ax.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved region comparison to {output_path}")


def plot_score_heatmap(tile_gdf: gpd.GeoDataFrame, output_path: str | Path):
    """Choropleth map of mean damage score per tile."""
    if tile_gdf.empty or "mean_score" not in tile_gdf.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    tile_gdf.plot(
        column="mean_score",
        cmap=_DAMAGE_CMAP,
        vmin=0,
        vmax=1,
        legend=True,
        legend_kwds={"label": "Mean Tarp Coverage Score", "orientation": "vertical"},
        ax=ax,
        edgecolor="grey",
        linewidth=0.3,
    )
    ax.set_title("Spatial Distribution of Damage Scores")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved score heatmap to {output_path}")


def plot_score_distribution_by_year(df: pd.DataFrame, output_path: str | Path):
    """Box plot of building scores grouped by year."""
    df_clean = df.dropna(subset=["year"])
    df_clean["year"] = df_clean["year"].astype(int)

    fig, ax = plt.subplots(figsize=(12, 5))
    years = sorted(df_clean["year"].unique())
    data_by_year = [df_clean[df_clean["year"] == y]["score"].values for y in years]

    ax.boxplot(data_by_year, labels=years, patch_artist=True,
               boxprops=dict(facecolor="lightblue", color="navy"),
               medianprops=dict(color="red", linewidth=2))
    ax.set_xlabel("Year")
    ax.set_ylabel("Building Score")
    ax.set_title("Distribution of Tarp Coverage Scores by Year")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
