from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import scanpy as sc
from scipy.spatial import KDTree

from gsMap.config import VisualizeConfig


def load_ldsc(ldsc_input_file):
    ldsc = pd.read_csv(
        ldsc_input_file,
        compression="gzip",
        dtype={"spot": str, "p": float},
        index_col="spot",
        usecols=["spot", "p"],
    )
    ldsc["logp"] = -np.log10(ldsc.p)
    return ldsc


# %%
def load_st_coord(adata, feature_series: pd.Series, annotation):
    spot_name = adata.obs_names.to_list()
    assert "spatial" in adata.obsm.keys(), "spatial coordinates are not found in adata.obsm"

    # to DataFrame
    space_coord = adata.obsm["spatial"]
    if isinstance(space_coord, np.ndarray):
        space_coord = pd.DataFrame(space_coord, columns=["sx", "sy"], index=spot_name)
    else:
        space_coord = pd.DataFrame(space_coord.values, columns=["sx", "sy"], index=spot_name)

    space_coord = space_coord[space_coord.index.isin(feature_series.index)]
    space_coord_concat = pd.concat([space_coord.loc[feature_series.index], feature_series], axis=1)
    space_coord_concat.head()
    if annotation is not None:
        annotation = pd.Series(
            adata.obs[annotation].values, index=adata.obs_names, name="annotation"
        )
        space_coord_concat = pd.concat([space_coord_concat, annotation], axis=1)
    return space_coord_concat


def estimate_point_size_for_plot(coordinates, DEFAULT_PIXEL_WIDTH=1000):
    tree = KDTree(coordinates)
    distances, _ = tree.query(coordinates, k=2)
    avg_min_distance = np.mean(distances[:, 1])
    # get the width and height of the plot
    width = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
    height = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])

    scale_factor = DEFAULT_PIXEL_WIDTH / max(width, height)
    pixel_width = width * scale_factor
    pixel_height = height * scale_factor

    point_size = np.ceil(avg_min_distance * scale_factor)
    return (pixel_width, pixel_height), point_size


def draw_scatter(
    space_coord_concat,
    title=None,
    fig_style: Literal["dark", "light"] = "light",
    point_size: int = None,
    width=800,
    height=600,
    annotation=None,
    color_by="logp",
):
    # Set theme based on fig_style
    if fig_style == "dark":
        px.defaults.template = "plotly_dark"
    else:
        px.defaults.template = "plotly_white"

    custom_color_scale = [
        (1, "#d73027"),  # Red
        (7 / 8, "#f46d43"),  # Red-Orange
        (6 / 8, "#fdae61"),  # Orange
        (5 / 8, "#fee090"),  # Light Orange
        (4 / 8, "#e0f3f8"),  # Light Blue
        (3 / 8, "#abd9e9"),  # Sky Blue
        (2 / 8, "#74add1"),  # Medium Blue
        (1 / 8, "#4575b4"),  # Dark Blue
        (0, "#313695"),  # Deep Blue
    ]
    custom_color_scale.reverse()

    # Create the scatter plot
    fig = px.scatter(
        space_coord_concat,
        x="sx",
        y="sy",
        color=color_by,
        symbol="annotation" if annotation is not None else None,
        title=title,
        color_continuous_scale=custom_color_scale,
        range_color=[0, max(space_coord_concat[color_by])],
    )

    # Update marker size if specified
    if point_size is not None:
        fig.update_traces(marker=dict(size=point_size, symbol="circle"))

    # Update layout for figure size
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
    )

    # Adjusting the legend
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.0,
            font=dict(
                size=10,
            ),
        )
    )

    # Update colorbar to be at the bottom and horizontal
    fig.update_layout(
        coloraxis_colorbar=dict(
            orientation="h",  # Make the colorbar horizontal
            x=0.5,  # Center the colorbar horizontally
            y=-0.0,  # Position below the plot
            xanchor="center",  # Anchor the colorbar at the center
            yanchor="top",  # Anchor the colorbar at the top to keep it just below the plot
            len=0.75,  # Length of the colorbar relative to the plot width
            title=dict(
                text="-log10(p)" if color_by == "logp" else color_by,  # Colorbar title
                side="top",  # Place the title at the top of the colorbar
            ),
        )
    )
    # Remove gridlines, axis labels, and ticks
    fig.update_xaxes(
        showgrid=False,  # Hide x-axis gridlines
        zeroline=False,  # Hide x-axis zero line
        showticklabels=False,  # Hide x-axis tick labels
        title=None,  # Remove x-axis title
        scaleanchor="y",  # Link the x-axis scale to the y-axis scale
    )

    fig.update_yaxes(
        showgrid=False,  # Hide y-axis gridlines
        zeroline=False,  # Hide y-axis zero line
        showticklabels=False,  # Hide y-axis tick labels
        title=None,  # Remove y-axis title
    )

    # Adjust margins to ensure no clipping and equal axis ratio
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=10),  # Adjust margins to prevent clipping
        height=width,  # Ensure the figure height matches the width for equal axis ratio
    )

    # Adjust the title location and font size
    fig.update_layout(
        title=dict(
            y=0.98,
            x=0.5,  # Center the title horizontally
            xanchor="center",  # Anchor the title at the center
            yanchor="top",  # Anchor the title at the top
            font=dict(
                size=20  # Increase the title font size
            ),
        )
    )

    return fig


def run_Visualize(config: VisualizeConfig):
    print(f"------Loading LDSC results of {config.ldsc_save_dir}...")
    ldsc = load_ldsc(
        ldsc_input_file=Path(config.ldsc_save_dir)
        / f"{config.sample_name}_{config.trait_name}.csv.gz"
    )

    print(f"------Loading ST data of {config.sample_name}...")
    adata = sc.read_h5ad(f"{config.hdf5_with_latent_path}")

    space_coord_concat = load_st_coord(adata, ldsc, annotation=config.annotation)
    fig = draw_scatter(
        space_coord_concat,
        title=config.fig_title,
        fig_style=config.fig_style,
        point_size=config.point_size,
        width=config.fig_width,
        height=config.fig_height,
        annotation=config.annotation,
    )

    # Visualization
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    output_file_html = output_dir / f"{config.sample_name}_{config.trait_name}.html"
    output_file_pdf = output_dir / f"{config.sample_name}_{config.trait_name}.pdf"
    output_file_csv = output_dir / f"{config.sample_name}_{config.trait_name}.csv"

    fig.write_html(str(output_file_html))
    fig.write_image(str(output_file_pdf))
    space_coord_concat.to_csv(str(output_file_csv))

    print(
        f"------The visualization result is saved in a html file: {output_file_html} which can interactively viewed in a web browser and a pdf file: {output_file_pdf}."
    )
    print(f"------The visualization data is saved in a csv file: {output_file_csv}.")
