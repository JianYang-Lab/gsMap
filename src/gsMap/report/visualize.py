import logging
import math
import tempfile
import warnings
from pathlib import Path

import distinctipy
import matplotlib
import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import KDTree

from gsMap.config.gwas import trait_label

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class MultiSampleSpatialRasterCache:
    """Cache coordinate-to-pixel mappings for fast multi-sample trait overviews."""

    def __init__(
        self,
        obs_data: pd.DataFrame,
        sample_name_list: list[str],
        resolution: int = 224,
    ) -> None:
        required = {"sample_name", "sx", "sy"}
        missing = sorted(required - set(obs_data.columns))
        if missing:
            raise KeyError(f"Missing spatial overview columns: {missing}")

        self.sample_name_list = [str(sample) for sample in sample_name_list]
        self.resolution = max(32, int(resolution))
        grouped_positions = {
            str(sample): np.asarray(positions, dtype=np.int64)
            for sample, positions in obs_data.groupby(
                "sample_name", sort=False, observed=True
            ).indices.items()
        }
        coordinates = obs_data[["sx", "sy"]].to_numpy(dtype=np.float64, copy=False)
        self.samples: dict[str, dict[str, np.ndarray | int]] = {}

        for sample_name in self.sample_name_list:
            positions = grouped_positions.get(sample_name, np.empty(0, dtype=np.int64))
            sample_coordinates = coordinates[positions]
            finite = np.isfinite(sample_coordinates).all(axis=1)
            positions = positions[finite]
            sample_coordinates = sample_coordinates[finite]
            if positions.size == 0:
                self.samples[sample_name] = {
                    "positions": positions,
                    "pixel_indices": np.empty(0, dtype=np.int32),
                    "height": 1,
                    "width": 1,
                }
                continue

            x = sample_coordinates[:, 0]
            y = sample_coordinates[:, 1]
            x_range = float(np.ptp(x))
            y_range = float(np.ptp(y))
            if x_range <= 0 and y_range <= 0:
                width = height = 1
            elif x_range >= y_range:
                width = self.resolution
                height = max(
                    1, round(self.resolution * y_range / max(x_range, np.finfo(float).eps))
                )
            else:
                height = self.resolution
                width = max(
                    1, round(self.resolution * x_range / max(y_range, np.finfo(float).eps))
                )

            if width == 1:
                x_pixel = np.zeros(x.size, dtype=np.int32)
            else:
                x_pixel = np.rint(
                    (x - x.min()) / max(x_range, np.finfo(float).eps) * (width - 1)
                ).astype(np.int32)
            if height == 1:
                y_pixel = np.zeros(y.size, dtype=np.int32)
            else:
                y_pixel = np.rint(
                    (y - y.min()) / max(y_range, np.finfo(float).eps) * (height - 1)
                ).astype(np.int32)

            self.samples[sample_name] = {
                "positions": positions,
                "pixel_indices": y_pixel * width + x_pixel,
                "height": height,
                "width": width,
            }

    def rasterize(self, sample_name: str, values: np.ndarray) -> np.ndarray:
        """Aggregate the strongest finite value in every display pixel."""

        sample = self.samples[str(sample_name)]
        positions = sample["positions"]
        height = int(sample["height"])
        width = int(sample["width"])
        if positions.size == 0:
            return np.full((height, width), np.nan, dtype=np.float32)

        sample_values = np.asarray(values[positions], dtype=np.float32)
        finite = np.isfinite(sample_values)
        raster = np.full(height * width, -np.inf, dtype=np.float32)
        if finite.any():
            np.maximum.at(raster, sample["pixel_indices"][finite], sample_values[finite])
        raster[~np.isfinite(raster)] = np.nan
        return raster.reshape(height, width)


class SpatialOverviewTileStore:
    """Encode every sample/trait tissue raster once for both overview directions."""

    def __init__(
        self,
        raster_cache: MultiSampleSpatialRasterCache,
        trait_names: list[str],
        *,
        plot_origin: str = "upper",
        temporary_dir: Path | None = None,
        max_ram_bytes: int = 512 * 1024**2,
    ) -> None:
        self.raster_cache = raster_cache
        self.sample_names = raster_cache.sample_name_list
        self.trait_names = [str(trait) for trait in trait_names]
        self.resolution = raster_cache.resolution
        self.plot_origin = plot_origin
        self.color_maxima = np.ones(len(self.trait_names), dtype=np.float32)
        self._temporary_path: Path | None = None

        shape = (
            len(self.sample_names),
            len(self.trait_names),
            self.resolution,
            self.resolution,
        )
        required_bytes = int(np.prod(shape, dtype=np.int64))
        if required_bytes <= max_ram_bytes:
            self.tiles = np.zeros(shape, dtype=np.uint8)
            self.storage_mode = "ram"
        else:
            temporary_dir = Path(temporary_dir or Path.cwd())
            temporary_dir.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                prefix=".gsmap_spatial_overview_",
                suffix=".uint8.dat",
                dir=temporary_dir,
                delete=False,
            )
            handle.close()
            self._temporary_path = Path(handle.name)
            self.tiles = np.memmap(self._temporary_path, mode="w+", dtype=np.uint8, shape=shape)
            self.tiles[:] = 0
            self.storage_mode = "memmap"
        logger.info(
            "Spatial overview tile cache: "
            f"{required_bytes / 1024**2:,.1f} MiB via {self.storage_mode}"
        )

    def encode_trait(self, trait_index: int, values: np.ndarray, color_maximum: float) -> None:
        """Rasterize and quantize one trait for every sample."""

        maximum = float(color_maximum)
        if not np.isfinite(maximum) or maximum <= 0:
            maximum = 1.0
        self.color_maxima[trait_index] = maximum

        for sample_index, sample_name in enumerate(self.sample_names):
            raster = self.raster_cache.rasterize(sample_name, values)
            if self.plot_origin == "lower":
                raster = np.flipud(raster)
            height, width = raster.shape
            y_offset = (self.resolution - height) // 2
            x_offset = (self.resolution - width) // 2
            target = self.tiles[
                sample_index,
                trait_index,
                y_offset : y_offset + height,
                x_offset : x_offset + width,
            ]
            finite = np.isfinite(raster)
            if finite.any():
                normalized = np.clip(raster[finite] / maximum, 0, 1)
                target[finite] = 1 + np.rint(normalized * 254).astype(np.uint8)

    def close(self) -> None:
        if isinstance(self.tiles, np.memmap):
            self.tiles.flush()
            mmap = getattr(self.tiles, "_mmap", None)
            if mmap is not None:
                mmap.close()
        if self._temporary_path is not None:
            self._temporary_path.unlink(missing_ok=True)
            self._temporary_path = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _overview_color_lut(colors: list[str]) -> np.ndarray:
    """Build a compact white/missing plus blue-to-red lookup table."""

    color_map = mcolors.LinearSegmentedColormap.from_list(
        "gsmap_spatial_overview", list(reversed(colors))
    )
    lookup = np.full((256, 3), 255, dtype=np.uint8)
    lookup[1:] = np.rint(color_map(np.linspace(0, 1, 255))[:, :3] * 255).astype(np.uint8)
    return lookup


def _overview_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / filename
    return ImageFont.truetype(str(path), size=size)


def _fit_overview_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    maximum_width: int,
) -> str:
    """Ellipsize a label to fit one overview cell."""

    label = str(text)
    if draw.textlength(label, font=font) <= maximum_width:
        return label
    suffix = "…"
    low, high = 0, len(label)
    while low < high:
        middle = (low + high + 1) // 2
        if draw.textlength(label[:middle] + suffix, font=font) <= maximum_width:
            low = middle
        else:
            high = middle - 1
    return label[:low] + suffix


def _draw_centered_overview_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    width: int,
    font: ImageFont.FreeTypeFont,
    fill: str = "#334155",
) -> None:
    label = _fit_overview_text(draw, text, font, width)
    text_width = draw.textlength(label, font=font)
    draw.text((x + (width - text_width) / 2, y), label, font=font, fill=fill)


def _atomic_save_overview_png(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.partial")
    temporary_path.unlink(missing_ok=True)
    try:
        image.save(temporary_path, format="PNG", compress_level=3, optimize=False)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


# %%


def estimate_plotly_point_size(coordinates, DEFAULT_PIXEL_WIDTH=1000):
    # Convert to numpy array if it's a DataFrame or other array-like object
    if hasattr(coordinates, "values"):
        coordinates = coordinates.values
    coordinates = np.asarray(coordinates)

    tree = KDTree(coordinates)
    distances, _ = tree.query(coordinates, k=2)
    avg_min_distance = np.median(distances[:, 1])
    # get the width and height of the plot
    width = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
    height = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])

    scale_factor = DEFAULT_PIXEL_WIDTH / max(width, height)
    pixel_width = width * scale_factor
    pixel_height = height * scale_factor

    point_size = avg_min_distance * scale_factor

    return (pixel_width, pixel_height), point_size


def estimate_matplotlib_scatter_marker_size(
    ax: matplotlib.axes.Axes,
    coordinates: np.ndarray,
    x_limits: tuple | None = None,
    y_limits: tuple | None = None,
    nearest_neighbor_radius: float | None = None,
) -> float:
    """
    Estimates the appropriate marker size to make adjacent markers touch.

    This function calculates the size 's' for a square marker (in points^2)
    such that its diameter in the plot corresponds to the average distance
    to the nearest neighbor for each point in the dataset. It accounts for
    the plot's aspect ratio and final rendered dimensions.

    Args:
        ax (matplotlib.axes.Axes): The subplot object. The function will
            temporarily set its limits and aspect ratio to ensure the
            transformation from data units to display units is accurate.
        coordinates (np.ndarray): A NumPy array of shape (n, 2)
            containing the (x, y) coordinates of the points.
        x_limits (Optional[tuple]): Optional (min, max) tuple to override automatic x-axis limits.
        y_limits (Optional[tuple]): Optional (min, max) tuple to override automatic y-axis limits.

    Returns:
        float: The estimated marker size 's' (in points^2) for use with
               ax.scatter().
    """
    # 1. Set up the axes' properties to ensure accurate transformations.
    # The aspect ratio and data limits must be set to correctly
    # calculate the relationship between data units and display units (inches/points).
    ax.set_aspect("equal")

    # Use provided limits if available, otherwise calculate from data
    if x_limits is not None:
        x_data_min, x_data_max = x_limits
    else:
        x_data_min, x_data_max = np.min(coordinates[:, 0]), np.max(coordinates[:, 0])

    if y_limits is not None:
        y_data_min, y_data_max = np.min(coordinates[:, 1]), np.max(coordinates[:, 1])
    else:
        y_data_min, y_data_max = np.min(coordinates[:, 1]), np.max(coordinates[:, 1])

    ax.set_xlim(x_data_min, x_data_max)
    ax.set_ylim(y_data_min, y_data_max)

    # Force a draw of the canvas to finalize the transformations.
    ax.figure.canvas.draw()

    # 2. Calculate the required marker radius in data units.
    # We find the average distance to the nearest neighbor for all points.
    # The desired radius is half of this distance.
    if nearest_neighbor_radius is None:
        tree = KDTree(coordinates)
        distances, _ = tree.query(coordinates, k=2)
        radius_data = np.mean(distances[:, 1]) / 2
    else:
        radius_data = nearest_neighbor_radius

    # 3. Convert the data radius to display units (points).
    # This requires transforming the axes' bounding box from data coordinates
    # to display coordinates (pixels), then to physical units (inches).

    # Get the bounding box in display (pixel) coordinates
    x_display_min, _ = ax.transData.transform((x_data_min, y_data_min))
    x_display_max, _ = ax.transData.transform((x_data_max, y_data_max))

    # Convert the display coordinates to inches
    x_inch_min, _ = ax.figure.dpi_scale_trans.inverted().transform((x_display_min, 0))
    x_inch_max, _ = ax.figure.dpi_scale_trans.inverted().transform((x_display_max, 0))

    width_inch = x_inch_max - x_inch_min
    width_data = x_data_max - x_data_min

    # Calculate the radius in inches. This scales the data radius by the
    # ratio of the plot's physical width to its data width.
    # This works because the aspect ratio is 'equal'.
    radius_inch = (radius_data / width_data) * width_inch

    # Convert inches to points (1 inch = 72 points).
    radius_points = radius_inch * 72

    # 4. Calculate the marker size 's'.
    # For ax.scatter, 's' is the marker area in points^2.
    # For a square marker, the area is (side)^2, where side = 2 * radius.
    square_marker_size = (2 * radius_points) ** 2

    return square_marker_size * 1.2


def _create_color_map(category_list: list, hex=False, rng=42) -> dict[str, tuple]:
    unique_categories = sorted(set(category_list), key=str)

    # Check for 'NaN' or nan and handle separately
    nan_values = [v for v in unique_categories if str(v).lower() in ["nan", "none", "null"]]
    other_categories = [v for v in unique_categories if v not in nan_values]

    n_colors = len(other_categories)

    # Generate N visually distinct colors for non-NaN categories
    if n_colors > 0:
        colors = distinctipy.get_colors(n_colors, rng=rng)
        color_map = dict(zip(other_categories, colors, strict=False))
    else:
        color_map = {}

    # Assign grey color to NaN values
    grey_rgb = (0.827, 0.827, 0.827)  # lightgrey
    for v in nan_values:
        color_map[v] = grey_rgb

    if hex:
        # Convert RGB tuples to hex format
        color_map = {category: distinctipy.get_hex(color_map[category]) for category in color_map}
        logger.debug("Generated categorical color map in hex format")
    return color_map


class VisualizeRunner:
    def __init__(self, config):
        self.config = config

    custom_colors_list = [
        "#d73027",
        "#f46d43",
        "#fdae61",
        "#fee090",
        "#e0f3f8",
        "#abd9e9",
        "#74add1",
        "#4575b4",
        "#313695",
    ]

    def _create_single_trait_multi_sample_tile_plot(
        self,
        tile_store: SpatialOverviewTileStore,
        trait_index: int,
        output_png_path: Path,
        max_cols: int = 8,
    ) -> None:
        """Compose one trait across samples from the shared uint8 tile cache."""

        sample_names = tile_store.sample_names
        n_cols = min(max(1, int(max_cols)), max(1, len(sample_names)))
        n_rows = max(1, math.ceil(len(sample_names) / n_cols))
        resolution = tile_store.resolution
        margin = 14
        header_height = 42
        label_height = 20
        gap = 8
        cell_width = resolution + gap
        cell_height = resolution + label_height + gap
        colorbar_width = 88
        image_width = margin * 2 + n_cols * cell_width + colorbar_width
        image_height = header_height + margin + n_rows * cell_height

        canvas = Image.new("RGB", (image_width, image_height), "white")
        draw = ImageDraw.Draw(canvas)
        title_font = _overview_font(18, bold=True)
        label_font = _overview_font(9)
        tick_font = _overview_font(9)
        trait = tile_store.trait_names[trait_index]
        _draw_centered_overview_text(
            draw,
            trait_label(self.config, trait),
            margin,
            10,
            image_width - 2 * margin,
            title_font,
            "#0f172a",
        )
        lookup = _overview_color_lut(self.custom_colors_list)

        for sample_index, sample_name in enumerate(sample_names):
            row, column = divmod(sample_index, n_cols)
            x = margin + column * cell_width
            y = header_height + row * cell_height
            tile = Image.fromarray(lookup[np.asarray(tile_store.tiles[sample_index, trait_index])])
            canvas.paste(tile, (x, y))
            draw.rectangle(
                (x, y, x + resolution - 1, y + resolution - 1),
                outline="#e2e8f0",
                width=1,
            )
            _draw_centered_overview_text(
                draw,
                sample_name,
                x,
                y + resolution + 2,
                resolution,
                label_font,
                "#475569",
            )

        bar_x = margin + n_cols * cell_width + 24
        bar_y = header_height + max(10, (image_height - header_height - 260) // 2)
        bar_height = min(260, max(80, image_height - bar_y - 44))
        gradient_indices = np.rint(np.linspace(255, 1, bar_height)).astype(np.uint8)
        gradient = np.repeat(lookup[gradient_indices, None, :], 16, axis=1)
        canvas.paste(Image.fromarray(gradient), (bar_x, bar_y))
        draw.rectangle((bar_x, bar_y, bar_x + 15, bar_y + bar_height - 1), outline="#cbd5e1")
        maximum = float(tile_store.color_maxima[trait_index])
        draw.text((bar_x + 22, bar_y - 5), f"{maximum:.2f}", font=tick_font, fill="#475569")
        draw.text((bar_x + 22, bar_y + bar_height - 10), "0", font=tick_font, fill="#475569")
        draw.text((bar_x - 4, bar_y - 24), "−log10(P)", font=tick_font, fill="#334155")
        _atomic_save_overview_png(canvas, output_png_path)

    def _create_single_sample_multi_trait_tile_plot(
        self,
        tile_store: SpatialOverviewTileStore,
        sample_index: int,
        output_png_path: Path,
        max_cols: int = 5,
    ) -> None:
        """Compose every trait for one sample, each with its own color bar."""

        trait_names = tile_store.trait_names
        n_cols = min(max(1, int(max_cols)), max(1, len(trait_names)))
        n_rows = max(1, math.ceil(len(trait_names) / n_cols))
        resolution = tile_store.resolution
        margin = 14
        header_height = 44
        gap = 10
        title_height = 22
        colorbar_height = 30
        cell_width = resolution + gap
        cell_height = title_height + resolution + colorbar_height + gap
        image_width = margin * 2 + n_cols * cell_width
        image_height = header_height + margin + n_rows * cell_height

        canvas = Image.new("RGB", (image_width, image_height), "white")
        draw = ImageDraw.Draw(canvas)
        header_font = _overview_font(18, bold=True)
        trait_font = _overview_font(9, bold=True)
        tick_font = _overview_font(8)
        sample_name = tile_store.sample_names[sample_index]
        _draw_centered_overview_text(
            draw,
            sample_name,
            margin,
            10,
            image_width - 2 * margin,
            header_font,
            "#0f172a",
        )
        lookup = _overview_color_lut(self.custom_colors_list)
        gradient_width = max(40, resolution - 40)
        gradient_indices = np.rint(np.linspace(1, 255, gradient_width)).astype(np.uint8)
        gradient = np.repeat(lookup[gradient_indices][None, :, :], 8, axis=0)
        gradient_image = Image.fromarray(gradient)

        for trait_index, trait in enumerate(trait_names):
            row, column = divmod(trait_index, n_cols)
            x = margin + column * cell_width
            y = header_height + row * cell_height
            _draw_centered_overview_text(
                draw, trait_label(self.config, trait), x, y + 2, resolution, trait_font, "#334155"
            )
            tile_y = y + title_height
            tile = Image.fromarray(lookup[np.asarray(tile_store.tiles[sample_index, trait_index])])
            canvas.paste(tile, (x, tile_y))
            draw.rectangle(
                (x, tile_y, x + resolution - 1, tile_y + resolution - 1),
                outline="#e2e8f0",
                width=1,
            )
            bar_x = x + 20
            bar_y = tile_y + resolution + 4
            canvas.paste(gradient_image, (bar_x, bar_y))
            draw.rectangle(
                (bar_x, bar_y, bar_x + gradient_width - 1, bar_y + 7),
                outline="#cbd5e1",
                width=1,
            )
            maximum = float(tile_store.color_maxima[trait_index])
            draw.text((bar_x, bar_y + 9), "0", font=tick_font, fill="#64748b")
            maximum_label = f"{maximum:.2f}"
            maximum_width = draw.textlength(maximum_label, font=tick_font)
            draw.text(
                (bar_x + gradient_width - maximum_width, bar_y + 9),
                maximum_label,
                font=tick_font,
                fill="#64748b",
            )

        _atomic_save_overview_png(canvas, output_png_path)

    def _create_single_trait_multi_sample_raster_plot(
        self,
        obs_ldsc_merged: pd.DataFrame,
        trait_abbreviation: str,
        sample_name_list: list[str],
        output_png_path: Path,
        n_rows: int,
        n_cols: int,
        raster_cache: MultiSampleSpatialRasterCache | None = None,
        subplot_width_inches: float = 1.8,
        dpi: int = 140,
    ) -> None:
        """Render a compact all-sample overview using cached spatial pixel bins."""

        if trait_abbreviation not in obs_ldsc_merged.columns:
            logger.warning(f"Trait {trait_abbreviation} is absent; skipping its overview")
            return
        if raster_cache is None:
            raster_cache = MultiSampleSpatialRasterCache(obs_ldsc_merged, sample_name_list)

        values = pd.to_numeric(obs_ldsc_merged[trait_abbreviation], errors="coerce").to_numpy(
            dtype=np.float32, copy=False
        )
        finite_values = values[np.isfinite(values)]
        color_maximum = float(np.quantile(finite_values, 0.999)) if finite_values.size else 1.0
        if not np.isfinite(color_maximum) or color_maximum <= 0:
            color_maximum = 1.0

        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            "gsmap_trait_overview", self.custom_colors_list
        ).reversed()
        custom_cmap.set_bad("white", alpha=0)
        normalizer = mcolors.Normalize(vmin=0, vmax=color_maximum, clip=True)
        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * subplot_width_inches, n_rows * subplot_width_inches),
            squeeze=False,
        )
        figure.patch.set_facecolor("white")
        figure.suptitle(
            trait_label(self.config, trait_abbreviation), fontsize=13, fontweight="bold", y=0.992
        )

        axes_flat = axes.ravel()
        image_origin = "upper" if self.config.plot_origin == "upper" else "lower"
        for axis, sample_name in zip(axes_flat, sample_name_list, strict=False):
            raster = raster_cache.rasterize(sample_name, values)
            axis.imshow(
                raster,
                cmap=custom_cmap,
                norm=normalizer,
                origin=image_origin,
                interpolation="nearest",
                aspect="equal",
            )
            axis.set_title(str(sample_name), fontsize=6, pad=1.5)
            axis.set_axis_off()
        for axis in axes_flat[len(sample_name_list) :]:
            axis.set_axis_off()

        figure.subplots_adjust(
            left=0.012,
            right=0.94,
            bottom=0.015,
            top=0.965,
            wspace=0.04,
            hspace=0.12,
        )
        colorbar_axis = figure.add_axes([0.955, 0.16, 0.012, 0.68])
        scalar_map = matplotlib.cm.ScalarMappable(norm=normalizer, cmap=custom_cmap)
        colorbar = figure.colorbar(scalar_map, cax=colorbar_axis)
        colorbar.set_label("$-\\log_{10}p$", fontsize=8, fontweight="bold")
        colorbar.ax.tick_params(labelsize=7)

        output_png_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_png_path.with_name(f".{output_png_path.name}.partial")
        temporary_path.unlink(missing_ok=True)
        try:
            figure.savefig(
                temporary_path,
                dpi=dpi,
                facecolor="white",
                format="png",
            )
            temporary_path.replace(output_png_path)
        finally:
            temporary_path.unlink(missing_ok=True)
            plt.close(figure)
