import matplotlib as mpl
import pyvista as pv
import numpy as np

from pandas.core.frame import DataFrame
from matplotlib.colors import LinearSegmentedColormap


p_color = ['#313695', '#4575b4', '#74add1','#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']



def _get_default_cmap():
    if "default_cmap" not in mpl.colormaps():
        colors = p_color
        nodes = np.linspace(0, 1, len(p_color))

        mpl.colormaps.register(LinearSegmentedColormap.from_list(
            "default_cmap", list(zip(nodes, colors))))
    return "default_cmap"


def create_plotter(
    jupyter=False,
    off_screen=False,
    window_size=(512, 512),
    background="white",
    shape=(1, 1),
    show_camera_orientation=True,
    show_axis_orientation=False
):

    # Create an initial plotting object.
    _get_default_cmap()
    plotter = pv.Plotter(
        off_screen=off_screen,
        window_size=window_size,
        notebook=False if jupyter is False else True,
        lighting="light_kit",
        shape=shape,
    )

    # Set the background color of the active render window.
    plotter.background_color = background

    # Add a camera orientation widget to the active renderer window.
    if jupyter != "trame":
        if show_camera_orientation:
            plotter.add_camera_orientation_widget()
        elif show_axis_orientation:
            plotter.add_axes(labels_off=True)
    return plotter


def add_point_labels(
    model,
    labels,
    key_added="groups",
    where="point_data",
    colormap="rainbow",
    alphamap=1.0,
    mask_color="black",
    mask_alpha=0.0,
    inplace=False,
):

    model = model.copy() if not inplace else model
    labels = np.asarray(labels).flatten()

    # Set color here if group is of string type. 
    if not np.issubdtype(labels.dtype, np.number):
        cu_arr = np.sort(np.unique(labels), axis=0).astype(object)
        
        raw_labels_hex = labels.copy().astype(object)
        raw_labels_alpha = labels.copy().astype(object)
        raw_labels_hex[raw_labels_hex =="mask"] = mpl.colors.to_hex(mask_color)
        raw_labels_alpha[raw_labels_alpha == "mask"] = mask_alpha

        # Set raw hex.
        if isinstance(colormap, str):
            if colormap in list(mpl.colormaps()):
                lscmap = mpl.colormaps[colormap]
                raw_hex_list = [mpl.colors.to_hex(lscmap(i)) for i in np.linspace(0, 1, len(cu_arr))]
                for label, color in zip(cu_arr, raw_hex_list):
                    raw_labels_hex[raw_labels_hex == label] = color
            else:
                raw_labels_hex[raw_labels_hex !="mask"] = mpl.colors.to_hex(colormap)
        elif isinstance(colormap, dict):
            for label, color in colormap.items():
                raw_labels_hex[raw_labels_hex ==label] = mpl.colors.to_hex(color)
        elif isinstance(colormap, list) or isinstance(colormap, np.ndarray):
            raw_hex_list = np.array([mpl.colors.to_hex(color)for color in colormap]).astype(object)
            for label, color in zip(cu_arr, raw_hex_list):
                raw_labels_hex[raw_labels_hex == label] = color
        else:
            raise ValueError(
                "`colormap` value is wrong." "\nAvailable `colormap` types are: `str`, `list` and `dict`.")

        # Set raw alpha.
        if isinstance(alphamap, float) or isinstance(alphamap, int):
            raw_labels_alpha[raw_labels_alpha != "mask"] = alphamap
        elif isinstance(alphamap, dict):
            for label, alpha in alphamap.items():
                raw_labels_alpha[raw_labels_alpha == label] = alpha
        elif isinstance(alphamap, list) or isinstance(alphamap, np.ndarray):
            raw_labels_alpha = np.asarray(alphamap).astype(object)
        else:
            raise ValueError(
                "`alphamap` value is wrong." "\nAvailable `alphamap` types are: `float`, `list` and `dict`."
            )

        # Set rgba.
        labels_rgba = [mpl.colors.to_rgba(c, alpha=a) for c, a in zip(raw_labels_hex, raw_labels_alpha)]
        labels_rgba = np.array(labels_rgba).astype(np.float32)

        # Added rgba of the labels.
        if where == "point_data":
            model.point_data[f"{key_added}_rgba"] = labels_rgba
        else:
            model.cell_data[f"{key_added}_rgba"] = labels_rgba

        plot_cmap = None
    else:
        plot_cmap = colormap

    # Added labels.
    if where == "point_data":
        model.point_data[key_added] = labels
    else:
        model.cell_data[key_added] = labels

    return model if not inplace else None, plot_cmap


def construct_pc(
    adata,
    layer="X",
    spatial_key="spatial",
    groupby=None,
    key_added="groups",
    mask=None,
    colormap="default_cmap",
    alphamap=1.0
):

    # Ensure mask is a list
    mask_list = mask if isinstance(mask, list) else [mask] if mask is not None else []

    # Extract spatial coordinates
    if isinstance(adata, DataFrame):
        cell_names = np.array(adata.index.tolist())
        try:
            bucket_xyz = adata[['sx','sy','sz']].values
        except KeyError:
            raise ValueError(f"Spatial coordinates ('sx','sy','sz') not found in meta data.")
    else:
        cell_names = np.array(adata.obs_names.tolist())
        if spatial_key not in adata.obsm:
            raise ValueError(f"Spatial key {spatial_key} not found in adata.obsm.")
        bucket_xyz = adata.obsm[spatial_key].astype(np.float64)
        if isinstance(bucket_xyz, DataFrame):
            bucket_xyz = bucket_xyz.values
        
    # Handle grouping
    if groupby is None:
        groups = np.array(["same"] * bucket_xyz.shape[0], dtype=str)
    elif isinstance(adata, DataFrame):
        # If adata is a DataFrame, check if groupby is a valid column
        if groupby not in adata.columns:
            raise ValueError(f"`groupby` column '{groupby}' not found in DataFrame.")
        groups = adata[groupby].map(lambda x: "mask" if x in mask_list else x).values
    else:
        # If adata is AnnData, check if groupby is in obs or var
        if groupby in adata.obs_keys():
            # Group by observation metadata
            groups = adata.obs[groupby].map(lambda x: "mask" if x in mask_list else x).values
        elif groupby in adata.var_names or set(groupby) <= set(adata.var_names):
            # Group by gene expression
            adata_X = adata.X if layer == "X" else adata.layers[layer]
            if isinstance(groupby, str):
                groupby = [groupby]
            groups = np.asarray(adata[:, groupby].X.sum(axis=1).flatten())
        else:
            raise ValueError(
                f"`groupby` value '{groupby}' is invalid. "
                "It must be a column in adata.obs, a gene in adata.var_names, "
                "or a list of genes in adata.var_names."
            )

    pc = pv.PolyData(bucket_xyz)
    _, plot_cmap = add_point_labels(
        model=pc,
        labels=groups,
        key_added=key_added,
        where="point_data",
        colormap=colormap,
        alphamap=alphamap,
        inplace=True,
    )

    # The obs_index of each coordinate in the original adata.
    pc.point_data["obs_index"] = cell_names

    return pc, plot_cmap
