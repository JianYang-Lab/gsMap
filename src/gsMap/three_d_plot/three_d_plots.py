import sys
import matplotlib as mpl
from typing import Optional, Union, Literal, List
from pyvista import MultiBlock, Plotter, PolyData
import pyvista as pv
import math
import logging
import numpy as np

from gsMap.three_d_plot.three_d_plot_prepare import create_plotter, _get_default_cmap, construct_pc
from gsMap.three_d_plot.three_d_plot_decorate import add_legend, add_outline, add_text, add_model


logger = logging.getLogger(__name__)

_get_default_cmap()


def wrap_to_plotter(
    plotter: Plotter,
    model: Union[PolyData, MultiBlock],
    key: Optional[str] = None,
    colormap: Optional[Union[str, list]] = None,

    # parameters for model settings
    ambient: float = 0.2,
    opacity: [float,list] = 1,
    point_size: float = 1,
    model_style: Literal["points", "surface", "wireframe"] = "surface",
    font_family: Literal["times", "courier", "arial"] = "arial",
    background: str = "black",
    cpo: Union[str, list] = "iso",
    clim: Optional[list] = None,
    legend_kwargs: Optional[dict] = None,
    outline_kwargs: Optional[dict] = None,
    text: Optional[str] = None,
    scalar_bar_title: Optional[str] = None,
    text_kwargs: Optional[dict] = None,
    show_outline: bool = False,
    show_text: bool = True,
    show_legend: bool = True
):
    """
    Wrap the model and its settings to a plotter.

    Parameters:
        plotter (Plotter): The plotter object to wrap the model to.
        model (Union[PolyData, MultiBlock]): The model to be added to the plotter.
        key (Optional[str]): The key to identify the model in the plotter.
        colormap (Optional[Union[str, list]]): The colormap to use for the model.
        ambient (float): The ambient lighting coefficient.
        opacity (float): The opacity of the model.
        point_size (float): The size of the points in the model.
        model_style (Literal["points", "surface", "wireframe"]): The style of the model.
        font_family (Literal["times", "courier", "arial"]): The font family to use.
        background (str): The background color of the plotter.
        cpo (Union[str, list]): The camera position.
        legend_kwargs (Optional[dict]): Additional keyword arguments for the legend.
        outline_kwargs (Optional[dict]): Additional keyword arguments for the outline.
        text (Optional[str]): The text to add to the plotter.
        scalar_bar_title (Optional[str]): The title of the scalar bar.
        text_kwargs (Optional[dict]): Additional keyword arguments for the text.
        show_outline (bool): Whether to show the outline.
        show_text (bool): Whether to show the text.
        show_legend (bool): Whether to show the legend.
    """

    # Set the bacic settings for the plotter.
    # plotter.camera_position = cpo
    bg_rgb = mpl.colors.to_rgb(background)
    cbg_rgb = (1 - bg_rgb[0], 1 - bg_rgb[1], 1 - bg_rgb[2])

    # Add model(s) basic settings to the plotter.
    add_model(
        plotter=plotter,
        model=model,
        key=key,
        colormap=colormap,
        ambient=ambient,
        opacity=opacity,
        point_size=point_size,
        model_style=model_style,
        clim = clim
    )

    # Add legends to the plotter.
    if show_legend:
        lg_kwargs = dict(
            categorical_legend_size=None,
            categorical_legend_loc=None,
            scalar_bar_size=None,
            scalar_bar_loc=None,
            scalar_bar_title_size=None,
            scalar_bar_label_size=None,
            scalar_bar_font_color=cbg_rgb,
            scalar_bar_n_labels=5,
            font_family=font_family,
            fmt="%.1e",
            vertical=True,
        )
        if not (legend_kwargs is None):
            lg_kwargs.update(
                (k, legend_kwargs[k]) for k in lg_kwargs.keys() & legend_kwargs.keys())

        add_legend(plotter=plotter, model=model, key=key,
                   colormap=colormap, scalar_bar_title=scalar_bar_title, **lg_kwargs)

    # Add an outline to the plotter.
    if show_outline:
        ol_kwargs = dict(
            outline_width=1.0,
            outline_color=cbg_rgb,
        )

        if not (outline_kwargs is None):
            ol_kwargs.update(
                (k, outline_kwargs[k]) for k in ol_kwargs.keys() & outline_kwargs.keys())

        add_outline(plotter=plotter, model=model, **ol_kwargs)

    # Add text to the plotter.
    if show_text:
        t_kwargs = dict(
            font_family=font_family,
            text_font_size=12,
            text_font_color=cbg_rgb,
            text_loc="upper_edge",
        )

        if not (text_kwargs is None):
            t_kwargs.update((k, text_kwargs[k])
                            for k in t_kwargs.keys() & text_kwargs.keys())

        add_text(plotter=plotter, text=text, **t_kwargs)


def three_d_plot(
    # parameters for plotter
    adata,
    spatial_key: str,
    keys: list,
    cmaps: Optional[Union[str, list, dict]] = 'default_cmap',
    scalar_bar_titles: Optional[Union[str, list]] = None,
    texts:  Optional[Union[str, list]] = None,

    window_size: Optional[tuple[int, int]] = None,
    off_screen: bool = False,
    shape: Optional[tuple] = None,
    show_camera_orientation: bool = True,
    show_axis_orientation: bool = False,
    jupyter: bool = True,

    # parameters for model settings
    ambient: float = 0.2,
    opacity: Union[float, list[float]] = 1,
    point_size: float = 1,
    clim: Optional[list] = None,
    model_style: Literal["points", "surface", "wireframe"] = "surface",
    font_family: Literal["times", "courier", "arial"] = "arial",
    background: str = "black",
    cpo: Union[str, list] = "iso",

    # parameters for show decoration
    show_outline: bool = False,
    show_text: bool = True,
    show_legend: bool = True,

    # parameters for legends, outline, and text
    legend_kwargs: Optional[dict] = None,
    outline_kwargs: Optional[dict] = None,
    text_kwargs: Optional[dict] = None
):
    """
    Generate a 3D plot using pyvista for spatial data visualization.

    Parameters:
    - adata: AnnData object containing spatial data.
    - spatial_key: Key in adata.obs that specifies the spatial coordinates.
    - keys: List of keys in adata.obs to group the data by.
    - cmaps: Colormap(s) to use for each group. Can be a string or a list of strings. Default is 'default_cmap'.
    - scalar_bar_titles: Titles for the scalar bars. Can be a string or a list of strings. Default is None.
    - texts: List of texts to display on the plot. Default is None.
    - window_size: Size of the plot window in pixels. Default is None.
        Note window_size is a tuple of (width, height), representing the width and height of the each subplot.
    - off_screen: Whether to render the plot off-screen. Default is False.
    - shape: Shape of the subplot grid. Default is None.
        For plotting  figues in a grid, shape should be a tuple of (n_rows, n_cols)
    - show_camera_orientation: Whether to show the camera orientation widget. Default is True.
    - jupyter: Whether to display the plot in a Jupyter notebook. Default is True.
    - ambient: Ambient lighting intensity. Default is 0.2.
    - opacity: Opacity of the plot objects. Default is 1.
    - point_size: Size of the points in the plot. Default is 1.
    - model_style: Style of the plot objects. Can be 'points', 'surface', or 'wireframe'. Default is 'surface'.
    - font_family: Font family for the plot text. Can be 'times', 'courier', or 'arial'. Default is 'arial'.
    - background: Background color of the plot. Default is 'black'.
    - cpo: Color by point option. Can be a string or a list. Default is 'iso'.
    - show_outline: Whether to show the plot outline. Default is False.
    - show_text: Whether to show the plot text. Default is True.
    - show_legend: Whether to show the plot legend. Default is True.
    - legend_kwargs (dict, optional): Additional keyword arguments for the legend, the default values are:
            categorical_legend_size: [tuple] = None, # (gap, size)
            categorical_legend_loc: [Literal["upper right", "upper left", "lower left", "lower right",
                                                    "center left", "center right", "lower center", "upper center"
                                                    "center"]] = None
            scalar_bar_title: Optional[str] = None
            scalar_bar_size: Optional[tuple] = None
            scalar_bar_loc: Optional[tuple] = None
            scalar_bar_title_size: Union[int, float] = None
            scalar_bar_label_size: Union[int, float] = None
            scalar_bar_font_color: Optional[str] = None
            scalar_bar_n_labels: int = 5
            fmt="%.1e",
            vertical: bool = True
        
    - outline_kwargs (dict, optional): Additional keyword arguments for the plot outline, the default values are:
            outline_width: float = 1.0
            outline_color: Optional[str] = None
            show_outline_labels: bool = False
            outline_font_size: Optional[int] = None
            outline_font_color: Optional[str] = None
            
    - text_kwargs (dict, optional): Additional keyword arguments for the text, the default values are:
            text_font_size: Optional[float] = None,
            text_font_color: Optional[str] = None,
            text_loc: Optional[Literal["lower_left", "lower_right", "upper_left",
                                        "upper_right", "lower_edge", "upper_edge",
                                        "right_edge", "left_edge",]] = None

    Returns:
    - plotter: The pyvista plotter object.
    """
    _get_default_cmap()
    if isinstance(cmaps, str):
        cmaps = [cmaps] * len(keys)

    if scalar_bar_titles is None or isinstance(scalar_bar_titles, str):
        scalar_bar_titles = [scalar_bar_titles] * len(keys)

    if texts is None or isinstance(texts, str):
        texts = [texts] * len(keys)

    # Build the pyvista object
    models = pv.MultiBlock()
    plot_cmaps = []
    for i, key in enumerate(keys):
        _model, _plot_cmap = construct_pc(adata=adata.copy(),
                                          spatial_key=spatial_key,
                                          groupby=key,
                                          key_added=key,
                                          colormap=cmaps[i])
        models[f"model_{i}"] = _model
        plot_cmaps.append(_plot_cmap)

    # Set the shape and window size of the plot
    n_window = len(keys)
    shape = (math.ceil(n_window / 3), n_window if n_window <
             3 else 3) if shape is None else shape
    if isinstance(shape, (tuple, list)):
        n_subplots = shape[0] * shape[1]
        subplots = []
        for i in range(n_subplots):
            col = math.floor(i / shape[1])
            ind = i - col * shape[1]
            subplots.append([col, ind])

    win_x, win_y = shape[1], shape[0]
    window_size = ((1500 * win_x, 1500 * win_y)
                   if window_size is None else (window_size[0] * win_x, window_size[1] * win_y))

    # Create the plotter
    plotter = create_plotter(
        background=background,
        off_screen=off_screen,
        shape=shape,
        show_camera_orientation=show_camera_orientation,
        show_axis_orientation=show_axis_orientation,
        window_size=window_size,
        jupyter=jupyter
    )

    # Set the plotter
    for (model, key, plot_cmap, subplot_index, scalar_bar_title, text) in zip(models, keys, plot_cmaps, subplots, scalar_bar_titles, texts):
        plotter.subplot(subplot_index[0], subplot_index[1])

        wrap_to_plotter(
            # parameters for plotter
            plotter=plotter,
            model=model,
            key=key,
            colormap=plot_cmap,

            # parameters for model settings
            clim=clim,
            ambient=ambient,
            opacity=opacity,
            point_size=point_size,
            model_style=model_style,
            font_family=font_family,
            background=background,
            cpo=cpo,

            # parameters for legends, outline, and text
            legend_kwargs=legend_kwargs,
            scalar_bar_title=scalar_bar_title,
            outline_kwargs=outline_kwargs,
            text_kwargs=text_kwargs,
            text=text,

            # parameters for show decoration
            show_outline=show_outline,
            show_text=show_text,
            show_legend=show_legend
        )

    plotter.link_views()
    plotter.camera_position = 'yz'
    
    return plotter


def three_d_plot_save(
    plotter: Plotter,
    filename: str,
    view_up_1=(0, 0, 0),
    view_up_2=(0, 0, 1),
    n_points: int = 150,
    factor: float = 2.0,
    shift: float = 0,
    step: int = 1,
    quality: int = 9,
    framerate: int = 10,
    save_mp4 : bool = False,
    save_gif : bool = False
):
    """
    Saves a 3D plot as an HTML, GIF, and MP4 file.

    Args:
        plotter (Plotter): The Plotter object used for generating the plot.
        filename (str): The base filename for saving the files.
        view_up_1 (tuple, optional): The initial view up direction. Defaults to (0.5, 0.5, 1).
        view_up_2 (tuple, optional): The final view up direction. Defaults to (0, 0, 1).
        n_points (int, optional): The number of points on the orbital path. Defaults to 150.
        factor (float, optional): The factor for scaling the orbital path. Defaults to 2.0.
        shift (float, optional): The shift value for the orbital path. Defaults to 0.
        step (int, optional): The step size for writing frames. Defaults to 1.
        quality (int, optional): The quality of the GIF file. Defaults to 9.
        framerate (int, optional): The framerate of the MP4 file. Defaults to 15.
    """
    # save html
    logger.info('saving 3d plot as html...')
    plotter.export_html(f'{filename}.html')

    # save gif
    if save_gif:
        logger.info('saving 3d plot as gif...')
        path = plotter.generate_orbital_path(factor=factor, shift=shift, viewup=view_up_1, n_points=n_points)
        plotter.open_gif(filename=f'{filename}.gif')
        plotter.orbit_on_path(path, write_frames=True, viewup=view_up_2, step=step)
        plotter.close()
        
    # save mp4
    if save_mp4:
        logger.info('saving 3d plot as mp4...')
        path = plotter.generate_orbital_path(factor=factor, shift=shift, viewup=view_up_1, n_points=n_points)
        plotter.open_movie(filename=f'{filename}.mp4', framerate=framerate, quality=quality)
        plotter.orbit_on_path(path, write_frames=True, viewup=view_up_2, step=step)
        plotter.close()



def rotate_around_xyz(
    camera_coordinates, 
    angle_x=0, 
    angle_y=0, 
    angle_z=0
):
    """
    Rotate a point around the x, y, and z axes.

    Parameters:
    - point: numpy array representing the camera coordinates
    - angle_x: rotation angle around the x-axis in degrees (default: 0)
    - angle_y: rotation angle around the y-axis in degrees (default: 0)
    - angle_z: rotation angle around the z-axis in degrees (default: 0)

    Returns:
    - point_rotated: numpy array representing the rotated point coordinates
    """
    angle_x_rad = np.radians(angle_x)
    angle_y_rad = np.radians(angle_y)
    angle_z_rad = np.radians(angle_z)
    
    # Rotation matrix for x-axis
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x_rad), -np.sin(angle_x_rad)],
        [0, np.sin(angle_x_rad), np.cos(angle_x_rad)]
    ])
    
    # Rotation matrix for y-axis
    rotation_matrix_y = np.array([
        [np.cos(angle_y_rad), 0, np.sin(angle_y_rad)],
        [0, 1, 0],
        [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad)]
    ])
    
    # Rotation matrix for z-axis
    rotation_matrix_z = np.array([
        [np.cos(angle_z_rad), -np.sin(angle_z_rad), 0],
        [np.sin(angle_z_rad), np.cos(angle_z_rad), 0],
        [0, 0, 1]
    ])
    
    # Apply rotations
    camera_rotated = np.dot(rotation_matrix_x, camera_coordinates)
    camera_rotated = np.dot(rotation_matrix_y, camera_rotated)
    camera_rotated = np.dot(rotation_matrix_z, camera_rotated)
    
    return camera_rotated
