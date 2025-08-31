import matplotlib as mpl
import numpy as np
from pyvista import MultiBlock

categorical_legend_loc_legal = ["upper right",
                                "upper left",
                                "lower left",
                                "lower right",
                                "center left",
                                "center right",
                                "lower center",
                                "upper center",
                                "center"]


def add_model(
    plotter,
    model,
    key=None,
    colormap=None,
    clim=None,
    ambient=0.2,
    opacity=1.0,
    model_style="surface",
    point_size=3.0,
):

    def _add_model(_p, _model, _key, _colormap, _style, _ambient, _opacity, _point_size,_clim):
        """Add any PyVista/VTK model to the scene."""
        if _style == "points":
            render_spheres, render_tubes, smooth_shading = True, False, True
        elif _style == "wireframe":
            render_spheres, render_tubes, smooth_shading = False, True, False
        else:
            render_spheres, render_tubes, smooth_shading = False, False, True
        mesh_kwargs = dict(
            style=_style,
            render_points_as_spheres=True,
            render_lines_as_tubes=render_tubes,
            point_size=_point_size,
            line_width=_point_size,
            ambient=_ambient,
            opacity=_opacity,
            smooth_shading=True,
            clim=_clim,
            show_scalar_bar=False,
        )

        if _colormap is None:
            added_kwargs = dict(
                scalars=f"{
                    _key}_rgba" if _key in _model.array_names else _model.active_scalars_name,
                rgba=True
            )
        else:
            added_kwargs = dict(
                scalars=_key if _key in _model.array_names else _model.active_scalars_name,
                cmap=_colormap
            )

        mesh_kwargs.update(added_kwargs)
        _p.add_mesh(_model, **mesh_kwargs)

    # Add model(s) to the plotter.
    _add_model(
        _p=plotter,
        _model=model,
        _key=key,
        _colormap=colormap,
        _style=model_style,
        _point_size=point_size,
        _ambient=ambient,
        _opacity=opacity,
        _clim=clim,
    )


def add_str_legend(
        plotter,
        labels,
        colors,
        font_family='arial',
        legend_size=None,
        legend_loc="center right"
):

    legend_data = np.concatenate(
        [labels.reshape(-1, 1).astype(object), colors.reshape(-1, 1).astype(object)], axis=1)
    legend_data = legend_data[legend_data[:, 0] != "mask", :]
    assert len(
        legend_data) != 0, "No legend can be added, please set `show_legend=False`."

    legend_entries = legend_data[np.lexsort(legend_data[:, ::-1].T)]
    if legend_size is None:
        legend_num = 10 if len(legend_entries) >= 10 else len(legend_entries)
        legend_size = (0.1 + 0.01 * legend_num, 0.1 + 0.012 * legend_num)

    plotter.add_legend(
        legend_entries.tolist(),
        face="none",
        font_family=font_family,
        bcolor=None,
        loc=legend_loc,
        size=legend_size
    )


def add_num_legend(
    plotter,
    title="",
    n_labels=5,
    title_font_size=None,
    label_font_size=None,
    font_color="black",
    font_family="arial",
    legend_size=(0.1, 0.4),
    legend_loc=(0.85, 0.3),
    vertical=True,
    fmt="%.2e",
):

    plotter.add_scalar_bar(
        title=title,
        n_labels=n_labels,
        title_font_size=title_font_size,
        label_font_size=label_font_size,
        color=font_color,
        font_family=font_family,
        use_opacity=True,
        width=legend_size[0],
        height=legend_size[1],
        position_x=legend_loc[0],
        position_y=legend_loc[1],
        vertical=vertical,
        fmt=fmt,
    )


def add_legend(
    plotter,
    model,
    key=None,
    colormap=None,
    categorical_legend_size=None,
    categorical_legend_loc=None,
    scalar_bar_title="",
    scalar_bar_size=None,
    scalar_bar_loc=None,
    scalar_bar_title_size=None,
    scalar_bar_label_size=None,
    scalar_bar_font_color="black",
    font_family="arial",
    fmt="%.2e",
    scalar_bar_n_labels=5,
    vertical=True,
):

    # if colormap is None: categorical
    # if colormap is not None: continuous

    if colormap is None:
        assert key is not None, "When colormap is None, key cannot be None at the same time."

        if categorical_legend_loc not in categorical_legend_loc_legal and categorical_legend_loc is None:
            categorical_legend_loc = 'center right'

        if isinstance(model, MultiBlock):
            keys = key if isinstance(key, list) else [key] * len(model)

            legend_label_data, legend_color_data = [], []
            for m, k in zip(model, keys):
                legend_label_data.append(np.asarray(m[k]).flatten())
                legend_color_data.append(np.asarray(
                    [mpl.colors.to_hex(i) for i in m[f"{k}_rgba"]]).flatten())
            legend_label_data = np.concatenate(legend_label_data, axis=0)
            legend_color_data = np.concatenate(legend_color_data, axis=0)
            print(legend_color_data)
        else:
            legend_label_data = np.asarray(model[key]).flatten()
            legend_color_data = np.asarray(
                [mpl.colors.to_hex(i) for i in model[f"{key}_rgba"]]).flatten()

        legend_data = np.concatenate(
            [legend_label_data.reshape(-1, 1), legend_color_data.reshape(-1, 1)], axis=1)
        unique_legend_data = np.unique(legend_data, axis=0)

        add_str_legend(
            plotter=plotter,
            labels=unique_legend_data[:, 0],
            colors=unique_legend_data[:, 1],
            font_family=font_family,
            legend_size=categorical_legend_size,
            legend_loc=categorical_legend_loc
        )
    else:
        if not isinstance(scalar_bar_size, tuple) and scalar_bar_size is None:
            scalar_bar_size = (0.1, 0.4)
        if not isinstance(scalar_bar_loc, tuple) and scalar_bar_loc is None:
            scalar_bar_loc = (0.85, 0.3)

        add_num_legend(
            plotter=plotter,
            legend_size=scalar_bar_size,
            legend_loc=scalar_bar_loc,
            title=scalar_bar_title,
            n_labels=scalar_bar_n_labels,
            title_font_size=scalar_bar_title_size,
            label_font_size=scalar_bar_label_size,
            font_color=scalar_bar_font_color,
            font_family=font_family,
            fmt=fmt,
            vertical=vertical
        )


def add_outline(
    plotter,
    model,
    outline_width=1.0,
    outline_color="black",
):

    model_outline = model.outline()
    plotter.add_bounding_box(
        color=outline_color,
        line_width=outline_width
    )


        
def add_text(
    plotter,
    text,
    font_family="arial",
    text_font_size=15,
    text_font_color="black",
    text_loc="upper_edge"
):

    plotter.add_text(
        text=text,
        font=font_family,
        color=text_font_color,
        font_size=text_font_size,
        position=text_loc if text_loc is not None else "upper_edge"
    )
