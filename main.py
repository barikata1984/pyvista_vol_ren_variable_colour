"""Volume-render a voxel mass distribution with PyVista.

Reads ``<object-dir>/ground_truth.csv`` (2 metadata rows + header, then one row
per voxel: ``x, y, z, mass, mass_density`` on a ``2**lod`` cubic grid), lays the
normalized mass density onto a uniform grid and volume-renders it. Optionally
shows a matplotlib 3D scatter, draws labelled axes, or writes a rotating
mp4/gif.

This is a PyVista re-implementation of an earlier Mayavi script (the
``mayavi_vol_ren_variable_colour`` project, which used tvtk
``ColorTransferFunction`` / ``PiecewiseFunction`` / ``LUTManager`` and moviepy);
the CLI and data handling are kept compatible. Notable points:
- The empty-voxel sentinel value (``TRANSPARENT_INPUT = -1``) and the scalar
  range ``[-1, 1]`` are preserved, so a filled voxel's normalized mass ``[0, 1]``
  maps to the upper half of the colormap (matching the Mayavi original's
  ``ColorTransferFunction.range = [-1, 1]``). See ``clim`` below to change this.
- ``cmap`` is a matplotlib colormap name (PyVista resolves it).
- The scalar field is quantized to ``uint16`` because VTK's CPU volume mapper
  (used when there is no GPU) renders nothing for float scalars.
- Camera placement mirrors Mayavi's ``mlab.view()`` conventions: ``azimuth`` is
  the angle in the x-y plane from the +x axis; ``elevation`` is the zenith angle
  from +z.
- No scalar bar is shown.
- The interactive window has an opacity slider (bottom-left) driving the same
  scalar-opacity function ``--alpha`` sets at startup.
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import vtk

from matplotlib import pyplot as plt
from matplotlib import colormaps as cm


TRANSPARENT_INPUT = -1.0  # sentinel scalar value for empty voxels
QUANT_MAX = 65535  # scalar field is quantized to uint16 (see _quantize_scalars)


def _quantize_scalars(scalars):
    """Map the conceptual scalar range ``[TRANSPARENT_INPUT, 1]`` onto ``uint16``.

    VTK's CPU volume mapper (fixed-point ray cast, used when there is no GPU) only
    accepts ``uint8``/``uint16`` scalars and renders nothing for floats, so the
    field is quantized. Empty voxels (``TRANSPARENT_INPUT``) become 0; the upper
    bound 1 maps to ``QUANT_MAX``.
    """
    span = 1.0 - TRANSPARENT_INPUT
    return np.round((scalars - TRANSPARENT_INPUT) / span * QUANT_MAX).astype(np.uint16)


def _scalar_opacity_function(opacity, scalar_range):
    """Build the volume's scalar-opacity ``vtkPiecewiseFunction``.

    Mirrors the Mayavi original's PiecewiseFunction: opacity 0 at the range
    minimum (empty voxels, quantized to 0), ramping to ``opacity`` at the
    midpoint, then flat ``opacity`` to the maximum. Used at startup and as the
    target of the interactive opacity slider.
    """
    lo, hi = float(scalar_range[0]), float(scalar_range[1])
    fn = vtk.vtkPiecewiseFunction()
    fn.AddPoint(lo, 0.0)
    fn.AddPoint(0.5 * (lo + hi), float(opacity))
    fn.AddPoint(hi, float(opacity))
    return fn


def _spherical_camera_position(
    azimuth_deg, elevation_deg, distance, focal=(0.0, 0.0, 0.0)
):
    """``camera_position`` triple following ``mlab.view()`` conventions.

    azimuth: angle in the x-y plane measured from the +x axis (deg).
    elevation: zenith angle measured from the +z axis (deg, 0-180).
    distance: camera distance from the focal point.
    """
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    fx, fy, fz = focal
    position = (
        fx + distance * math.sin(el) * math.cos(az),
        fy + distance * math.sin(el) * math.sin(az),
        fz + distance * math.cos(el),
    )
    return [position, focal, (0.0, 0.0, 1.0)]


def _init_pyvista_volume_rendering(
    scalars,
    spacing,
    origin,
    cmap=None,
    opacity_function=None,
    clim=None,
    shade=True,
    off_screen=False,
    size=720,
):
    """PyVista counterpart of ``_init_mayavi_volume_rendering``.

    ``scalars`` is a ``(res, res, res)`` array indexed ``[i, j, k]`` with coords
    ``(ndc_coords[i], ndc_coords[j], ndc_coords[k])``. It is laid onto a uniform
    grid; VTK ``ImageData`` wants point data in Fortran order w.r.t. ``(x, y, z)``.
    """
    plotter = pv.Plotter(off_screen=off_screen, window_size=[size, size])
    plotter.background_color = "white"

    grid = pv.ImageData(dimensions=scalars.shape, spacing=spacing, origin=origin)
    grid.point_data["mass_distr"] = scalars.flatten(order="F")

    volume = plotter.add_volume(
        grid,
        scalars="mass_distr",
        cmap="viridis" if cmap is None else cmap,
        clim=clim,
        shade=shade,
        show_scalar_bar=False,
    )
    if opacity_function is not None:
        volume.prop.SetScalarOpacity(opacity_function)
    return plotter, volume


def _generate_mass_distr_video(frames, mass_distr_img_dir, output_name, fps):
    """Assemble already-rendered RGB frames into ``.mp4`` and ``.gif`` (replaces moviepy)."""
    import imageio.v2 as imageio

    video_path = os.path.join(mass_distr_img_dir, output_name)
    imageio.mimwrite(f"{video_path}.mp4", frames, fps=fps, codec="libx264")
    imageio.mimsave(f"{video_path}.gif", frames, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-dir")
    parser.add_argument(
        "--lod", type=int, default=7
    )  # kept for CLI parity; res is read from data
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--matplotlib", action="store_true")
    parser.add_argument("--axes", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument(
        "--save",
        default=None,
        help="render a single still to this path instead of opening a window",
    )
    parser.add_argument("--generate-video", action="store_true")
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    print(f"{args=}\n")

    # Main part ========================================================
    object_dir = Path(args.object_dir)
    csv_filepath = object_dir / "ground_truth.csv"
    metadata_df = pd.read_csv(
        csv_filepath,
        nrows=1,  # num metadata rows after the metadata header
    )

    print(f"{metadata_df}")

    main_df = pd.read_csv(
        csv_filepath,
        skiprows=2,  # exclude the metadata and its header
    ).loc[:, "x":"mass_density"]

    print(f"{main_df.to_numpy().shape=}")

    _x, _y, _z, _mass_distr = main_df.to_numpy().T[:4]

    res = math.ceil(math.pow(_x.shape[0], 1 / 3))
    ndc_coords = (2 * np.arange(0, res) - res + 1.0) / res
    print(f"{res=}")
    aabb_scale = _x.max() / ndc_coords.max()
    print(f"{aabb_scale=}")  # [m]

    x = _x / aabb_scale
    y = _y / aabb_scale
    z = _z / aabb_scale

    # set plot data
    cmap = cm.get_cmap(args.cmap)
    c = _mass_distr / _mass_distr.max()
    s = np.zeros_like(_mass_distr)
    s[_mass_distr.nonzero()] = 1

    matplotlib_scatter = args.matplotlib
    if matplotlib_scatter:
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_aspect("equal")
        size_const = 0.3
        ax.scatter(x, y, z, s=size_const * s, c=c, alpha=args.alpha, cmap=cmap)
        plt.show(block=False)

    # volume render with pyvista =======================================
    # Setup input data (same as the Mayavi version)
    meshgrid_shape = (res, res, res)
    scalars = TRANSPARENT_INPUT * np.ones(meshgrid_shape)
    zero_to_one_md = _mass_distr / _mass_distr.max()
    zero_to_one_md = zero_to_one_md.reshape(meshgrid_shape)
    scalars[zero_to_one_md.nonzero()] = zero_to_one_md[zero_to_one_md.nonzero()]

    #    print(f"Total mass: {_mass_distr[_mass_distr.nonzero()].sum()}")

    scalars = _quantize_scalars(
        scalars
    )  # uint16; empty voxels -> 0, upper bound -> QUANT_MAX

    # Uniform grid geometry in NDC space: points sit at `ndc_coords` along each axis.
    step = 2.0 / res
    origin = (float(ndc_coords[0]),) * 3
    spacing = (step, step, step)

    # clim spans the whole quantized range [0, QUANT_MAX] (i.e. conceptual [-1, 1]),
    # so a filled voxel's normalized mass [0, 1] maps to the upper half of `cmap`
    # (matching the Mayavi original's `ColorTransferFunction.range = [-1, 1]`). Use
    # clim=[QUANT_MAX // 2, QUANT_MAX] to make filled voxels span the full colormap.
    clim = [0, QUANT_MAX]
    opacity_fn = _scalar_opacity_function(args.alpha, clim)

    plotter, volume = _init_pyvista_volume_rendering(
        scalars,
        spacing=spacing,
        origin=origin,
        cmap=args.cmap,
        opacity_function=opacity_fn,
        clim=clim,
        shade=False,
        off_screen=args.offscreen,
        size=args.size,
    )

    #    plotter.add_text(f"object: {metadata_df.at[0, 'id']}",
    #                     position=(0.02, 0.05), viewport=True, font="courier", color="black")
    #    plotter.add_text(f"aabb scale: {metadata_df.at[0, 'aabb_scale']} [m]",
    #                     position=(0.02, 0.01), viewport=True, font="courier", color="black")

    if args.axes:
        plotter.show_grid(xtitle="X", ytitle="Y", ztitle="Z", color="black")

    if args.generate_video:
        max_epochs = 200
        num_camera_turn = 2
        azimuth = 0.0
        azimuth_tick = num_camera_turn * 360.0 / max_epochs

        mass_distr_img_dir = object_dir / "mass_distr"
        os.makedirs(mass_distr_img_dir, exist_ok=True)

        # NOTE: pair --generate-video with --offscreen on headless machines.
        frames = []
        for epoch in range(max_epochs):
            distance = 5.9  # this val provides the tightest extra space around the bbox
            azimuth = (azimuth + azimuth_tick) % 360.0
            plotter.camera_position = _spherical_camera_position(
                azimuth, 45.0, distance
            )

            filename = f"{epoch:04}.png"
            frames.append(
                plotter.screenshot(str(mass_distr_img_dir / filename), return_img=True)
            )

        _generate_mass_distr_video(
            frames, mass_distr_img_dir, "mass_distr", fps=args.fps
        )
        plotter.close()
    else:
        azimuth = 2 * 360 / 200 * (200 - 1)
        plotter.camera_position = _spherical_camera_position(azimuth, 45.0, 5.0)
        if args.save is not None:
            plotter.screenshot(args.save)
            print(f"saved still to {args.save}")
            plotter.close()
        else:
            # Opacity slider (bottom-left): rebuilds the scalar-opacity function.
            plotter.add_slider_widget(
                lambda value: volume.prop.SetScalarOpacity(
                    _scalar_opacity_function(value, clim)
                ),
                rng=[0.0, 1.0],
                value=args.alpha,
                title="opacity",
                pointa=(0.025, 0.08),
                pointb=(0.30, 0.08),
                style="modern",
            )
            plotter.show()
