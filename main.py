"""Volume-render a voxel mass distribution with PyVista.

Reads ``<object-dir>/ground_truth.csv`` (2 metadata rows + header, then one row
per voxel: ``x, y, z, mass, mass_density`` on a ``2**lod`` cubic grid), lays the
normalized mass onto a uniform grid and volume-renders it. Optionally shows a
matplotlib 3D scatter, draws labelled axes, or writes a rotating mp4/gif.

This is a PyVista re-implementation of an earlier Mayavi script (the
``mayavi_vol_ren_variable_colour`` project, which used tvtk
``ColorTransferFunction`` / ``PiecewiseFunction`` / ``LUTManager`` and moviepy);
the CLI and data handling are kept compatible. Notable points:
- The empty-voxel sentinel value (``TRANSPARENT_INPUT = -1``) and the scalar
  range ``[-1, 1]`` are preserved, so a filled voxel's normalized mass ``[0, 1]``
  maps to the upper half of the colormap (matching the Mayavi original's
  ``ColorTransferFunction.range = [-1, 1]``). See ``CLIM`` below to change this.
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

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from matplotlib import colormaps as cm
from matplotlib import pyplot as plt


# --- scalar field -----------------------------------------------------------
TRANSPARENT_INPUT = -1.0  # sentinel scalar value for empty voxels
QUANT_MAX = 65535  # the scalar field is quantized to uint16 (see _quantize_scalars)
# clim spans the whole quantized range, i.e. conceptual [-1, 1]: a filled voxel's
# normalized mass [0, 1] maps to the upper half of the colormap (matching the Mayavi
# original's ``ColorTransferFunction.range = [-1, 1]``). Use [QUANT_MAX // 2,
# QUANT_MAX] to make filled voxels span the full colormap instead.
CLIM = (0, QUANT_MAX)

# --- data layout ------------------------------------------------------------
GROUND_TRUTH_FILENAME = "ground_truth.csv"
N_METADATA_ROWS = 1  # data rows after the metadata header, before the voxel header

# --- rendering --------------------------------------------------------------
WINDOW_BG_COLOR = "white"
CAMERA_ELEVATION_DEG = 45.0
STILL_CAMERA_DISTANCE = 5.0
OPACITY_SLIDER_RANGE = (0.0, 1.0)

# --- rotation video ---------------------------------------------------------
VIDEO_DIRNAME = "mass_distr"  # also the basename of the .mp4 / .gif written there
VIDEO_N_FRAMES = 200
VIDEO_N_TURNS = 2
VIDEO_CAMERA_DISTANCE = 5.9  # tightest extra space around the bbox


@dataclass
class MassDistribution:
    """A voxel mass distribution loaded from ``ground_truth.csv``."""

    metadata: pd.DataFrame  # single-row dataframe (id, aabb_scale, total_mass, ...)
    scalars: (
        np.ndarray
    )  # (res, res, res) uint16; empty voxels -> 0, normalized mass -> [.., QUANT_MAX]
    res: int  # grid resolution per axis
    ndc_coords: np.ndarray  # (res,) NDC coordinates of the voxel centres along one axis
    aabb_scale: float  # metres per NDC unit
    points_ndc: (
        np.ndarray
    )  # (3, N) voxel-centre coordinates in NDC space (for the scatter)
    mass: np.ndarray  # (N,) raw per-voxel mass (for the scatter colours)

    @property
    def grid_spacing(self) -> tuple[float, float, float]:
        step = 2.0 / self.res
        return (step, step, step)

    @property
    def grid_origin(self) -> tuple[float, float, float]:
        return (float(self.ndc_coords[0]),) * 3


def _quantize_scalars(scalars: np.ndarray) -> np.ndarray:
    """Map the conceptual scalar range ``[TRANSPARENT_INPUT, 1]`` onto ``uint16``.

    VTK's CPU volume mapper (fixed-point ray cast, used when there is no GPU) only
    accepts ``uint8``/``uint16`` scalars and renders nothing for floats, so the
    field is quantized. Empty voxels (``TRANSPARENT_INPUT``) become 0; the upper
    bound 1 maps to ``QUANT_MAX``.
    """
    span = 1.0 - TRANSPARENT_INPUT
    return np.round((scalars - TRANSPARENT_INPUT) / span * QUANT_MAX).astype(np.uint16)


def _scalar_opacity_function(
    opacity: float, scalar_range=CLIM
) -> vtk.vtkPiecewiseFunction:
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
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    focal: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list:
    """``Plotter.camera_position`` triple following Mayavi ``mlab.view()`` conventions.

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


def load_mass_distribution(object_dir: Path) -> MassDistribution:
    """Load ``<object-dir>/ground_truth.csv`` into a :class:`MassDistribution`."""
    csv_path = Path(object_dir) / GROUND_TRUTH_FILENAME

    metadata = pd.read_csv(csv_path, nrows=N_METADATA_ROWS)
    print(f"{metadata}")

    voxels = pd.read_csv(csv_path, skiprows=N_METADATA_ROWS + 1).loc[
        :, "x":"mass_density"
    ]
    print(f"{voxels.to_numpy().shape=}")
    _x, _y, _z, mass = voxels.to_numpy().T[:4]

    res = math.ceil(math.pow(_x.shape[0], 1 / 3))
    ndc_coords = (2 * np.arange(0, res) - res + 1.0) / res
    print(f"{res=}")
    aabb_scale = _x.max() / ndc_coords.max()
    print(f"{aabb_scale=}")  # [m]

    # Volume scalars: TRANSPARENT_INPUT everywhere, normalized mass on filled voxels.
    shape = (res, res, res)
    scalars = TRANSPARENT_INPUT * np.ones(shape)
    normalized_mass = (mass / mass.max()).reshape(shape)
    filled = normalized_mass.nonzero()
    scalars[filled] = normalized_mass[filled]
    scalars = _quantize_scalars(scalars)

    return MassDistribution(
        metadata=metadata,
        scalars=scalars,
        res=res,
        ndc_coords=ndc_coords,
        aabb_scale=aabb_scale,
        points_ndc=np.stack([_x, _y, _z]) / aabb_scale,
        mass=mass,
    )


def make_volume_plotter(
    md: MassDistribution,
    *,
    cmap: str = "viridis",
    alpha: float = 0.99,
    shade: bool = False,
    off_screen: bool = False,
    size: int = 720,
) -> tuple[pv.Plotter, pv.Volume]:
    """Create a :class:`pyvista.Plotter` showing ``md`` as a volume.

    ``md.scalars`` is indexed ``[i, j, k]`` with coords
    ``(ndc_coords[i], ndc_coords[j], ndc_coords[k])``; VTK ``ImageData`` wants
    point data in Fortran order w.r.t. ``(x, y, z)``.
    """
    plotter = pv.Plotter(off_screen=off_screen, window_size=[size, size])
    plotter.background_color = WINDOW_BG_COLOR

    grid = pv.ImageData(
        dimensions=md.scalars.shape, spacing=md.grid_spacing, origin=md.grid_origin
    )
    grid.point_data["mass_distr"] = md.scalars.flatten(order="F")

    volume = plotter.add_volume(
        grid,
        scalars="mass_distr",
        cmap=cmap,
        clim=list(CLIM),
        shade=shade,
        show_scalar_bar=False,
    )
    volume.prop.SetScalarOpacity(_scalar_opacity_function(alpha))
    return plotter, volume


def add_labelled_axes(plotter: pv.Plotter) -> None:
    """Draw a labelled bounding box (Mayavi ``mlab.axes`` equivalent)."""
    plotter.show_grid(xtitle="X", ytitle="Y", ztitle="Z", color="black")
    # Optional metadata overlay (the Mayavi original showed these via mlab.text):
    #   plotter.add_text(f"object: {md.metadata.at[0, 'id']}",
    #                    position=(0.02, 0.05), viewport=True, font="courier", color="black")
    #   plotter.add_text(f"aabb scale: {md.metadata.at[0, 'aabb_scale']} [m]",
    #                    position=(0.02, 0.01), viewport=True, font="courier", color="black")


def add_opacity_slider(
    plotter: pv.Plotter, volume: pv.Volume, *, initial_alpha: float
) -> None:
    """Add a bottom-left slider that rebuilds the volume's scalar-opacity function."""
    plotter.add_slider_widget(
        lambda value: volume.prop.SetScalarOpacity(_scalar_opacity_function(value)),
        rng=list(OPACITY_SLIDER_RANGE),
        value=initial_alpha,
        title="opacity",
        pointa=(0.025, 0.08),
        pointb=(0.30, 0.08),
        style="modern",
    )


def show_matplotlib_scatter(
    md: MassDistribution, *, alpha: float, cmap_name: str
) -> None:
    """Show a non-blocking matplotlib 3D scatter of the filled voxels."""
    cmap = cm.get_cmap(cmap_name)
    colors = md.mass / md.mass.max()
    sizes = np.zeros_like(md.mass)
    sizes[md.mass.nonzero()] = 1
    x, y, z = md.points_ndc

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("equal")
    size_const = 0.3
    ax.scatter(x, y, z, s=size_const * sizes, c=colors, alpha=alpha, cmap=cmap)
    plt.show(block=False)


def _write_video(
    frames: list[np.ndarray], out_dir: Path, basename: str, fps: int
) -> None:
    """Assemble already-rendered RGB frames into ``.mp4`` and ``.gif`` (replaces moviepy)."""
    import imageio.v2 as imageio

    base = Path(out_dir) / basename
    imageio.mimwrite(f"{base}.mp4", frames, fps=fps, codec="libx264")
    imageio.mimsave(f"{base}.gif", frames, fps=fps)


def render_rotation_video(plotter: pv.Plotter, out_dir: Path, *, fps: int) -> None:
    """Orbit the camera ``VIDEO_N_TURNS`` times, saving each frame, then write mp4/gif.

    Pair this with ``off_screen=True`` (the ``--offscreen`` flag) on headless machines.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    azimuth = 0.0
    azimuth_tick = VIDEO_N_TURNS * 360.0 / VIDEO_N_FRAMES
    frames: list[np.ndarray] = []
    for epoch in range(VIDEO_N_FRAMES):
        azimuth = (azimuth + azimuth_tick) % 360.0
        plotter.camera_position = _spherical_camera_position(
            azimuth, CAMERA_ELEVATION_DEG, VIDEO_CAMERA_DISTANCE
        )
        frames.append(
            plotter.screenshot(str(out_dir / f"{epoch:04}.png"), return_img=True)
        )

    _write_video(frames, out_dir, VIDEO_DIRNAME, fps=fps)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--object-dir")
    parser.add_argument(
        "--lod",
        type=int,
        default=7,
        help="kept for CLI parity; the resolution is read from the data",
    )
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"{args=}\n")

    object_dir = Path(args.object_dir)
    md = load_mass_distribution(object_dir)

    if args.matplotlib:
        show_matplotlib_scatter(md, alpha=args.alpha, cmap_name=args.cmap)

    plotter, volume = make_volume_plotter(
        md,
        cmap=args.cmap,
        alpha=args.alpha,
        shade=False,
        off_screen=args.offscreen,
        size=args.size,
    )
    if args.axes:
        add_labelled_axes(plotter)

    if args.generate_video:
        render_rotation_video(plotter, object_dir / VIDEO_DIRNAME, fps=args.fps)
        plotter.close()
        return

    # Static view: the camera angle of the rotation video's last frame.
    azimuth = VIDEO_N_TURNS * 360.0 / VIDEO_N_FRAMES * (VIDEO_N_FRAMES - 1)
    plotter.camera_position = _spherical_camera_position(
        azimuth, CAMERA_ELEVATION_DEG, STILL_CAMERA_DISTANCE
    )
    if args.save is not None:
        plotter.screenshot(args.save)
        print(f"saved still to {args.save}")
        plotter.close()
        return

    add_opacity_slider(plotter, volume, initial_alpha=args.alpha)
    plotter.show()


if __name__ == "__main__":
    main()
