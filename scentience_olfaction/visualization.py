"""Physical-unit plume views. Matplotlib is optional and imported on demand."""

from __future__ import annotations

import numpy as np


def concentration_slice(plume, *, xlim=(-2.0, 16.0), ylim=(-5.0, 5.0),
                        height_m=1.0, resolution=(100, 60), species=None):
    """Sample an XY plane without advancing transport or sensor random state."""
    nx, ny = resolution
    if not all(isinstance(v, int) and v >= 2 for v in resolution):
        raise ValueError("resolution must contain two integers >= 2")
    if (not np.isfinite((*xlim, *ylim, height_m)).all()
            or xlim[0] >= xlim[1] or ylim[0] >= ylim[1]):
        raise ValueError("Slice bounds must be finite and ordered")
    if not plume.species_names:
        raise ValueError("The plume has no species to visualize")
    name = species or plume.species_names[0]
    if name not in plume.species_names:
        raise ValueError(f"Unknown slice species {name!r}; choose {plume.species_names}")
    x, y = np.linspace(*xlim, nx), np.linspace(*ylim, ny)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, height_m)))
    concentration = plume.sample_species(points)[:, plume.species_names.index(name)]
    return x, y, concentration.reshape(ny, nx)


def plot_plume(plume, *, ax=None, species=None, height_m=1.0,
               xlim=(-2.0, 16.0), ylim=(-5.0, 5.0), log_scale=True):
    """Concentration heatmap, source locations, and local wind vectors.

    A logarithmic color scale reveals dilute filaments; the colorbar always
    gives ppm. The plane is a concentration query, not a particle-density map.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    if not plume.species_names:
        raise ValueError("The plume has no species to visualize")
    name = species or plume.species_names[0]
    x, y, values = concentration_slice(plume, xlim=xlim, ylim=ylim,
                                     height_m=height_m, species=name, resolution=(240, 120))
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.6))
    positive = values[values > 0]
    vmax = max(float(values.max()), 1e-6)
    vmin = max(vmax * 1e-4, float(positive.min())) if positive.size else vmax * 1e-4
    norm = LogNorm(vmin=min(vmin, vmax * 0.99), vmax=vmax) if log_scale else Normalize(0, vmax)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eef2f5")
    mesh = ax.pcolormesh(x, y, values, shading="auto", cmap=cmap, norm=norm)
    ax.figure.colorbar(mesh, ax=ax, label=f"{name.replace('_', ' ')} [ppm]")
    for e in plume.emitters:
        ax.scatter(*e.position[:2], marker="*", s=110, color="#ffb454", edgecolors="black")
    xx, yy = np.meshgrid(np.linspace(*xlim, 9), np.linspace(*ylim, 5))
    points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, height_m)))
    wind = plume.airflow.velocity(points)
    q = ax.quiver(xx, yy, wind[:, 0].reshape(xx.shape), wind[:, 1].reshape(xx.shape),
                  color="#738598", alpha=0.6, scale=16)
    ax.quiverkey(q, 0.88, 1.05, 1.0, "1 m/s", labelpos="E", color="black")
    ax.set(xlabel="x [m]", ylabel="y [m]", xlim=xlim, ylim=ylim,
           title=f"Concentration at z = {height_m:g} m · t = {plume.t:.1f} s")
    ax.set_aspect("equal")
    return ax


def render_navigation(plume, position, heading, height_m):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 4), dpi=100, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    lo, hi = plume.cfg.domain_min, plume.cfg.domain_max
    plot_plume(plume, ax=ax, height_m=height_m, xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    ax.scatter(*position, color="#ff686b", s=45, zorder=4)
    ax.arrow(*position, np.cos(heading), np.sin(heading), color="#ff686b", width=0.08)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()
