import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np

# Add the advection_diffusion_simulator path
sys.path.insert(0, os.path.abspath("advection_diffusion_simulator"))

from utils import read_pkl


def plot_simulation_results(
    data_path="results/coupled_simulation_data.pkl",
    output_image_path="results/simulation_results.png",
    show_plot=True,
):
    """
    Load saved simulation results and plot concentration field with velocity field quiver arrows.

    Parameters:
    -----------
    data_path : str
        Path to input pickle file containing simulation data.
    output_image_path : str
        Path where output image figure will be saved.
    show_plot : bool
        Whether to call plt.show() after rendering.
    """
    print("=" * 80)
    print("Coupled SWE Advection-Diffusion Visualization")
    print("=" * 80)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Simulation data file not found: {data_path}")

    print(f"Loading simulation data from: {data_path}...")
    data = read_pkl(data_path)

    solutions = data["solutions"]
    meshgrid = data["meshgrid"]
    time_steps = data["time_steps"]
    velocity_interp = data["velocity_interpolator"]
    solution_interp = data.get("solution_interpolator")
    ocean_mask = data.get("ocean_mask")

    print(f"  Loaded {len(solutions)} solution snapshots.")
    print(f"  Meshgrid shape: X={meshgrid[0].shape}, Y={meshgrid[1].shape}")
    print(f"  Time range: {time_steps[0]:.1f}s to {time_steps[-1]:.1f}s")
    if ocean_mask is not None:
        print(f"  Bathymetry Ocean Mask: Active ({np.sum(ocean_mask)} ocean points)")

    # Verify solution interpolator if available
    if solution_interp is not None:
        mid_x = float(np.mean(meshgrid[0]))
        mid_y = float(np.mean(meshgrid[1]))
        mid_t = float(time_steps[len(time_steps) // 2])
        u_sample = solution_interp(mid_x, mid_y, mid_t)
        v_sample = velocity_interp(mid_x, mid_y, mid_t)
        print(f"\nInterpolator Verification:")
        print(f"  Concentration u(lon={mid_x:.4f}, lat={mid_y:.4f}, t={mid_t:.1f}s) = {float(u_sample):.4f}")
        print(f"  Velocity v(lon={mid_x:.4f}, lat={mid_y:.4f}, t={mid_t:.1f}s) = {v_sample}")

    print("\nGenerating 3x3 subplot grid visualization...")
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    indices = np.linspace(0, len(solutions) - 1, 9, dtype=np.int64)
    Lon, Lat = meshgrid[0], meshgrid[1]

    for ax, idx in zip(axes, indices):
        ax.set_aspect("equal")
        t_val = time_steps[idx]

        # Evaluate concentration solution using solution_interpolator on meshgrid
        if solution_interp is not None:
            u_eval = solution_interp(Lon, Lat, t_val)
        else:
            print("Using raw solutions")
            u_eval = solutions[idx]

        # Apply ocean mask if available
        if ocean_mask is not None:
            u_display = np.ma.masked_where(~ocean_mask, u_eval)
            ax.set_facecolor("#dcdcdc")  # Light gray for land background
        else:
            u_display = u_eval

        # Contour plot of concentration solution
        c = ax.contourf(Lon, Lat, u_display, levels=10, cmap="viridis")
        ax.set_title(f"t = {t_val:.1f}s")
        plt.colorbar(c, ax=ax)

        # Overlay velocity field quiver plot using velocity_interpolator
        vel = velocity_interp(Lon, Lat, t_val)
        U = vel[..., 0]
        V = vel[..., 1]

        # Subsample grid for legible arrows
        subsample = 3
        X_q = Lon[::subsample, ::subsample]
        Y_q = Lat[::subsample, ::subsample]
        U_q = U[::subsample, ::subsample].copy()
        V_q = V[::subsample, ::subsample].copy()

        if ocean_mask is not None:
            ocean_mask_q = ocean_mask[::subsample, ::subsample]
            U_q[~ocean_mask_q] = np.nan
            V_q[~ocean_mask_q] = np.nan

        speed_q = np.sqrt(U_q**2 + V_q**2)

        q = ax.quiver(X_q, Y_q, U_q, V_q, speed_q, cmap="inferno", alpha=0.8)
        plt.colorbar(q, ax=ax, label="Velocity (m/s)")

    plt.suptitle("Coupled Advection-Diffusion: Concentration Field & Velocity Vectors (Ocean Masked)")
    plt.tight_layout()

    out_dir = os.path.dirname(output_image_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(output_image_path, dpi=150)
    print(f"  ✓ Saved visualization plot to: {output_image_path}")

    if show_plot:
        plt.show()

    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot coupled simulation results.")
    parser.add_argument(
        "--data",
        type=str,
        default="results/coupled_simulation_data.pkl",
        help="Path to input simulation pickle file (default: results/coupled_simulation_data.pkl)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="results/simulation_results.png",
        help="Path to save output image file (default: results/simulation_results.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plot interactively",
    )
    args = parser.parse_args()
    plot_simulation_results(data_path=args.data, output_image_path=args.save, show_plot=not args.no_show)
