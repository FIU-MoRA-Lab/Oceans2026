import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# Ensure module path is accessible
sys.path.insert(0, os.path.abspath("advection_diffusion_simulator"))
from auv_optimizer import AUVFeatureOptimizer


def fetch_satellite_image(
    lon_min, lon_max, lat_min, lat_max, mode="RGB", size=(800, 800), cache_dir="results/sat_cache"
):
    """
    Fetch satellite imagery from Esri World Imagery REST API for the specified bounding box.
    Caches images locally to avoid redundant network requests.
    Mode: 'RGB' for true color, 'L' for single-channel grayscale (for colormapping).
    """
    import io
    import urllib.request
    from PIL import Image

    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = os.path.join(
        cache_dir, f"sat_{lon_min:.4f}_{lon_max:.4f}_{lat_min:.4f}_{lat_max:.4f}.png"
    )

    if os.path.exists(cache_filename):
        try:
            return np.array(Image.open(cache_filename).convert(mode))
        except Exception:
            pass

    url = (
        f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export?"
        f"bbox={lon_min},{lat_min},{lon_max},{lat_max}&bboxSR=4326&imageSR=4326&size={size[0]},{size[1]}&f=image"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            img_data = response.read()
        raw_img = Image.open(io.BytesIO(img_data))
        raw_img.save(cache_filename)
        return np.array(raw_img.convert(mode))
    except Exception as e:
        print(f"[Warning] Could not fetch satellite image: {e}")
        return None


def run_and_visualize(
    sim_path="coupled_simulations/port_canaveral_simulation_data.pkl",
    output_dir="results",
    duration=3000.0,
    n_steps=40,
    v_max=1.0,
    save_anim=True,
    cmap="YlOrRd",
    sat_cmap="Greens",
):
    """
    Run trajectory optimization and baseline comparisons, and generate plots & animations.
    """
    os.makedirs(output_dir, exist_ok=True)
    sim_filename = os.path.basename(sim_path)
    sim_name = sim_filename.replace("_simulation_data.pkl", "").replace(".pkl", "")
    optimizer = AUVFeatureOptimizer(sim_path)

    # Pick a starting position in the ocean near the tracer plume
    Lon, Lat = optimizer.data["meshgrid"]
    bath = optimizer.data["interpolated_bathymetry"]
    sol_grid = optimizer.data["solutions"]

    # Locate peak concentration coordinate as reference target
    max_idx = np.unravel_index(np.argmax(sol_grid), sol_grid.shape)
    t_peak_idx, y_peak_idx, x_peak_idx = max_idx
    peak_lon = optimizer.lon_grid[x_peak_idx]
    peak_lat = optimizer.lat_grid[y_peak_idx]

    # Pick start position offset from peak in ocean region
    start_lon = peak_lon #- 0.05
    start_lat = peak_lat - 0.04
    if optimizer.is_land(start_lon, start_lat):
        # Fallback to mean lon/lat if land
        ocean_indices = np.argwhere(bath < -0.5)
        if len(ocean_indices) > 0:
            idx = ocean_indices[len(ocean_indices) // 3]
            start_lat = optimizer.lat_grid[idx[0]]
            start_lon = optimizer.lon_grid[idx[1]]

    print("\n" + "=" * 80)
    print(f"AUV Path Optimization & Visualization ({os.path.basename(sim_path)})")
    print(f"  Start Coordinate: lon={start_lon:.4f}, lat={start_lat:.4f}")
    print(f"  Feature Peak:      lon={peak_lon:.4f}, lat={peak_lat:.4f}")
    print(f"  Duration: {duration}s ({n_steps} steps, dt={duration/n_steps:.1f}s)")
    print("=" * 80 + "\n")

    # 1. Run 3 Policies
    print("[1/3] Running Pure Drift policy...")
    res_drift = optimizer.run_pure_drift(
        start_lon, start_lat, t0=0.0, duration=duration, n_steps=n_steps
    )

    print("[2/3] Running Unoptimized Direct Pursuit policy...")
    res_pursuit = optimizer.run_unoptimized_pursuit(
        start_lon, start_lat, t0=0.0, duration=duration, n_steps=n_steps, v_max=v_max
    )

    print("[3/3] Running Energy-Optimal Feature Exploration Optimizer...")
    res_opt = optimizer.optimize_trajectory(
        start_lon,
        start_lat,
        t0=0.0,
        duration=duration,
        n_steps=n_steps,
        v_max=v_max,
        w_energy=1.0,
        w_explore=25.0,
        w_land=1e6,
    )

    # Metrics Summary
    print("\n" + "=" * 80)
    print(f"{'Policy':<30} | {'Energy (J)':<12} | {'Exploration Reward':<18} | {'Efficiency (R/E)':<16} | {'Land Collisions':<15}")
    print("-" * 100)
    policies = [
        ("Pure Drift", res_drift["metrics"]),
        ("Unoptimized Pursuit", res_pursuit["metrics"]),
        ("Energy-Optimal (Ours)", res_opt["metrics"]),
    ]
    for name, m in policies:
        print(
            f"{name:<30} | {m['total_energy_J']:<12.2f} | {m['total_explore_reward']:<18.2f} | {m['energy_efficiency_ratio']:<16.4f} | {m['land_violations']:<15}"
        )
    print("=" * 100 + "\n")

    # ---------------------------------------------------------
    # STATIC SUMMARY PLOT (results/robot_exploration_path.png)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.8, 1, 1], height_ratios=[1, 1])

    ax_map = fig.add_subplot(gs[:, 0])
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_explore = fig.add_subplot(gs[0, 2])
    ax_eff = fig.add_subplot(gs[1, 1:])

    # --- Map Plot ---
    t_mid = duration / 2.0
    lon_min, lon_max = optimizer.swe_cfg["lon_range"]
    lat_min, lat_max = optimizer.swe_cfg["lat_range"]
    
    use_rgb = (sat_cmap is None) or (str(sat_cmap).lower() == "rgb")
    sat_mode = "RGB" if use_rgb else "L"
    sat_img = fetch_satellite_image(lon_min, lon_max, lat_min, lat_max, mode=sat_mode)

    ax_map.set_aspect("equal")
    if sat_img is not None:
        if use_rgb:
            ax_map.imshow(sat_img, extent=[lon_min, lon_max, lat_min, lat_max], origin="upper", aspect="equal")
        else:
            ax_map.imshow(sat_img, extent=[lon_min, lon_max, lat_min, lat_max], origin="upper", aspect="equal", cmap=sat_cmap)
    else:
        ax_map.set_facecolor("#dcdcdc")  # Fallback to gray for land background

    # Tracer background contour (masked to ocean non-land areas)
    X_lon, Y_lat = np.meshgrid(optimizer.lon_grid, optimizer.lat_grid)
    C_mid = np.zeros_like(X_lon)
    for i in range(len(optimizer.lat_grid)):
        for j in range(len(optimizer.lon_grid)):
            if bath[i, j] < 0.0:
                C_mid[i, j] = optimizer.sol_interp(optimizer.lon_grid[j], optimizer.lat_grid[i], t_mid)
            else:
                C_mid[i, j] = np.nan

    land_mask = bath >= 0.0
    C_mid_masked = np.ma.masked_where(land_mask, C_mid)

    cntr = ax_map.contourf(X_lon, Y_lat, C_mid_masked, levels=10, cmap=cmap, alpha=0.65)
    cbar = fig.colorbar(cntr, ax=ax_map, shrink=0.75, pad=0.02)
    cbar.set_label("Concentration Field (Ocean)", fontsize=12, fontweight="bold")

    # Current vector quiver overlay (masked to non-land ocean areas)
    skip = 4
    X_q = X_lon[::skip, ::skip]
    Y_q = Y_lat[::skip, ::skip]
    U_c = np.zeros_like(X_q)
    V_c = np.zeros_like(Y_q)
    land_mask_q = land_mask[::skip, ::skip]

    for i_idx, i in enumerate(range(0, len(optimizer.lat_grid), skip)):
        for j_idx, j in enumerate(range(0, len(optimizer.lon_grid), skip)):
            if not land_mask_q[i_idx, j_idx]:
                v_vec = optimizer.vel_interp(optimizer.lon_grid[j], optimizer.lat_grid[i], t_mid)
                U_c[i_idx, j_idx] = v_vec[0]
                V_c[i_idx, j_idx] = v_vec[1]
            else:
                U_c[i_idx, j_idx] = np.nan
                V_c[i_idx, j_idx] = np.nan

    speed_q = np.sqrt(U_c**2 + V_c**2)
    q = ax_map.quiver(X_q, Y_q, U_c, V_c, speed_q, cmap="inferno", alpha=0.8)
    qbar = fig.colorbar(q, ax=ax_map, shrink=0.75, pad=0.02)
    qbar.set_label("Velocity (m/s)", fontsize=12, fontweight="bold")

    # Trajectories
    t_drift = res_drift["trajectory"]
    t_pursuit = res_pursuit["trajectory"]
    t_opt = res_opt["trajectory"]

    ax_map.plot(
        t_drift["lons"],
        t_drift["lats"],
        "o--",
        color="gray",
        linewidth=2,
        markersize=4,
        label="Pure Drift",
    )
    ax_map.plot(
        t_pursuit["lons"],
        t_pursuit["lats"],
        "^--",
        color="crimson",
        linewidth=2.2,
        markersize=5,
        label="Unoptimized Pursuit",
    )
    ax_map.plot(
        t_opt["lons"],
        t_opt["lats"],
        "*-",
        color="teal",
        linewidth=3.0,
        markersize=8,
        label="Energy-Optimal Exploration (Ours)",
    )

    # Start and End points
    ax_map.plot(start_lon, start_lat, "k*", markersize=14, markeredgecolor="yellow", label="Start")
    ax_map.plot(t_opt["lons"][-1], t_opt["lats"][-1], "s", color="darkgreen", markersize=10, label="Opt End")

    ax_map.set_title(f"AUV Trajectories on Ocean Feature & Current Field\n({os.path.basename(sim_path)})", fontsize=14, fontweight="bold")
    ax_map.set_xlabel("Longitude (°E)", fontsize=13)
    ax_map.set_ylabel("Latitude (°N)", fontsize=13)
    ax_map.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax_map.grid(True, linestyle=":", alpha=0.4)

    # --- Performance Bar Charts ---
    labels = ["Pure Drift", "Unoptimized\nPursuit", "Energy-Optimal\n(Ours)"]
    colors = ["gray", "crimson", "teal"]

    # Energy Chart
    energies = [res_drift["metrics"]["total_energy_J"], res_pursuit["metrics"]["total_energy_J"], res_opt["metrics"]["total_energy_J"]]
    bars_e = ax_energy.bar(labels, energies, color=colors, alpha=0.85, edgecolor="black")
    ax_energy.set_title("Total Propulsion Energy (J)", fontsize=13, fontweight="bold")
    ax_energy.set_ylabel("Energy (Joules)", fontsize=12)
    for bar in bars_e:
        yval = bar.get_height()
        ax_energy.text(bar.get_x() + bar.get_width()/2.0, yval + max(energies)*0.02, f"{yval:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Exploration Reward Chart
    rewards = [res_drift["metrics"]["total_explore_reward"], res_pursuit["metrics"]["total_explore_reward"], res_opt["metrics"]["total_explore_reward"]]
    bars_r = ax_explore.bar(labels, rewards, color=colors, alpha=0.85, edgecolor="black")
    ax_explore.set_title("Feature Exploration Reward", fontsize=13, fontweight="bold")
    ax_explore.set_ylabel("Sampled Concentration", fontsize=12)
    for bar in bars_r:
        yval = bar.get_height()
        ax_explore.text(bar.get_x() + bar.get_width()/2.0, yval + max(rewards)*0.02, f"{yval:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Efficiency Ratio Chart
    effs = [res_drift["metrics"]["energy_efficiency_ratio"], res_pursuit["metrics"]["energy_efficiency_ratio"], res_opt["metrics"]["energy_efficiency_ratio"]]
    bars_eff = ax_eff.bar(labels, effs, color=colors, alpha=0.85, edgecolor="black", width=0.4)
    ax_eff.set_title("Energy Efficiency Ratio (Exploration Reward / Energy)", fontsize=13, fontweight="bold")
    ax_eff.set_ylabel("Efficiency (Reward/Joule)", fontsize=12)
    for bar in bars_eff:
        yval = bar.get_height()
        ax_eff.text(bar.get_x() + bar.get_width()/2.0, yval + max(effs)*0.02, f"{yval:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    static_png_path = os.path.join(output_dir, "robot_exploration_path.png")
    static_png_path = os.path.join(output_dir, f"{sim_name}_robot_exploration_path.png")
    plt.savefig(static_png_path, dpi=200)
    print(f"✓ Saved static path summary plot to: {static_png_path}")

    # ---------------------------------------------------------
    # ANIMATED GIF GENERATION (results/robot_exploration_animation.gif)
    # ANIMATED GIF & SNAPSHOT GENERATION
    # ---------------------------------------------------------
    if save_anim:
        print("\nGenerating animated GIF trajectory progression...")
        print("\nGenerating animated GIF trajectory progression and snapshots...")
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "cm"

        fig_anim, ax_anim = plt.subplots(figsize=(7, 5), dpi=200)

        # Dynamic contour setup
        C_init = np.zeros_like(X_lon)
        for i in range(len(optimizer.lat_grid)):
            for j in range(len(optimizer.lon_grid)):
                if bath[i, j] < 0.0:
                    C_init[i, j] = optimizer.sol_interp(optimizer.lon_grid[j], optimizer.lat_grid[i], 0.0)
                else:
                    C_init[i, j] = np.nan

        ax_anim.set_aspect("equal")
        if sat_img is not None:
            if use_rgb:
                ax_anim.imshow(sat_img, extent=[lon_min, lon_max, lat_min, lat_max], origin="upper", aspect="equal")
            else:
                ax_anim.imshow(sat_img, extent=[lon_min, lon_max, lat_min, lat_max], origin="upper", aspect="equal", cmap=sat_cmap)
        else:
            ax_anim.set_facecolor("#dcdcdc")

        C_init_masked = np.ma.masked_where(land_mask, C_init)
        levels = 10
        cntr_anim = [ax_anim.contourf(X_lon, Y_lat, C_init_masked, levels=levels, cmap=cmap, alpha=0.65)]

        # Initial Velocity Field Quiver setup (cool cmap color-coded by water speed)
        skip = 4
        X_q = X_lon[::skip, ::skip]
        Y_q = Y_lat[::skip, ::skip]
        land_mask_q = land_mask[::skip, ::skip]

        U_c = np.zeros_like(X_q)
        V_c = np.zeros_like(Y_q)
        for i_idx, i in enumerate(range(0, len(optimizer.lat_grid), skip)):
            for j_idx, j in enumerate(range(0, len(optimizer.lon_grid), skip)):
                if not land_mask_q[i_idx, j_idx]:
                    v_vec = optimizer.vel_interp(optimizer.lon_grid[j], optimizer.lat_grid[i], 0.0)
                    U_c[i_idx, j_idx] = v_vec[0]
                    V_c[i_idx, j_idx] = v_vec[1]
                else:
                    U_c[i_idx, j_idx] = np.nan
                    V_c[i_idx, j_idx] = np.nan

        speed_q = np.sqrt(U_c**2 + V_c**2)
        v_speed_max = 0.35
        quiver_anim = [ax_anim.quiver(X_q, Y_q, U_c, V_c, speed_q, cmap="cool", alpha=0.85, clim=(0, v_speed_max))]

        # Colorbar 1: Water Feature Concentration
        cbar_c = fig_anim.colorbar(cntr_anim[0], ax=ax_anim, shrink=0.75, pad=0.02)
        cbar_c.set_label("Water Feature Concentration", fontsize=10)
        cbar_c.ax.tick_params(labelsize=9)

        # Colorbar 2: Water Current Velocity Speed (cool cmap)
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        sm_vel = cm.ScalarMappable(cmap="cool", norm=mcolors.Normalize(vmin=0, vmax=v_speed_max))
        cbar_v = fig_anim.colorbar(sm_vel, ax=ax_anim, shrink=0.75, pad=0.08)
        cbar_v.set_label("Water Current Speed (m/s)", fontsize=10)
        cbar_v.ax.tick_params(labelsize=9)

        (line_drift,) = ax_anim.plot([], [], "o--", color="gray", lw=2, label="Pure Drift")
        (line_pursuit,) = ax_anim.plot([], [], "^--", color="crimson", lw=2.2, label="Unoptimized Pursuit")
        (line_opt,) = ax_anim.plot([], [], "*-", color="teal", lw=3.0, ms=8, label="Energy-Optimal (Ours)")

        (head_drift,) = ax_anim.plot([], [], "o", color="gray", ms=8)
        (head_pursuit,) = ax_anim.plot([], [], "^", color="crimson", ms=9)
        (head_opt,) = ax_anim.plot([], [], "*", color="teal", ms=12)

        ax_anim.set_title("AUV Trajectory Progression (t = 0 s)", fontsize=11)
        ax_anim.set_xlabel("Longitude (°E)", fontsize=10)
        ax_anim.set_ylabel("Latitude (°N)", fontsize=10)
        ax_anim.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_anim.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_anim.tick_params(axis="both", which="major", labelsize=9)
        ax_anim.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax_anim.grid(True, linestyle=":", alpha=0.4)

        def update_frame(k):
            t_curr = t_opt["times"][k]
            ax_anim.set_title(
                f"AUV Exploration & Flow Exploitation (t = {int(t_curr)} s / {int(duration)} s)",
                fontsize=11,
            )

            # Update concentration contour field for current frame time t_curr (masked to ocean)
            C_curr = np.zeros_like(X_lon)
            for i in range(len(optimizer.lat_grid)):
                for j in range(len(optimizer.lon_grid)):
                    if bath[i, j] < 0.0:
                        C_curr[i, j] = optimizer.sol_interp(optimizer.lon_grid[j], optimizer.lat_grid[i], t_curr)
                    else:
                        C_curr[i, j] = np.nan

            C_curr_masked = np.ma.masked_where(land_mask, C_curr)

            if hasattr(cntr_anim[0], "collections"):
                for coll in list(cntr_anim[0].collections):
                    coll.remove()
            cntr_anim[0] = ax_anim.contourf(X_lon, Y_lat, C_curr_masked, levels=levels, cmap=cmap, alpha=0.65)

            # Update velocity field quiver plot (cool cmap)
            if len(quiver_anim) > 0 and quiver_anim[0] is not None:
                quiver_anim[0].remove()

            U_c_frame = np.zeros_like(X_q)
            V_c_frame = np.zeros_like(Y_q)
            for i_idx, i in enumerate(range(0, len(optimizer.lat_grid), skip)):
                for j_idx, j in enumerate(range(0, len(optimizer.lon_grid), skip)):
                    if not land_mask_q[i_idx, j_idx]:
                        v_vec = optimizer.vel_interp(optimizer.lon_grid[j], optimizer.lat_grid[i], t_curr)
                        U_c_frame[i_idx, j_idx] = v_vec[0]
                        V_c_frame[i_idx, j_idx] = v_vec[1]
                    else:
                        U_c_frame[i_idx, j_idx] = np.nan
                        V_c_frame[i_idx, j_idx] = np.nan

            speed_q_frame = np.sqrt(U_c_frame**2 + V_c_frame**2)
            quiver_anim[0] = ax_anim.quiver(X_q, Y_q, U_c_frame, V_c_frame, speed_q_frame, cmap="cool", alpha=0.85, clim=(0, v_speed_max))

            line_drift.set_data(t_drift["lons"][: k + 1], t_drift["lats"][: k + 1])
            line_pursuit.set_data(t_pursuit["lons"][: k + 1], t_pursuit["lats"][: k + 1])
            line_opt.set_data(t_opt["lons"][: k + 1], t_opt["lats"][: k + 1])

            head_drift.set_data([t_drift["lons"][k]], [t_drift["lats"][k]])
            head_pursuit.set_data([t_pursuit["lons"][k]], [t_pursuit["lats"][k]])
            head_opt.set_data([t_opt["lons"][k]], [t_opt["lats"][k]])

            return line_drift, line_pursuit, line_opt, head_drift, head_pursuit, head_opt

        # Save 3 snapshot PNGs at 0, n_steps // 2, and n_steps
        snapshots = [
            (0, "first"),
            (n_steps // 2, "mid"),
            (n_steps, "last"),
        ]
        for k_idx, label in snapshots:
            update_frame(k_idx)
            snap_path = os.path.join(output_dir, f"{sim_name}_{label}.png")
            fig_anim.savefig(snap_path, dpi=200, bbox_inches="tight")
            snap_path_full = os.path.join(output_dir, f"{sim_name}_robot_exploration_animation_{label}.png")
            fig_anim.savefig(snap_path_full, dpi=200, bbox_inches="tight")
            print(f"✓ Saved snapshot ({label}, step {k_idx}) to: {snap_path}")

        anim = animation.FuncAnimation(fig_anim, update_frame, frames=n_steps + 1, interval=150, blit=False)
        anim_path = os.path.join(output_dir, "robot_exploration_animation.gif")
        anim_path = os.path.join(output_dir, f"{sim_name}_robot_exploration_animation.gif")
        anim.save(anim_path, writer="pillow", fps=7, dpi=200)
        plt.close(fig_anim)
        print(f"✓ Saved trajectory animation GIF to: {anim_path}")

    plt.close("all")
    print("\nVisualization generation complete!")
    print(f"\nVisualization generation complete for {sim_name}!")


if __name__ == "__main__":
    import glob

    parser = argparse.ArgumentParser(description="AUV Energy-Efficient Feature Exploration Trajectory Visualizer")
    parser.add_argument(
        "--sim",
        type=str,
        default="coupled_simulations/port_canaveral_simulation_data.pkl",
        help="Path to simulation pickle file",
        default="all",
        help="Path to simulation pickle file, or 'all' to run all simulations in coupled_simulations/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for generated plots",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3000.0,
        help="Trajectory duration in seconds",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=35,
        help="Number of trajectory steps",
    )
    parser.add_argument(
        "--v_max",
        type=float,
        default=1.0,
        help="Max AUV thrust speed in m/s",
    )
    parser.add_argument(
        "--no_anim",
        action="store_true",
        help="Skip generating animated GIF",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="YlOrRd",
        help="Matplotlib colormap for water feature (default: YlOrRd, options: viridis, Greens, YlGnBu, cool, etc.)",
    )
    parser.add_argument(
        "--sat_cmap",
        type=str,
        default="Greens",
        help="Matplotlib colormap for satellite background terrain (default: Greens; set to 'rgb' for natural color RGB)",
    )

    args = parser.parse_args()
    run_and_visualize(
        sim_path=args.sim,
        output_dir=args.output_dir,
        duration=args.duration,
        n_steps=args.n_steps,
        v_max=args.v_max,
        save_anim=not args.no_anim,
        cmap=args.cmap,
        sat_cmap=args.sat_cmap,
    )

    if args.sim.lower() == "all":
        sim_files = sorted(glob.glob("coupled_simulations/*.pkl"))
    elif os.path.isdir(args.sim):
        sim_files = sorted(glob.glob(os.path.join(args.sim, "*.pkl")))
    else:
        sim_files = [args.sim]

    for sim_file in sim_files:
        print(f"\n{'='*80}\nProcessing: {sim_file}\n{'='*80}")
        run_and_visualize(
            sim_path=sim_file,
            output_dir=args.output_dir,
            duration=args.duration,
            n_steps=args.n_steps,
            v_max=args.v_max,
            save_anim=not args.no_anim,
            cmap=args.cmap,
            sat_cmap=args.sat_cmap,
        )

