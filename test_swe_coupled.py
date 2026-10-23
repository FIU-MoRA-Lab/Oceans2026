import argparse
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import yaml

# Add the advection_diffusion_simulator path
sys.path.insert(0, os.path.abspath("advection_diffusion_simulator"))

from advection_diffusion_simulator.config import AdvectionDiffusionConfig
from advection_diffusion_simulator.coordinate_mapper import GeographicCoordinateMapper
from advection_diffusion_simulator.solver import ADSolver
from utils import get_interpolator, read_pkl


def load_yaml_config(yaml_path):
    """Load configuration dictionary from YAML file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def run_coupled_simulation(yaml_path="configs/swe_coupled_config.yaml"):
    """
    Run coupled SWE velocity field advection-diffusion simulation based on YAML configuration.

    Parameters:
    -----------
    yaml_path : str
        Path to YAML configuration file.

    Returns:
    --------
    dict: Output dictionary containing solutions, meshgrid, time_steps, interpolators, and metadata.
    """
    print("=" * 80)
    print("Coupled SWE Velocity Field Advection-Diffusion Simulation")
    print("=" * 80)
    print(f"Loading configuration from: {yaml_path}")
    cfg = load_yaml_config(yaml_path)

    # 1. Load SWE pickle
    input_cfg = cfg.get("input", {})
    swe_path = input_cfg.get("swe_pickle_path", "swe_data/key_west_result.pkl")
    speed_mult = input_cfg.get("speed_multiplier", 5.0)
    speed_thresh = input_cfg.get("speed_threshold", 0.1)

    print(f"\nLoading velocity field from: {swe_path}...")
    swe_data = read_pkl(swe_path)
    config_swe = swe_data["config"]

    # Extract speed components u and v: shape (T, 2, nx, ny)
    speed_matrix = speed_mult * swe_data["solution"][:, 1:3, :, :]
    mask = np.abs(speed_matrix) > speed_thresh

    print(
        f"  Speed stats (> {speed_thresh}): mean={np.mean(speed_matrix[mask]):.4f}, "
        f"max={np.max(speed_matrix[mask]):.4f}, min={np.min(speed_matrix[mask]):.4f}"
    )
    print(f"  Loaded velocity matrix with shape: {speed_matrix.shape}")

    # 2. Maximum velocity coordinate
    t_idx, c_idx, x_idx, y_idx = np.unravel_index(
        np.argmax(np.abs(speed_matrix)), speed_matrix.shape
    )
    init_cfg = cfg.get("initial_condition", {})
    central_x = init_cfg.get("central_x", -81.75)
    central_y = init_cfg.get("central_y", 24.475)
    buffer = init_cfg.get("buffer", 1000 / 111111)

    lon_min = config_swe["lon_range"][0] - buffer
    lon_max = config_swe["lon_range"][1] + buffer
    lat_min = config_swe["lat_range"][0] - buffer
    lat_max = config_swe["lat_range"][1] + buffer

    nx, ny = config_swe["nx"], config_swe["ny"]
    x_coords = np.linspace(lon_min, lon_max, nx)
    y_coords = np.linspace(lat_min, lat_max, ny)

    max_lon = x_coords[x_idx]
    max_lat = y_coords[y_idx]
    print(
        f"  Max velocity coordinate: lon={max_lon:.5f}, lat={max_lat:.5f} (t_idx={t_idx})"
    )

    # 3. Create velocity interpolator
    pde_cfg = cfg.get("pde_params", {})
    t_final_sim = pde_cfg.get("t_final_sim", 20000.0)
    dt_sim = pde_cfg.get("dt_sim", 1000.0)
    mesh_resolution = pde_cfg.get("mesh_resolution", 20)

    print("\nCreating velocity interpolator...")
    velocity_interp = get_interpolator(
        speed_matrix, config=config_swe, t_final=t_final_sim
    )

    # 4. Domain & AD Config
    polygon_points = [
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
    ]

    out_cfg = cfg.get("output", {})
    output_dir = out_cfg.get("output_dir", "results")

    config_ad = AdvectionDiffusionConfig(
        polygon_points=np.asarray(polygon_points),
        mesh_resolution=mesh_resolution,
        t_final=t_final_sim,
        dt=dt_sim,
        diffusivity=pde_cfg.get("diffusivity", 0.01),
        element_degree=pde_cfg.get("element_degree", 1),
        element_family=pde_cfg.get("element_family", "P"),
        are_coordinates_lonlat=pde_cfg.get("are_coordinates_lonlat", True),
        # output_dir=output_dir,
    )

    print(f"\nSimulation Parameters:")
    print(f"  t_final={config_ad.t_final}s, dt={config_ad.dt}s")
    print(f"  Diffusivity: {config_ad.diffusivity} m^2/s")
    print(f"  Mesh resolution: {mesh_resolution}")
    print(f"  Output directory: {output_dir}")

    # 5. Define AD functions
    def forcing_term(x, y):
        return 0.0

    mapper = GeographicCoordinateMapper(
        lon0=np.mean(config_ad.polygon_points[:, 0]),
        lat0=np.mean(config_ad.polygon_points[:, 1]),
    )

    def velocity_field(x, y, t):
        lon, lat = mapper.metric_to_coord(x, y)
        vx, vy = velocity_interp(lon, lat, t)
        return [vx, vy]

    sigma = init_cfg.get("sigma", 2000.0)
    amplitude = init_cfg.get("amplitude", 27.0)

    def initial_condition(x, y):
        center_x, center_y = mapper.coord_to_metric(central_x, central_y)
        return (amplitude * 3 / 4) * np.exp(
            -((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma**2)
        ) + amplitude / 4

    def boundary_condition(x, y):
        return 0.0

    # 6. Initialize and run solver
    print("\nInitializing ADSolver...")
    solver = ADSolver(config=config_ad)
    solver.set_forcing_term(forcing_term)
    solver.set_velocity_field(velocity_field)
    solver.set_initial_condition(initial_condition)
    solver.set_boundary_condition(boundary_condition)

    print("Setting up solver...")
    solver.setup_solver()
    print(f"  Mesh vertices: {solver.mesh.num_vertices()}")
    print(f"  Function space dim: {solver.function_space.dim()}")

    print("\nRunning simulation...")
    result = solver.solve()
    print(f"Simulation finished! Time steps: {len(result.solutions)}")

    # 7. Create solution interpolator if requested
    solution_interp = None
    if out_cfg.get("create_solution_interpolator", True):
        print("\nCreating PDE solution interpolator...")
        sol_nx = result.solutions.shape[1]
        sol_ny = result.solutions.shape[2]
        solution_interp = get_interpolator(
            result.solutions,
            lon_range=(lon_min, lon_max),
            lat_range=(lat_min, lat_max),
            nx=sol_nx,
            ny=sol_ny,
            t_final=result.time_steps[-1],
            dt=config_ad.dt,
        )

    # 8. Compute Bathymetry / Ocean Mask
    bathymetry = swe_data.get("bathymetry")
    ocean_mask = None
    interpolated_bathymetry = None
    if bathymetry is not None:
        from scipy.interpolate import RegularGridInterpolator

        print("Computing bathymetry ocean mask...")
        swe_x = np.linspace(
            config_swe["lon_range"][0], config_swe["lon_range"][1], config_swe["nx"]
        )
        swe_y = np.linspace(
            config_swe["lat_range"][0], config_swe["lat_range"][1], config_swe["ny"]
        )
        b_rgi = RegularGridInterpolator(
            (swe_x, swe_y), bathymetry, bounds_error=False, fill_value=1.0
        )

        Lon, Lat = result.meshgrid[0], result.meshgrid[1]
        pts = np.stack([Lon, Lat], axis=-1)
        interpolated_bathymetry = b_rgi(pts)
        ocean_mask = interpolated_bathymetry < 0.0
        print(
            f"  Ocean mask created: {np.sum(ocean_mask)} ocean grid points out of {ocean_mask.size} total."
        )

    # 9. Package output dictionary
    output_data = {
        "solutions": result.solutions,
        "meshgrid": result.meshgrid,
        "time_steps": result.time_steps,
        "mesh_mask": result.mesh_mask,
        "bathymetry": bathymetry,
        "interpolated_bathymetry": interpolated_bathymetry,
        "ocean_mask": ocean_mask,
        "velocity_interpolator": velocity_interp,
        "solution_interpolator": solution_interp,
        "config_ad": config_ad,
        "yaml_config": cfg,
        "swe_config": config_swe,
    }

    # 10. Save pickle output data
    save_pkl_filename = out_cfg.get("save_pkl_filename", "coupled_simulation_data.pkl")
    os.makedirs(output_dir, exist_ok=True)
    pkl_output_path = os.path.join(output_dir, save_pkl_filename)
    print(f"Saving simulation data to: {pkl_output_path}...")
    with open(pkl_output_path, "wb") as f:
        pickle.dump(output_data, f)
    print(f"✓ Saved simulation data pickle successfully.")

    print("\n" + "=" * 80)
    print("Simulation execution complete!")
    print("=" * 80)

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run coupled SWE advection-diffusion simulation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/swe_coupled_config.yaml",
        help="Path to YAML configuration file (default: configs/swe_coupled_config.yaml)",
    )
    args = parser.parse_args()
    run_coupled_simulation(args.config)
