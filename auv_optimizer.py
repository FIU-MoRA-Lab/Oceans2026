import os
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

# Ensure advection_diffusion_simulator module path is accessible
sys.path.insert(0, os.path.abspath("advection_diffusion_simulator"))
from advection_diffusion_simulator.coordinate_mapper import GeographicCoordinateMapper
from utils import read_pkl


class AUVFeatureOptimizer:
    """
    Trajectory Optimizer for Autonomous Underwater Vehicles (AUVs) exploring dynamic ocean features
    (e.g., tracer/pollutant plumes) while utilizing ocean currents to minimize propulsion energy
    and avoiding land boundaries.
    """

    def __init__(self, simulation_pkl_path):
        """
        Initialize optimizer by loading coupled simulation output pickle.

        Parameters
        ----------
        simulation_pkl_path : str
            Path to pickled simulation dataset (from coupled_simulations/).
        """
        if not os.path.exists(simulation_pkl_path):
            raise FileNotFoundError(f"Simulation file not found: {simulation_pkl_path}")

        print(f"[AUVOptimizer] Loading simulation data from {simulation_pkl_path}...")
        self.data = read_pkl(simulation_pkl_path)

        self.vel_interp = self.data["velocity_interpolator"]
        self.sol_interp = self.data["solution_interpolator"]
        self.swe_cfg = self.data["swe_config"]

        Lon, Lat = self.data["meshgrid"]
        self.lon_grid = Lon[0, :] if Lon.ndim == 2 else Lon
        self.lat_grid = Lat[:, 0] if Lat.ndim == 2 else Lat

        # Geographic coordinate mapper centered on domain center
        self.lon0 = float(np.mean(self.swe_cfg["lon_range"]))
        self.lat0 = float(np.mean(self.swe_cfg["lat_range"]))
        self.mapper = GeographicCoordinateMapper(lon0=self.lon0, lat0=self.lat0)

        # Bathymetry interpolator: depth b (b < 0 is ocean, b >= 0 is land)
        bath = self.data.get("interpolated_bathymetry")
        if bath is None:
            bath = np.full((len(self.lat_grid), len(self.lon_grid)), -10.0)

        self.bath_interp = RegularGridInterpolator(
            (self.lat_grid, self.lon_grid),
            bath,
            bounds_error=False,
            fill_value=1.0,  # Treat out-of-bounds as land boundary
        )

        print(
            f"[AUVOptimizer] Initialized. Domain lon: {self.swe_cfg['lon_range']}, lat: {self.swe_cfg['lat_range']}"
        )

    def is_land(self, lon, lat, safety_margin=0.0):
        """Check if coordinate (lon, lat) is land or out of bounds."""
        depth = float(self.bath_interp([[lat, lon]])[0])
        return depth + safety_margin >= 0.0

    def simulate_trajectory(self, thrust_controls, x0_m, y0_m, t0, dt):
        """
        Simulate AUV path given a sequence of thrust vector controls [vx, vy] in metric space.

        Parameters
        ----------
        thrust_controls : np.ndarray
            Thrust vector array of shape (N, 2) in m/s.
        x0_m, y0_m : float
            Starting metric coordinates (meters relative to domain center).
        t0 : float
            Starting time (seconds).
        dt : float
            Time step duration (seconds).

        Returns
        -------
        dict : Trajectory coordinates, controls, sampled features, and bathymetry depth.
        """
        N = len(thrust_controls)
        x_m = np.zeros(N + 1)
        y_m = np.zeros(N + 1)
        lons = np.zeros(N + 1)
        lats = np.zeros(N + 1)
        times = np.zeros(N + 1)
        concentrations = np.zeros(N + 1)
        bathymetry = np.zeros(N + 1)
        current_vels = np.zeros((N + 1, 2))

        x_m[0] = x0_m
        y_m[0] = y0_m
        lons[0], lats[0] = self.mapper.metric_to_coord(x0_m, y0_m)
        times[0] = t0
        concentrations[0] = float(self.sol_interp(lons[0], lats[0], t0))
        bathymetry[0] = float(self.bath_interp([[lats[0], lons[0]]])[0])
        current_vels[0] = self.vel_interp(lons[0], lats[0], t0)

        for k in range(N):
            curr_x, curr_y = x_m[k], y_m[k]
            curr_lon, curr_lat = lons[k], lats[k]
            curr_t = times[k]

            v_curr = self.vel_interp(curr_lon, curr_lat, curr_t)
            v_thr = thrust_controls[k]
            v_tot = v_thr + v_curr

            # Euler step in metric space (meters)
            next_x = curr_x + v_tot[0] * dt
            next_y = curr_y + v_tot[1] * dt
            next_t = curr_t + dt

            next_lon, next_lat = self.mapper.metric_to_coord(next_x, next_y)

            x_m[k + 1] = next_x
            y_m[k + 1] = next_y
            lons[k + 1] = next_lon
            lats[k + 1] = next_lat
            times[k + 1] = next_t
            concentrations[k + 1] = float(self.sol_interp(next_lon, next_lat, next_t))
            bathymetry[k + 1] = float(self.bath_interp([[next_lat, next_lon]])[0])
            current_vels[k + 1] = v_curr

        return {
            "x_m": x_m,
            "y_m": y_m,
            "lons": lons,
            "lats": lats,
            "times": times,
            "concentrations": concentrations,
            "bathymetry": bathymetry,
            "current_vels": current_vels,
            "thrust_controls": thrust_controls,
            "dt": dt,
        }

    def _cost_function(
        self,
        u_flat,
        x0_m,
        y0_m,
        t0,
        dt,
        N,
        w_energy,
        w_explore,
        w_land,
        v_max,
        safety_margin=0.02,
    ):
        thrust_controls = u_flat.reshape(N, 2)
        traj = self.simulate_trajectory(thrust_controls, x0_m, y0_m, t0, dt)

        # 1. Energy cost: sum of ||v_thr||^2 * dt
        thrust_sq = np.sum(thrust_controls**2, axis=1)
        energy_cost = np.sum(thrust_sq) * dt

        # 2. Exploration reward: sum of tracer concentration sampled
        explore_reward = np.sum(traj["concentrations"]) * dt

        # 3. Land boundary penalty (depth b >= 0 is land)
        land_penetrations = np.maximum(0.0, traj["bathymetry"] + safety_margin)
        land_penalty = np.sum(land_penetrations**2) * w_land

        # 4. Max speed constraint penalty
        speed_excess = np.maximum(0.0, np.sqrt(thrust_sq) - v_max)
        speed_penalty = np.sum(speed_excess**2) * 1e4

        total_cost = (
            w_energy * energy_cost
            - w_explore * explore_reward
            + land_penalty
            + speed_penalty
        )
        return total_cost

    def optimize_trajectory(
        self,
        x0_lon,
        y0_lat,
        t0=0.0,
        duration=2000.0,
        n_steps=40,
        v_max=1.0,
        w_energy=1.0,
        w_explore=20.0,
        w_land=1e8,
        init_strategy="pursuit",
    ):
        """
        Compute optimal thrust controls to explore concentration feature while saving energy and avoiding land.

        Parameters
        ----------
        x0_lon, y0_lat : float
            Starting longitude and latitude.
        t0 : float
            Start time in seconds.
        duration : float
            Total trajectory simulation duration in seconds.
        n_steps : int
            Number of control steps.
        v_max : float
            Maximum thrust speed (m/s).
        w_energy, w_explore, w_land : float
            Weighting factors for energy, exploration, and land boundary avoidance.
        init_strategy : str
            Initial guess strategy ('zero', 'pursuit').

        Returns
        -------
        dict : Optimization result with optimized trajectory and performance metrics.
        """
        dt = duration / n_steps
        x0_m, y0_m = self.mapper.coord_to_metric(x0_lon, y0_lat)

        # Initial control vector guess
        if init_strategy == "zero":
            u0 = np.zeros((n_steps, 2))
        else:
            # Pursuit initial guess towards local concentration gradient or peak
            u0 = np.zeros((n_steps, 2))
            curr_lon, curr_lat = x0_lon, y0_lat
            for k in range(n_steps):
                # Small step probe to find gradient direction
                c_curr = self.sol_interp(curr_lon, curr_lat, t0 + k * dt)
                dx_c = (
                    self.sol_interp(curr_lon + 0.005, curr_lat, t0 + k * dt) - c_curr
                )
                dy_c = (
                    self.sol_interp(curr_lon, curr_lat + 0.005, t0 + k * dt) - c_curr
                )
                norm = np.hypot(dx_c, dy_c) + 1e-8
                u0[k] = [v_max * 0.5 * dx_c / norm, v_max * 0.5 * dy_c / norm]
                curr_lon += (u0[k, 0] * dt) / 111111.0
                curr_lat += (u0[k, 1] * dt) / 111111.0

        u0_flat = u0.ravel()
        bounds = [(-v_max, v_max)] * (n_steps * 2)

        print(
            f"[AUVOptimizer] Running optimization for {n_steps} steps (dt={dt:.1f}s, duration={duration:.1f}s)..."
        )
        res = minimize(
            self._cost_function,
            u0_flat,
            args=(
                x0_m,
                y0_m,
                t0,
                dt,
                n_steps,
                w_energy,
                w_explore,
                w_land,
                v_max,
            ),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 150, "ftol": 1e-5},
        )

        opt_thrust = res.x.reshape(n_steps, 2)
        opt_traj = self.simulate_trajectory(opt_thrust, x0_m, y0_m, t0, dt)

        # Calculate metrics
        thrust_sq = np.sum(opt_thrust**2, axis=1)
        total_energy = float(np.sum(thrust_sq) * dt)
        total_explore = float(np.sum(opt_traj["concentrations"]) * dt)
        land_violations = int(np.sum(opt_traj["bathymetry"] >= 0.0))

        metrics = {
            "total_energy_J": total_energy,
            "total_explore_reward": total_explore,
            "energy_efficiency_ratio": total_explore / (total_energy + 1e-5),
            "land_violations": land_violations,
            "opt_success": res.success,
            "opt_message": str(res.message),
        }

        print(
            f"[AUVOptimizer] Optimization Finished. Success: {res.success}. Energy: {total_energy:.2f}, "
            f"Exploration Reward: {total_explore:.2f}, Land Violations: {land_violations}"
        )

        return {
            "trajectory": opt_traj,
            "metrics": metrics,
            "thrust_controls": opt_thrust,
        }

    def run_pure_drift(self, x0_lon, y0_lat, t0=0.0, duration=2000.0, n_steps=40):
        """Simulate baseline unpowered drifting policy (v_thr = 0)."""
        dt = duration / n_steps
        x0_m, y0_m = self.mapper.coord_to_metric(x0_lon, y0_lat)
        zero_thrust = np.zeros((n_steps, 2))
        traj = self.simulate_trajectory(zero_thrust, x0_m, y0_m, t0, dt)

        total_energy = 0.0
        total_explore = float(np.sum(traj["concentrations"]) * dt)
        land_violations = int(np.sum(traj["bathymetry"] >= 0.0))

        return {
            "trajectory": traj,
            "metrics": {
                "total_energy_J": total_energy,
                "total_explore_reward": total_explore,
                "energy_efficiency_ratio": total_explore / (total_energy + 1e-5),
                "land_violations": land_violations,
            },
        }

    def run_unoptimized_pursuit(
        self, x0_lon, y0_lat, t0=0.0, duration=2000.0, n_steps=40, v_max=1.0
    ):
        """
        Simulate baseline naive active pursuit policy: constant max thrust towards feature peak
        without taking current drift into account.
        """
        dt = duration / n_steps
        x0_m, y0_m = self.mapper.coord_to_metric(x0_lon, y0_lat)

        # Estimate feature peak coordinate
        sol_grid = self.data["solutions"]
        max_idx = np.unravel_index(np.argmax(sol_grid), sol_grid.shape)
        t_peak_idx, y_peak_idx, x_peak_idx = max_idx
        target_lon = self.lon_grid[x_peak_idx]
        target_lat = self.lat_grid[y_peak_idx]
        target_xm, target_ym = self.mapper.coord_to_metric(target_lon, target_lat)

        thrust_controls = np.zeros((n_steps, 2))
        curr_xm, curr_ym = x0_m, y0_m

        for k in range(n_steps):
            dx = target_xm - curr_xm
            dy = target_ym - curr_ym
            dist = np.hypot(dx, dy) + 1e-6
            # Always apply full thrust directly towards peak target
            thrust_controls[k] = [v_max * dx / dist, v_max * dy / dist]

            # Simulate step to update next position estimate
            curr_lon, curr_lat = self.mapper.metric_to_coord(curr_xm, curr_ym)
            v_c = self.vel_interp(curr_lon, curr_lat, t0 + k * dt)
            curr_xm += (thrust_controls[k, 0] + v_c[0]) * dt
            curr_ym += (thrust_controls[k, 1] + v_c[1]) * dt

        traj = self.simulate_trajectory(thrust_controls, x0_m, y0_m, t0, dt)

        total_energy = float(np.sum(np.sum(thrust_controls**2, axis=1)) * dt)
        total_explore = float(np.sum(traj["concentrations"]) * dt)
        land_violations = int(np.sum(traj["bathymetry"] >= 0.0))

        return {
            "trajectory": traj,
            "metrics": {
                "total_energy_J": total_energy,
                "total_explore_reward": total_explore,
                "energy_efficiency_ratio": total_explore / (total_energy + 1e-5),
                "land_violations": land_violations,
            },
        }
