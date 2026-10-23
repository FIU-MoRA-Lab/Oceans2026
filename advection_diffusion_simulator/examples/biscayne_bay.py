import os
import sys

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from advection_diffusion_simulator import utils
from advection_diffusion_simulator.config import AdvectionDiffusionConfig
from advection_diffusion_simulator.solver import ADSolver

# ============================================================================
# Configuration
# ============================================================================


def main():
    print("=" * 80)
    print("Advection-Diffusion Simulation - Biscayne Bay Example")
    print("=" * 80)

    # ========================================================================
    # Define Domain (Polygon)
    # ========================================================================

    print("\nDefining domain...")

    # TODO: Define your polygon domain points
    # Example: Simplified Biscayne Bay polygon (counterclockwise)
    # These points will be automatically sorted counterclockwise
    polygon_points = [
        [-80.13725460, 25.91157340],
        [-80.13679060, 25.91166500],
        [-80.13728140, 25.91236710],
        [-80.13773740, 25.91295090],
        [-80.13797610, 25.91293890],
        [-80.13793050, 25.91284240],
        [-80.13782590, 25.91283990],
        [-80.13779370, 25.91274340],
        [-80.13783400, 25.91270240],
    ]

    # Mesh resolution (higher = finer mesh)
    mesh_resolution = 30

    print(f"Polygon domain with {len(polygon_points)} points")
    print(f"Mesh resolution: {mesh_resolution}")

    # ========================================================================
    # Create Configuration
    # ========================================================================

    config = AdvectionDiffusionConfig(
        # Domain (will be overridden by set_domain)
        polygon_points=np.asarray(polygon_points),
        mesh_resolution=mesh_resolution,
        # Time parameters
        t_final=100,  # seconds
        dt=0.5,  # seconds
        # Physical parameters
        diffusivity=0.1,  # Should be in m^2/s; needs to be greater to kill numerical unstability(TODO: adjust based on your application)
        # Element parameters
        element_degree=2,
        element_family="P",
        are_coordinates_lonlat=True,  # Set to True if using lon/lat coordinates
        # Output
        output_dir=None,  # "_output_biscayne",
        # multiple_output_times=True,
    )

    print(f"\nConfiguration:")
    print(f"  Time: t_final={config.t_final}s, dt={config.dt}s")
    print(f"  Diffusivity: {config.diffusivity} m^2/s")
    print(f"  Element: {config.element_family}, degree={config.element_degree}")

    # ========================================================================
    # Define Physical Functions
    # ========================================================================

    print("\nDefining physical functions...")

    # TODO: Define forcing term (source/sink)
    def forcing_term(x, y):
        """
        Forcing term f(x, y) - represents sources/sinks

        Example: Point source at center
        """
        # Placeholder: No forcing
        return 0.0

        # Example: Gaussian source at (0.5, 0.5)
        # center_x, center_y = 0.5, 0.5
        # sigma = 0.1
        # return 10.0 * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

    # TODO: Define velocity field (advection)
    def velocity_field(x, y, t):
        """
        Velocity field v(x, y) - represents flow/wind

        Returns: [vx, vy] velocity vector
        """
        # Placeholder: Constant velocity
        return np.array([0.5, -0.8])

        # Example: Circular flow
        # center_x, center_y = 0.5, 0.5
        # return np.array([-(y - center_y), (x - center_x)])

        # Example: Hurricane-like velocity
        # speed = 10.0  # m/s
        # direction = np.array([1/np.sqrt(2), -1/np.sqrt(2)])  # Southeast
        # return speed * direction

    # TODO: Define initial condition
    def initial_condition(x, y):
        """
        Initial concentration/temperature u0(x, y)
        """
        # Placeholder: Gaussian blob at center-left
        center_x, center_y = 20.0, -20.0
        sigma = 10.0
        return 27.0 * np.exp(
            -((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma**2)
        )

        # Example: Multiple sources
        # blob1 = 30.0 * np.exp(-((x - 0.3)**2 + (y - 0.3)**2) / (2 * 0.05**2))
        # blob2 = 40.0 * np.exp(-((x - 0.7)**2 + (y - 0.7)**2) / (2 * 0.05**2))
        # return blob1 + blob2

    # TODO: Define boundary condition
    def boundary_condition(x, y):
        """
        Dirichlet boundary condition g(x, y)
        """
        # Placeholder: Zero at boundaries
        return 0.0

        # Example: Temperature gradient
        # return 10.0 * x  # Linear in x-direction

    # ========================================================================
    # Initialize Solver
    # ========================================================================

    print("\nInitializing ADSolver...")
    solver = ADSolver(config=config)

    # You can also set domain after initialization if needed:
    # solver.set_domain(polygon_points, mesh_resolution=mesh_resolution)

    # You can set time parameters individually if desired:
    # solver.set_time_parameters(t_final=2.0, dt=0.05)

    # You can set diffusivity individually if desired:
    # solver.set_diffusivity(0.01)

    # ========================================================================
    # Set Problem Functions
    # ========================================================================

    print("Setting problem functions...")
    solver.set_forcing_term(forcing_term)
    solver.set_velocity_field(velocity_field)
    solver.set_initial_condition(initial_condition)
    solver.set_boundary_condition(boundary_condition)

    print("  ✓ Forcing term set")
    print("  ✓ Velocity field set")
    print("  ✓ Initial condition set")
    print("  ✓ Boundary condition set")

    # ========================================================================
    # Setup and Run Solver
    # ========================================================================

    print("\nSetting up solver...")
    solver.setup_solver()
    print(f"  Mesh created with {solver.mesh.num_vertices()} vertices")
    print(f"  Function space dimension: {solver.function_space.dim()}")

    print("\nRunning simulation...")
    result = solver.solve()

    print(f"\nSimulation complete!")
    print(f"  Number of time steps: {len(result.solutions)}")
    print(f"  Time steps: {result.time_steps}")
    print(f"  Final time: {result.time_steps[-1]:.3f}s")

    # ========================================================================
    # Post-processing
    # ========================================================================

    print("\n" + "=" * 80)
    print("Post-processing Results")
    print("=" * 80)

    # TODO: Add visualization
    print("\nVisualization:")
    print("  TODO: Use matplotlib or ParaView to visualize results")
    print(f"  Results saved to: {config.output_dir}")

    # Example: Plot using matplotlib
    try:
        import matplotlib.pyplot as plt

        print("\nGenerating plots...")

        # Plot initial and final solutions

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()

        indices = np.linspace(0, result.solutions.shape[0] - 1, 9, dtype=np.int64)
        for ax, idx in zip(axes, indices):
            ax.set_aspect("equal")
            # ax.set_xticks([])
            # ax.set_yticks([])
            u_plot = np.ma.masked_where(~result.mesh_mask, result.solutions[idx])
            c = ax.contourf(*result.meshgrid, u_plot, levels=50, cmap="viridis")
            ax.set_title(f"t={result.time_steps[idx]:.1f}s")
            plt.colorbar(c, ax=ax)

        # Final solution
        # df.plot(solver.mesh, axes=axes[1])
        # c = df.plot(result.solutions[-1], axes=axes[1])
        # axes[1].set_title(f"Final Solution (t={result.time_steps[-1]:.2f}s)")
        # axes[1].set_xlabel("x")
        # axes[1].set_ylabel("y")
        # plt.colorbar(c, ax=axes[1])

        # plt.tight_layout()

        # Save figure
        if config.output_dir:
            os.makedirs(config.output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(config.output_dir, "solution_comparison.png"), dpi=150
            )
            print(f"  Saved plot to {config.output_dir}/solution_comparison.png")

        plt.show()

    except ImportError:
        print("  matplotlib not available for plotting")
    except Exception as e:
        print(f"  Error during plotting: {e}")

    print("\n" + "=" * 80)
    print("Simulation completed successfully!")
    print("=" * 80)

    return result


if __name__ == "__main__":
    result = main()
