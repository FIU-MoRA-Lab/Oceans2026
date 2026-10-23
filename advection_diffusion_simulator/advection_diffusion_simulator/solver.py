#!/usr/bin/env python
# encoding: utf-8

import os
from typing import Callable, Optional, Tuple

import dolfin as df
import numpy as np
import tqdm

from .config import AdvectionDiffusionConfig
from .coordinate_mapper import GeographicCoordinateMapper
from .result import ADResult
from .utils import MyVariableExpression, create_polygon_mesh, fenics_2_tensor


class ADSolver:
    """
    Solver class for 2D advection-diffusion equation using FEniCS.

    Solves: du/dt + v·∇u - d∇²u = f
    where:
        u: concentration/temperature field
        v: velocity field
        d: diffusivity coefficient
        f: forcing term

    Parameters
    ----------
    config : Optional[AdvectionDiffusionConfig]
        Configuration object for the solver
    """

    def __init__(
        self,
        config: Optional[AdvectionDiffusionConfig] = None,
    ) -> None:
        # Configuration
        self.config = config or AdvectionDiffusionConfig()

        # Mesh
        self.mesh: Optional[df.Mesh] = None

        # Functions defining the problem
        self.forcing_term_func: Optional[Callable] = None
        self.velocity_field_func: Optional[Callable] = None
        self.initial_condition_func: Optional[Callable] = None
        self.boundary_condition_func: Optional[Callable] = None

        # Function spaces (initialized when mesh is created)
        self.function_space: Optional[df.FunctionSpace] = None
        self.vector_function_space: Optional[df.VectorFunctionSpace] = None

    def set_time_parameters(self, t_final: float, dt: float) -> None:
        """
        Set the time parameters for the simulation.

        Parameters
        ----------
        t_final : float
            Final simulation time
        dt : float
            Time step size
        """
        self.config.t_final = t_final
        self.config.dt = dt

    def set_domain(self, points, mesh_resolution: Optional[int] = None) -> None:
        """
        Set up the domain from a collection of points defining a polygon.

        Parameters
        ----------
        points : list or numpy.ndarray
            Collection of points defining the polygon.
            Can be a list of tuples/lists or a numpy array of shape (n, 2).
            Points will be automatically sorted counterclockwise.
        mesh_resolution : Optional[int]
            Resolution for polygon mesh generation. If None, uses config default.

        Note
        ----
        Points will be automatically sorted counterclockwise before mesh generation.
        If domain changes and mesh was already created, it will be regenerated
        on next solve() call.

        Examples
        --------
        >>> # Rectangular domain
        >>> solver.set_domain([[0, 0], [1, 0], [1, 1], [0, 1]], mesh_resolution=20)
        >>>
        >>> # Arbitrary polygon
        >>> solver.set_domain([[1, 0], [0.75, 0.75], [0, 1], [0.25, 0.25]], mesh_resolution=25)
        """
        # Convert to numpy array and validate
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be a 2D array with shape (n, 2)")
        if len(points) < 3:
            raise ValueError("points must have at least 3 points")

        self.config.polygon_points = points
        if mesh_resolution is not None:
            self.config.mesh_resolution = mesh_resolution

        # Invalidate mesh and function spaces to force regeneration
        self.mesh = None
        self.function_space = None
        self.vector_function_space = None

    def set_diffusivity(self, diffusivity: float) -> None:
        """
        Set the diffusivity coefficient.

        Parameters
        ----------
        diffusivity : float
            Diffusivity coefficient (must be non-negative)
        """
        if diffusivity < 0:
            raise ValueError("Diffusivity must be non-negative")
        self.config.diffusivity = diffusivity

    def set_forcing_term(self, forcing_func: Callable[[float, float], float]) -> None:
        """
        Set the forcing term (source term).

        Parameters
        ----------
        forcing_func : Callable[[float, float], float]
            Function f(x, y) that returns the forcing term value
        """
        self.forcing_term_func = forcing_func

    def set_velocity_field(
        self, velocity_func: Callable[[float, float], np.ndarray]
    ) -> None:
        """
        Set the velocity field for advection.

        Parameters
        ----------
        velocity_func : Callable[[float, float], np.ndarray]
            Function v(x, y) that returns velocity vector [vx, vy]
        """
        self.velocity_field_func = velocity_func

    def set_initial_condition(
        self, initial_func: Callable[[float, float], float]
    ) -> None:
        """
        Set the initial condition.

        Parameters
        ----------
        initial_func : Callable[[float, float], float]
            Function u0(x, y) that returns initial value
        """
        self.initial_condition_func = initial_func

    def set_boundary_condition(
        self, boundary_func: Callable[[float, float], float]
    ) -> None:
        """
        Set the Dirichlet boundary condition.

        Parameters
        ----------
        boundary_func : Callable[[float, float], float]
            Function g(x, y) that returns boundary value
        """
        self.boundary_condition_func = boundary_func

    def _create_mesh(self) -> df.Mesh:
        """Create the computational mesh."""

        # Use polygon mesh
        if self.config.are_coordinates_lonlat:
            # If coordinates are in lon/lat, convert to metric
            mapper = GeographicCoordinateMapper(
                lon0=np.mean(self.config.polygon_points[:, 0]),
                lat0=np.mean(self.config.polygon_points[:, 1]),
            )
            x, y = mapper.coord_to_metric(
                self.config.polygon_points[:, 0], self.config.polygon_points[:, 1]
            )
            print(x, y)
            # raise
            self.polygon_points_metric = np.column_stack((x, y))
            self.coordinate_mesh, _ = create_polygon_mesh(
                self.config.polygon_points,
                resolution=self.config.mesh_resolution,
                element_degree=self.config.element_degree,
                element_family=self.config.element_family,
            )
            mesh, _ = create_polygon_mesh(
                self.polygon_points_metric,
                resolution=self.config.mesh_resolution,
                element_degree=self.config.element_degree,
                element_family=self.config.element_family,
            )
            return mesh

        mesh, _ = create_polygon_mesh(
            self.config.polygon_points,
            resolution=self.config.mesh_resolution,
            element_degree=self.config.element_degree,
            element_family=self.config.element_family,
        )

        return mesh

    def _validate_configuration(self) -> None:
        """Validate that all required configuration has been set."""
        errors = []

        if self.forcing_term_func is None:
            errors.append("Forcing term has not been set.")
        if self.velocity_field_func is None:
            errors.append("Velocity field has not been set.")
        if self.initial_condition_func is None:
            errors.append("Initial condition has not been set.")
        if self.boundary_condition_func is None:
            errors.append("Boundary condition has not been set.")

        if errors:
            raise ValueError(
                "Advection-diffusion configuration errors:\n" + "\n".join(errors)
            )

    def _initialize_function_spaces(self) -> None:
        """Initialize FEniCS function spaces."""
        element = df.FiniteElement(
            self.config.element_family,
            self.mesh.ufl_cell(),
            self.config.element_degree,
        )

        self.function_space = df.FunctionSpace(self.mesh, element)
        self.vector_function_space = df.VectorFunctionSpace(
            self.mesh,
            self.config.element_family,
            self.config.element_degree,
        )

    def setup_solver(self) -> None:
        """
        Set up the FEniCS problem (mesh, function spaces, boundary conditions).
        """
        self._validate_configuration()

        # Create mesh
        self.mesh = self._create_mesh()

        # Initialize function spaces
        self._initialize_function_spaces()

        # Ensure output directory exists
        if self.config.output_dir is not None:
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir)

    def get_temporal_velocity_field(self, t=None):
        if t is None:
            velocity_field_func = lambda x, y: self.velocity_field_func(x, y)
        else:
            velocity_field_func = lambda x, y: self.velocity_field_func(x, y, t)

        velocity_field_expr = MyVariableExpression(velocity_field_func, dimension=2)
        velocity_interpolated = df.interpolate(
            velocity_field_expr, self.vector_function_space
        )
        return velocity_interpolated

    def solve(self) -> ADResult:
        """
        Run the simulation.

        Returns
        -------
        ADResult
            Result object containing solutions and metadata
        """
        if self.mesh is None or self.function_space is None:
            self.setup_solver()

        # Time stepping
        time_values = np.arange(0, self.config.t_final, self.config.dt)

        # Set up boundary condition
        bc_expr = MyVariableExpression(self.boundary_condition_func)
        bc_interpolated = df.interpolate(bc_expr, self.function_space)

        def u0_boundary(x, on_boundary):
            return on_boundary

        bc = df.DirichletBC(self.function_space, bc_interpolated, u0_boundary)

        # Set up forcing term
        forcing_expr = MyVariableExpression(self.forcing_term_func)
        forcing = df.interpolate(forcing_expr, self.function_space)

        # Set up velocity field
        # velocity_expr = MyVariableExpression(self.velocity_field_func, dimension=2)
        # velocity_f = df.interpolate(velocity_expr, self.vector_function_space)

        # Set up initial condition
        u0_expr = MyVariableExpression(self.initial_condition_func)
        u = df.interpolate(u0_expr, self.function_space)

        # Constants
        dt = df.Constant(self.config.dt)
        d = df.Constant(self.config.diffusivity)

        # Trial and test functions
        u_trial = df.TrialFunction(self.function_space)
        v_test = df.TestFunction(self.function_space)

        # Store solutions
        solutions = [df.interpolate(u0_expr, self.function_space)]
        gradients = [df.grad(solutions[0])]

        print(
            "Solving the advection-diffusion equation:\n"
            "  du/dt + v·∇u - d∇²u = f\n"
            f"Domain: [{self.config.x_range[0]}, {self.config.x_range[1]}] x "
            f"[{self.config.y_range[0]}, {self.config.y_range[1]}]\n"
            f"Time: [0, {self.config.t_final}] with dt={self.config.dt}\n"
            f"Diffusivity: {self.config.diffusivity}"
        )

        # Time stepping loop
        n = df.FacetNormal(self.mesh)
        for t in tqdm.tqdm(time_values[1:], desc="Time steps"):
            # Weak formulation
            u_sol = df.Function(self.function_space)
            velocity_f = self.get_temporal_velocity_field(t)

            a = (
                u_trial * v_test
                + dt
                * (
                    df.dot(d * df.grad(u_trial), df.grad(v_test))
                    + df.dot(df.grad(u_trial), velocity_f) * v_test
                )
            ) * df.dx  # + df.dot(velocity_f, n) * u_trial * v_test * df.ds(1)

            L = (dt * forcing + u) * v_test * df.dx

            # Solve
            df.solve(a == L, u_sol)
            u.assign(u_sol)

            # Store solution
            solutions.append(u_sol)
            gradients.append(df.project(df.grad(u_sol), self.vector_function_space))

        # Create meshgrid for result
        solutions, mesh_mask, meshgrid = fenics_2_tensor(
            solutions=solutions,
            mesh=self.mesh,
            polygon_coords=self.config.polygon_points
            if not self.config.are_coordinates_lonlat
            else self.polygon_points_metric,
            grid_resolution=self.config.meshgrid_resolution,
        )
        if self.config.are_coordinates_lonlat:
            # If original coordinates were lon/lat, convert meshgrid back to lon/lat
            x_min, x_max = self.config.x_range
            y_min, y_max = self.config.y_range
            geo_meshgrid = np.meshgrid(
                np.linspace(x_min, x_max, num=self.config.meshgrid_resolution[0]),
                np.linspace(y_min, y_max, num=self.config.meshgrid_resolution[1]),
            )

        # Create result object
        result = ADResult(
            meshgrid=meshgrid
            if not self.config.are_coordinates_lonlat
            else geo_meshgrid,
            solutions=solutions,
            config=self.config,
            time_steps=np.concatenate([[0], time_values[1:]]),
            mesh_mask=mesh_mask,
        )

        # Save if output directory specified
        if self.config.output_dir is not None:
            result.save(os.path.join(self.config.output_dir, "result.pkl"))

        return result
