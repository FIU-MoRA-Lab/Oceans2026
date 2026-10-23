import pickle
import sys

import dolfin as df
import fenics as fe
import h5py
import matplotlib.pyplot as plt
import mshr
import numpy as np
import pandas as pd
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    RBFInterpolator,
    RegularGridInterpolator,
    griddata,
)
from shapely.geometry import Point, Polygon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern


def fenics_2_pyfunc(solutions, mesh, grid_resolution=(100, 100), name=None):
    """
    Converts FEniCS solutions to Python functions and saves them to a file.

    Args:
        solutions (list): List of FEniCS solutions.
        mesh (fenics.Mesh): FEniCS mesh.
        name (str): Name of the output file
        grid_resolution (tuple): Number of points along the x-axis and y-axis for interpolation (default: (100, 100)).

    Returns:
        tuple: Tuple containing lists interpolated solutions and gradients.
    """
    mesh_coordinates = mesh.coordinates()
    x_min, y_min = np.min(mesh_coordinates, axis=0)
    x_max, y_max = np.max(mesh_coordinates, axis=0)
    nx, ny = grid_resolution

    # Extract the values of the FEniCS function at the mesh coordinates
    interpolated_solutions, interpolated_grad = [], []
    for solution in solutions:
        u_values = [solution(coord) for coord in mesh_coordinates]

        # Create a 2D interpolator
        interp_solution = LinearNDInterpolator(mesh_coordinates, u_values)

        # Reshape the coordinates and values for creating a 2D gradient
        x_vals, y_vals = np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x_vals, y_vals)
        interpolated_values = interp_solution(X, Y)

        grad_X, grad_Y = np.gradient(interpolated_values.T, x_vals, y_vals)
        grad = np.dstack((grad_X, grad_Y))
        interp_grad = RegularGridInterpolator((x_vals, y_vals), grad)

        interpolated_solutions.append(interp_solution)
        interpolated_grad.append(interp_grad)

    if isinstance(name, str):
        with open(name, "wb") as file:
            pickle.dump([interpolated_solutions, interpolated_grad], file)

    return interpolated_solutions, interpolated_grad


def fenics_2_tensor(
    solutions, mesh, polygon_coords, grid_resolution=(100, 100), name=None
):
    """
    Converts FEniCS solutions to a tensor and saves them to a file.

    Args:
        solutions (list): List of FEniCS solutions.
        mesh (fenics.Mesh): FEniCS mesh.
        polygon_coords (numpy.ndarray): Coordinates of the polygon.
        grid_resolution (tuple): Number of points along the x-axis and y-axis for interpolation (default: (100, 100)).
        name (str): Name of the output file (default: None).

    Returns:
        tuple: Tuple containing interpolated solutions and mesh mask.
    """
    mesh_coordinates = mesh.coordinates()
    x_min, y_min = np.min(mesh_coordinates, axis=0)
    x_max, y_max = np.max(mesh_coordinates, axis=0)
    nx, ny = grid_resolution

    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, num=nx), np.linspace(y_min, y_max, num=ny)
    )
    mesh_mask = mask_from_polygon((X, Y), polygon_coords)
    interpolated_solutions = np.zeros((len(solutions), nx, ny))
    for i in range(len(solutions)):
        interpolated_solutions[i] = griddata(
            mesh.coordinates(), solutions[i].compute_vertex_values(mesh), (X, Y)
        )

    if isinstance(name, str):
        with open(name, "wb") as file:
            pickle.dump([interpolated_solutions, mesh_mask], file)

    return interpolated_solutions, mesh_mask, (X, Y)


class MyVariableExpression(df.UserExpression):
    """
    A custom expression class for defining variable expressions.

    Parameters:
    scalar_function : callable
        A scalar function that takes two arguments (x, y) and returns either a scalar or an array.
    dimension : int, optional
        The dimension of the expression output. Default is 1.

    Attributes:
    scalar_function : callable
        A scalar function that takes two arguments (x, y) and returns either a scalar or an array.
    dimension : int
        The dimension of the expression output.

    Methods:
    eval(value, x)
        Evaluate the expression at the point (x, y) and store the result in the provided value array.
    value_shape()
        Return the shape of the expression output.

    Example usage:
    scalar_function = lambda x, y: x + y
    variable_expression = MyVariableExpression(scalar_function)
    ```
    """

    def __init__(self, scalar_function, dimension=1, **kwargs):
        """
        Initialize the MyVariableExpression object.

        Parameters:
        scalar_function : callable
            A scalar function that takes two arguments (x, y) and returns either a scalar or an array.
        dimension : int, optional
            The dimension of the expression output. Default is 1.
        kwargs : dict
            Additional keyword arguments to pass to the parent class constructor.
        """
        self.scalar_function = scalar_function
        self.dimension = dimension
        super().__init__(**kwargs)

    def eval(self, value, x):
        """
        Evaluate the expression at the point (x, y) and store the result in the provided value array.

        Parameters:
        value : array_like
            Array to store the evaluated expression value.
        x : array_like
            Array containing the coordinates (x, y) at which to evaluate the expression.
        """
        result = self.scalar_function(x[0], x[1])
        if self.dimension == 1:
            value[0] = result
        else:
            for i in range(self.dimension):
                value[i] = result[i]

    def value_shape(self):
        """
        Return the shape of the expression output.

        Returns:
        shape : tuple
            Shape of the expression output.
        """
        if self.dimension == 1:
            return ()
        else:
            return (self.dimension,)


def sort_coordinates(coords):
    """
    Sorts coordinates in counterclockwise sense.

    Args:
        coords (numpy.ndarray): Coordinates to be sorted.

    Returns:
        numpy.ndarray: Sorted coordinates.
    """
    centroid = np.mean(coords, axis=0)
    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_coords = coords[sorted_indices]
    return sorted_coords


def mask_from_polygon(meshgrid, polygon_coordinates):
    """
    Create a mask matrix indicating whether points in a meshgrid are inside a polygon.
    Args:
    - meshgrid: Tuple of arrays from np.meshgrid.
    -polygon_coordinates: Polygon's coordinates; sorted counterclockwise.

    Returns:
    - Sorted coordinates.
    """
    X, Y = meshgrid
    mask = np.zeros_like(X, dtype=bool)
    polygon = Polygon(polygon_coordinates)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = Point(X[i, j], Y[i, j])
            mask[i, j] = polygon.contains(point)

    return mask


def create_polygon_mesh(
    points, resolution=5, element_degree=1, element_family="Lagrange"
):
    """
    Create a polygon mesh from a collection of points.

    Args:
        points (list or numpy.ndarray): Collection of points defining the polygon.
                                        Can be a list of tuples/lists or a numpy array of shape (n, 2).
        resolution (int): Mesh resolution parameter for mshr.generate_mesh (default: 5).
        element_degree (int): Degree of the Lagrange element (default: 1).
        element_family (str): Element family for the finite element (default: 'Lagrange').

    Returns:
        tuple: (mesh, element) where:
            - mesh: Generated FEniCS mesh
            - element: FEniCS FiniteElement object

    Example:
        >>> points = [[1, 0], [0.75, 0.75], [0, 1], [0.25, 0.25]]
        >>> mesh, element = create_polygon_mesh(points, resolution=5)
    """
    # Convert points to numpy array if needed
    points = np.asarray(points)

    # Sort coordinates counterclockwise
    sorted_points = sort_coordinates(points)

    # Create df.Point objects for mshr.Polygon
    df_points = [df.Point(float(p[0]), float(p[1])) for p in sorted_points]

    # Create the polygon domain
    domain = mshr.Polygon(df_points)

    # Generate mesh
    mesh = mshr.generate_mesh(domain, resolution)

    # Create the finite element
    element = df.FiniteElement(element_family, mesh.ufl_cell(), element_degree)

    return mesh, element


def create_interpolator_from_data(coords, values, method="rbf"):
    """
    Create a 2D interpolator from given coordinates and data using linear, cubic, or GP interpolation.

    Args:
        coords (numpy.ndarray): Coordinates (N, 2) of the data points.
        values (numpy.ndarray): Values at the coordinates.
        method (str): Method to use ('linear', 'cubic', or 'gp').

    Returns:
        callable: A function that returns the interpolated value at a point.
    """
    print(coords.shape, values.shape)
    if method == "linear":
        interp = LinearNDInterpolator(coords, values)
        return lambda point: interp(point)

    elif method == "cubic":
        # CloughTocher2DInterpolator is a C1 smooth, piecewise cubic interpolant
        interp = CloughTocher2DInterpolator(coords, values)
        return lambda point: interp(point)
    elif method == "rbf":
        interp = RBFInterpolator(
            coords
            * 111111,  # Scale coordinates to convert degrees to meters (approximate)
            values,
            smoothing=1,
            # neighbors=500,
            kernel="gaussian",
            epsilon=0.1,
        )
        return lambda point: interp(
            point * 111111
        )  # Scale input point as well for consistency

    else:
        raise ValueError(f"Method {method} not supported.")


def _load_structured_data(file_path, column_names=None, data_len=None, chunk_size=50):
    """Load structured HDF5 data into a pandas DataFrame, optionally loading only specific columns."""
    with h5py.File(file_path, "r") as f:
        dataset = f["data"]
        print("Available columns:", dataset[0].dtype.names)
        column_names.extend(["Longitude", "Latitude"])
        dataset_len = len(dataset) if data_len is None else min(len(dataset), data_len)
        print(f"Dataset length: {dataset_len} {chunk_size}")
        if column_names:
            all_chunks = []
            for i in range(0, dataset_len, chunk_size):
                print(f"Loading rows {i} to {min(i + chunk_size, dataset_len)}...")
                chunk = dataset[i : i + chunk_size][column_names]
                all_chunks.append(chunk)
            structured_array = np.concatenate(all_chunks)
        else:
            structured_array = dataset[:]
            print("Available columns:", structured_array.dtype.names)

    df = pd.DataFrame()
    for name in structured_array.dtype.names:
        col = structured_array[name]
        if col.ndim > 1:
            col = list(col)  # Store multi-dimensional arrays as objects
        df[name] = col
    print(f"Data shape: {df.shape}")
    df = df[
        (df["Longitude"] != 0) & (df["Latitude"] != 0)
    ]  # Filter out rows with zero longitude
    return df


def load_into_interpolator(file_path, column_names=None, method="rbf", **kwargs):
    """Load structured data from a file and create an interpolator function."""
    df = _load_structured_data(
        file_path,
        column_names,
        data_len=kwargs.get("data_len"),
        chunk_size=kwargs.get("chunk_size", 50),
    )
    print(df.describe())
    coords = df[["Longitude", "Latitude"]].values
    values = df[column_names[0]].values
    return create_interpolator_from_data(coords, values, method=method)


if __name__ == "__main__" and "fenics" in sys.modules:
    # Create a FEniCS mesh and function space
    import sys

    if sys.argv[-1] == "test_fenics":
        nx = 10
        ny = nx
        mesh = fe.UnitSquareMesh(10, 10)
        V = fe.FunctionSpace(mesh, "P", 2)

        # Define a FEniCS function
        u = fe.Expression("pow(x[0]-0.5,2)+pow(x[1]-0.25,2)", degree=2)
        u_function = fe.interpolate(u, V)
        mesh_coordinates = mesh.coordinates()
        x_min, y_min = np.min(mesh_coordinates, axis=0)
        x_max, y_max = np.max(mesh_coordinates, axis=0)
        x_vals, y_vals = np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny)

        # Create a mesh grid
        X, Y = np.meshgrid(x_vals, y_vals)

        solutions, grads = fenics_2_pyfunc(
            [u_function], mesh, grid_resolution=(100, 100)
        )
        with open("test_grad.pkl", "wb") as file:
            pickle.dump([solutions, grads], file)

        del solutions, grads

        with open("test_grad.pkl", "rb") as file:
            solutions, grads = pickle.load(file)
        test_pt = np.array([0.5, 0.5])
        print(solutions[0](test_pt))
        print(grads[0](test_pt))
        grad = grads[0]((X, Y))

        plt.figure(figsize=(8, 6))

        plt.contourf(X, Y, solutions[0](X, Y), cmap="viridis")
        plt.colorbar()
        print("h", grads[0](np.array([0.1, 0.8])))
        plt.quiver(X, Y, grad[:, :, 0], grad[:, :, 1], scale=10, color="black")

        plt.show()
    elif sys.argv[-1] == "test_load_structured_data":
        file_path = "data/asv_datasets/iteration_2/20250609_173859.h5"
        # file_path = "20260130group1-1.csv"
        column_names = ["Temperature (C)"]
        df = _load_structured_data(file_path, column_names, 1000)[:]
        # df = pd.read_csv(file_path)[column_names + ["Longitude", "Latitude"]]
        # df = df[(df["Longitude"] != 0) & (df["Latitude"] != 0)][:1000:5]
        print(df.describe())
        # Prepare data for interpolation
        coords = df[["Longitude", "Latitude"]].values
        values = df[column_names[0]].values

        # Create interpolator using Gaussian Process
        interp_func = create_interpolator_from_data(coords, values)

        # Define a grid for evaluation
        xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 50)
        yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 50)
        Xi, Yi = np.meshgrid(xi, yi)

        # Evaluate the interpolator on the grid
        grid_points = np.c_[Xi.ravel(), Yi.ravel()]
        Zi = interp_func(grid_points).reshape(Xi.shape)
        # Zi = np.clip(
        #     Zi, 0.9 * values.min(), 1.1 * values.max()
        # )  # Clip to original data range
        # Plot the comparison
        plt.figure(figsize=(12, 8))
        # Plot the interpolated data as a contour plot
        cntr = plt.contourf(Xi, Yi, Zi, levels=20, cmap="viridis", alpha=0.8)
        # cntr.set_clim(values.min(), values.max())
        plt.colorbar(cntr, label=column_names[0])

        # Plot the real data points as a scatter plot
        sc = plt.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            cmap="coolwarm",
            edgecolors="black",
            label="Original Data",
        )
        plt.colorbar(sc, label="Temperature (C)")

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("GP Interpolation (Contour) vs. Original Data (Scatter)")
        plt.legend()
        plt.show()
