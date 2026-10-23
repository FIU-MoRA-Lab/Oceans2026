import pickle
import sys


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core.numeric" and name == "_frombuffer":
            try:
                # pyrefly: ignore [missing-import]
                import numpy._core.numeric as num_numeric

                return getattr(num_numeric, "_frombuffer")
            except ModuleNotFoundError:
                return self._custom_frombuffer

        if module.startswith("numpy._core"):
            try:
                __import__(module)
            except ModuleNotFoundError:
                module = module.replace("numpy._core", "numpy.core")

        return super().find_class(module, name)

    @staticmethod
    def _custom_frombuffer(buf, dtype, shape, order, axis_order=None):
        import numpy as np

        array = np.frombuffer(buf, dtype=dtype)
        if order == "K" and axis_order is not None:
            return array.reshape(shape, order="C").transpose(axis_order)
        return array.reshape(shape, order=order)


def read_pkl(path):
    with open(path, "rb") as f:
        return CustomUnpickler(f).load()


class GridInterpolator:
    """
    Picklable wrapper around scipy.interpolate.RegularGridInterpolator.
    Supports evaluating 2D vector fields (e.g. velocity [vx, vy]) and 2D scalar fields (e.g. concentration u).
    """

    def __init__(self, rgi, is_vector=True):
        self.rgi = rgi
        self.is_vector = is_vector

    def __call__(self, x, y, t):
        import numpy as np

        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        t_arr = np.asarray(t)

        broadcasted = np.broadcast(x_arr, y_arr, t_arr)
        shape = broadcasted.shape

        x_flat = np.atleast_1d(x_arr).ravel()
        y_flat = np.atleast_1d(y_arr).ravel()
        t_flat = np.atleast_1d(t_arr).ravel()

        t_flat, x_flat, y_flat = np.broadcast_arrays(t_flat, x_flat, y_flat)
        pts = np.stack([t_flat, x_flat, y_flat], axis=-1)

        res = self.rgi(pts)

        if len(shape) == 0:
            return res[0]
        else:
            if self.is_vector:
                return res.reshape(shape + (2,))
            else:
                return res.reshape(shape)


def get_interpolator(
    matrix,
    config=None,
    lon_range=None,
    lat_range=None,
    nx=None,
    ny=None,
    t_final=None,
    dt=None,
):
    """
    Creates a picklable GridInterpolator object interp(x, y, t) that returns interpolated
    vector or scalar values at coordinates (x, y) and time t.

    Parameters:
    -----------
    matrix : numpy.ndarray
        Water speed matrix of shape (T, 2, ny, nx) or scalar matrix of shape (T, nx, ny).
    config : dict, optional
        Configuration dictionary containing grid metadata.
    lon_range : tuple, optional
        Range of longitude (min, max).
    lat_range : tuple, optional
        Range of latitude (min, max).
    nx : int, optional
        Number of grid points in the x (longitude) direction.
    ny : int, optional
        Number of grid points in the y (latitude) direction.
    t_final : float, optional
        Final simulation time.
    dt : float, optional
        Time step size.

    Returns:
    --------
    GridInterpolator: callable interp(x, y, t) -> numpy.ndarray
    """
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    if config is not None:
        lon_range = lon_range or config.get("lon_range")
        lat_range = lat_range or config.get("lat_range")
        nx = nx or config.get("nx")
        ny = ny or config.get("ny")
        t_final = t_final or config.get("t_final")
        dt = dt or config.get("dt")

    if any(param is None for param in [lon_range, lat_range, nx, ny, t_final, dt]):
        raise ValueError(
            "Missing required grid parameters. Provide 'config' or individual parameters."
        )

    T = matrix.shape[0]
    time_coords = np.linspace(0, t_final, T)
    x_coords = np.linspace(lon_range[0], lon_range[1], nx)
    y_coords = np.linspace(lat_range[0], lat_range[1], ny)

    if matrix.ndim == 4:
        # Transpose vector matrix from (T, 2, ny, nx) to (T, nx, ny, 2)
        is_vector = True
        values = matrix.transpose(0, 3, 2, 1)
    elif matrix.ndim == 3:
        # Scalar matrix: FEniCS solutions are shape (T, ny, nx) where axis 1 is lat (ny) and axis 2 is lon (nx).
        # RegularGridInterpolator needs shape (T, nx, ny) corresponding to (time, lon, lat).
        is_vector = False
        values = matrix.transpose(0, 2, 1)
    else:
        raise ValueError(f"Unsupported matrix dimension: {matrix.ndim}")

    rgi = RegularGridInterpolator(
        (time_coords, x_coords, y_coords), values, bounds_error=False, fill_value=0.0
    )

    return GridInterpolator(rgi, is_vector=is_vector)

