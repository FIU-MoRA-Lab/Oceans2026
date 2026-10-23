"""Coordinate mapping between geographic and metric coordinates."""

from typing import Union

import numpy as np
import numpy.typing as npt

from .logging_config import get_logger

logger = get_logger(__name__)


class GeographicCoordinateMapper:
    """
    Map between lon/lat (degrees) and x/y (meters) coordinates.

    Parameters
    ----------
    lon0 : float
        Reference longitude in degrees (mapped to x=0)
    lat0 : float
        Reference latitude in degrees (mapped to y=0)
    R : float, default=6371000.0
        Earth radius in meters
    """

    def __init__(self, lon0: float, lat0: float, R: float = 6371000.0) -> None:
        self.lon0: float = np.deg2rad(lon0)
        self.lat0: float = np.deg2rad(lat0)
        self.R: float = R
        self.cos_lat0: float = np.cos(self.lat0)

        logger.debug(
            f"Initialized mapper: lon0={np.rad2deg(self.lon0):.4f}°, lat0={np.rad2deg(self.lat0):.4f}°"
        )

    def coord_to_metric(
        self,
        lon: Union[float, npt.NDArray[np.float64]],
        lat: Union[float, npt.NDArray[np.float64]],
    ) -> tuple[
        Union[float, npt.NDArray[np.float64]], Union[float, npt.NDArray[np.float64]]
    ]:
        """
        Convert (lon, lat) in degrees to (x, y) in meters.

        Parameters
        ----------
        lon : float or np.ndarray
            Longitude(s) in degrees
        lat : float or np.ndarray
            Latitude(s) in degrees

        Returns
        -------
        x : float or np.ndarray
            X coordinate(s) in meters
        y : float or np.ndarray
            Y coordinate(s) in meters
        """
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)

        x = self.R * (lon_rad - self.lon0) * self.cos_lat0
        y = self.R * (lat_rad - self.lat0)

        return x, y

    def metric_to_coord(
        self,
        x: Union[float, npt.NDArray[np.float64]],
        y: Union[float, npt.NDArray[np.float64]],
    ) -> tuple[
        Union[float, npt.NDArray[np.float64]], Union[float, npt.NDArray[np.float64]]
    ]:
        """
        Convert (x, y) in meters to (lon, lat) in degrees.

        Parameters
        ----------
        x : float or np.ndarray
            X coordinate(s) in meters
        y : float or np.ndarray
            Y coordinate(s) in meters

        Returns
        -------
        lon : float or np.ndarray
            Longitude(s) in degrees
        lat : float or np.ndarray
            Latitude(s) in degrees
        """
        lon_rad = self.lon0 + x / (self.R * self.cos_lat0)
        lat_rad = self.lat0 + y / self.R

        lon = np.rad2deg(lon_rad)
        lat = np.rad2deg(lat_rad)

        return lon, lat
