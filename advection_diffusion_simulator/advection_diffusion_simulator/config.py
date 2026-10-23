#!/usr/bin/env python
# encoding: utf-8

from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import utils


@dataclass
class AdvectionDiffusionConfig:
    """Configuration for advection-diffusion solver."""

    # Mesh parameters, polygon domain
    polygon_points: np.ndarray = np.array(
        []
    )  # Array of shape (n, 2) with polygon vertices
    polygon_points: Optional[np.ndarray] = None  # Array of shape (n, 2) with polygon vertices
    mesh_resolution: int = 20  # Resolution for polygon mesh generation
    are_coordinates_lonlat: bool = (
        True  # Whether polygon points are in lon/lat (degrees)
    )
    meshgrid_resolution: tuple[int, int] = (
        100,
        100,
    )  # Resolution for output meshgrid (X, Y)

    # Time parameters
    t_final: float = 1.0
    dt: float = 0.1

    # Physical parameters
    diffusivity: float = 0.005

    # Element parameters
    element_degree: int = 2
    element_family: str = "P"

    # Output parameters
    output_dir: Optional[str] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.dt <= 0 or self.t_final <= 0:
            raise ValueError("Time parameters must be positive")
        if self.diffusivity < 0:
            raise ValueError("Diffusivity must be non-negative")
        if self.polygon_points is not None:
            # Convert to numpy array and validate
            self.polygon_points = np.asarray(self.polygon_points)
            if self.polygon_points.ndim != 2 or self.polygon_points.shape[1] != 2:
                raise ValueError("polygon_points must be a 2D array with shape (n, 2)")
            if len(self.polygon_points) < 3:
                raise ValueError("polygon_points must have at least 3 points")
            # self.polygon_points = utils.sort_coordinates(self.polygon_points)
            self.x_range = (
                np.min(self.polygon_points[:, 0]),
                np.max(self.polygon_points[:, 0]),
            )
            self.y_range = (
                np.min(self.polygon_points[:, 1]),
                np.max(self.polygon_points[:, 1]),
            )
