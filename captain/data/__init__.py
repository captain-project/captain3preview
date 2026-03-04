"""Data structures for spatial data and extinction risk classification."""

from __future__ import annotations

from captain.data.extinction_risk import ExtinctionRisk, ExtinctionRiskStatic
from captain.data.spatial_data import (
    SpatialData,
    load_spatial_data,
    load_spatial_data_from_dir,
)

__all__ = [
    "SpatialData",
    "ExtinctionRisk",
    "ExtinctionRiskStatic",
    "load_spatial_data",
    "load_spatial_data_from_dir",
]
