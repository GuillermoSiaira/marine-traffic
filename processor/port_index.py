"""
World Port Index loader.
Downloads the NGA World Port Index (free, public domain) and provides
a fast nearest-port lookup via a KD-tree.
"""

import io
import math
import struct
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import requests
import structlog

log = structlog.get_logger()

WPI_URL  = "https://msi.nga.mil/api/publications/download?type=view&key=16920959/SFH00000/UpdatedPub150.csv"
WPI_CACHE = Path("/data/wpi_ports.parquet")

# Radius used to decide a vessel is "in port" (nautical miles)
PORT_RADIUS_NM = 5.0


class Port(NamedTuple):
    port_id:   str
    port_name: str
    country:   str
    lat:       float
    lon:       float


def _nm_to_deg(nm: float) -> float:
    """Approximate: 1 degree lat ≈ 60 NM."""
    return nm / 60.0


def load_port_index() -> tuple[pd.DataFrame, np.ndarray]:
    """Return (ports_df, xy_array) where xy_array is (N,2) lat/lon in radians."""
    if WPI_CACHE.exists():
        df = pd.read_parquet(WPI_CACHE)
    else:
        log.info("downloading_wpi")
        r = requests.get(WPI_URL, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)

        # WPI column names vary by release; pick what we need
        rename = {}
        for col in df.columns:
            cl = col.lower()
            if "port_nm"   in cl or "port name" in cl:  rename[col] = "port_name"
            elif "wpi_"    in cl and "no" in cl:         rename[col] = "port_id"
            elif "country" in cl:                        rename[col] = "country"
            elif "lat_deg" in cl:                        rename[col] = "lat"
            elif "long_deg" in cl:                       rename[col] = "lon"
        df = df.rename(columns=rename)

        keep = ["port_id", "port_name", "country", "lat", "lon"]
        df   = df[[c for c in keep if c in df.columns]].dropna(subset=["lat","lon"])
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat","lon"]).reset_index(drop=True)

        WPI_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(WPI_CACHE)
        log.info("wpi_loaded", ports=len(df))

    coords = np.radians(df[["lat","lon"]].values.astype(np.float64))
    return df, coords


class PortIndex:
    """Fast nearest-port lookup using a brute-force vectorised search.
    For ~3700 ports this is plenty fast (<1ms per query).
    """

    EARTH_RADIUS_NM = 3440.065  # nautical miles

    def __init__(self):
        self.df, self._coords_rad = load_port_index()

    def nearest(self, lat: float, lon: float) -> tuple[Port | None, float]:
        """Return (Port, distance_nm) for the closest port."""
        pt = np.radians([lat, lon])
        # Haversine vectorised
        dlat = self._coords_rad[:, 0] - pt[0]
        dlon = self._coords_rad[:, 1] - pt[1]
        a    = np.sin(dlat/2)**2 + np.cos(pt[0]) * np.cos(self._coords_rad[:,0]) * np.sin(dlon/2)**2
        dist = 2 * self.EARTH_RADIUS_NM * np.arcsin(np.sqrt(a))
        idx  = int(np.argmin(dist))
        row  = self.df.iloc[idx]
        port = Port(
            port_id   = str(row.get("port_id",   idx)),
            port_name = str(row.get("port_name", "Unknown")),
            country   = str(row.get("country",   "")),
            lat       = float(row["lat"]),
            lon       = float(row["lon"]),
        )
        return port, float(dist[idx])

    def in_port(self, lat: float, lon: float) -> Port | None:
        """Return Port if vessel is within PORT_RADIUS_NM, else None."""
        port, dist = self.nearest(lat, lon)
        return port if dist <= PORT_RADIUS_NM else None
