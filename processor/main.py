"""
Daily Processor
Reads raw Parquet files for a given date, detects port calls,
builds voyage segments and the O-D matrix.

Usage:
    python main.py --date 2026-05-01
    python main.py --date yesterday
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
from dotenv import load_dotenv

from port_index import PortIndex, PORT_RADIUS_NM

load_dotenv()
log = structlog.get_logger()

RAW_DIR       = Path("/data/raw")
PROCESSED_DIR = Path("/data/processed")

# A vessel is considered "stopped" when SOG < this threshold
STOPPED_SOG_KN = 0.5
# Minimum time stopped to count as a port call (minutes)
MIN_STOP_MINUTES = 20


# ── Schemas ───────────────────────────────────────────────────────────────────

PORT_CALLS_SCHEMA = pa.schema([
    pa.field("mmsi",           pa.int64()),
    pa.field("ship_name",      pa.string()),
    pa.field("ship_type",      pa.int16()),
    pa.field("port_id",        pa.string()),
    pa.field("port_name",      pa.string()),
    pa.field("country",        pa.string()),
    pa.field("port_lat",       pa.float32()),
    pa.field("port_lon",       pa.float32()),
    pa.field("arrived_at",     pa.timestamp("us", tz="UTC")),
    pa.field("departed_at",    pa.timestamp("us", tz="UTC")),
    pa.field("duration_hours", pa.float32()),
])

VOYAGE_SCHEMA = pa.schema([
    pa.field("mmsi",           pa.int64()),
    pa.field("ship_name",      pa.string()),
    pa.field("ship_type",      pa.int16()),
    pa.field("origin_port",    pa.string()),
    pa.field("origin_country", pa.string()),
    pa.field("dest_port",      pa.string()),
    pa.field("dest_country",   pa.string()),
    pa.field("departed_at",    pa.timestamp("us", tz="UTC")),
    pa.field("arrived_at",     pa.timestamp("us", tz="UTC")),
    pa.field("duration_hours", pa.float32()),
    pa.field("distance_nm",    pa.float32()),
    pa.field("avg_speed_kn",   pa.float32()),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in nautical miles."""
    R = 3440.065
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def load_day(date: datetime) -> pd.DataFrame:
    """Load all raw Parquet files for a given date into one DataFrame."""
    pattern = (
        f"year={date.year}/month={date.month:02d}/day={date.day:02d}/**/*.parquet"
    )
    files = list(RAW_DIR.glob(pattern))
    if not files:
        log.warning("no_files_found", date=date.date())
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True)
    df  = df.drop_duplicates(subset=["mmsi", "timestamp"]).sort_values(
              ["mmsi", "timestamp"])
    log.info("data_loaded", date=date.date(), records=len(df),
             vessels=df["mmsi"].nunique(), files=len(files))
    return df


def detect_port_calls(df: pd.DataFrame, port_idx: PortIndex) -> pd.DataFrame:
    """
    Detect port calls from raw AIS positions.

    Strategy:
      1. Filter rows where vessel is stopped: SOG < threshold OR nav_status in {1,5}
         (1=at anchor, 5=moored)
      2. For each stopped segment, look up nearest port
      3. Group consecutive stopped positions at the same port into one call
      4. Filter out stops shorter than MIN_STOP_MINUTES
    """
    calls = []

    # AIS nav_status codes that indicate "in port"
    IN_PORT_STATUS = {1, 5}  # At Anchor, Moored

    stopped = df[
        (df["sog"] < STOPPED_SOG_KN) | (df["nav_status"].isin(IN_PORT_STATUS))
    ].copy()

    if stopped.empty:
        return pd.DataFrame(columns=[f.name for f in PORT_CALLS_SCHEMA])

    # Annotate each stopped position with its nearest port
    log.info("annotating_ports", stopped_records=len(stopped))
    ports_found = []
    for _, row in stopped.iterrows():
        port = port_idx.in_port(row["lat"], row["lon"])
        ports_found.append(port.port_id if port else None)
    stopped["port_id"] = ports_found
    stopped = stopped.dropna(subset=["port_id"])

    # Group by vessel + port, merge consecutive stops at same port
    for mmsi, vessel_df in stopped.groupby("mmsi"):
        vessel_df = vessel_df.sort_values("timestamp")
        meta = df[df["mmsi"] == mmsi].iloc[0]

        current_port = None
        seg_start    = None
        seg_end      = None

        for _, row in vessel_df.iterrows():
            if row["port_id"] != current_port:
                # Save previous segment
                if current_port and seg_start:
                    duration = (seg_end - seg_start).total_seconds() / 3600
                    if duration >= MIN_STOP_MINUTES / 60:
                        port, _ = port_idx.nearest(
                            vessel_df[vessel_df["port_id"] == current_port]["lat"].mean(),
                            vessel_df[vessel_df["port_id"] == current_port]["lon"].mean(),
                        )
                        calls.append({
                            "mmsi":           int(mmsi),
                            "ship_name":      str(meta["ship_name"]),
                            "ship_type":      int(meta["ship_type"]),
                            "port_id":        current_port,
                            "port_name":      port.port_name,
                            "country":        port.country,
                            "port_lat":       float(port.lat),
                            "port_lon":       float(port.lon),
                            "arrived_at":     seg_start,
                            "departed_at":    seg_end,
                            "duration_hours": float(duration),
                        })
                current_port = row["port_id"]
                seg_start    = row["timestamp"]
            seg_end = row["timestamp"]

    if not calls:
        return pd.DataFrame(columns=[f.name for f in PORT_CALLS_SCHEMA])

    result = pd.DataFrame(calls)
    log.info("port_calls_detected", calls=len(result),
             vessels=result["mmsi"].nunique())
    return result


def build_voyages(port_calls: pd.DataFrame) -> pd.DataFrame:
    """
    Build voyage segments from consecutive port calls per vessel.
    Each voyage = (origin port call) → (next port call).
    """
    voyages = []
    for mmsi, calls in port_calls.groupby("mmsi"):
        calls = calls.sort_values("arrived_at").reset_index(drop=True)
        for i in range(len(calls) - 1):
            orig = calls.iloc[i]
            dest = calls.iloc[i + 1]

            departed  = orig["departed_at"]
            arrived   = dest["arrived_at"]
            if pd.isna(departed) or pd.isna(arrived):
                continue

            duration  = (arrived - departed).total_seconds() / 3600
            if duration <= 0:
                continue

            distance = haversine_nm(
                orig["port_lat"], orig["port_lon"],
                dest["port_lat"], dest["port_lon"],
            )
            avg_speed = distance / duration if duration > 0 else 0.0

            voyages.append({
                "mmsi":           int(mmsi),
                "ship_name":      str(orig["ship_name"]),
                "ship_type":      int(orig["ship_type"]),
                "origin_port":    str(orig["port_name"]),
                "origin_country": str(orig["country"]),
                "dest_port":      str(dest["port_name"]),
                "dest_country":   str(dest["country"]),
                "departed_at":    departed,
                "arrived_at":     arrived,
                "duration_hours": float(duration),
                "distance_nm":    float(distance),
                "avg_speed_kn":   float(avg_speed),
            })

    if not voyages:
        return pd.DataFrame(columns=[f.name for f in VOYAGE_SCHEMA])

    result = pd.DataFrame(voyages)
    log.info("voyages_built", voyages=len(result))
    return result


def build_od_matrix(voyages: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate voyage segments into an Origin-Destination matrix.
    Each row: (origin, destination, voyage_count, avg_distance_nm, avg_duration_h)
    """
    if voyages.empty:
        return pd.DataFrame()

    od = (
        voyages.groupby(["origin_port", "dest_port", "origin_country", "dest_country"])
        .agg(
            voyage_count   = ("mmsi",           "count"),
            avg_distance_nm= ("distance_nm",    "mean"),
            avg_duration_h = ("duration_hours", "mean"),
            avg_speed_kn   = ("avg_speed_kn",   "mean"),
        )
        .reset_index()
        .sort_values("voyage_count", ascending=False)
    )
    log.info("od_matrix_built", routes=len(od))
    return od


def save(df: pd.DataFrame, path: Path, schema=None):
    if df.empty:
        log.warning("empty_dataframe_skipped", path=str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=schema, safe=False) if schema else \
            pa.Table.from_pandas(df)
    pq.write_table(table, path, compression="snappy")
    log.info("saved", path=str(path), rows=len(df),
             size_mb=round(path.stat().st_size / 1_048_576, 2))


def process_date(date: datetime):
    date_str = date.strftime("%Y-%m-%d")
    out_dir  = PROCESSED_DIR / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("processing_date", date=date_str)

    df = load_day(date)
    if df.empty:
        log.warning("no_data", date=date_str)
        return

    # Save daily positions
    pos_path = out_dir / f"positions_{date_str}.parquet"
    save(df, pos_path)

    # Port calls
    port_idx = PortIndex()
    port_calls = detect_port_calls(df, port_idx)
    save(port_calls, out_dir / f"port_calls_{date_str}.parquet", PORT_CALLS_SCHEMA)

    # Voyages
    voyages = build_voyages(port_calls)
    save(voyages, out_dir / f"voyages_{date_str}.parquet", VOYAGE_SCHEMA)

    # O-D matrix
    od = build_od_matrix(voyages)
    save(od, out_dir / f"od_matrix_{date_str}.parquet")

    log.info("processing_complete", date=date_str, output=str(out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="yesterday",
                        help="Date to process: YYYY-MM-DD or 'yesterday'")
    args = parser.parse_args()

    if args.date == "yesterday":
        target = datetime.now(timezone.utc) - timedelta(days=1)
    else:
        target = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    process_date(target)
