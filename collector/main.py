"""
AIS Stream Collector
Connects to aisstream.io WebSocket, buffers messages, writes Parquet batches,
uploads to IPFS via Pinata.
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
import websockets
from dotenv import load_dotenv
from prometheus_client import Counter, Gauge, start_http_server
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()

log = structlog.get_logger()

# ── Config ────────────────────────────────────────────────────────────────────
AISSTREAM_API_KEY = os.environ["AISSTREAM_API_KEY"]
PINATA_JWT        = os.environ["PINATA_JWT"]
DATA_DIR          = Path(os.getenv("DATA_DIR", "/data/raw"))
FLUSH_INTERVAL    = int(os.getenv("FLUSH_INTERVAL_SECONDS", "600"))
FLUSH_MAX_MSGS    = int(os.getenv("FLUSH_MAX_MESSAGES", "50000"))
BOUNDING_BOXES    = json.loads(os.getenv("BOUNDING_BOXES", "[[[-90,-180],[90,180]]]"))

AISSTREAM_URI = "wss://stream.aisstream.io/v0/stream"

# ── Prometheus metrics ────────────────────────────────────────────────────────
msgs_received  = Counter("ais_messages_received_total", "AIS messages received")
msgs_written   = Counter("ais_messages_written_total",  "AIS messages written to Parquet")
files_uploaded = Counter("ais_files_uploaded_total",    "Parquet files uploaded to IPFS")
buffer_size    = Gauge("ais_buffer_size",               "Current message buffer size")
last_flush     = Gauge("ais_last_flush_timestamp",      "Timestamp of last flush")

# ── Arrow schema ──────────────────────────────────────────────────────────────
AIS_SCHEMA = pa.schema([
    pa.field("mmsi",       pa.int64()),
    pa.field("ship_name",  pa.string()),
    pa.field("ship_type",  pa.int16()),
    pa.field("lat",        pa.float32()),
    pa.field("lon",        pa.float32()),
    pa.field("sog",        pa.float32()),   # speed over ground, knots
    pa.field("cog",        pa.float32()),   # course over ground, degrees
    pa.field("heading",    pa.int16()),     # true heading
    pa.field("nav_status", pa.int8()),      # AIS navigational status
    pa.field("msg_type",   pa.string()),    # AIS message type
    pa.field("timestamp",  pa.timestamp("us", tz="UTC")),
])


def parse_message(raw: dict) -> dict | None:
    """Extract relevant fields from an aisstream.io message."""
    try:
        msg_type = raw.get("MessageType", "")
        meta     = raw.get("MetaData", {})
        msg      = raw.get("Message", {})

        # Position reports are under different keys depending on class
        pos = (
            msg.get("PositionReport")
            or msg.get("StandardClassBPositionReport")
            or msg.get("ExtendedClassBPositionReport")
        )
        if pos is None:
            return None

        ts_raw = meta.get("time_utc", "")
        # Format: "2026-05-01 12:00:00.000000 +0000 UTC"
        try:
            ts = pd.Timestamp(ts_raw.replace(" UTC", "").strip(), tz="UTC")
        except Exception:
            ts = pd.Timestamp.now(tz="UTC")

        return {
            "mmsi":       int(meta.get("MMSI", 0)),
            "ship_name":  str(meta.get("ShipName", "")).strip(),
            "ship_type":  int(meta.get("ShipType", 0)),
            "lat":        float(pos.get("Latitude",  pos.get("Lat", 0.0))),
            "lon":        float(pos.get("Longitude", pos.get("Lon", 0.0))),
            "sog":        float(pos.get("Sog", 0.0)),
            "cog":        float(pos.get("Cog", 0.0)),
            "heading":    int(pos.get("TrueHeading", 511)),   # 511 = not available
            "nav_status": int(pos.get("NavigationalStatus", 15)),
            "msg_type":   msg_type,
            "timestamp":  ts,
        }
    except Exception as e:
        log.debug("parse_error", error=str(e))
        return None


class MessageBuffer:
    def __init__(self):
        self.records: list[dict] = []
        self.created_at = time.monotonic()

    def add(self, record: dict):
        self.records.append(record)

    def should_flush(self) -> bool:
        age = time.monotonic() - self.created_at
        return age >= FLUSH_INTERVAL or len(self.records) >= FLUSH_MAX_MSGS

    def size(self) -> int:
        return len(self.records)


def write_parquet(records: list[dict]) -> Path:
    """Write records to a partitioned Parquet file. Returns file path."""
    df = pd.DataFrame(records)

    # Coerce types
    for col, dtype in [
        ("mmsi", "int64"), ("ship_type", "Int16"), ("heading", "Int16"),
        ("nav_status", "Int8"), ("lat", "float32"), ("lon", "float32"),
        ("sog", "float32"), ("cog", "float32"),
    ]:
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    now  = datetime.now(timezone.utc)
    part = DATA_DIR / f"year={now.year}" / f"month={now.month:02d}" / \
           f"day={now.day:02d}" / f"hour={now.hour:02d}"
    part.mkdir(parents=True, exist_ok=True)

    path = part / f"batch_{int(now.timestamp())}.parquet"
    table = pa.Table.from_pandas(df, schema=AIS_SCHEMA, safe=False)
    pq.write_table(table, path, compression="snappy")

    log.info("parquet_written", path=str(path), records=len(records),
             size_mb=round(path.stat().st_size / 1_048_576, 2))
    return path


@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(5))
async def upload_to_ipfs(path: Path, session: aiohttp.ClientSession) -> str:
    """Pin file to IPFS via Pinata. Returns IPFS CID."""
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    headers = {"Authorization": f"Bearer {PINATA_JWT}"}

    metadata = json.dumps({
        "name": path.name,
        "keyvalues": {"dataset": "open-ais-historical", "file": path.name},
    })

    data = aiohttp.FormData()
    data.add_field("pinataMetadata", metadata, content_type="application/json")
    data.add_field("file", open(path, "rb"), filename=path.name,
                   content_type="application/octet-stream")

    async with session.post(url, headers=headers, data=data) as resp:
        resp.raise_for_status()
        result = await resp.json()
        cid = result["IpfsHash"]
        log.info("ipfs_uploaded", cid=cid, file=path.name)
        files_uploaded.inc()
        return cid


async def flush(buffer: MessageBuffer, session: aiohttp.ClientSession) -> MessageBuffer:
    """Write buffer to Parquet, upload to IPFS, return fresh buffer."""
    if buffer.size() == 0:
        return MessageBuffer()

    records = buffer.records
    msgs_written.inc(len(records))
    last_flush.set(time.time())

    try:
        path = write_parquet(records)
        await upload_to_ipfs(path, session)
    except Exception as e:
        log.error("flush_error", error=str(e))

    return MessageBuffer()


async def stream(session: aiohttp.ClientSession):
    """Main WebSocket loop — reconnects indefinitely on disconnect."""
    subscribe_msg = json.dumps({
        "APIKey": AISSTREAM_API_KEY,
        "BoundingBoxes": BOUNDING_BOXES,
        "FilterMessageTypes": [
            "PositionReport",
            "StandardClassBPositionReport",
            "ExtendedClassBPositionReport",
        ],
    })

    buf = MessageBuffer()

    while True:
        try:
            log.info("connecting", uri=AISSTREAM_URI)
            async with websockets.connect(
                AISSTREAM_URI,
                ping_interval=20,
                ping_timeout=30,
                max_size=2**20,
            ) as ws:
                await ws.send(subscribe_msg)
                log.info("subscribed", bounding_boxes=BOUNDING_BOXES)

                async for raw_msg in ws:
                    msgs_received.inc()
                    data = json.loads(raw_msg)
                    record = parse_message(data)
                    if record:
                        buf.add(record)
                        buffer_size.set(buf.size())

                    if buf.should_flush():
                        buf = await flush(buf, session)
                        buffer_size.set(0)

        except websockets.ConnectionClosed as e:
            log.warning("connection_closed", code=e.code, reason=e.reason)
            buf = await flush(buf, session)
            await asyncio.sleep(5)
        except Exception as e:
            log.error("stream_error", error=str(e))
            buf = await flush(buf, session)
            await asyncio.sleep(10)


async def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    start_http_server(8000)   # Prometheus metrics endpoint
    log.info("collector_starting", data_dir=str(DATA_DIR),
             flush_interval=FLUSH_INTERVAL, flush_max=FLUSH_MAX_MSGS)

    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        await stream(session)


if __name__ == "__main__":
    asyncio.run(main())
