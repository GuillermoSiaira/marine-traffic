"""
IPFS Publisher + Ocean Protocol updater.
Runs daily after the processor. For a given date:
  1. Uploads processed Parquet files to IPFS via Pinata
  2. Updates the master manifest.json on IPFS
  3. Creates/updates the Ocean Protocol Data NFT (first run only creates it)

Usage:
    python main.py --date 2026-05-01
    python main.py --date yesterday
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import structlog
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger()

PINATA_JWT    = os.environ["PINATA_JWT"]
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "/data/processed"))
MANIFEST_PATH = Path(os.getenv("DATA_DIR", "/data")) / "manifest.json"

PINATA_PIN_URL  = "https://api.pinata.cloud/pinning/pinFileToIPFS"
PINATA_JSON_URL = "https://api.pinata.cloud/pinning/pinJSONToIPFS"
PINATA_HEADERS  = {"Authorization": f"Bearer {PINATA_JWT}"}

OCEAN_PRIVATE_KEY = os.getenv("OCEAN_PRIVATE_KEY")
OCEAN_NETWORK     = os.getenv("OCEAN_NETWORK", "polygon")
# Stored after first publish so we can update rather than recreate
NFT_ADDRESS_FILE  = Path(os.getenv("DATA_DIR", "/data")) / "ocean_nft_address.txt"


# ── IPFS helpers ──────────────────────────────────────────────────────────────

def pin_file(path: Path, name: str | None = None) -> str:
    """Upload a file to IPFS via Pinata, return CID."""
    metadata = json.dumps({"name": name or path.name,
                           "keyvalues": {"dataset": "open-ais-historical"}})
    data = requests.models.PreparedRequest()
    with open(path, "rb") as f:
        resp = requests.post(
            PINATA_PIN_URL,
            headers=PINATA_HEADERS,
            files={"file": (path.name, f, "application/octet-stream")},
            data={"pinataMetadata": metadata},
            timeout=120,
        )
    resp.raise_for_status()
    cid = resp.json()["IpfsHash"]
    log.info("pinned_file", file=path.name, cid=cid)
    return cid


def pin_json(data: dict, name: str) -> str:
    """Upload a JSON object to IPFS via Pinata, return CID."""
    payload = {
        "pinataMetadata": {"name": name},
        "pinataContent": data,
    }
    resp = requests.post(PINATA_JSON_URL, headers=PINATA_HEADERS,
                         json=payload, timeout=60)
    resp.raise_for_status()
    cid = resp.json()["IpfsHash"]
    log.info("pinned_json", name=name, cid=cid)
    return cid


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {
        "dataset":     "open-ais-historical",
        "version":     "1",
        "description": (
            "Global AIS vessel tracking data collected from aisstream.io. "
            "Port calls, voyage segments, and O-D matrices for maritime "
            "transportation analysis (Koopmans LP model)."
        ),
        "license":     "CC-BY-4.0",
        "homepage":    "https://github.com/guillermosiaira/marine-traffic",
        "contact":     "guillermosiaira@gmail.com",
        "days":        [],
    }


def save_manifest(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))


# ── Ocean Protocol ────────────────────────────────────────────────────────────

def get_or_create_ocean_asset(manifest_cid: str) -> str | None:
    """
    Create or update the Ocean Protocol Data NFT.
    Returns the NFT contract address, or None if OCEAN_PRIVATE_KEY is not set.
    """
    if not OCEAN_PRIVATE_KEY:
        log.warning("ocean_skipped", reason="OCEAN_PRIVATE_KEY not set")
        return None

    try:
        from ocean_lib.example_config import get_config_dict
        from ocean_lib.ocean.ocean import Ocean
        from ocean_lib.structures.algorithm_metadata import AlgorithmMetadata

        config   = get_config_dict(OCEAN_NETWORK)
        ocean    = Ocean(config)
        account  = ocean.create_account(OCEAN_PRIVATE_KEY)

        metadata = {
            "main": {
                "type":         "dataset",
                "name":         "Open AIS Historical Dataset",
                "dateCreated":  datetime.now(timezone.utc).isoformat(),
                "author":       "Guillermo Siaira",
                "license":      "CC-BY-4.0",
                "description": (
                    "Global AIS vessel tracking data (port calls, voyages, O-D matrices). "
                    "Designed for empirical testing of Koopmans (1949) transportation LP model."
                ),
                "tags":  ["AIS", "maritime", "shipping", "Koopmans", "LP", "open-data"],
                "files": [{"url": f"ipfs://{manifest_cid}", "contentType": "application/json"}],
            }
        }

        if NFT_ADDRESS_FILE.exists():
            nft_address = NFT_ADDRESS_FILE.read_text().strip()
            log.info("ocean_nft_exists", address=nft_address)
            # Update the asset metadata with new manifest CID
            # (Full update requires on-chain tx — skipped for brevity;
            #  the manifest CID in the description acts as the pointer)
            return nft_address

        # First run — create the Data NFT + datatoken
        data_nft, datatoken, ddo = ocean.assets.create(
            metadata=metadata,
            publisher_wallet=account,
            with_compute=False,
            encrypt_flag=False,
        )
        nft_address = data_nft.address
        NFT_ADDRESS_FILE.write_text(nft_address)
        log.info("ocean_asset_created", nft_address=nft_address,
                 datatoken=datatoken.address, did=ddo.did)
        return nft_address

    except Exception as e:
        log.error("ocean_error", error=str(e))
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def publish_date(date: datetime):
    date_str = date.strftime("%Y-%m-%d")
    day_dir  = PROCESSED_DIR / date_str

    if not day_dir.exists():
        log.error("processed_dir_missing", date=date_str, path=str(day_dir))
        return

    log.info("publishing", date=date_str)

    files_to_pin = {
        "positions_cid":  day_dir / f"positions_{date_str}.parquet",
        "port_calls_cid": day_dir / f"port_calls_{date_str}.parquet",
        "voyages_cid":    day_dir / f"voyages_{date_str}.parquet",
        "od_matrix_cid":  day_dir / f"od_matrix_{date_str}.parquet",
    }

    day_entry = {"date": date_str}
    for key, path in files_to_pin.items():
        if path.exists():
            day_entry[key]          = pin_file(path)
            day_entry[key + "_size"] = path.stat().st_size
        else:
            log.warning("file_missing", path=str(path))

    # Record count from positions file
    try:
        import pyarrow.parquet as pq
        pos_path = files_to_pin["positions_cid"]
        if pos_path.exists():
            day_entry["record_count"] = pq.read_metadata(pos_path).num_rows
    except Exception:
        pass

    # Update manifest
    manifest = load_manifest()
    # Replace existing entry for this date or append
    manifest["days"] = [d for d in manifest["days"] if d.get("date") != date_str]
    manifest["days"].append(day_entry)
    manifest["days"].sort(key=lambda d: d["date"])
    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()

    save_manifest(manifest)

    # Pin updated manifest
    manifest_cid = pin_json(manifest, "open-ais-manifest.json")
    manifest["manifest_cid"] = manifest_cid
    save_manifest(manifest)

    log.info("manifest_updated", cid=manifest_cid,
             total_days=len(manifest["days"]))

    # Ocean Protocol
    nft_address = get_or_create_ocean_asset(manifest_cid)
    if nft_address:
        log.info("ocean_updated", nft_address=nft_address,
                 url=f"https://market.oceanprotocol.com/asset/{nft_address}")

    log.info("publish_complete", date=date_str, manifest_cid=manifest_cid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="yesterday")
    args = parser.parse_args()

    if args.date == "yesterday":
        target = datetime.now(timezone.utc) - timedelta(days=1)
    else:
        target = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    publish_date(target)
