# Open AIS Historical Dataset

Decentralized, open-source historical AIS vessel tracking dataset.
Collects global vessel positions 24/7, publishes daily snapshots to IPFS/Filecoin,
and lists them on Ocean Protocol for open access and community funding.

## Why this project exists

Tjalling Koopmans won the 1975 Nobel Prize in Economics for formalizing the
**transportation problem** using linear programming — originally motivated by
optimizing Allied merchant shipping in WWII. His model:

```
minimize  Σ c_ij · x_ij          (total transport cost)
subject to Σ_j x_ij ≤ S_i        (supply at origin i)
           Σ_i x_ij ≥ D_j        (demand at destination j)
               x_ij ≥ 0
```

The **dual solution** produces shadow prices at each port that, in a competitive
market, equal equilibrium freight rates. Koopmans could never test this empirically
— AIS data didn't exist. Now it does, and it's free.

This project builds the dataset needed to empirically test Koopmans' model:
- Are observed cargo flows efficient (do they minimize total transport cost)?
- Do port congestion levels correspond to the dual shadow prices?
- Where is the maritime system furthest from the theoretical optimum?

## Architecture

```
aisstream.io (free WebSocket)
        │
        ▼
  AIS Collector (Docker, Akash Network)
  - Streams all global vessel positions
  - Detects port calls via speed + NavigationalStatus
  - Writes hourly Parquet batches
        │
        ▼
  Daily Processor
  - Builds port call events (vessel, origin port, arrival/departure timestamps)
  - Builds voyage segments (origin → destination, duration, distance)
  - Computes O-D matrix between ports
        │
        ▼
  IPFS Publisher (Pinata → Filecoin)
  - Uploads processed daily datasets to IPFS
  - Maintains manifest.json with all CIDs
  - Publishes/updates Data NFT on Ocean Protocol (Polygon)
        │
        ▼
  Ocean Protocol Marketplace
  - Dataset discoverable at market.oceanprotocol.com
  - Open access (free) or subscription to fund infra
  - Apply for Ocean Shipyard Grant
```

## Stack

| Layer | Tool | Why |
|---|---|---|
| AIS data | [aisstream.io](https://aisstream.io) | Free global WebSocket stream |
| Compute | [Akash Network](https://akash.network) | Decentralized, ~70% cheaper than GCP |
| Storage | [IPFS + Filecoin](https://web3.storage) | Decentralized, content-addressed |
| Archival | [Pinata](https://pinata.cloud) | Simple pinning API, free tier |
| Marketplace | [Ocean Protocol](https://oceanprotocol.com) | Data NFT + funding |
| Analysis | Python + SciPy | Koopmans LP model |

## Quickstart (local)

```bash
# 1. Copy env file and fill in your keys
cp .env.example .env

# 2. Run collector locally
docker compose up collector

# 3. Run processor on collected data
docker compose run processor

# 4. Publish to IPFS
docker compose run publisher
```

## Get your API keys

1. **aisstream.io** — free account at https://aisstream.io/authenticate
2. **Pinata** — free 1GB at https://app.pinata.cloud (use JWT auth)
3. **Ocean Protocol** — just needs your MetaMask wallet on Polygon network

## Deploy to Akash

```bash
# Install Akash CLI
# https://docs.akash.network/guides/cli/install

# Deploy collector
akash tx deployment create infra/akash/deploy.sdl.yml --from <your-wallet>
```

## Data schema

### Port calls (`port_calls.parquet`)
| Field | Type | Description |
|---|---|---|
| mmsi | int64 | Vessel identifier |
| ship_name | string | Vessel name |
| ship_type | int16 | AIS vessel type code |
| port_id | string | NGA World Port Index ID |
| port_name | string | Port name |
| country | string | Port country (ISO 3166) |
| lat | float32 | Port latitude |
| lon | float32 | Port longitude |
| arrived_at | timestamp | Arrival UTC |
| departed_at | timestamp | Departure UTC (null if still in port) |
| duration_hours | float32 | Time in port |

### Voyage segments (`voyages.parquet`)
| Field | Type | Description |
|---|---|---|
| mmsi | int64 | Vessel identifier |
| origin_port | string | Departure port name |
| dest_port | string | Arrival port name |
| departed_at | timestamp | Departure UTC |
| arrived_at | timestamp | Arrival UTC |
| duration_hours | float32 | Voyage duration |
| distance_nm | float32 | Great-circle distance in nautical miles |
| avg_speed_kn | float32 | Average speed in knots |

### Daily positions (`positions_YYYY-MM-DD.parquet`)
| Field | Type | Description |
|---|---|---|
| mmsi | int64 | Vessel identifier |
| ship_name | string | |
| lat | float32 | |
| lon | float32 | |
| sog | float32 | Speed over ground (knots) |
| cog | float32 | Course over ground (degrees) |
| nav_status | int8 | AIS navigational status |
| timestamp | timestamp | UTC |

## IPFS Manifest

Each day's upload creates/updates `manifest.json` on IPFS:

```json
{
  "dataset": "open-ais-historical",
  "version": "1",
  "license": "CC-BY-4.0",
  "contact": "guillermosiaira@gmail.com",
  "days": [
    {
      "date": "2026-05-01",
      "positions_cid": "bafybeig...",
      "port_calls_cid": "bafybeih...",
      "voyages_cid": "bafybeia...",
      "record_count": 1847293
    }
  ]
}
```

## License

Data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Code: [MIT](LICENSE)

## Funding

This project applies for the [Ocean Shipyard Grant](https://oceanprotocol.com/fund) program.
If you use this data, consider purchasing a datatoken on Ocean Market to fund
ongoing infrastructure costs.
