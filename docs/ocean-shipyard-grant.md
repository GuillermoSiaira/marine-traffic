# Ocean Shipyard Grant Application
## Open AIS Historical Dataset — Koopmans LP Applied to Maritime Traffic

---

### Project Summary

We are building the first **open, decentralized, quality-scored AIS historical dataset**
published as a Data NFT on Ocean Protocol. The dataset enables empirical testing of
Tjalling Koopmans' 1949 Nobel Prize-winning transportation linear programming model
on real global maritime data — work that was theoretically complete 75 years ago but
never empirically validated due to the absence of data.

**Funding requested:** $30,000 USD equivalent in OCEAN tokens
**Timeline:** 12 months
**GitHub:** https://github.com/GuillermoSiaira/marine-traffic

---

### The Problem

**No open maritime dataset exists for serious quantitative research.**

Every major AIS data provider — MarineTraffic (now Kpler), Spire Maritime, Vortexa,
exactEarth — is closed, expensive, and inaccessible to academic researchers,
independent analysts, and public interest organizations.

The consequence is that:

1. **A Nobel-Prize-winning model sits untested.** Koopmans (1949) proved mathematically
   that competitive freight markets should produce shadow prices at ports equal to
   marginal transport costs. This has never been empirically verified at scale because
   the data did not exist in open form.

2. **Maritime inefficiency is invisible.** ~80% of global trade moves by sea.
   Identifying where the system deviates from optimal allocation — and why — has
   direct policy implications for trade, emissions, and economic development.
   Currently only well-funded private actors can do this analysis.

3. **Dark vessel activity goes unstudied.** Sanctions evasion, illegal fishing, and
   AIS spoofing are documentable from AIS data but require longitudinal historical
   records that no open source provides.

---

### Our Solution

A fully decentralized pipeline that:

1. **Collects** real-time global AIS data 24/7 via aisstream.io (free WebSocket)
2. **Processes** it daily: detects port calls, builds voyage segments, computes
   origin-destination matrices between ~3,700 world ports (NGA World Port Index)
3. **Scores quality** per day: quantifies spoofing, dark periods, coverage gaps,
   ghost jumps — transparency that no commercial provider offers
4. **Publishes** to IPFS/Filecoin as daily Parquet snapshots with a persistent manifest
5. **Lists** on Ocean Market as a Data NFT (ERC-721) with open access datatokens

The **analysis layer** runs Koopmans' transportation LP on the accumulated data,
producing shadow prices per port and identifying deviations from optimal flow.
A graph analysis module (NetworkX + Claude API) interprets the network topology
in real-world terms: which ports are chokepoints, which communities represent
natural trading blocs, and where observed flows diverge most from theory.

---

### Why This Needs a Grant

Our current infrastructure runs on **free-tier services** (aisstream.io terrestrial AIS,
Pinata 1GB IPFS). This produces a useful but incomplete dataset with a known, critical
limitation: **terrestrial AIS covers only ~60 nautical miles from shore.**

Trans-oceanic routes — the backbone of global trade — are invisible to us.
A vessel leaving Rotterdam for Singapore disappears from our dataset for 20 days
and reappears at the destination. The port-call detection works; the voyage
reconstruction does not.

**The grant would fund satellite AIS coverage**, which eliminates this gap entirely.

#### Budget breakdown

| Item | Monthly | Annual |
|---|---|---|
| Satellite AIS feed (Spire Maritime entry tier) | $800 | $9,600 |
| Historical AIS backfill — 5 years (one-time) | — | $4,000 |
| Hetzner VPS compute (collector + processor) | $12 | $144 |
| Pinata IPFS storage (growing dataset) | $40 | $480 |
| Filecoin long-term archival | $20 | $240 |
| Development and maintenance (part-time) | $1,300 | $15,600 |
| **Total** | | **~$30,000** |

After 12 months, the project pursues multi-source sustainability rather than relying
on a single revenue stream. Ocean Market datatoken sales provide a first revenue layer,
but the primary path to ongoing funding is:

- **Follow-on grants:** The Filecoin Foundation Grant Program and EU Horizon Europe
  both fund open scientific data infrastructure. A working dataset with one year of
  satellite coverage and documented academic use is a strong basis for both.
- **Academic partnerships:** Universities and research institutes routinely fund
  access to proprietary datasets; an open alternative with a citable methodology
  is attractive to economics and operations research departments.
- **NGO partnerships:** Organizations such as Global Fishing Watch, Transparency
  International, and environmental compliance bodies have operational need for
  longitudinal AIS data and budgets to fund it.

The base dataset remains CC-BY open access permanently regardless of funding outcome.
Datatokens gate only the premium satellite tier and Compute-to-Data features.

---

### Ocean Protocol Integration

We use Ocean Protocol as the **primary distribution and monetization layer**:

- **Data NFT (ERC-721):** The manifest.json — an index of all daily datasets with
  their IPFS CIDs — is minted as a Data NFT on Polygon. This gives the dataset
  a permanent, verifiable on-chain identity.

- **Datatokens (ERC-20):** Access to the satellite-quality tier (complete ocean
  routes, commodity estimates) is gated by datatokens, priced to fund infrastructure.
  The terrestrial-quality tier (port calls, coastal routing) is always free.

- **Ocean Market listing:** The dataset is discoverable at market.oceanprotocol.com,
  making it available to the entire Ocean ecosystem and its data buyers.

- **Compute-to-Data (future):** Once we accumulate 6+ months of satellite data,
  we will enable Compute-to-Data so researchers can run the Koopmans LP model
  against the full dataset without the raw data leaving IPFS — privacy-preserving
  quantitative research at scale.

---

### The Koopmans Connection — Why It Matters

Tjalling Koopmans shared the 1975 Nobel Prize in Economics for proving that the
**transportation problem** — minimizing total shipping cost subject to supply and
demand constraints — produces dual variables (shadow prices) that, in competitive
markets, represent equilibrium freight rates.

```
minimize    Σ c_ij · x_ij          (total transport cost)
subject to  Σ_j x_ij = S_i         (all supply shipped from port i)
            Σ_i x_ij = D_j         (all demand met at port j)
            x_ij ≥ 0
```

The shadow prices u_i, v_j of this LP satisfy:
- u_i + v_j ≤ c_ij for all routes (no route is "too cheap")
- u_i + v_j = c_ij for all routes actually used (active routes are efficient)

**Koopmans built this model on wartime shipping data and never had enough data to
test it empirically.** We now have the tools to do it. With daily AIS data, we can:

1. Measure whether observed shipping flows minimize total transport cost
2. Compare model shadow prices to observed freight rate indices (Baltic Dry Index, etc.)
3. Identify systematically inefficient corridors and hypothesize structural causes
4. Quantify the economic cost of maritime disruptions (Suez blockages, Red Sea attacks)
   in Koopmanian terms — not just "X days of delay" but "Y% deviation from optimal allocation"

This connects a 75-year-old Nobel-Prize-winning theory to live economic data for the
first time, in a fully reproducible, open-source, decentralized framework.

---

### Graph Analysis + LLM Interpretation

Beyond the LP, we model the shipping network as a directed weighted graph:

- **Nodes:** world ports (attributes: country, region, cargo specialization)
- **Edges:** observed shipping routes (weight: voyage frequency; distance: NM)
- **Metrics:** PageRank (hub importance), betweenness centrality (chokepoints),
  Louvain community detection (natural trading blocs)

The graph metrics complement the LP: betweenness centrality identifies the same
ports that have high shadow prices in the dual solution, but from a topological
perspective. An LLM interpretation layer (Claude API) bridges the mathematics
to real-world context — explaining *why* a port has high centrality in terms of
geography, commodity flows, and historical patterns.

This is a novel methodological contribution: **Koopmans LP + graph theory + LLM
interpretation on open AIS data.**

---

### Team

**Guillermo Siaira** — Project lead
Independent researcher and developer. Background in economics and distributed systems.
Motivated by the intersection of classical economic theory and decentralized data
infrastructure. Contact: guillermosiaira@gmail.com

Open to collaborators, especially maritime economists and graph theory researchers.
The project is explicitly designed as a community effort — all code is MIT licensed,
all data is CC-BY.

---

### Traction and Current Status

- ✅ Full data pipeline implemented and open-sourced on GitHub
- ✅ AIS collector running via aisstream.io (terrestrial coverage)
- ✅ Daily processor: port call detection, voyage segments, O-D matrix
- ✅ Quality validator: per-day quality score (0-1) published with each dataset
- ✅ IPFS publisher: daily Parquet snapshots pinned via Pinata
- ✅ Ocean Protocol Data NFT: manifest published on Polygon
- ✅ Koopmans LP model: transportation problem solver with shadow price output
- ✅ Graph analysis: NetworkX + LLM interpretation module
- 🔄 Accumulating first weeks of live data (started May 2026)
- ⏳ Satellite AIS integration: pending funding

**GitHub:** https://github.com/GuillermoSiaira/marine-traffic
**Ocean Market:** [link once NFT is published]

---

### Impact Metrics (12-month targets)

| Metric | Target |
|---|---|
| Days of satellite AIS data published | 365 |
| Unique vessels tracked | >100,000 |
| Port pairs in O-D matrix | >50,000 |
| GitHub stars | >500 |
| Academic citations / uses | >10 |
| Datatoken sales | >$200/month |
| Follow-on grant applications submitted | ≥2 (Filecoin Foundation, EU Horizon) |
| Academic or NGO partnerships initiated | ≥1 |
| Ocean Market dataset downloads | >1,000 |

---

### Broader Impact

**For researchers:** The first replicable, open dataset for maritime transportation
economics. Enables validation of Koopmans, Samuelson's spatial price equilibrium,
and modern supply chain resilience models.

**For policy:** Quantitative evidence on maritime efficiency for port authorities,
shipping regulators, and trade ministries — without commercial data vendor dependency.

**For Ocean Protocol:** A flagship use case demonstrating that Ocean can host serious
scientific datasets, not just financial data. Maritime data is a $2B+ market;
an open alternative creates ecosystem gravity.

**For the public:** Dark vessel detection, sanctions evasion patterns, and illegal
fishing corridors become observable and documentable by civil society, journalists,
and NGOs — currently only possible for well-funded institutions.

---

### License

- Code: MIT
- Data: Creative Commons CC-BY 4.0
- Models/analysis outputs: CC-BY 4.0

All outputs are permanently open. Datatokens gate premium features, not the core data.
