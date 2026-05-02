"""
Graph Analysis + LLM Interpretation
=====================================
Builds a directed weighted graph from the O-D matrix,
computes network metrics, and uses Claude to interpret
what the topology means in real-world maritime terms.

Usage:
    python graph_analysis.py --od data/processed/2026-05-01/od_matrix_2026-05-01.parquet
    python graph_analysis.py --od data/processed/ --days 30 --top 50
"""

import argparse
import json
import os
from pathlib import Path

import anthropic
import networkx as nx
import numpy as np
import pandas as pd
import structlog
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph(od: pd.DataFrame, min_voyages: int = 3) -> nx.DiGraph:
    """
    Build a directed weighted graph from the O-D matrix.

    Nodes  = ports (with country attribute)
    Edges  = routes (weight = voyage_count, distance = avg_distance_nm)
    """
    G = nx.DiGraph()

    for _, row in od.iterrows():
        if row["voyage_count"] < min_voyages:
            continue

        orig = row["origin_port"]
        dest = row["dest_port"]

        if not G.has_node(orig):
            G.add_node(orig, country=row.get("origin_country", ""))
        if not G.has_node(dest):
            G.add_node(dest, country=row.get("dest_country", ""))

        G.add_edge(
            orig, dest,
            weight      = float(row["voyage_count"]),
            distance_nm = float(row.get("avg_distance_nm", 0)),
            duration_h  = float(row.get("avg_duration_h", 0)),
            speed_kn    = float(row.get("avg_speed_kn", 0)),
        )

    log.info("graph_built", nodes=G.number_of_nodes(),
             edges=G.number_of_edges(), min_voyages=min_voyages)
    return G


# ── Network metrics ───────────────────────────────────────────────────────────

def compute_metrics(G: nx.DiGraph) -> dict:
    """
    Compute standard network metrics relevant to maritime transport.
    Returns a dict with both per-node and global metrics.
    """
    # Undirected version for some metrics
    U = G.to_undirected()

    # PageRank — "importance" considering incoming traffic volume
    pagerank = nx.pagerank(G, weight="weight", max_iter=200)

    # Betweenness centrality — ports that lie on many shortest paths (chokepoints)
    betweenness = nx.betweenness_centrality(U, weight="distance_nm", normalized=True)

    # In/out degree and strength (weighted degree)
    in_strength  = {n: sum(d["weight"] for _, _, d in G.in_edges(n,  data=True)) for n in G}
    out_strength = {n: sum(d["weight"] for _, _, d in G.out_edges(n, data=True)) for n in G}

    # Community detection (Louvain on undirected graph)
    try:
        from networkx.algorithms.community import louvain_communities
        communities_raw = louvain_communities(U, weight="weight", seed=42)
        community_map   = {}
        for i, comm in enumerate(communities_raw):
            for node in comm:
                community_map[node] = i
        n_communities = len(communities_raw)
    except Exception:
        community_map = {n: 0 for n in G.nodes()}
        n_communities = 1

    # Build per-node summary
    nodes = []
    for node in G.nodes():
        nodes.append({
            "port":         node,
            "country":      G.nodes[node].get("country", ""),
            "pagerank":     round(pagerank.get(node, 0), 6),
            "betweenness":  round(betweenness.get(node, 0), 6),
            "in_voyages":   round(in_strength.get(node, 0), 1),
            "out_voyages":  round(out_strength.get(node, 0), 1),
            "community":    community_map.get(node, -1),
            "in_degree":    G.in_degree(node),
            "out_degree":   G.out_degree(node),
        })

    nodes_df = pd.DataFrame(nodes).sort_values("pagerank", ascending=False)

    # Global metrics
    global_metrics = {
        "n_ports":         G.number_of_nodes(),
        "n_routes":        G.number_of_edges(),
        "n_communities":   n_communities,
        "density":         round(nx.density(G), 6),
        "total_voyages":   int(sum(d["weight"] for _, _, d in G.edges(data=True))),
        "avg_distance_nm": round(
            np.mean([d["distance_nm"] for _, _, d in G.edges(data=True)
                     if d["distance_nm"] > 0]) if G.number_of_edges() > 0 else 0, 1
        ),
    }

    # Top ports by each metric
    top_pagerank    = nodes_df.head(15)[["port","country","pagerank","community"]].to_dict("records")
    top_bottlenecks = nodes_df.sort_values("betweenness", ascending=False).head(15)[
                          ["port","country","betweenness","community"]].to_dict("records")
    top_importers   = nodes_df.sort_values("in_voyages", ascending=False).head(10)[
                          ["port","country","in_voyages"]].to_dict("records")
    top_exporters   = nodes_df.sort_values("out_voyages", ascending=False).head(10)[
                          ["port","country","out_voyages"]].to_dict("records")

    # Community summaries
    community_summaries = []
    for cid in sorted(set(community_map.values())):
        members = [n for n, c in community_map.items() if c == cid]
        sub     = nodes_df[nodes_df["community"] == cid]
        community_summaries.append({
            "id":         cid,
            "size":       len(members),
            "top_ports":  sub.sort_values("pagerank", ascending=False)
                            .head(5)["port"].tolist(),
            "countries":  sub["country"].value_counts().head(3).to_dict(),
        })

    return {
        "global":       global_metrics,
        "nodes":        nodes_df,
        "top_pagerank": top_pagerank,
        "bottlenecks":  top_bottlenecks,
        "top_importers":top_importers,
        "top_exporters":top_exporters,
        "communities":  community_summaries,
    }


# ── LLM interpretation ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a maritime economics expert with deep knowledge of:
- Global shipping networks and trade routes
- Port geography and strategic chokepoints (Suez, Malacca, Panama, Bosporus, Hormuz)
- Commodity flows (oil, LNG, containers, bulk cargo, grain)
- Historical shipping disruptions and their network effects
- Koopmans (1949) transportation LP theory and its implications for maritime efficiency
- The relationship between network topology and shadow prices in transport economics

You are analyzing real AIS vessel tracking data processed into a shipping network graph.
Be specific, insightful, and connect mathematical observations to real-world causes.
When shadow prices are provided, interpret them as Koopmans equilibrium freight rates."""


def llm_interpret(metrics: dict, shadow_prices: pd.DataFrame | None = None,
                  question: str | None = None) -> str:
    """
    Send graph metrics + optional shadow prices to Claude for interpretation.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build the context payload
    context = {
        "network_summary":    metrics["global"],
        "top_hubs_pagerank":  metrics["top_pagerank"],
        "chokepoints":        metrics["bottlenecks"],
        "top_importers":      metrics["top_importers"],
        "top_exporters":      metrics["top_exporters"],
        "trading_communities":metrics["communities"],
    }

    if shadow_prices is not None and not shadow_prices.empty:
        context["shadow_prices_top20"] = shadow_prices.head(20).to_dict("records")

    default_question = """Analyze this maritime shipping network and answer:

1. TOPOLOGY: What does the network structure reveal about global trade patterns?
   Which ports are true hubs vs. simple waypoints? What does the community structure tell us?

2. CHOKEPOINTS: Which ports have disproportionate betweenness centrality?
   What real-world geographic or logistical factors explain this?

3. EFFICIENCY (Koopmans lens): If shadow prices are provided, do they align with
   expectations from port geography? Where is the network furthest from optimum and why?

4. ANOMALIES: Are there any surprising patterns — ports with unexpected centrality,
   communities that shouldn't cluster together, routes that seem suboptimal?

5. RESILIENCE: Based on the topology, which single port removal would most disrupt
   global maritime trade? What historical precedent supports this?

Be specific. Name real ports, real trade corridors, real commodities."""

    user_msg = f"""Here is the shipping network data:

```json
{json.dumps(context, indent=2, default=str)}
```

{question or default_question}"""

    log.info("calling_claude", model="claude-opus-4-7")

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    return response.content[0].text


def interactive_mode(G: nx.DiGraph, metrics: dict,
                     shadow_prices: pd.DataFrame | None = None):
    """REPL for asking custom questions about the network."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    context = {
        "network_summary":     metrics["global"],
        "top_hubs":            metrics["top_pagerank"],
        "chokepoints":         metrics["bottlenecks"],
        "trading_communities": metrics["communities"],
    }
    if shadow_prices is not None and not shadow_prices.empty:
        context["shadow_prices"] = shadow_prices.head(20).to_dict("records")

    history = [{"role": "user", "content":
                f"Network data:\n```json\n{json.dumps(context, indent=2, default=str)}\n```\n"
                f"I'll ask questions about this shipping network."}]

    print("\n=== Interactive Graph Analysis (Ctrl+C to exit) ===")
    print("Ejemplos:")
    print("  '¿Qué pasaría si cerramos el Estrecho de Malacca?'")
    print("  '¿Por qué Rotterdam tiene tanto peso?'")
    print("  '¿Qué comunidad representa el corredor Asia-Europa?'\n")

    while True:
        try:
            question = input("Tu pregunta > ").strip()
            if not question:
                continue
            history.append({"role": "user", "content": question})
            resp = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=history,
            )
            answer = resp.content[0].text
            history.append({"role": "assistant", "content": answer})
            print(f"\nClaude:\n{answer}\n")
        except KeyboardInterrupt:
            print("\nSaliendo.")
            break


# ── Main ──────────────────────────────────────────────────────────────────────

def load_od(path: str, days: int | None) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("**/od_matrix_*.parquet"))
        if days:
            files = files[-days:]
        dfs = [pd.read_parquet(f) for f in files]
        if not dfs:
            raise FileNotFoundError(f"No od_matrix files in {p}")
        df = pd.concat(dfs, ignore_index=True)
        return (df.groupby(["origin_port","dest_port","origin_country","dest_country"])
                  .agg(voyage_count=("voyage_count","sum"),
                       avg_distance_nm=("avg_distance_nm","mean"),
                       avg_duration_h=("avg_duration_h","mean"),
                       avg_speed_kn=("avg_speed_kn","mean"))
                  .reset_index())
    return pd.read_parquet(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--od",          required=True)
    parser.add_argument("--days",        type=int, default=None)
    parser.add_argument("--top",         type=int, default=20,
                        help="Min voyages for a route to be included")
    parser.add_argument("--shadow",      default=None,
                        help="Path to shadow_prices CSV from koopmans_lp.py")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive Q&A mode after initial analysis")
    parser.add_argument("--out",         default=None)
    args = parser.parse_args()

    od = load_od(args.od, args.days)
    G  = build_graph(od, min_voyages=args.top)

    print(f"\nGrafo: {G.number_of_nodes()} puertos, {G.number_of_edges()} rutas")

    metrics = compute_metrics(G)

    shadow_prices = None
    if args.shadow and Path(args.shadow).exists():
        shadow_prices = pd.read_csv(args.shadow)

    print("\n=== Análisis LLM (Claude) ===\n")
    interpretation = llm_interpret(metrics, shadow_prices)
    print(interpretation)

    if args.out:
        metrics["nodes"].to_csv(f"{args.out}_nodes.csv", index=False)
        Path(f"{args.out}_interpretation.txt").write_text(interpretation)
        log.info("saved", prefix=args.out)

    if args.interactive:
        interactive_mode(G, metrics, shadow_prices)
