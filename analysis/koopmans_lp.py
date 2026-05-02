"""
Koopmans Transportation Problem
================================
Empirical test of Koopmans (1949) "Optimum Utilization of the Transportation System".

The model:
    minimize    Σ_{i,j} c_ij · x_ij          (total transport cost)
    subject to  Σ_j x_ij  = S_i  ∀i          (all supply shipped)
                Σ_i x_ij  = D_j  ∀j          (all demand met)
                x_ij ≥ 0

The DUAL produces port-level shadow prices (u_i, v_j) such that:
    u_i + v_j ≤ c_ij  for all routes
    u_i + v_j = c_ij  for all routes actually used in optimum

Koopmans' insight: in a competitive market, equilibrium freight rates
should equal these shadow prices. Port congestion = routes where observed
flows deviate most from optimal.

Usage:
    python koopmans_lp.py --od data/processed/2026-05-01/od_matrix_2026-05-01.parquet
    python koopmans_lp.py --od data/processed/ --days 7  (aggregate last N days)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import structlog

log = structlog.get_logger()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_od_matrix(path: str | Path, days: int | None = None) -> pd.DataFrame:
    """
    Load O-D matrix from Parquet.
    If path is a directory and days is set, aggregate last N days.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("**/od_matrix_*.parquet"))
        if days:
            files = files[-days:]
        dfs = [pd.read_parquet(f) for f in files]
        if not dfs:
            raise FileNotFoundError(f"No od_matrix files in {p}")
        df = pd.concat(dfs, ignore_index=True)
        # Re-aggregate
        df = (
            df.groupby(["origin_port", "dest_port", "origin_country", "dest_country"])
            .agg(voyage_count=("voyage_count", "sum"),
                 avg_distance_nm=("avg_distance_nm", "mean"),
                 avg_duration_h=("avg_duration_h", "mean"),
                 avg_speed_kn=("avg_speed_kn", "mean"))
            .reset_index()
        )
    else:
        df = pd.read_parquet(p)

    log.info("od_loaded", routes=len(df), total_voyages=int(df["voyage_count"].sum()))
    return df


# ── Cost matrix ───────────────────────────────────────────────────────────────

def build_cost_matrix(df: pd.DataFrame, cost_per_nm: float = 1.0) -> tuple:
    """
    Build the cost matrix c_ij for the LP.

    Cost proxy: great-circle distance × cost_per_nm.
    In practice, cost_per_nm encodes fuel, canal fees, port charges, etc.
    Here we normalise to 1.0 so shadow prices are in "distance units".

    Returns:
        origins  — list of port names
        dests    — list of port names
        supply   — array S_i (total departures from each origin)
        demand   — array D_j (total arrivals at each destination)
        cost_mat — 2D array c_ij (origins × destinations)
        flows    — 2D array x_ij_observed (from data, for comparison)
    """
    origins = sorted(df["origin_port"].unique().tolist())
    dests   = sorted(df["dest_port"].unique().tolist())
    n_orig  = len(origins)
    n_dest  = len(dests)

    orig_idx = {p: i for i, p in enumerate(origins)}
    dest_idx = {p: j for j, p in enumerate(dests)}

    # Supply = number of voyages departing each origin
    supply = np.zeros(n_orig)
    for _, row in df.iterrows():
        supply[orig_idx[row["origin_port"]]] += row["voyage_count"]

    # Demand = number of voyages arriving at each destination
    demand = np.zeros(n_dest)
    for _, row in df.iterrows():
        demand[dest_idx[row["dest_port"]]] += row["voyage_count"]

    # Cost matrix (inf for routes not observed — could occur if O-D set partial)
    cost_mat  = np.full((n_orig, n_dest), np.inf)
    flows_obs = np.zeros((n_orig, n_dest))

    for _, row in df.iterrows():
        i = orig_idx[row["origin_port"]]
        j = dest_idx[row["dest_port"]]
        cost_mat[i, j]  = row["avg_distance_nm"] * cost_per_nm
        flows_obs[i, j] = row["voyage_count"]

    # Replace inf with a large penalty (10× max observed cost) so LP is feasible
    max_cost = cost_mat[cost_mat != np.inf].max() if (cost_mat != np.inf).any() else 1.0
    cost_mat[cost_mat == np.inf] = max_cost * 10

    return origins, dests, supply, demand, cost_mat, flows_obs


# ── LP solver ─────────────────────────────────────────────────────────────────

def solve_transportation_lp(
    supply: np.ndarray,
    demand: np.ndarray,
    cost_mat: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Solve the balanced transportation problem via scipy.optimize.linprog.

    The standard form linearises x_ij as a flat vector of length n*m,
    with supply and demand equality constraints.

    Returns:
        x_opt      — optimal flow matrix (n_orig × n_dest)
        total_cost — optimal total cost
        u          — dual prices at origins (shadow prices, length n_orig)
        v          — dual prices at destinations (shadow prices, length n_dest)
    """
    n, m = cost_mat.shape

    # Balance supply and demand (required for equality constraints)
    total_supply = supply.sum()
    total_demand = demand.sum()
    if not np.isclose(total_supply, total_demand, rtol=1e-3):
        # Add a dummy origin or destination to balance
        log.warning("unbalanced_od",
                    supply=round(total_supply, 1), demand=round(total_demand, 1))
        if total_supply > total_demand:
            # Add dummy destination absorbing excess
            excess = total_supply - total_demand
            demand = np.append(demand, excess)
            cost_mat = np.hstack([cost_mat, np.zeros((n, 1))])
            m += 1
        else:
            excess = total_demand - total_supply
            supply = np.append(supply, excess)
            cost_mat = np.vstack([cost_mat, np.zeros((1, m))])
            n += 1

    # Flatten cost vector
    c_flat = cost_mat.flatten()

    # Equality constraints: supply (n rows) + demand (m rows)
    A_eq = np.zeros((n + m, n * m))
    for i in range(n):
        A_eq[i, i*m:(i+1)*m] = 1.0           # row sum = supply[i]
    for j in range(m):
        A_eq[n + j, j::m] = 1.0              # col sum = demand[j]

    b_eq = np.concatenate([supply, demand])

    bounds = [(0, None)] * (n * m)

    result = linprog(c_flat, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.status != 0:
        raise RuntimeError(f"LP failed: {result.message}")

    x_opt      = result.x.reshape(n, m)
    total_cost = result.fun

    # Dual variables — marginal costs at each node
    # scipy puts dual in result.ineqlin / eqlin
    duals = result.eqlin.marginals if hasattr(result, "eqlin") else np.zeros(n + m)
    u = duals[:n]    # shadow prices at origins
    v = duals[n:]    # shadow prices at destinations

    return x_opt, total_cost, u, v


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(df: pd.DataFrame) -> dict:
    """
    Run the full Koopmans analysis:
    1. Build cost matrix from observed O-D data
    2. Solve LP to find optimal flows
    3. Compare observed vs. optimal flows
    4. Report shadow prices (= theoretical equilibrium freight rates)
    5. Identify most congested / inefficient routes
    """
    origins, dests, supply, demand, cost_mat, flows_obs = build_cost_matrix(df)

    log.info("solving_lp", origins=len(origins), destinations=len(dests),
             variables=len(origins)*len(dests))

    x_opt, total_cost_opt, u, v = solve_transportation_lp(supply, demand, cost_mat)

    # Observed total cost
    total_cost_obs = (cost_mat[:len(origins), :len(dests)] * flows_obs).sum()

    # Efficiency ratio: how close is observed cost to optimal?
    efficiency = total_cost_opt / total_cost_obs if total_cost_obs > 0 else None

    # Route-level deviation: |observed - optimal| / max(optimal, 1)
    n_o, n_d = len(origins), len(dests)
    deviation = np.abs(flows_obs - x_opt[:n_o, :n_d]) / \
                np.maximum(x_opt[:n_o, :n_d], 1)

    # Top inefficient routes (highest deviation)
    route_deviations = []
    for i, orig in enumerate(origins):
        for j, dest in enumerate(dests):
            if flows_obs[i, j] > 0 or x_opt[i, j] > 0.5:
                route_deviations.append({
                    "origin":    orig,
                    "dest":      dest,
                    "observed":  round(float(flows_obs[i, j]), 1),
                    "optimal":   round(float(x_opt[i, j]), 1),
                    "cost_nm":   round(float(cost_mat[i, j]), 1),
                    "deviation": round(float(deviation[i, j]), 3),
                })
    routes_df = pd.DataFrame(route_deviations).sort_values("deviation", ascending=False)

    # Shadow prices per port
    shadow_origins = pd.DataFrame({
        "port":         origins,
        "shadow_price": np.round(u[:n_o], 4),
        "role":         "origin",
    })
    shadow_dests = pd.DataFrame({
        "port":         dests,
        "shadow_price": np.round(v[:n_d], 4),
        "role":         "destination",
    })
    shadow_prices = pd.concat([shadow_origins, shadow_dests], ignore_index=True)
    shadow_prices = shadow_prices.sort_values("shadow_price", ascending=False)

    return {
        "total_cost_optimal":  round(total_cost_opt, 2),
        "total_cost_observed": round(total_cost_obs, 2),
        "efficiency_ratio":    round(efficiency, 4) if efficiency else None,
        "routes":              routes_df,
        "shadow_prices":       shadow_prices,
    }


def print_report(results: dict):
    print("\n" + "="*60)
    print("  KOOPMANS TRANSPORTATION PROBLEM — RESULTS")
    print("="*60)
    print(f"\n  Optimal total cost:   {results['total_cost_optimal']:>12,.1f} nm·voyage")
    print(f"  Observed total cost:  {results['total_cost_observed']:>12,.1f} nm·voyage")
    if results["efficiency_ratio"]:
        eff_pct = results["efficiency_ratio"] * 100
        slack   = (1 - results["efficiency_ratio"]) * 100
        print(f"  Efficiency ratio:     {eff_pct:>11.1f}%")
        print(f"  Potential savings:    {slack:>11.1f}%  ← room for optimisation")

    print("\n  TOP 10 SHADOW PRICES (equilibrium freight rates by port)")
    print("  " + "-"*50)
    top = results["shadow_prices"].head(10)
    for _, row in top.iterrows():
        print(f"  {row['port'][:35]:<35} {row['shadow_price']:>8.2f}  [{row['role']}]")

    print("\n  TOP 10 MOST INEFFICIENT ROUTES")
    print("  " + "-"*55)
    top_routes = results["routes"].head(10)
    for _, row in top_routes.iterrows():
        print(f"  {row['origin'][:20]:<20} → {row['dest'][:20]:<20}"
              f"  obs={row['observed']:>6.0f}  opt={row['optimal']:>6.1f}"
              f"  dev={row['deviation']:.2f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Koopmans LP analysis on collected AIS data"
    )
    parser.add_argument("--od",   required=True, help="Path to od_matrix Parquet or directory")
    parser.add_argument("--days", type=int,       help="Aggregate last N days (if --od is dir)")
    parser.add_argument("--out",  default=None,   help="Save results to CSV prefix")
    args = parser.parse_args()

    df      = load_od_matrix(args.od, args.days)
    results = analyse(df)
    print_report(results)

    if args.out:
        results["routes"].to_csv(f"{args.out}_routes.csv", index=False)
        results["shadow_prices"].to_csv(f"{args.out}_shadow_prices.csv", index=False)
        log.info("saved", prefix=args.out)
