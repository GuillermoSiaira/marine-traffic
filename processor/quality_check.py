"""
AIS Data Quality Validator
===========================
AIS data tiene problemas conocidos y documentados.
Este módulo los detecta, los cuantifica y produce un reporte de calidad
que se publica junto con cada dataset en IPFS.

Problemas que detectamos:
  1. Spoofing       — posiciones falsas (evasión de sanciones, pesca ilegal)
  2. Dark periods   — barcos que "desaparecen" (AIS apagado deliberadamente)
  3. Cobertura      — aisstream.io es terrestre: gaps en alta mar
  4. MMSI duplicado — varios barcos con mismo identificador
  5. Posiciones imposibles — velocidades o coordenadas inválidas
  6. Saltos fantasma — teleportaciones entre posiciones consecutivas
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

# Umbrales físicos
MAX_SPEED_KN        = 35.0    # ningún buque mercante supera esto
MAX_JUMP_NM         = 200.0   # salto imposible en <10 min entre posiciones
DARK_PERIOD_HOURS   = 6.0     # gap de >6h = posible AIS apagado
COASTAL_RADIUS_DEG  = 1.0     # ~60 NM — cobertura terrestre aproximada


@dataclass
class QualityReport:
    date:                   str
    total_records:          int
    unique_vessels:         int

    # Cobertura
    coverage_note:          str
    pct_records_coastal:    float   # % de registros dentro de cobertura terrestre

    # Problemas detectados
    impossible_speeds:      int
    pct_impossible_speeds:  float
    ghost_jumps:            int
    pct_ghost_jumps:        float
    duplicate_mmsi:         int
    dark_periods_detected:  int
    avg_dark_period_hours:  float

    # Posiciones sospechosas de spoofing
    spoofing_candidates:    int     # MMSI con posiciones en 2 océanos simultáneamente
    pct_spoofing:           float

    # Resumen
    overall_quality_score:  float   # 0-1, estimación conservadora
    warnings:               list[str]
    recommendation:         str


def haversine_nm_vectorised(lat1, lon1, lat2, lon2) -> np.ndarray:
    R   = 3440.065
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def check_quality(df: pd.DataFrame, date_str: str) -> QualityReport:
    warnings    = []
    total       = len(df)
    n_vessels   = df["mmsi"].nunique()

    if total == 0:
        return QualityReport(date=date_str, total_records=0, unique_vessels=0,
                             coverage_note="No data", pct_records_coastal=0,
                             impossible_speeds=0, pct_impossible_speeds=0,
                             ghost_jumps=0, pct_ghost_jumps=0,
                             duplicate_mmsi=0, dark_periods_detected=0,
                             avg_dark_period_hours=0, spoofing_candidates=0,
                             pct_spoofing=0, overall_quality_score=0,
                             warnings=["No data for this date"],
                             recommendation="No data to evaluate")

    # ── 1. Velocidades imposibles ─────────────────────────────────────────────
    impossible_mask = df["sog"] > MAX_SPEED_KN
    n_impossible    = int(impossible_mask.sum())
    pct_impossible  = n_impossible / total * 100
    if pct_impossible > 0.5:
        warnings.append(f"{pct_impossible:.1f}% of records have impossible speed (>{MAX_SPEED_KN}kn)")

    # ── 2. Posiciones (0,0) y fuera de rango ──────────────────────────────────
    null_pos = ((df["lat"] == 0) & (df["lon"] == 0)).sum()
    if null_pos > 0:
        warnings.append(f"{null_pos} records at position (0,0) — likely null/default values")

    # ── 3. Saltos fantasma (teleportaciones) ──────────────────────────────────
    df_sorted  = df.sort_values(["mmsi", "timestamp"])
    df_sorted  = df_sorted[~impossible_mask].copy()
    df_shifted = df_sorted.groupby("mmsi")[["lat","lon","timestamp"]].shift(1)

    valid_pairs = df_shifted["lat"].notna()
    time_diff_h = (
        df_sorted["timestamp"] - df_shifted["timestamp"]
    ).dt.total_seconds() / 3600

    # Only check pairs within 10 minutes
    short_interval = (time_diff_h > 0) & (time_diff_h < 10/60)
    if short_interval.any():
        dist = haversine_nm_vectorised(
            df_sorted.loc[short_interval, "lat"].values,
            df_sorted.loc[short_interval, "lon"].values,
            df_shifted.loc[short_interval, "lat"].values,
            df_shifted.loc[short_interval, "lon"].values,
        )
        ghost_mask  = dist > MAX_JUMP_NM
        n_ghosts    = int(ghost_mask.sum())
        pct_ghosts  = n_ghosts / total * 100
    else:
        n_ghosts, pct_ghosts = 0, 0.0

    if pct_ghosts > 0.1:
        warnings.append(f"{n_ghosts} ghost jumps detected (>{MAX_JUMP_NM}NM in <10min)")

    # ── 4. MMSI duplicados ────────────────────────────────────────────────────
    # Vessels sharing MMSI but with different ship names
    mmsi_names   = df.groupby("mmsi")["ship_name"].nunique()
    dup_mmsi     = int((mmsi_names > 1).sum())
    if dup_mmsi > 0:
        warnings.append(f"{dup_mmsi} MMSI numbers used by >1 vessel name (misconfiguration or fraud)")

    # ── 5. Dark periods ───────────────────────────────────────────────────────
    gaps     = []
    for mmsi, grp in df_sorted.groupby("mmsi"):
        ts   = grp["timestamp"].sort_values()
        diff = ts.diff().dt.total_seconds() / 3600
        dark = diff[diff > DARK_PERIOD_HOURS]
        gaps.extend(dark.tolist())

    n_dark      = len(gaps)
    avg_dark    = float(np.mean(gaps)) if gaps else 0.0
    if n_dark > 0:
        warnings.append(f"{n_dark} dark periods (>{DARK_PERIOD_HOURS}h gap) — vessel may have disabled AIS")

    # ── 6. Spoofing candidates ────────────────────────────────────────────────
    # Vessel appearing in two distant locations within 1 hour = impossible
    spoofing = 0
    vessel_sample = df["mmsi"].value_counts().head(500).index  # check top 500 vessels
    for mmsi in vessel_sample:
        grp = df[df["mmsi"] == mmsi].sort_values("timestamp")
        if len(grp) < 2:
            continue
        ts_h = (grp["timestamp"] - grp["timestamp"].min()).dt.total_seconds() / 3600
        # Check any pair within 1 hour
        for i in range(len(grp) - 1):
            if ts_h.iloc[i+1] - ts_h.iloc[i] < 1.0:
                d = haversine_nm_vectorised(
                    np.array([grp["lat"].iloc[i]]),
                    np.array([grp["lon"].iloc[i]]),
                    np.array([grp["lat"].iloc[i+1]]),
                    np.array([grp["lon"].iloc[i+1]]),
                )
                if d[0] > 200:  # >200 NM in <1h = physically impossible
                    spoofing += 1
                    break

    pct_spoofing = spoofing / min(n_vessels, 500) * 100

    # ── 7. Cobertura ──────────────────────────────────────────────────────────
    # aisstream.io = terrestrial AIS. Coverage ~40-60 NM from coast.
    # We approximate: records near known AIS receiver areas (Europe, N.America, E.Asia)
    # Simple proxy: lat/lon within busy coastal bands
    coastal_mask = (
        # Europe + Mediterranean
        ((df["lat"].between(35, 72)) & (df["lon"].between(-10, 40))) |
        # East Asia
        ((df["lat"].between(20, 45)) & (df["lon"].between(110, 145))) |
        # Eastern North America
        ((df["lat"].between(25, 50)) & (df["lon"].between(-80, -60))) |
        # West coast Americas
        ((df["lat"].between(15, 50)) & (df["lon"].between(-130, -75))) |
        # Middle East / Persian Gulf
        ((df["lat"].between(20, 35)) & (df["lon"].between(45, 65)))
    )
    pct_coastal = float(coastal_mask.mean() * 100)

    coverage_note = (
        "aisstream.io uses terrestrial AIS receivers (~40-60 NM coastal coverage). "
        "Mid-ocean positions are NOT captured. This dataset represents coastal and "
        "port activity only. High-seas routing data has systematic gaps."
    )

    # ── Quality score ─────────────────────────────────────────────────────────
    score = 1.0
    score -= min(pct_impossible / 100, 0.2)     # up to -20 for bad speeds
    score -= min(pct_ghosts    / 100, 0.15)     # up to -15 for ghost jumps
    score -= min(pct_spoofing  / 100, 0.25)     # up to -25 for spoofing
    score -= min(dup_mmsi / max(n_vessels, 1), 0.1)
    score -= (1 - pct_coastal / 100) * 0.3     # penalise low coastal coverage
    score  = max(0.0, round(score, 3))

    if score > 0.8:
        recommendation = "Good quality for port-call and coastal routing analysis."
    elif score > 0.6:
        recommendation = "Moderate quality. Use with caution for quantitative claims. " \
                         "Validate key findings against secondary sources."
    else:
        recommendation = "Low quality. High spoofing or coverage gaps detected. " \
                         "Results should be treated as indicative only."

    return QualityReport(
        date=date_str, total_records=total, unique_vessels=n_vessels,
        coverage_note=coverage_note, pct_records_coastal=round(pct_coastal, 1),
        impossible_speeds=n_impossible, pct_impossible_speeds=round(pct_impossible, 2),
        ghost_jumps=n_ghosts, pct_ghost_jumps=round(pct_ghosts, 2),
        duplicate_mmsi=dup_mmsi, dark_periods_detected=n_dark,
        avg_dark_period_hours=round(avg_dark, 1),
        spoofing_candidates=spoofing, pct_spoofing=round(pct_spoofing, 2),
        overall_quality_score=score, warnings=warnings,
        recommendation=recommendation,
    )


def save_quality_report(report: QualityReport, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"quality_report_{report.date}.json"
    path.write_text(json.dumps(asdict(report), indent=2))
    log.info("quality_report_saved", path=str(path), score=report.overall_quality_score)

    print(f"\n{'='*55}")
    print(f"  DATA QUALITY REPORT — {report.date}")
    print(f"{'='*55}")
    print(f"  Records:        {report.total_records:>10,}")
    print(f"  Vessels:        {report.unique_vessels:>10,}")
    print(f"  Coastal coverage:{report.pct_records_coastal:>9.1f}%")
    print(f"  Bad speeds:     {report.pct_impossible_speeds:>9.2f}%")
    print(f"  Ghost jumps:    {report.pct_ghost_jumps:>9.2f}%")
    print(f"  MMSI duplicates:{report.duplicate_mmsi:>10,}")
    print(f"  Dark periods:   {report.dark_periods_detected:>10,}  (avg {report.avg_dark_period_hours:.1f}h)")
    print(f"  Spoofing candidates: {report.spoofing_candidates:>6,}  ({report.pct_spoofing:.1f}%)")
    print(f"\n  QUALITY SCORE: {report.overall_quality_score:.2f}/1.00")
    print(f"  {report.recommendation}")
    if report.warnings:
        print(f"\n  Warnings:")
        for w in report.warnings:
            print(f"    ⚠ {w}")
    print(f"\n  NOTE: {report.coverage_note[:80]}...")
    print()
    return path
