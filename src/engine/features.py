"""Point-in-time (PIT) features with Bayesian shrinkage for the six Fighter Rating Card attributes.

Pipeline: Gold fact -> ``build_pit_counts`` (cumulative history *strictly before* each bout, plus one
``CURRENT`` row per fighter holding their full career) -> ``fit_priors`` (division priors from a training
window only) -> ``compute_features`` (shrunk rates, 18 features).

Shrinkage (Gamma-Poisson / Beta-Binomial posterior mean, one formula for every rate):
    shrunk = (numerator + k * prior) / (exposure + k)  ==  (M * raw + k * mu) / (M + k)
where exposure M is minutes, strikes attempted, takedowns attempted, wins, ... and k is a pseudo-exposure.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.engine.medallion import UFCWarehouse

ATTRIBUTES = ["ATT", "DEF", "WRE", "SUB", "STA", "PSY"]


def shrink_rate(numerator, exposure, prior, pseudo_exposure):
    """Posterior-mean rate: pulls thin-sample rates toward ``prior`` in proportion to missing exposure.

    Args:
        numerator: Observed event count (scalar, array or Series).
        exposure: Observed exposure in the same units as ``pseudo_exposure`` (minutes, attempts, wins ...).
        prior: Division-level prior rate per unit of exposure.
        pseudo_exposure: ``k``, how much exposure the prior is "worth".

    Returns:
        Shrunk rate per unit of exposure. Zero exposure returns the prior exactly.
    """
    return (numerator + pseudo_exposure * prior) / (exposure + pseudo_exposure)


@dataclass(frozen=True)
class Component:
    """A shrinkable rate: ``numer`` events per ``exposure / div`` units, with pseudo-exposure ``pseudo``."""

    numer: str
    exposure: str
    pseudo: float
    scale: float = 1.0
    div: float = 1.0


COMPONENTS: Dict[str, Component] = {
    "slpm": Component("sig_landed", "stat_sec", 30, div=60.0),
    "sig_acc": Component("sig_landed", "sig_att", 150),
    "kd_rate": Component("kd", "stat_sec", 30, scale=15.0, div=60.0),
    "win_sec_avg": Component("win_sec", "wins", 3),
    "absorb_rate": Component("sig_absorbed", "sig_att_faced", 150),
    "sapm": Component("sig_absorbed", "stat_sec", 30, div=60.0),
    "td_conceded_rate": Component("td_conceded", "td_faced", 8),
    "kd_suffered_rate": Component("kd_suffered", "stat_sec", 30, scale=15.0, div=60.0),
    "td_rate": Component("td_landed", "stat_sec", 30, scale=15.0, div=60.0),
    "td_acc": Component("td_landed", "td_att", 8),
    "ctrl_share": Component("ctrl_sec", "stat_sec", 600),
    "sub_rate": Component("sub_att", "stat_sec", 30, scale=15.0, div=60.0),
    "sub_win_rate": Component("sub_wins", "bouts", 5),
    "r1_slpm": Component("r1_landed", "r1_sec", 10, div=60.0),
    "late_slpm": Component("late_landed", "late_sec", 10, div=60.0),
    "late_acc": Component("late_landed", "late_att", 60),
    "bounce": Component("after_loss_win", "after_loss", 3),
    "loss_dur": Component("loss_dur_sum", "loss_dur_n", 2),
    "opp_wr": Component("opp_wr_sum", "opp_n", 5),
}


@dataclass(frozen=True)
class FeatureSpec:
    """One model feature. ``direction`` is the domain-expected sign (+1 = higher is better for the attribute)."""

    attribute: str
    direction: int
    comp: str
    transform: str = "id"          # id | one_minus | ratio
    comp2: Optional[str] = None    # denominator component for ``ratio``


FEATURES: Dict[str, FeatureSpec] = {
    "slpm": FeatureSpec("ATT", 1, "slpm"),
    "sig_acc": FeatureSpec("ATT", 1, "sig_acc"),
    "kd_per15": FeatureSpec("ATT", 1, "kd_rate"),
    "avg_win_sec": FeatureSpec("ATT", -1, "win_sec_avg"),
    "sig_def": FeatureSpec("DEF", 1, "absorb_rate", "one_minus"),
    "sapm": FeatureSpec("DEF", -1, "sapm"),
    "td_def": FeatureSpec("DEF", 1, "td_conceded_rate", "one_minus"),
    "kd_suffered_per15": FeatureSpec("DEF", -1, "kd_suffered_rate"),
    "td_per15": FeatureSpec("WRE", 1, "td_rate"),
    "td_acc": FeatureSpec("WRE", 1, "td_acc"),
    "ctrl_share": FeatureSpec("WRE", 1, "ctrl_share"),
    "sub_att_per15": FeatureSpec("SUB", 1, "sub_rate"),
    "sub_win_rate": FeatureSpec("SUB", 1, "sub_win_rate"),
    "late_pace_retention": FeatureSpec("STA", 1, "late_slpm", "ratio", "r1_slpm"),
    "late_sig_acc": FeatureSpec("STA", 1, "late_acc"),
    "bounce_back_rate": FeatureSpec("PSY", 1, "bounce"),
    "loss_durability": FeatureSpec("PSY", 1, "loss_dur"),
    "opp_strength": FeatureSpec("PSY", 1, "opp_wr"),
}

# ---------------------------------------------------------------- PIT counts (DuckDB SQL)
_SIMPLE: Dict[str, str] = {
    "bouts": "1",
    "stat_sec": "CASE WHEN f.has_round_stats THEN f.total_fight_sec ELSE 0 END",
    "sig_landed": "coalesce(f.sig_landed, 0)",
    "sig_att": "coalesce(f.sig_att, 0)",
    "sig_absorbed": "coalesce(f.sig_absorbed, 0)",
    "sig_att_faced": "coalesce(f.sig_att_faced, 0)",
    "kd": "coalesce(f.kd, 0)",
    "kd_suffered": "coalesce(f.kd_suffered, 0)",
    "td_landed": "coalesce(f.td_landed, 0)",
    "td_att": "coalesce(f.td_att, 0)",
    "td_conceded": "coalesce(f.td_conceded, 0)",
    "td_faced": "coalesce(f.td_faced, 0)",
    "ctrl_sec": "coalesce(f.ctrl_sec, 0)",
    "sub_att": "coalesce(f.sub_att, 0)",
    "r1_landed": "coalesce(r.r1_landed, 0)",
    "r1_sec": "coalesce(r.r1_sec, 0)",
    "late_landed": "coalesce(r.late_landed, 0)",
    "late_att": "coalesce(r.late_att, 0)",
    "late_sec": "coalesce(r.late_sec, 0)",
    "wins": "CASE WHEN f.result = 'WIN' THEN 1 ELSE 0 END",
    "losses": "CASE WHEN f.result = 'LOSS' THEN 1 ELSE 0 END",
    "sub_wins": "CASE WHEN f.result = 'WIN' AND f.method_group = 'SUB' THEN 1 ELSE 0 END",
    "win_sec": "CASE WHEN f.result = 'WIN' THEN f.total_fight_sec ELSE 0 END",
    "loss_dur_n": "CASE WHEN f.result = 'LOSS' AND d.scheduled_rounds > 0 THEN 1 ELSE 0 END",
    "loss_dur_sum": "CASE WHEN f.result = 'LOSS' AND d.scheduled_rounds > 0 "
                    "THEN f.total_fight_sec / (d.scheduled_rounds * 300.0) ELSE 0 END",
}
METRICS = list(_SIMPLE) + ["after_loss", "after_loss_win", "opp_n", "opp_wr_sum"]


def _pit_sql() -> str:
    simple = ",\n         ".join(f"{expr} AS {name}" for name, expr in _SIMPLE.items())
    metric_cols = ",\n       ".join(f"CAST(coalesce({m}, 0) AS DOUBLE) AS {m}" for m in METRICS)
    prior_cols = ",\n       ".join(
        f"CAST(coalesce(sum({m}) OVER w, 0) AS DOUBLE) AS prior_{m}" for m in METRICS
    )
    return f"""
CREATE OR REPLACE TABLE gold.fact_pit_counts AS
WITH rnd AS (
    SELECT x.bout_id, x.fighter,
           sum(CASE WHEN x.round_no = 1 THEN x.sig_landed END) AS r1_landed,
           sum(CASE WHEN x.round_no = 1 THEN x.rs END) AS r1_sec,
           sum(CASE WHEN x.round_no >= 3 THEN x.sig_landed END) AS late_landed,
           sum(CASE WHEN x.round_no >= 3 THEN x.sig_att END) AS late_att,
           sum(CASE WHEN x.round_no >= 3 THEN x.rs END) AS late_sec
    FROM (SELECT s.bout_id, s.fighter, s.round_no, s.sig_landed, s.sig_att,
                 CASE WHEN s.round_no < b.end_round THEN 300
                      WHEN s.round_no = b.end_round THEN b.end_time_sec END AS rs
          FROM stg_round_stats s JOIN stg_bouts b ON b.bout_id = s.bout_id) x
    GROUP BY 1, 2
), real AS (
    SELECT f.bout_id, f.event_date, f.fighter, f.opponent, f.side, f.result, f.weight_class, f.gender,
           {simple},
           CASE WHEN lag(f.result) OVER w0 = 'LOSS' THEN 1 ELSE 0 END AS after_loss,
           CASE WHEN lag(f.result) OVER w0 = 'LOSS' AND f.result = 'WIN' THEN 1 ELSE 0 END AS after_loss_win,
           FALSE AS is_current
    FROM gold.fact_fighter_bouts f
    JOIN gold.dim_bout d ON d.bout_id = f.bout_id
    LEFT JOIN rnd r ON r.bout_id = f.bout_id AND r.fighter = f.fighter
    WINDOW w0 AS (PARTITION BY f.fighter ORDER BY f.event_date, f.bout_id)
), cur AS (
    SELECT 'CURRENT' AS bout_id, (CAST(? AS DATE) + INTERVAL 1 DAY)::DATE AS event_date, fighter,
           latest_weight_class AS weight_class, gender, TRUE AS is_current
    FROM gold.dim_fighter
), b1 AS (
    SELECT * FROM real UNION ALL BY NAME SELECT * FROM cur
), b2 AS (
    SELECT *, coalesce(sum(wins) OVER pw, 0) AS pw_wins, coalesce(sum(bouts) OVER pw, 0) AS pw_bouts
    FROM b1
    WINDOW pw AS (PARTITION BY fighter ORDER BY event_date, bout_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
), b3 AS (
    SELECT b.*, CASE WHEN o.fighter IS NOT NULL THEN 1 ELSE 0 END AS opp_n,
           CASE WHEN o.fighter IS NOT NULL THEN (o.pw_wins + 1.0) / (o.pw_bouts + 2.0) ELSE 0 END AS opp_wr_sum
    FROM b2 b LEFT JOIN b2 o ON o.bout_id = b.bout_id AND o.fighter = b.opponent
)
SELECT bout_id, event_date, fighter, opponent, side, result, weight_class, gender, is_current,
       {metric_cols},
       {prior_cols}
FROM b3
WINDOW w AS (PARTITION BY fighter ORDER BY event_date, bout_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);
"""


def build_pit_counts(wh: UFCWarehouse, as_of: Optional[date] = None) -> pd.DataFrame:
    """Build ``gold.fact_pit_counts``: per fighter-bout metrics plus ``prior_*`` cumulative counts.

    ``prior_*`` columns only include bouts strictly *before* the row (point-in-time). One extra
    ``CURRENT`` row per fighter (``is_current``) carries the full career for forward-looking ratings.
    """
    as_of = as_of or date.today()
    wh.conn.execute(_pit_sql(), [as_of])
    return wh.query("SELECT * FROM gold.fact_pit_counts ORDER BY fighter, event_date, bout_id")


# ---------------------------------------------------------------- priors + features
def _pooled_rates(frame: pd.DataFrame) -> Dict[str, float]:
    sums = frame[sorted({c.numer for c in COMPONENTS.values()} | {c.exposure for c in COMPONENTS.values()})].sum()
    out: Dict[str, float] = {}
    for name, comp in COMPONENTS.items():
        exposure = sums[comp.exposure] / comp.div
        out[name] = float(sums[comp.numer] / exposure) if exposure > 0 else np.nan
    return out


def fit_priors(counts: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Pool per-division prior rates from bouts on or before ``cutoff`` only (no future information).

    Returns one row per (gender, weight_class) plus a global fallback row with gender = weight_class = '*'.
    """
    dates = pd.to_datetime(counts["event_date"])
    window = counts[(~counts["is_current"].astype(bool)) & (dates <= pd.Timestamp(cutoff))]
    rows = [
        {"gender": g, "weight_class": w, **_pooled_rates(frame)}
        for (g, w), frame in window.groupby(["gender", "weight_class"])
    ]
    rows.append({"gender": "*", "weight_class": "*", **_pooled_rates(window)})
    return pd.DataFrame(rows)


def compute_features(counts: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """Shrunk feature matrix aligned to ``counts`` (18 columns, one per entry of ``FEATURES``)."""
    glob = priors[(priors["gender"] == "*") & (priors["weight_class"] == "*")].iloc[0]
    keyed = counts[["gender", "weight_class"]].merge(priors, on=["gender", "weight_class"], how="left")
    keyed.index = counts.index
    values: Dict[str, pd.Series] = {}
    for name, comp in COMPONENTS.items():
        mu = keyed[name].fillna(glob[name]).fillna(0.0)
        exposure = counts[f"prior_{comp.exposure}"] / comp.div
        values[name] = shrink_rate(counts[f"prior_{comp.numer}"], exposure, mu, comp.pseudo)
    out = pd.DataFrame(index=counts.index)
    for fname, spec in FEATURES.items():
        base = values[spec.comp] * COMPONENTS[spec.comp].scale
        if spec.transform == "one_minus":
            out[fname] = 1.0 - base
        elif spec.transform == "ratio":
            out[fname] = values[spec.comp] / np.maximum(values[spec.comp2], 1e-9)
        else:
            out[fname] = base
    return out
