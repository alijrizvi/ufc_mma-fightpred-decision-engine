"""Out-of-time Elastic-Net, promotion gate, model versions and FIFA-style Fighter Rating Cards.

Design decisions (see ADR 017-018, 022):
  * Training rows are bouts oriented at random (hash of bout_id) because UFCStats lists the winner first
    ~63% of the time - listing order is a label leak.
  * Model: logistic regression on standardized (left - right) feature differences, no intercept, so swapping
    the fighters flips the probability exactly.
  * Card weights come from the learned coefficients, floored at zero where they contradict the domain-expected
    direction (contradictions are reported, never hidden).
  * Ratings are 65 + 34 * percentile, scaled against active, non-provisional fighters of the same division.
"""
from __future__ import annotations

import inspect
import json
import warnings
import zlib
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.engine.features import ATTRIBUTES, FEATURES, compute_features, fit_priors

DEFAULT_TRAIN_START = date(2014, 1, 1)   # first year with consistently complete stats (features still use full history)
DEFAULT_CS = (0.01, 0.03, 0.1, 0.3, 1.0)


# ---------------------------------------------------------------- orientation + frames
def orientation_flip(bout_id: str) -> bool:
    """Deterministic pseudo-random coin per bout, independent of who won or how UFCStats lists the fighters."""
    return zlib.crc32(bout_id.encode("utf-8")) % 2 == 1


def build_training_frame(counts: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """One row per decided bout: feature differences (left - right) and label ``y`` = left fighter won.

    Orientation is a hash of ``bout_id`` (see ``orientation_flip``), never the upstream listing order.
    """
    real = counts[(~counts["is_current"].astype(bool)) & counts["result"].isin(["WIN", "LOSS"])]
    side_a = real[real["side"] == "A"].drop_duplicates("bout_id")
    side_b = real[real["side"] == "B"].drop_duplicates("bout_id")
    a_row = pd.Series(side_a.index.to_numpy(), index=side_a["bout_id"].to_numpy())
    b_row = pd.Series(side_b.index.to_numpy(), index=side_b["bout_id"].to_numpy())
    bouts = a_row.index.intersection(b_row.index)
    flip = np.array([orientation_flip(b) for b in bouts])
    left = np.where(flip, b_row.loc[bouts].to_numpy(), a_row.loc[bouts].to_numpy())
    right = np.where(flip, a_row.loc[bouts].to_numpy(), b_row.loc[bouts].to_numpy())
    out = pd.DataFrame(feats.loc[left].to_numpy() - feats.loc[right].to_numpy(), columns=feats.columns)
    out.insert(0, "bout_id", np.asarray(bouts))
    out.insert(1, "event_date", pd.to_datetime(counts.loc[left, "event_date"]).to_numpy())
    out.insert(2, "left_fighter", counts.loc[left, "fighter"].to_numpy())
    out.insert(3, "right_fighter", counts.loc[right, "fighter"].to_numpy())
    out["y"] = (counts.loc[left, "result"].to_numpy() == "WIN").astype(int)
    return out.sort_values(["event_date", "bout_id"]).reset_index(drop=True)


def split_out_of_time(frame: pd.DataFrame, cutoff: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train = on or before ``cutoff``; validation = strictly after. The model never sees the future."""
    dates = pd.to_datetime(frame["event_date"])
    return frame[dates <= pd.Timestamp(cutoff)].copy(), frame[dates > pd.Timestamp(cutoff)].copy()


# ---------------------------------------------------------------- model
@dataclass
class FittedModel:
    """Standardized logistic model: P(left wins) = sigmoid(sum_f coef_f * x_f / scale_f)."""

    feature_names: List[str]
    coef: Dict[str, float]
    scale: Dict[str, float]
    C: float
    l1_ratio: float

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        z = sum(self.coef[f] * X[f] / self.scale[f] for f in self.feature_names)
        return 1.0 / (1.0 + np.exp(-z))


def _logit(C: float, l1_ratio: float) -> LogisticRegression:
    kwargs = dict(C=C, l1_ratio=l1_ratio, solver="saga", fit_intercept=False, max_iter=5000, tol=1e-4, random_state=0)
    if inspect.signature(LogisticRegression).parameters["penalty"].default != "deprecated":
        kwargs["penalty"] = "elasticnet"  # older scikit-learn needs it spelled out
    return LogisticRegression(**kwargs)


def fit_elastic_net(
    train: pd.DataFrame, feature_cols: Sequence[str], l1_ratio: float = 0.5,
    Cs: Sequence[float] = DEFAULT_CS, holdout_frac: float = 0.2,
) -> FittedModel:
    """Fit L1+L2 logistic regression; pick ``C`` on the most recent ``holdout_frac`` of the training window."""
    ordered = train.sort_values("event_date")
    X = ordered[list(feature_cols)]
    scale = X.std(ddof=0).replace(0.0, 1.0)
    Z, y = (X / scale).to_numpy(), ordered["y"].to_numpy()
    cut = int(len(ordered) * (1 - holdout_frac))
    best_C, best_loss = Cs[0], np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for C in Cs:
            fit = _logit(C, l1_ratio).fit(Z[:cut], y[:cut])
            loss = log_loss(y[cut:], fit.predict_proba(Z[cut:])[:, 1], labels=[0, 1])
            if loss < best_loss:
                best_C, best_loss = C, loss
        final = _logit(best_C, l1_ratio).fit(Z, y)
    return FittedModel(
        feature_names=list(feature_cols),
        coef={f: float(c) for f, c in zip(feature_cols, final.coef_[0])},
        scale={f: float(scale[f]) for f in feature_cols},
        C=float(best_C), l1_ratio=float(l1_ratio),
    )


def evaluate(model: FittedModel, frame: pd.DataFrame) -> Dict[str, float]:
    """Out-of-sample metrics. Coin-flip baselines: log_loss 0.693, brier 0.25, auc 0.5, accuracy 0.5."""
    p = model.predict_proba(frame).clip(1e-6, 1 - 1e-6)
    y = frame["y"].to_numpy()
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(((p > 0.5).astype(int) == y).mean()),
        "n": int(len(frame)),
    }


# ---------------------------------------------------------------- versions + gate (ADR 017)
@dataclass
class ModelVersion:
    """A saved, auditable weekly fit."""

    version_id: str
    created_at: str
    train_cutoff: str
    metrics: Dict[str, float]
    model: FittedModel
    priors: List[dict] = field(default_factory=list)
    notes: str = ""


@dataclass
class GateDecision:
    promote: bool
    reasons: List[str]


def evaluate_gate(
    prev: Optional[ModelVersion], new: ModelVersion, log_loss_tolerance: float = 0.005,
    drift_threshold: float = 0.20, top_k: int = 5, max_drifted: int = 1,
) -> GateDecision:
    """Promote ``new`` only if it is not meaningfully worse AND the model's story is stable.

    Holds the previous version when log-loss worsens beyond tolerance, any top-k coefficient flips sign, or
    more than ``max_drifted`` top-k coefficients move by over ``drift_threshold`` (relative).
    """
    if prev is None:
        return GateDecision(True, ["first model: nothing to compare against"])
    reasons: List[str] = []
    if new.metrics["log_loss"] > prev.metrics["log_loss"] + log_loss_tolerance:
        reasons.append(f"log_loss worsened {prev.metrics['log_loss']:.4f} -> {new.metrics['log_loss']:.4f}")
    top = sorted(prev.model.coef, key=lambda f: -abs(prev.model.coef[f]))[:top_k]
    drifted = 0
    for f in top:
        before, after = prev.model.coef[f], new.model.coef.get(f, 0.0)
        if before * after < 0:
            reasons.append(f"sign flip on '{f}' ({before:+.2f} -> {after:+.2f})")
        elif before != 0 and abs(after - before) / abs(before) > drift_threshold:
            drifted += 1
    if drifted > max_drifted:
        reasons.append(f"drift: {drifted} of top-{top_k} coefficients moved more than {drift_threshold:.0%}")
    return GateDecision(not reasons, reasons or ["passed performance and stability checks"])


def save_version(registry_dir: Path | str, version: ModelVersion) -> Path:
    """Write ``<registry>/<version_id>/model.json``."""
    folder = Path(registry_dir) / version.version_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model.json"
    path.write_text(json.dumps(asdict(version), indent=2, default=str))
    return path


def load_latest_version(registry_dir: Path | str) -> Optional[ModelVersion]:
    """Load the lexicographically newest version (ids are ISO timestamps), or None if the registry is empty."""
    root = Path(registry_dir)
    if not root.exists():
        return None
    folders = sorted(p for p in root.iterdir() if (p / "model.json").exists())
    if not folders:
        return None
    raw = json.loads((folders[-1] / "model.json").read_text())
    raw["model"] = FittedModel(**raw["model"])
    return ModelVersion(**raw)


# ---------------------------------------------------------------- cards
@dataclass
class AttributeWeights:
    weights: Dict[str, float]
    contradictions: List[str]
    fallback: bool


def attribute_weights(model: FittedModel) -> Dict[str, AttributeWeights]:
    """Per-attribute feature weights from learned coefficients, sign-constrained to the domain direction."""
    out: Dict[str, AttributeWeights] = {}
    for attr in ATTRIBUTES:
        names = [f for f, s in FEATURES.items() if s.attribute == attr]
        weights, contradictions = {}, []
        for f in names:
            oriented = FEATURES[f].direction * model.coef.get(f, 0.0)
            if oriented < 0:
                contradictions.append(f)
            weights[f] = max(0.0, oriented)
        fallback = sum(weights.values()) == 0
        if fallback:
            weights = {f: 1.0 for f in names}
        out[attr] = AttributeWeights(weights, contradictions, fallback)
    return out


def percentile_of(value: float, population: np.ndarray) -> float:
    """Mid-rank percentile of ``value`` within ``population`` (ties count half)."""
    population = np.asarray(population, dtype=float)
    return float((population < value).mean() + 0.5 * (population == value).mean())


def _division_label(gender: str, weight_class: str) -> str:
    return f"Women's {weight_class}" if gender == "F" else weight_class


def build_cards(
    features: pd.DataFrame, fighters: pd.DataFrame, model: FittedModel, min_population: int = 15,
) -> pd.DataFrame:
    """Build 65-99 rating cards.

    Args:
        features: One row per fighter: ``fighter`` plus every column of ``FEATURES`` (current career).
        fighters: ``fighter, gender, latest_weight_class, status, is_provisional, ufc_bouts, last_fight_date``.
        model: Fitted model supplying the attribute weights.
        min_population: Divisions smaller than this are scaled against the same-gender active population.

    Returns:
        ``fighter, division, ATT..PSY`` (ints in [65, 99]) plus eligibility columns. Percentiles are measured
        against active (not INACTIVE), non-provisional fighters only.
    """
    df = features.merge(fighters, on="fighter", how="inner").reset_index(drop=True)
    weights = attribute_weights(model)
    for attr in ATTRIBUTES:
        df[f"_c_{attr}"] = sum(
            w * FEATURES[f].direction * df[f] / model.scale[f] for f, w in weights[attr].weights.items()
        )
    eligible = (df["status"] != "INACTIVE") & (~df["is_provisional"].astype(bool))
    ratings = {attr: np.zeros(len(df), dtype=int) for attr in ATTRIBUTES}
    for (gender, division), group in df.groupby(["gender", "latest_weight_class"]):
        pop_mask = eligible & (df["gender"] == gender) & (df["latest_weight_class"] == division)
        if pop_mask.sum() < min_population:
            pop_mask = eligible & (df["gender"] == gender)
        for attr in ATTRIBUTES:
            pop = df.loc[pop_mask, f"_c_{attr}"].to_numpy()
            for i in group.index:
                ratings[attr][i] = int(np.rint(65 + 34 * percentile_of(df.loc[i, f"_c_{attr}"], pop)))
    out = pd.DataFrame({
        "fighter": df["fighter"],
        "division": [_division_label(g, w) for g, w in zip(df["gender"], df["latest_weight_class"])],
        **{attr: ratings[attr] for attr in ATTRIBUTES},
        "is_provisional": df["is_provisional"].astype(bool),
        "status": df["status"],
        "ufc_bouts": df["ufc_bouts"],
        "last_fight_date": df["last_fight_date"],
    })
    return out


# ---------------------------------------------------------------- orchestration
def run_training(
    counts: pd.DataFrame, cutoff: date, train_start: date = DEFAULT_TRAIN_START,
    l1_ratio: float = 0.5, version_id: Optional[str] = None,
) -> Tuple[ModelVersion, pd.DataFrame, pd.DataFrame]:
    """Fit on bouts in [train_start, cutoff], validate on bouts after cutoff.

    Returns ``(version, priors, feature_matrix)``. Priors come from the training window only.
    """
    priors = fit_priors(counts, cutoff)
    feats = compute_features(counts, priors)
    frame = build_training_frame(counts, feats)
    window = frame[pd.to_datetime(frame["event_date"]) >= pd.Timestamp(train_start)]
    train, valid = split_out_of_time(window, cutoff)
    model = fit_elastic_net(train, list(FEATURES), l1_ratio=l1_ratio)
    metrics = {**evaluate(model, valid), "train_n": int(len(train))}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    version = ModelVersion(
        version_id=version_id or stamp, created_at=stamp, train_cutoff=str(cutoff),
        metrics=metrics, model=model, priors=priors.to_dict("records"),
    )
    return version, priors, feats
