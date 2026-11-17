import pathlib

import pytest


@pytest.fixture
def fixture_raw():
    return pathlib.Path(__file__).parent / "fixtures" / "ufcstats"


@pytest.fixture
def fx_window():
    from datetime import date
    return {"cutoff": date(2025, 6, 20), "train_start": date(2024, 1, 1)}


@pytest.fixture
def fast_training(monkeypatch):
    """Swap only the ML fit for a fixed model so pipeline tests stay deterministic."""
    from src.engine import pipeline
    from src.engine.features import FEATURES, compute_features, fit_priors
    from src.engine.ratings import FittedModel, ModelVersion

    def fake(counts, cutoff, train_start=None, l1_ratio=0.5, version_id=None, created_at=None):
        priors = fit_priors(counts, cutoff)
        names = list(FEATURES)
        model = FittedModel(names, {f: 0.1 for f in names}, {f: 1.0 for f in names}, 1.0, l1_ratio)
        version = ModelVersion(
            version_id=version_id, created_at=created_at, train_cutoff=str(cutoff),
            metrics={"log_loss": 0.69, "auc": 0.5, "train_n": 0}, model=model,
            priors=priors.to_dict("records"),
        )
        return version, priors, compute_features(counts, priors)

    monkeypatch.setattr(pipeline, "run_training", fake)
    monkeypatch.setattr("src.engine.medallion.MIN_BOUTS_FOR_CARD", 1)  # fixture is tiny
