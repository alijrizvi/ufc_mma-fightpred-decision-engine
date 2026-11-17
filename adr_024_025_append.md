
## ADR 024: Semantic Layer Is a Registry Over the Published Cards
* **Date:** 2026-11-17
* **Status:** Accepted
* **Context:** `tools.py` guessed metrics with `getattr` on a legacy profile, and `metrics.yml` described interval_telemetry measures nothing used.
* **Alternatives Considered:** Compute-on-query SQL measures (Cube.dev style); keep both layers.
* **Decision:** `config/metrics.yml` (v2) is a registry: metric name -> card field, description, valid range, over `artifacts/cards/cards.json` only. `src/engine/semantic.py` loads it and raises `UnknownMetricError`, `UnknownFighterError` or `MetricRangeError` (all `ValueError`s). `tools.query_semantic_metric` must use it.
* **Consequences:** A typo'd metric fails loudly with the valid names. SQL measures can come later if slicing is needed. The registry holds the 6 rating attributes plus `ufc_bouts`; model-level metrics stay in `metrics.json`.
* **Non-Technical Explanation:** "A menu with fixed item names: ask for something off the menu and you get told what is available, not a made-up dish."

## ADR 025: Structured Claims, One Tolerance, Wilson Intervals
* **Date:** 2026-11-17
* **Status:** Accepted
* **Decision:** Narrators return `Narration(text, claims)`. `AnalyticalAuditor.verify_structured` checks each `Claim` (fighter identity, win probability, method share, CI, provenance) against ground truth; the prose is still scanned for stray percentages. One constant, `AUDIT_TOLERANCE_PCT = 1.5`, is shared by auditor, copilot and UI. Simulation results carry 95% Wilson intervals (`src/engine/intervals.py`).
* **Consequences:** Swapped fighters and invented methods are caught, which regex alone could not. A bare-string narrator is still accepted (prose audit only). Empty claims are refused as unverifiable. At N=200 a 1.5 point tolerance is valid only because the auditor compares against the same simulation output.
* **Non-Technical Explanation:** "The writer hands in an itemized receipt, and the inspector checks every line against the till."
