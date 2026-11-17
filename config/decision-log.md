# CageQuant Architecture Decision Records (ADRs)

## ADR 001: 10-Second Discrete Interval Simulation Model
* **Date:** 2026-11-01
* **Status:** Accepted
* **Context:** Continuous differential equation fight modeling lacks granular event triggers and real-time probability update hooks required for fight telemetry.
* **Decision:** Segment bout simulation into discrete 10-second intervals (30 intervals per 5-minute round).
* **Consequences:** Enables deterministic state transitions, step-by-step feature logging, and high-frequency win probability updating.

## ADR 002: Four Combat Pillars (xCT, SDR, SVI, Vitality HP)
* **Date:** 2026-11-01
* **Status:** Accepted
* **Context:** Raw strike totals fail to account for power variance, defensive mitigation, and cardio depletion.
* **Decision:** Formulate four normalized domain metrics: Expected Cumulative Damage (xCT), Stamina Drain Ratio (SDR), Strike Volatility Index (SVI), and Vitality (HP).
* **Consequences:** Standardizes feature inputs for predictive ML algorithms and tactical stance modifiers.

## ADR 003: DuckDB Embedded Analytical Engine
* **Date:** 2026-11-02
* **Status:** Accepted
* **Context:** Need high-performance SQL query execution on structured fighter telemetry without deploying external database servers.
* **Decision:** Embedded DuckDB as the in-memory semantic analytical storage layer.
* **Consequences:** Sub-millisecond SQL query execution, zero infrastructure deployment cost, and seamless integration with Pandas DataFrames.

### ADR 004: Test-Driven Verification & Test Coverage Standards
- **Date:** 2026-08-15
- **Status:** Accepted
- **Decision:** Enforce 100% test coverage across core mathematical paths (DuckDB ETL metrics, XGBoost probability calibration, and Receipt Auditor verification) using `pytest --cov=src/engine --cov=src/agent`. Mocked LLM fixtures are used in CI to ensure zero-cost deterministic execution gates.

## ADR 005: Streamlit & Dual-Axis Plotly Trajectory Visuals
* **Date:** 2026-11-03
* **Status:** Accepted
* **Context:** Stakeholders require real-time visual tracking of vitality degradation alongside rolling probability win metrics.
* **Decision:** Build interactive front-end using Streamlit and sub-second Plotly stacked subplots with step-by-step playback animation.
* **Consequences:** High-impact visual storytelling for portfolio reviews with custom playback pacing controls.

## ADR 006: XGBoost Classifier Calibration via Brier Score Loss
* **Date:** 2026-11-04
* **Status:** Accepted
* **Context:** Heuristic prior logits lacked empirical validation on interval telemetry features.
* **Decision:** Train `xgboost.XGBClassifier` on DuckDB telemetry feature matrix $X$ and calibrate output probabilities using Brier Score Loss ($BS \le 0.25$).
* **Consequences:** Provides empirically calibrated interval win probabilities ($P_{Win}$) bound to telemetry states.

## ADR 007: Stochastic Damage Volatility & Monte Carlo Risk Engine
* **Date:** 2026-11-05
* **Status:** Accepted
* **Context:** Deterministic damage calculations caused Monte Carlo $N=1,000$ iterations to collapse into identical single-winner outcomes.
* **Decision:** Inject Gaussian stochastic variance ($\mu=1.0, \sigma=0.35$) into interval strike damage calculations.
* **Consequences:** Monte Carlo runs branch into realistic, smooth probability distributions, confidence intervals, and varied method-of-victory breakdowns.

## ADR 008: Dynamic Tactical Stance Trade-Off Engine
* **Date:** 2026-11-05
* **Status:** Accepted
* **Context:** Static fighter profiles do not reflect dynamic tactical adjustments or physical power-scaling tradeoffs during fight progression.
* **Decision:** Implement `TacticalStanceEngine` modeling combat stances (Standard, High-Output Power, High-Efficiency Flow, Ascended Peak Output) with trade-offs between damage output, stamina decay, and defense.
* **Consequences:** Enables scenario modeling where aggressive power boosts incur cardio penalties unless neutralized by mastered baseline efficiency.

## ADR 009: Unsupervised PCA, K-Means Archetypes, and KNN Peer Engine
* **Date:** 2026-11-05
* **Status:** Accepted
* **Context:** Fighter profiles require automated categorization into intuitive combat archetypes and historical peer matching.
* **Decision:** Implement `FighterArchetypeEngine` using StandardScaler, PCA dimensional reduction, K-Means clustering ($K=4$), and KNN Euclidean distance peer search.
* **Consequences:** Automatically assigns fighters to tactical archetypes and identifies top-2 nearest historical match-ups for any parameter configuration.

## ADR 010: Transition to FastMCP Tool-Calling Agentic Architecture
* **Date:** 2026-11-07
* **Status:** Accepted
* **Context:** Passive text summarization fails to provide verifiable, dynamic tool interaction.
* **Decision:** Implement a modular `src/agent/` package containing FastMCP tool interfaces (`tools.py`), an orchestration copilot (`copilot.py`), and a verification step (`auditor.py`).
* **Consequences:** Enables dynamic tool invocation and multi-step reasoning capabilities.

## ADR 011: Structured Receipt Auditor for Anti-Hallucination Verification
* **Date:** 2026-11-07
* **Status:** Accepted
* **Context:** Regex-only text parsing is brittle against diverse LLM formatting outputs.
* **Decision:** Build a structured receipt validation auditor in `src/agent/auditor.py` that extracts numerical assertions and matches them against tool output states within a $\pm 1.5\%$ tolerance.
* **Consequences:** Prevents hallucinated metrics from rendering in output briefs.

## ADR 012: Deterministic CI Evals with Mocked LLM Interfaces
* **Date:** 2026-11-07
* **Status:** Accepted
* **Context:** Live LLM API calls in CI pipelines introduce latency, financial cost, and non-deterministic test failures.
* **Decision:** Decouple CI test suites using mocked LLM response generators for GitHub Actions, running live API benchmarking in a separate evaluation pipeline (`evals/`).
* **Consequences:** Ensures 100% reproducible, zero-cost CI builds.

## ADR 013: Model Calibration Evaluation via Brier Skill Score (BSS)
* **Date:** 2026-11-07
* **Status:** Accepted
* **Context:** Absolute Brier Score thresholds ($BS \le 0.25$) are easily satisfied by naive $50/50$ random guessing.
* **Decision:** Evaluate model calibration using Brier Skill Score ($BSS = 1 - BS_{\text{model}} / BS_{\text{baseline}}$), asserting $BSS > 0.0$ against empirical baseline win rates.
* **Consequences:** Ensures predictive models provide verifiable statistical lift over random guessing.

## ADR 014: Light-Touch Agent Architecture over Framework Dependency
* **Date:** 2026-11-07
* **Status:** Accepted
* **Context:** Heavy multi-agent frameworks (LangGraph, CrewAI) introduce significant abstractions and dependency overhead.
* **Decision:** Build a lightweight, explicit agent orchestration loop in Python using FastMCP and native SDK calls.
* **Consequences:** Minimizes latency, keeps dependencies lean, and retains complete control over context and prompt assembly.

## ADR 015: Real UFC Data via Public CSV Mirror (Replaces Synthetic Seed Data)
* **Date:** 2026-11-15
* **Status:** Accepted (risk knowingly accepted)
* **Context:** Seed profiles and Gaussian training frames were circular: models learned our own formula. CageQuant's terms forbid scraping, and no UFC equivalent of FBRef or StatsBomb open data exists.
* **Decision:** Ingest UFCStats-derived CSVs from `Greco1899/scrape_ufc_stats` weekly. Keep the project non-commercial and credit all sources (see LEGAL.md).
* **Alternatives Considered:** Scraping CageQuant (ToS conflict), writing our own UFCStats scraper (same terms question, more maintenance), staying synthetic (circular).
* **Consequences:** Real outcomes enable honest training. Dependency on one maintainer, so pin the commit, cache snapshots, and fail loudly on schema change. Legal gray area is documented, not ignored.
* **Non-Technical Explanation:** "We switched from practicing against our own made-up answers to real fight results."

## ADR 016: Medallion Architecture with Star Schema in DuckDB
* **Date:** 2026-11-15
* **Status:** Accepted
* **Context:** Raw CSVs are messy (strikes stored as "20 of 47", times as "1:23") and ad-hoc cleaning is not auditable.
* **Decision:** Bronze (verbatim weekly snapshots plus audit columns) -> Silver (typed, deduplicated, bout-level, shrunk rates) -> Gold (`dim_fighter`, `dim_bout`, `fact_bouts`, rating cards). Bout-first, then fighter-level views.
* **Consequences:** Every number traces back to a raw row, and each layer is unit-testable. More tables to maintain.
* **Non-Technical Explanation:** "Raw data is the delivery box, Silver is the cleaned shelf, Gold is the finished product on display."

## ADR 017: Weekly Elastic-Net Refit with Versioned Models and a Promotion Gate
* **Date:** 2026-11-15
* **Status:** Proposed [ALI to confirm thresholds]
* **Context:** New fights arrive weekly, so ratings must stay current without silently degrading.
* **Decision:** Each Sunday, refit the L1+L2 (Elastic-Net) win model on all bouts. Train only on fights before a cutoff date and validate on later fights (out-of-time, never a random split). Save each fit as a version: coefficients, training data cutoff, metrics, git commit. Promote a new version only if it passes the gate; otherwise keep the previous one.
* **Proposed gate [ALI]:** out-of-time log-loss and Brier score no worse than the current version beyond a small tolerance; coefficient signs pass sanity checks (e.g., a higher opponent-adjusted strike differential never lowers win odds).
* **Consequences:** Ratings can drift week to week and every change is explainable and reversible. XGBoost remains the separate calibrated win-probability model (ADR 006).

## ADR 018: PSY Is a Proxy Attribute
* **Date:** 2026-11-15
* **Status:** Accepted
* **Context:** Psychology cannot be measured directly from fight statistics.
* **Decision:** PSY v1 uses behavioral proxies only: bounce-back after a loss, sustained output when trailing after earlier rounds, results against strong opponents (opponent UFC win rate at fight time), loss durability (time lasted in losses), and opponent-weighted win/loss streaks. Always labeled "PSY (proxy)". We run an ablation (model with vs. without PSY) and report it. Fast-win time lives in ATT only, to avoid double counting.
* **Consequences:** Honest and falsifiable. The auditor must never present PSY as measured fact. If the model assigns PSY near-zero weight, we report that finding.

## ADR 019: Rating Eligibility & Confidence Flags
* **Date:** 2026-11-15
* **Status:** Proposed [ALI to confirm]
* **Decision:** A fighter needs at least 7 UFC bouts for a full card; fewer gets a "provisional" card using Bayesian shrinkage. `stale` flag if no fight within [ALI: e.g., 18-24 months]. A separate `cold_streak` form alert (4 straight losses) is shown as information, not as a data-quality flag.
* **Consequences:** Users can see when a rating rests on thin or old evidence.

## ADR 020: Strangler Migration - New `medallion.py` Beside Legacy `etl.py`
* **Date:** 2026-11-15
* **Status:** Accepted
* **Context:** `OctagonDataEngine` (synthetic seed data) is imported by 10+ modules and tests. Deleting it in one step would break the whole suite.
* **Decision:** Build the real-data warehouse as `UFCWarehouse` in new `src/engine/medallion.py`, fed by `src/engine/ingest.py`. Migrate consumers (`train`, `simulation`, `archetypes`, agent tools, `app.py`) one slice at a time, then delete the legacy class and fold `medallion.py` into `etl.py`.
* **Consequences:** The suite stays green at every step. Two engines coexist briefly, which is a deliberate, temporary cost.
* **Non-Technical Explanation:** "We built the new kitchen next to the old one and moved dishes over one at a time, instead of closing the restaurant."

## ADR 021: Data-Quality Rules Found on Real Data
* **Date:** 2026-11-15
* **Status:** Accepted
* **Decision:** (1) Bouts whose event has no upstream date stay in Silver but are excluded from Gold, because point-in-time features need a date (2 of 9,003 bouts today; guarded at <0.1%). (2) Fighters are name-keyed because upstream stats carry no fighter ID; duplicate names are flagged `name_collision` and their physical attributes are nulled, not guessed. (3) A missing control time (`--`) stays NULL, never 0. (4) Impossible rates (landed > attempted) are clamped to [0, 1].
* **Consequences:** Honest gaps instead of silent guesses. `latest_weight_class` (not the most common class) should define a fighter's division population.

## ADR 022: Latency Budget - Monte Carlo Capped at 200, Playback Defaults to Medium (0.2s)
* **Date:** 2026-11-16
* **Status:** Accepted
* **Context:** The live replay re-runs the simulation on every interaction. The old slider allowed up to 500 sims (default 250), and the PRD target was N=1,000. A replay that makes the user wait is a replay nobody uses.
* **Alternatives Considered:** Keep 500; cache simulations per matchup; run N=1,000 in a background job.
* **Decision:** Cap the UI at N=200 (default 100, steps of 50). Playback options are Slow 0.4s, Medium 0.2s (default), Fast 0.1s, Turbo 0.03s. All limits live in `src/engine/app_config.py` and are pinned by tests.
* **Consequences:** Win-probability estimates are noisier (standard error is about 3.5 points at N=200 for p=0.5). The UI must show N and avoid false precision. The PRD's N=1,000 remains the offline/eval target.
* **Non-Technical Explanation:** "We chose a quick, good-enough estimate that arrives while you are still watching, instead of a perfect one that makes you wait."

## ADR 023: Publish-Last Weekly Pipeline with Injected Clock
* **Date:** 2026-11-16
* **Status:** Accepted
* **Context:** The weekly refresh must be safe to re-run and must never show half-finished or worse ratings.
* **Decision:** `pipeline.run_weekly` does all heavy work first (ingest, medallion, features, train), asks the champion/challenger gate, and only then writes `cards.json` and `metrics.json` atomically (temp file plus `os.replace`). The date is injected (`--as-of`, `created_at`, `version_id=v<date>`), so the same inputs give byte-identical outputs. Pipeline tests stub only the ML fit because the fixture has 3 decided bouts; the model itself is covered by Slice 2 tests and the real-data tests.
* **Consequences:** A crash or a gate hold leaves last week's cards untouched. The fixture cannot validate model quality, which is a known limit.
* **Non-Technical Explanation:** "We bake in the back kitchen and only move bread to the shelf when it is done, so customers never see half-baked cards."
