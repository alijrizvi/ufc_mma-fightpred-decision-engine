# 🥊 Production-Grade UFC Decision Intelligence Platform

[![CI](https://github.com/alijrizvi/ufc_mma-fightpred-decision-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/alijrizvi/ufc_mma-fightpred-decision-engine/actions/workflows/ci.yml)
[![Weekly refresh](https://github.com/alijrizvi/ufc_mma-fightpred-decision-engine/actions/workflows/weekly.yml/badge.svg)](https://github.com/alijrizvi/ufc_mma-fightpred-decision-engine/actions/workflows/weekly.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)
![MCP](https://img.shields.io/badge/agent-FastMCP-green)

**[▶ Live demo](https://ufc-mma-cagequant-fightpred-decision-engine.streamlit.app/)** · FIFA-style fighter rating cards, a Monte Carlo fight simulator, and an AI copilot whose every number is audited before you see it.

> **Production-Oriented** project - not a full-scale intelligent AI data/ML system.
> **Educational project, not betting advice.** See [LEGAL.md](LEGAL.md).

---

## 1. What It Does (30 seconds)

1. **Rates fighters.** Six attributes (ATT, DEF, WRE, SUB, STA, PSY) scaled 65-99, built from real UFCStats data.
2. **Simulates fights.** Pick two fighters, run up to 200 Monte Carlo bouts, get win probabilities, finish methods, and 95% confidence intervals.
3. **Explains itself safely.** A copilot writes the tactical brief, and an auditor checks every claim against the simulation. A wrong number is refused, never shown.

Think of it as a **restaurant**: the data pipeline is the kitchen, the semantic layer is the fixed menu, the model is the chef, the copilot is the waiter, and the auditor is the health inspector.

---

## 2. Stats & Modeling Toolkit (what's inside)

| Technique | Where it's used | Why it matters |
|---|---|---|
| **Point-in-time features** | `features.py` | Only data known *before* each fight is used, so no future leakage. |
| **Bayesian shrinkage** | `features.py` | Fighters with few bouts get pulled toward average (shown as a *provisional* badge). |
| **Elastic-Net logistic regression** | `ratings.py` | Learns which stats drive wins; antisymmetric by design (swap corners, probability flips). |
| **Out-of-time validation** | `ratings.py` | Train up to 2023, validate 2024-26, the way the model is really used. |
| **Promotion gate** | `pipeline.py` | A new model replaces the champion only if it holds or beats it. |
| **Percentile scaling (65-99)** | `ratings.py` | Ratings are ranks within a division, so they are easy to read. |
| **Calibrated XGBoost + Brier Skill Score** | `train.py`, `simulation.py` | Powers the animated replay; judged on probability quality, not just accuracy. |
| **Monte Carlo simulation** | `monte_carlo.py` | Turns one matchup into a distribution of outcomes. |
| **Wilson confidence intervals** | `intervals.py` | Honest uncertainty on a simulated win rate. |
| **PCA + K-Means + KNN** | `archetypes.py` | Groups fighters into styles and finds historical look-alikes. |
| **Rating-shift report** | `pipeline.py` | Week-over-week movement per fighter (the useful drift signal for percentile ratings). |

**Honest result:** validated AUC is **0.643 vs a 0.631 baseline**. That is baseline-competitive, not magic. MMA is a high-noise sport, and we say so instead of hiding it.

---

## 3. Insights for Fight Fans

- **A probability is not a verdict.** A 62% win rate with a wide interval is a lean, not a lock. Always read the interval.
- **Ratings are relative.** 90 means "top of their division right now", not "unbeatable". Cross-division matchups are approximations.
- **Provisional means thin evidence.** Few UFC bouts = more shrinkage toward average.
- **Method breakdowns show how a fighter wins**, not only whether. Compare finish shares, not just the headline number.
- **Fights are noisy.** Our own validation says the edge over a simple baseline is small. Treat the cards as a conversation starter.

> 📝 *Add your own 2-3 findings here after exploring the cards (e.g. a surprising rating, a stylistic matchup).*

---

## 4. Architecture

```text
 UFCStats CSVs ──► ingest.py (schema-checked) ──► medallion.py (Bronze → Silver → Gold, DuckDB)
                                                        │
                       features.py (point-in-time) ◄────┘
                                │
                 ratings.py (train → gate → cards)  ──►  pipeline.py (weekly, publish-last)
                                                              │
                                              artifacts/cards/cards.json + metrics.json
                                                              │
   config/metrics.yml ► semantic.py ► tools.py ◄── profiles.py (ratings → simulator)
                                         │
                      harness.py (plan → run → narrate → audit → retry/refuse)
                         ├─ copilot.py   (narrator: Claude or deterministic mock)
                         └─ auditor.py   (structured claims, ±1.5 pt tolerance)
                                         │
                            app.py (Streamlit)   mcp_server.py (FastMCP)
```

**Key design choices** (all logged in [`config/decision-log.md`](config/decision-log.md)):

- **Governed metrics:** `config/metrics.yml` is the single menu of metric names. A typo fails loudly, never guesses.
- **The LLM phrases, never calculates.** Numbers come only from `tools.py`.
- **Publish-last pipeline:** a bad week can never overwrite a good one.
- **Deterministic CI:** the LLM is mocked, so tests are free and repeatable.

---

## 5. A Week in the Life (sample end-to-end workflow)

*Numbers below are illustrative; the flow is exactly what the code does.*

**Monday 09:00 UTC: the robot wakes up** (`weekly.yml`)

1. **Fetch.** `pipeline --fetch` downloads the four UFCStats CSVs (events, results, fight stats, fighter details). Each file is staged and schema-checked first. If upstream changed a column, the job stops and last week's data stays untouched.
2. **Land it in layers** (`medallion.py`, DuckDB). *Bronze* = raw copy, *Silver* = cleaned and typed, *Gold* = one row per fighter and per bout. Data-quality rules run here.
3. **Build features** (`features.py`). For every fight, only what was known *before* it is counted, then shrunk toward average for fighters with few bouts.
4. **Train and score** (`ratings.py`). An Elastic-Net model learns on fights up to 2023 and is checked on 2024-26. Fighter attributes become percentile ratings from 65 to 99.
5. **Two gates.** The *model gate* asks: is this version as good as the champion? The *card gate* asks: no empty file, no duplicates, no missing or out-of-range ratings, and no roster that shrank by more than 20%? A failed model gate quietly holds. A failed card gate fails the job loudly.
6. **Publish last.** `cards.json` and `metrics.json` are written atomically (including a rating-shift report), then the bot commits them to `main`. CI runs again on that commit.

**Anytime afterwards: you open the Streamlit app** (`app.py`)

7. **The app loads the published cards** through the semantic layer (`semantic.py` + `config/metrics.yml`). Only the 8 showcase fighters appear in the selectors, and each card renders under its fighter (`card_view.py`).
8. **You pick Fighter A vs Fighter B.** `profiles.py` converts each fighter's six ratings into 12 combat parameters (power, blocking, takedowns, stamina decay, and so on).
9. **Simulate.** `monte_carlo.py` runs up to 200 bouts. Out come win rates, finish-method shares, and (via `intervals.py`) 95% Wilson confidence intervals. The animated replay uses the XGBoost interval predictor.

**When you ask the Copilot for a brief** (`src/agent/`)

10. **Plan → Execute.** `harness.py` calls `tools.py`, the *only* place numbers are produced. This is the ground truth.
11. **Narrate.** `copilot.py` asks the narrator (Claude, or a deterministic mock in CI) for two things: a short brief for humans and a list of structured **claims** (fighter names, win probabilities, CI, method share, simulation count).
12. **Audit.** `auditor.py` checks every claim against ground truth within ±1.5 points, and scans the prose for stray percentages. Swapped fighters, invented methods, and made-up intervals are all caught.
13. **Pass → show, fail → retry once with feedback, fail again → refuse.** The UI then displays either the verified brief with a "verified" caption or a plain refusal. Nothing unverified reaches the screen.

**The same tools, other clients.** `mcp_server.py` wraps `tools.py` for FastMCP, so Claude Desktop or Cursor can call identical, audited functions.

```text
Mon 09:00  fetch → bronze/silver/gold → features → train → model gate → card gate → publish → commit
Any time   app.py → cards → profiles → Monte Carlo (+CI) → copilot brief → auditor → screen
```

---

## 6. Repository Skeleton

```text
├── .github/workflows/
│   ├── ci.yml            # ruff + pytest + evals + docker build on every push/PR
│   └── weekly.yml        # scheduled data refresh and card publish
├── config/
│   ├── metrics.yml       # semantic layer (governed metric registry)
│   ├── decision-log.md   # architecture decision records (ADRs)
│   └── ubiquitous_language.md
├── data/raw/             # UFCStats CSV snapshot
├── artifacts/cards/      # published cards.json, metrics.json, model registry
├── src/
│   ├── engine/           # data, features, ratings, simulation, pipeline
│   └── agent/            # tools, harness, copilot, auditor, MCP server
├── tests/                # pytest guardrails
├── evals/                # golden scenarios (incl. adversarial lies)
├── app.py                # Streamlit app
├── Dockerfile
├── PRD.md  design-doc.md  LEGAL.md
```

---

## 7. Quick Start

```bash
# clone and install
git clone https://github.com/alijrizvi/ufc_mma-fightpred-decision-engine.git
cd ufc_mma-fightpred-decision-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the app
streamlit run app.py

# verify everything
ruff check . && pytest tests/ evals/

# refresh the data and republish cards
python -m src.engine.pipeline --raw-dir data/raw --out-dir artifacts/cards --fetch

# expose tools to MCP clients
python -m src.agent.mcp_server

# container
docker build -t cagequant . && docker run -p 8501:8501 cagequant
```

**Optional:** set `ANTHROPIC_API_KEY` to use live Claude narration. Without it, the copilot uses a deterministic mock.

---

## 8. Build Your Own (the path I followed)

1. **Frame the decision first.** Write `PRD.md` and a glossary before any code.
2. **Build the safety net.** CI, linting, and one passing test on day 1.
3. **Land data in layers.** Raw → clean → analytics tables, with quality tests.
4. **Define metrics once.** A registry file plus a loader that rejects unknown names.
5. **Baseline first, then model.** Split by *time*, never randomly.
6. **Automate the refresh.** An idempotent pipeline with a promotion gate and a cron.
7. **Add the agent last.** Tools → auditor → harness → evals, with adversarial tests.
8. **Package and tell the story.** Docker, README, and a write-up on your *judgment*.

Each step follows **Red → Green → Refactor**: write the failing test, make it pass, then clean up.

---

## 9. Standing on Giants' Shoulders

| Who | What I borrowed |
|---|---|
| **Andres Vourakis** | Project-based, stakeholder-first data science and portfolio storytelling |
| **Matt Dancho** | Business-driven data science: translate models into ROI and decisions |
| **Matt Pocock** | AI-coding workflow: shared alignment ("grill me"), deep modules, tracer-bullet slices, TDD |
| **Chip Huyen & Goku Mohandas** | Production ML: data contracts, monitoring, CI/CD |
| **Cassie Kozyrkov** | Decision intelligence: frame the decision before the model |
| **Sebastian Raschka & Andrej Karpathy** | First-principles understanding of models |
| **Demetrios Brinkmann (MLOps Community)** | Real-world deployment and testing culture |

---

## 10. Known Limitations

- Ratings map to simulator parameters with a **modelling assumption** (linear, ADR 026), not a fitted estimate.
- The animated replay's interval predictor still trains on **synthetic seeded telemetry**.
- The UI caps simulations at **200** for latency (ADR 022).
- Source data is a community mirror of UFCStats; terms may restrict automated use (ADR 015).

## 11. Legal & Attribution

Educational, non-commercial, and unaffiliated with the UFC. Data comes from UFCStats via the `Greco1899/scrape_ufc_stats` CSV mirror. Full terms in [LEGAL.md](LEGAL.md).

---

*Built by [Ali Jazib Rizvi](https://github.com/alijrizvi). Questions and critiques welcome, and so are pull requests.*
