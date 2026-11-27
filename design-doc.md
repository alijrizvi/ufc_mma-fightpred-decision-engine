# Technical Design Document: CageQuant UFC Decision Engine

## 1. System Overview & Component Architecture

UFC Fight Predictor Engine, inspired and powered by _CageQuant_, is a modular decision intelligence framework for 10-second discrete fight simulation, probability modeling, stochastic risk calculation, and verified agentic analytics.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SYSTEM COMPONENT FLOW MATRIX                          │
├──────────────────┬──────────────────────────────────────────────────────────┤
│ Layer            │ Core Modules & Responsibilities                          │
├──────────────────┼──────────────────────────────────────────────────────────┤
│ Presentation     │ app.py (Streamlit UI, Plotly animated dual-axis charts) │
│ Agentic AI       │ src/agent/ (tools.py, copilot.py, auditor.py)            │
│ ML Scoring       │ src/engine/models.py (Calibrated XGBoost Classifier)     │
│ Risk & Tradeoffs │ src/engine/simulation.py, stances.py, monte_carlo.py     │
│ Unsupervised ML  │ src/engine/archetypes.py (PCA + K-Means + KNN)           │
│ Data / Governance│ src/engine/etl.py (DuckDB), config/metrics.yml (Semantic) │
└──────────────────┴──────────────────────────────────────────────────────────┘
```

## Data & Agentic Architecture + Data Flow

DATA SPINE ───────────────────────────────────────── 1. Data Architecture
UFCStats CSVs ─▶ ingest.py ─▶ BRONZE ─▶ SILVER ─▶ GOLD ─▶ features.py
 (4 files)       schema check    raw      clean    bouts    point-in-time
                                                               │
   ratings.py: Elastic-Net · 2024-26 validation · ratings 65-99 ◀┘
                              │
              MODEL GATE ─▶ CARD GATE ─▶ atomic publish
                                              │
          artifacts/cards/: cards.json · metrics.json · registry/vDATE


GOVERNANCE ───────────────────────────────── 2. Semantic Layer + Contracts
metrics.yml ─▶ semantic.py ─▶ profiles.py ─▶ 12 combat parameters
 (the menu)    3 typed errors   ratings→params
 ATT DEF WRE SUB STA PSY ufc_bouts
Contracts (schemas.py, Pydantic):
 MetricQuery · Simulation · Claim · Narration · CopilotAnalysisResult


AGENT HARNESS ────────────────────────────── 3. Agentic AI Architecture
      Streamlit app  ·  FastMCP clients (Claude Desktop, Cursor)
                              │
                              ▼
 harness.py:  plan ─▶ execute ─▶ narrate ─▶ audit ─▶ retry once ─▶ refuse
                 │         │          │
            tools.py   copilot.py  auditor.py
            GROUND     NARRATOR    INSPECTOR
            TRUTH      Claude/mock claims vs truth
            Monte      brief +     tolerance ±1.5 pts
            Carlo +    claims
            Wilson CI
                              │
          PASS ─▶ verified brief on screen   ·   FAIL ─▶ refusal


OPS ──────────────────────────────────────────────── 4. MLOps Guardrails
Weekly cron ─▶ fetch ─▶ train ─▶ 2 gates ─▶ publish ─▶ bot commit ─▶ CI
CI: ruff · pytest · 25 evals (100%) · docker build      ADRs: decision-log.md
Not used: MLflow · DVC · Evidently (file registry + rating-shift report instead)
