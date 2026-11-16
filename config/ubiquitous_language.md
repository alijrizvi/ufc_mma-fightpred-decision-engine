# Ubiquitous Language Glossary

* **Interval Telemetry:** Discretized 10-second data segment generated from empirical strike distributions (30 segments per round).
* **Brier Skill Score (BSS):** Relative measure of predictive probability calibration over naive empirical baselines.
* **High-Output Power Stance:** Combat mode boosting damage ($1.40\times$) at the cost of high stamina drain ($2.50\times$) and defensive penalty ($0.75\times$).
* **High-Efficiency Flow Stance:** Combat mode optimizing stamina conservation ($0.60\times$) while preserving defensive baseline.
* **Ascended Peak Output State:** High-performance stance unlocked under high vitality ($HP > 60$, Stamina $> 70$).
* **FastMCP Tool:** Standardized API function exposed by the engine for agentic tool calling.
* **Analytical Auditor:** Verification module validating LLM numerical assertions against tool ground truth.
* **Fighter Rating Card:** FIFA-style card with six attributes scaled 65-99 by percentile rank within the division population.
* **ATT / DEF / WRE / SUB / STA / PSY:** Attack, Defense, Wrestling/Grappling, Submission, Stamina/Cardio, Psychology (proxy).
* **Point-in-Time:** Computed using only information available before a given fight date, so training never sees the future.
* **Provisional Card:** A rating built on fewer than 7 UFC bouts, heavily shrunk toward the division prior.
* **Stale Flag:** Marks a fighter whose last bout is older than the inactivity threshold.
* **Cold Streak:** Four consecutive losses. A form alert, not a data-quality flag.
* **Model Version:** A saved, labeled Elastic-Net fit (coefficients, data cutoff, metrics, commit) that can be audited or rolled back.
* **Promotion Gate:** The automatic check a new weekly fit must pass before replacing the current version.
* **Bayesian Shrinkage:** Pulling a thin-sample estimate toward the division average, weighted by cage minutes.
* **Medallion Layers:** Bronze (raw), Silver (cleaned), Gold (analysis-ready star schema).
