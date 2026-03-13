# Market Direction Probability Dashboard

## What This Is

This project is a quantitative finance application that turns historical market data into a directional probability model and a self-contained interactive dashboard. It is for analysts, traders, and educators who want to upload index data, generate next-period UP/DOWN probabilities from multiple ML models, and explore the result through interactive 3D visualizations.

## Core Value

Generate a reproducible market-direction probability dashboard from user-supplied data with transparent modeling, evaluation, and interactive visual outputs.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Ingest CSV/XLSX historical market datasets with an explicit primary index column.
- [ ] Engineer directional-model features, train multiple time-series ML models, and compute an ensemble forecast.
- [ ] Export a single self-contained HTML dashboard with interactive 2D/3D charts and evaluation diagnostics.
- [ ] Support live-market value overrides for the current feature vector.
- [ ] Support Monte Carlo simulation overlays and combined signal reporting.

### Out of Scope

- Multi-user authentication or hosted SaaS features — not required for the first working release.
- Real-time brokerage integrations or automated trading execution — this is an analytics tool, not an execution platform.
- Guaranteed predictive performance claims — inconsistent with the model limitations stated in the source prompt guide.

## Context

The source material is a prompt guide describing an AI-generated quant finance dashboard with logistic regression, random forest, gradient boosting, feature engineering, walk-forward evaluation, Monte Carlo simulation, and a Bloomberg-style presentation layer. The repository is greenfield with no existing application code. The implementation should prioritize reproducibility, transparency, and a clean path from uploaded data to a browser-openable HTML artifact.

## Constraints

- **Tech stack**: Python data/ML stack — best fit for pandas, scikit-learn, and Plotly-based dashboard generation.
- **Input compatibility**: CSV and Excel inputs — the guide explicitly expects both file types.
- **Output format**: Single HTML file — must open locally without a server.
- **Modeling honesty**: No financial-advice framing — the dashboard and docs must preserve educational-use disclaimers.
- **Time-series correctness**: Forward-looking leakage is unacceptable — all feature generation and validation must respect chronology.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use Python CLI plus HTML exporter instead of a hosted web app | The prompt guide’s end state is a downloadable self-contained HTML dashboard | — Pending |
| Treat live data as optional overrides on the latest feature row | Matches the source guide without requiring continuous market feeds | — Pending |
| Keep Monte Carlo as part of the first implementation scope | It is an explicit step in the source guide and part of the top-level signal story | — Pending |

---
*Last updated: 2026-03-13 after initialization*
