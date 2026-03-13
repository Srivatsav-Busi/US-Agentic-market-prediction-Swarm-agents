# Roadmap: Market Direction Probability Dashboard

## Overview

Build a local-first quant finance application that moves from trustworthy data ingestion and time-series feature engineering through model training, evaluation, and self-contained dashboard export. The roadmap keeps chronology-sensitive modeling foundations ahead of visualization work so the final product is explainable, reproducible, and usable from a single command.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Project Scaffold and Data Intake** - Create the Python project, CLI entry points, and robust historical-data loading.
- [x] **Phase 2: Feature Engineering Pipeline** - Build leakage-safe return, lag, momentum, volatility, and spread features.
- [x] **Phase 3: Directional Modeling and Ensemble Forecasting** - Train the three ML models and generate ensemble predictions.
- [x] **Phase 4: Evaluation and Diagnostics** - Add walk-forward evaluation, confusion matrices, importances, and correlation analysis.
- [x] **Phase 5: Dashboard Export and Presentation Layer** - Generate the self-contained HTML dashboard with interactive visuals.
- [x] **Phase 6: Live Overrides and Monte Carlo Layer** - Add current-value overrides, simulation outputs, and combined signal presentation.

## Phase Details

### Phase 1: Project Scaffold and Data Intake
**Goal**: Establish the repository structure and a dependable ingestion path for CSV/XLSX market datasets.
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04
**Success Criteria** (what must be TRUE):
  1. User can run a CLI command that accepts an input file and primary index column.
  2. Historical data is loaded, normalized, sorted by date, and cleaned consistently.
  3. The project has installable dependencies, sample configuration, and basic tests.
**Plans**: 2 plans

Plans:
- [x] 01-01: Create the Python package, CLI, configuration, and dependency manifest.
- [x] 01-02: Implement dataset loading, schema normalization, and ingestion tests.

### Phase 2: Feature Engineering Pipeline
**Goal**: Create reusable, chronology-safe engineered features from market history.
**Depends on**: Phase 1
**Requirements**: FEAT-01, FEAT-02, FEAT-03, FEAT-04
**Success Criteria** (what must be TRUE):
  1. Numeric asset columns produce log-return features.
  2. Lag, moving-average deviation, momentum, volatility, and spread features are available for downstream modeling.
  3. Feature creation does not leak future information into training rows.
**Plans**: 2 plans

Plans:
- [x] 02-01: Build return and lag feature generation for the primary index and auxiliary series.
- [x] 02-02: Add moving-average, momentum, volatility, and yield-spread features with tests.

### Phase 3: Directional Modeling and Ensemble Forecasting
**Goal**: Train the ML models and produce calibrated directional forecasts for the latest feature vector.
**Depends on**: Phase 2
**Requirements**: MODL-01, MODL-02, MODL-03, MODL-04, MODL-05
**Success Criteria** (what must be TRUE):
  1. The application trains logistic regression, random forest, and gradient boosting models with time-series splits.
  2. The latest row receives per-model UP/DOWN probabilities and an ensemble probability.
  3. Expected move and confidence metrics are computed from historical conditional returns.
**Plans**: 2 plans

Plans:
- [x] 03-01: Implement target generation, model training, and cross-validation scoring.
- [x] 03-02: Implement latest-row prediction, ensemble weighting, and summary metrics.

### Phase 4: Evaluation and Diagnostics
**Goal**: Expose model quality and interpretability metrics needed to understand the forecast.
**Depends on**: Phase 3
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):
  1. Walk-forward accuracy is computed across the historical sample.
  2. Confusion matrices and feature-importance outputs are available for each model.
  3. Feature correlation data is produced for dashboard rendering.
**Plans**: 2 plans

Plans:
- [x] 04-01: Add rolling walk-forward evaluation and confusion-matrix generation.
- [x] 04-02: Add feature-importance and feature-correlation diagnostics.

### Phase 5: Dashboard Export and Presentation Layer
**Goal**: Produce the browser-openable HTML dashboard with the required hero metrics and interactive visualizations.
**Depends on**: Phase 4
**Requirements**: DASH-01, DASH-02, DASH-03, DASH-04, DASH-05
**Success Criteria** (what must be TRUE):
  1. The application exports a single HTML file containing all required data and visuals.
  2. The dashboard includes ensemble metrics, per-model cards, 2D evaluation visuals, and 3D interactive charts.
  3. The presentation includes disclaimers and a coherent Bloomberg-style visual treatment.
**Plans**: 2 plans

Plans:
- [x] 05-01: Build the dashboard data contract and Plotly visualization set.
- [x] 05-02: Build the HTML template, styling, and export workflow.

### Phase 6: Live Overrides and Monte Carlo Layer
**Goal**: Add current-market override support and Monte Carlo simulation outputs with combined signal messaging.
**Depends on**: Phase 5
**Requirements**: LIVE-01, LIVE-02, MC-01, MC-02, MC-03
**Success Criteria** (what must be TRUE):
  1. User can provide live override values and see them reflected in the forecast output.
  2. Monte Carlo simulation metrics and visuals are included in the dashboard.
  3. The dashboard presents a combined ML plus Monte Carlo directional signal with confidence context.
**Plans**: 2 plans

Plans:
- [x] 06-01: Add live-feature override ingestion and prediction refresh logic.
- [x] 06-02: Add GBM Monte Carlo simulation, combined signal calculation, and dashboard sections.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Project Scaffold and Data Intake | 2/2 | Complete | 2026-03-13 |
| 2. Feature Engineering Pipeline | 2/2 | Complete | 2026-03-13 |
| 3. Directional Modeling and Ensemble Forecasting | 2/2 | Complete | 2026-03-13 |
| 4. Evaluation and Diagnostics | 2/2 | Complete | 2026-03-13 |
| 5. Dashboard Export and Presentation Layer | 2/2 | Complete | 2026-03-13 |
| 6. Live Overrides and Monte Carlo Layer | 2/2 | Complete | 2026-03-13 |
