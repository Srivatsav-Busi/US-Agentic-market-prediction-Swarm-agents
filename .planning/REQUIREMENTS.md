# Requirements: Market Direction Probability Dashboard

**Defined:** 2026-03-13
**Core Value:** Generate a reproducible market-direction probability dashboard from user-supplied data with transparent modeling, evaluation, and interactive visual outputs.

## v1 Requirements

### Data Ingestion

- [ ] **DATA-01**: User can load historical market data from CSV files.
- [ ] **DATA-02**: User can load historical market data from Excel files.
- [ ] **DATA-03**: User can identify the primary index column explicitly through CLI arguments or config.
- [ ] **DATA-04**: Application sorts observations chronologically and handles missing numeric values safely.

### Feature Engineering

- [ ] **FEAT-01**: Application computes log returns for numeric market columns.
- [ ] **FEAT-02**: Application computes lagged returns for the primary index and key auxiliary assets when present.
- [ ] **FEAT-03**: Application computes moving-average deviation, momentum, and rolling-volatility features without leakage.
- [ ] **FEAT-04**: Application computes yield-spread features when domestic and US bond columns are available.

### Modeling

- [ ] **MODL-01**: Application creates a binary next-period direction target from the primary index return.
- [ ] **MODL-02**: Application trains Logistic Regression, Random Forest, and Gradient Boosting models using time-series cross-validation.
- [ ] **MODL-03**: Application generates per-model UP/DOWN probabilities for the latest feature vector.
- [ ] **MODL-04**: Application computes a weighted ensemble based on cross-validation accuracy.
- [ ] **MODL-05**: Application computes expected move and confidence metrics from ensemble outputs.

### Evaluation

- [ ] **EVAL-01**: Application reports walk-forward accuracy over time.
- [ ] **EVAL-02**: Application produces confusion matrices for each model.
- [ ] **EVAL-03**: Application reports feature importance using model-appropriate metrics.
- [ ] **EVAL-04**: Application reports a feature-correlation heatmap.

### Dashboard

- [ ] **DASH-01**: Application exports a self-contained HTML dashboard that opens locally in a browser.
- [ ] **DASH-02**: Dashboard includes ensemble hero metrics and individual model cards.
- [ ] **DASH-03**: Dashboard includes interactive 3D charts for probability evolution, feature importance, and model-scatter exploration.
- [ ] **DASH-04**: Dashboard includes 2D evaluation visuals including rolling accuracy and confusion matrices.
- [ ] **DASH-05**: Dashboard preserves educational-use and non-financial-advice disclaimers.

### Live Data

- [ ] **LIVE-01**: User can supply current market values as overrides for the latest prediction run.
- [ ] **LIVE-02**: Dashboard displays a live-data badge with the as-of date when overrides are used.

### Monte Carlo

- [ ] **MC-01**: Application can run Geometric Brownian Motion simulations from the current price.
- [ ] **MC-02**: Application reports expected price, median, P5, P95, and directional probabilities from the simulation.
- [ ] **MC-03**: Dashboard includes Monte Carlo 3D visuals and a combined ML plus Monte Carlo signal.

## v2 Requirements

### Data Sources

- **LIVE-03**: Application can fetch live market values directly from configurable providers.
- **MULTI-01**: Application can compare multiple target indices in one run.
- **SOC-01**: Application can export a separate social-summary card HTML artifact.

### Advanced Features

- **FEAT-05**: Application supports optional technical indicators such as RSI, MACD, Bollinger position, and ADX.
- **REG-01**: Application supports market regime classification and regime-specific performance analysis.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Automated trading or order routing | Outside the educational analytics scope |
| High-frequency or intraday prediction engine | The source guide focuses on monthly or weekly directional modeling |
| Guaranteed live web scraping adapters | Provider variability adds fragility beyond v1 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| FEAT-01 | Phase 2 | Pending |
| FEAT-02 | Phase 2 | Pending |
| FEAT-03 | Phase 2 | Pending |
| FEAT-04 | Phase 2 | Pending |
| MODL-01 | Phase 3 | Pending |
| MODL-02 | Phase 3 | Pending |
| MODL-03 | Phase 3 | Pending |
| MODL-04 | Phase 3 | Pending |
| MODL-05 | Phase 3 | Pending |
| EVAL-01 | Phase 4 | Pending |
| EVAL-02 | Phase 4 | Pending |
| EVAL-03 | Phase 4 | Pending |
| EVAL-04 | Phase 4 | Pending |
| DASH-01 | Phase 5 | Pending |
| DASH-02 | Phase 5 | Pending |
| DASH-03 | Phase 5 | Pending |
| DASH-04 | Phase 5 | Pending |
| DASH-05 | Phase 5 | Pending |
| LIVE-01 | Phase 6 | Pending |
| LIVE-02 | Phase 6 | Pending |
| MC-01 | Phase 6 | Pending |
| MC-02 | Phase 6 | Pending |
| MC-03 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 27 total
- Mapped to phases: 27
- Unmapped: 0

---
*Requirements defined: 2026-03-13*
*Last updated: 2026-03-13 after initialization*
