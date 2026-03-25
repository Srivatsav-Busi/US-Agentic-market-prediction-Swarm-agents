# Graph-Native Prediction Upgrade Plan

## Summary

Upgrade the repo from a graph-assisted reasoning system to a graph-informed prediction system in three implementation waves:

1. Add graph priors and graph-aware evidence weighting to the statistical engine.
2. Add graph-derived numeric features to the ML training and inference dataset.
3. Add temporal graph deltas across runs and use them in both engines.

This keeps the current architecture intact while making the graph materially affect predictions. Do not introduce GNNs or full graph-learning infrastructure in v1. Use deterministic engineered graph features, persisted run-level graph summaries, and clear ablations.

## Target Architecture

`raw evidence -> graph-first ingestion -> graph summaries / graph priors / graph deltas -> signal features + evidence quality -> statistical engine + hybrid ML -> ensemble decision -> persisted artifact + graph snapshot`

Key principle:

- The graph is not the final model.
- The graph is a structured feature and prior generator for the existing models.

## Implementation Waves

### Wave 1: Statistical Engine Integration

Goal:

- Make the graph directly influence posterior probabilities and evidence quality with low-risk deterministic logic.

Key changes:

- Introduce a graph analytics module that computes run-level graph priors from the in-memory `KnowledgeGraph`.
- Add a `GraphPredictionPriors` structure with fields like:
  - `bullish_path_strength`
  - `bearish_path_strength`
  - `consensus_score`
  - `contradiction_score`
  - `contagion_score`
  - `credibility_weighted_pressure`
  - `cross_category_connectivity`
  - `freshness_propagation_score`
- Extend the statistical evidence aggregation flow so `_aggregate_bayesian_evidence(...)` accepts graph priors.
- Adjust posterior combination logic so graph priors act as bounded modifiers, not replacements, for statistical probabilities.
- Add graph-aware evidence-quality adjustments:
  - penalize duplicate evidence from tightly connected low-diversity source clusters
  - boost independent corroboration from weakly connected source clusters
  - penalize high contradiction density across paths tied to the same thesis
- Surface graph diagnostics in the artifact:
  - `graph_priors`
  - `graph_evidence_adjustments`
  - `graph_conflict_summary`
  - `graph_feature_summary`

Design constraints:

- Cap graph influence to prevent it overwhelming the baseline model.
- If graph quality is low or graph is sparse, priors collapse to neutral.
- Graph priors must be reproducible from the same artifact.

Files/modules to change:

- `src/market_direction_dashboard/statistical_engine.py`
- new graph analytics module under `src/market_direction_dashboard/graph_features/` or similar
- `src/market_direction_dashboard/pipeline.py` to compute graph priors before `synthesize_prediction`

### Wave 2: ML Feature Engine Integration

Goal:

- Turn graph summaries into stable numeric features for hybrid ML training and inference.

Key changes:

- Define a graph feature vector schema for daily runs. Recommended first set:
  - evidence node count by category
  - provider concentration index
  - source redundancy ratio
  - contradiction density
  - disconnected corroboration score
  - bullish path strength
  - bearish path strength
  - market-to-macro connectivity
  - inflation-to-yields-to-equities chain strength
  - freshness-weighted evidence mass
  - regime-cluster intensity by category
- Create a graph summary builder that produces these features from the run graph.
- Persist daily graph summary rows alongside existing history so ML retraining has historical graph features.
- Extend `_build_ml_dataset(...)` to join graph summary features by date.
- Ensure inference-time graph feature generation uses exactly the same schema and default behavior as training.
- Add feature-availability flags so the model can degrade when graph history is sparse.

Design constraints:

- Use only deterministic engineered features in v1.
- Avoid graph features that depend on future information.
- All graph features must be timestamp-safe and aligned to the prediction date.

Files/modules to change:

- `src/market_direction_dashboard/forecasting/hybrid_ml.py`
- storage models/repositories for graph summary persistence
- `src/market_direction_dashboard/pipeline.py` to save graph feature summaries when persistence is enabled

### Wave 3: Retrieval-Assisted Feature Extraction

Goal:

- Improve signal extraction by using graph neighborhoods and paths before final feature construction.

Key changes:

- Split feature extraction into two passes:
  - pass 1: raw item features from current logic
  - pass 2: graph-retrieved compound features from connected evidence clusters
- Add retrieval routines for:
  - category neighborhood lookup
  - source-to-theme path lookup
  - cross-category motif detection
  - contradiction triangle / conflict cluster detection
- Create second-pass feature templates such as:
  - `inflation_yields_equity_pressure`
  - `policy_uncertainty_credit_stress`
  - `oil_usd_growth_conflict`
  - `independent_macro_confirmation`
  - `headline_crowding_without_market_confirmation`
- Merge pass-2 features into the existing `signal_features` list with provenance metadata pointing to graph entities/edges.

Design constraints:

- Second-pass features should be additive and explainable.
- Keep feature names deterministic and compact.
- Preserve backward compatibility for downstream consumers of `signal_features`.

Files/modules to change:

- current feature extraction path
- graph retrieval helpers
- `src/market_direction_dashboard/pipeline.py` ordering so graph retrieval happens before final feature set is frozen

### Wave 4: Temporal Graph Delta Modeling

Goal:

- Capture how narrative structure evolves across days, not just what exists today.

Key changes:

- Persist daily graph summaries and lightweight graph snapshots keyed by run date.
- Add graph delta computation between adjacent runs:
  - new entities introduced
  - removed entities
  - edge churn
  - category intensity change
  - directional path change
  - contradiction change
  - corroboration change
  - provider diversity change
- Feed graph deltas into:
  - statistical regime-transition logic as prior modifiers
  - hybrid ML as temporal explanatory features
- Add artifact diagnostics:
  - `graph_delta_summary`
  - `theme_acceleration`
  - `narrative_reversal_flags`

Design constraints:

- Delta computation must tolerate missing prior runs.
- Use summary-level deltas first; do not diff full large graphs in hot path if avoidable.
- Store compact delta rows for fast historical joins.

Files/modules to change:

- graph persistence layer
- storage layer for graph summaries/deltas
- `statistical_engine.py`
- `forecasting/hybrid_ml.py`

### Wave 5: Graph-Aware Trust and Quality Layer

Goal:

- Upgrade evidence quality scoring from item-local to graph-aware.

Key changes:

- Extend normalization / quality scoring with graph-derived signals:
  - cluster redundancy penalty
  - independent corroboration boost
  - contradiction penalty
  - stale-cluster penalty
  - source monoculture penalty
- Add cluster-level quality summary into `data_quality_summary`.
- Feed quality adjustments into both the statistical posterior combiner and challenge layer.

Design constraints:

- Graph-aware quality should modify, not replace, current freshness/credibility logic.
- Keep outputs auditable at item and cluster levels.

Files/modules to change:

- source normalization / validation
- statistical evidence combiner
- challenge/trust-gate logic

## Public Interfaces And Types

Add new internal/public artifact fields:

- `graph_priors`
- `graph_feature_summary`
- `graph_evidence_adjustments`
- `graph_delta_summary`
- `graph_quality_summary`

Add new internal types:

- `GraphPredictionPriors`
- `GraphFeatureVector`
- `GraphDeltaSummary`
- `GraphQualityAdjustments`

Storage additions:

- daily graph summary table keyed by `prediction_date` and `target`
- graph delta table keyed by `run_id` or `prediction_date`
- optional model-feature-schema version tracking for graph features

Backward compatibility:

- Existing output fields remain unchanged.
- New graph fields are additive.
- If graph data is unavailable, all new graph-driven logic must degrade to neutral defaults.

## Subagent Implementation Plan

Use parallel workers with disjoint ownership.

### Subagent 1: Graph Analytics Worker

Ownership:

- new graph analytics package
- graph priors
- graph feature vector builder
- graph delta calculator

Deliverables:

- deterministic graph metrics from `KnowledgeGraph`
- tests for sparse, contradictory, redundant, and corroborated graphs
- versioned feature schema

### Subagent 2: Statistical Engine Worker

Ownership:

- `statistical_engine.py`
- evidence combiner integration
- graph prior blending
- graph-aware posterior diagnostics

Deliverables:

- bounded graph prior integration
- neutral fallback behavior
- ablation toggles and trace steps

### Subagent 3: ML Pipeline Worker

Ownership:

- `forecasting/hybrid_ml.py`
- dataset joins with graph summaries
- inference/training schema parity
- retraining integration

Deliverables:

- graph feature joins by date
- missing-data handling
- importance reporting for graph features

### Subagent 4: Retrieval / Feature Extraction Worker

Ownership:

- second-pass graph retrieval
- compound graph-derived signal features
- pipeline orchestration around feature extraction order

Deliverables:

- retrieval-assisted feature templates
- provenance-aware merged feature set
- tests for motif detection

### Subagent 5: Storage / Persistence Worker

Ownership:

- DB models and repositories for graph summaries and deltas
- persistence hooks in pipeline
- schema migrations or creation updates

Deliverables:

- summary/delta persistence
- historical load APIs for ML/statistics
- compact query paths

### Subagent 6: Evaluation / Validation Worker

Ownership:

- offline backtests
- ablation framework
- regression thresholds
- artifact validation

Deliverables:

- compare baseline vs graph-priors vs graph+ML-features vs full graph pipeline
- metrics by regime and by data availability
- guardrails against overfitting

## Execution Order

1. Build graph analytics package and feature schema.
2. Integrate graph priors into the statistical engine.
3. Add graph-aware quality adjustments.
4. Persist graph summaries per run.
5. Join graph summaries into hybrid ML training/inference.
6. Add graph retrieval-assisted second-pass signal features.
7. Add temporal graph deltas.
8. Run ablations and tighten influence caps.
9. Expose final graph diagnostics in JSON/HTML.

## Testing And Acceptance Criteria

Core correctness:

- Graph prior computation is deterministic for the same graph input.
- Sparse/noisy graphs collapse to neutral priors.
- Contradictory graph structures reduce conviction rather than amplify it.
- Missing graph history does not break ML training/inference.

Statistical integration:

- Posterior probabilities change when graph priors are present.
- Graph prior impact is capped and traceable.
- Quality penalties/boosts are visible in diagnostics.

ML integration:

- Graph feature columns are identical between training and inference.
- Retraining succeeds with and without graph history.
- Feature importance output can show graph features explicitly.

Retrieval-assisted features:

- Known graph motifs create expected compound features.
- No duplicate/contradictory feature explosion from second-pass retrieval.

Temporal deltas:

- Delta summaries are correct for added/removed themes and edges.
- Regime-transition features are unavailable-safe when previous run is missing.

Performance:

- Graph metric computation stays within acceptable daily-run latency.
- No material slowdown from path exploration on typical artifact sizes.

Evaluation:

- Run ablations:
  - baseline
  - baseline + graph priors
  - baseline + graph quality
  - baseline + graph priors + graph ML features
  - full system
- Acceptance target:
  - improved calibration or directional accuracy versus baseline
  - no large degradation in low-data days
  - better explainability and confidence behavior in contradictory evidence regimes

## Risks And Controls

Main risks:

- graph features overfit narrative noise
- graph priors overpower statistical evidence
- temporal deltas leak future information if aligned incorrectly
- complexity increases without measurable prediction gain

Controls:

- influence caps on all graph priors
- schema versioning for graph features
- strict date alignment for persisted graph summaries
- ablation-first rollout
- config flags to disable each graph enhancement independently

## Assumptions

- The in-run `knowledge_graph` remains the canonical graph input in v1.
- Neo4j persistence remains optional and not required for core prediction.
- Existing statistical and ML models stay in place; this plan augments rather than replaces them.
- The first release should optimize for deterministic, auditable improvements rather than deep graph learning.
