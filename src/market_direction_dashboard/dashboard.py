from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

from .evaluation import top_feature_importance


def render_dashboard(result: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    figures = _build_figures(result)
    figure_html: dict[str, str] = {}
    first = True
    for key, fig in figures.items():
        figure_html[key] = to_html(fig, full_html=False, include_plotlyjs="inline" if first else False)
        first = False

    payload = json.dumps(result, default=_json_default)
    cards = []
    for name, model in result["models"].items():
        cards.append(
            f"""
            <div class="card model-card">
              <div class="card-label">{name}</div>
              <div class="prob-row"><span>UP</span><strong>{model['latest_probability_up'] * 100:.1f}%</strong></div>
              <div class="bar"><span style="width:{model['latest_probability_up'] * 100:.1f}%"></span></div>
              <div class="prob-row"><span>DOWN</span><strong>{model['latest_probability_down'] * 100:.1f}%</strong></div>
              <div class="meta-grid">
                <div><span>CV Accuracy</span><strong>{model['cv_accuracy'] * 100:.1f}%</strong></div>
                <div><span>AUC</span><strong>{model['auc'] * 100:.1f}%</strong></div>
              </div>
            </div>
            """
        )

    ticker_items = [
        f"{result['index_column']} {result['latest_price']:.2f}",
        f"UP {result['ensemble']['probability_up'] * 100:.1f}%",
        f"DOWN {result['ensemble']['probability_down'] * 100:.1f}%",
        f"EXP MOVE {result['ensemble']['expected_move'] * 100:.2f}%",
        f"CONF {result['ensemble']['confidence']:.1f}%",
    ]
    if result.get("live_badge"):
        ticker_items.append(result["live_badge"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{result['project_name']}</title>
  <style>
    :root {{
      --bg: #0a0c10;
      --panel: #11151c;
      --panel-alt: #171d27;
      --accent: #ff9b2f;
      --accent-soft: rgba(255, 155, 47, 0.2);
      --text: #f5f7fb;
      --muted: #9ca8ba;
      --good: #4fd18b;
      --bad: #ff6b6b;
      --grid: rgba(255,255,255,0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(255,155,47,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(79,209,139,0.12), transparent 22%),
        linear-gradient(180deg, #05070a 0%, #0b0e13 55%, #07090d 100%);
    }}
    .shell {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    .ticker {{
      overflow: hidden;
      white-space: nowrap;
      border: 1px solid var(--accent-soft);
      border-radius: 999px;
      padding: 12px 0;
      background: rgba(255,255,255,0.03);
      margin-bottom: 24px;
    }}
    .ticker-track {{
      display: inline-block;
      padding-left: 100%;
      animation: ticker 28s linear infinite;
    }}
    .ticker-track span {{
      margin-right: 40px;
      color: var(--accent);
      font-weight: 700;
      letter-spacing: 0.12em;
      font-size: 0.78rem;
    }}
    @keyframes ticker {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-100%); }} }}
    .hero {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .card {{
      background: linear-gradient(180deg, rgba(23,29,39,0.92), rgba(12,16,22,0.92));
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 24px;
      padding: 22px;
      box-shadow: 0 22px 60px rgba(0,0,0,0.28);
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: clamp(2rem, 4vw, 4.4rem); }}
    .hero p {{ margin: 0; color: var(--muted); max-width: 65ch; }}
    .metric-grid {{
      margin-top: 22px;
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }}
    .metric {{
      padding: 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.05);
    }}
    .metric span, .card-label, .meta-grid span, .section-kicker {{
      color: var(--muted);
      display: block;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    .metric strong {{
      display: block;
      margin-top: 8px;
      font-size: 1.7rem;
    }}
    .signal {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      margin-top: 16px;
      padding: 10px 14px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 18px;
      margin-bottom: 18px;
    }}
    .bar {{
      margin: 10px 0 14px;
      height: 10px;
      background: rgba(255,255,255,0.08);
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--accent), #ffd17e);
    }}
    .prob-row, .meta-grid {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }}
    .meta-grid {{ margin-top: 16px; }}
    .meta-grid div {{ flex: 1; }}
    .section {{
      margin-bottom: 18px;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      margin-bottom: 14px;
    }}
    .section-head h2 {{
      margin: 0;
      font-size: 1.25rem;
    }}
    .chart-grid-2 {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 18px;
    }}
    .footer-note {{
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.5;
      margin-top: 18px;
    }}
    @media (max-width: 980px) {{
      .hero, .grid-3, .chart-grid-2, .metric-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="ticker"><div class="ticker-track">{"".join(f"<span>{item}</span>" for item in ticker_items)}</div></div>
    <section class="hero">
      <div class="card">
        <span class="section-kicker">AI Market Direction Probability Model</span>
        <h1>{result['index_column']}</h1>
        <p>{result['summary_text']}</p>
        <div class="metric-grid">
          <div class="metric"><span>Probability Up</span><strong>{result['ensemble']['probability_up'] * 100:.1f}%</strong></div>
          <div class="metric"><span>Probability Down</span><strong>{result['ensemble']['probability_down'] * 100:.1f}%</strong></div>
          <div class="metric"><span>Expected Move</span><strong>{result['ensemble']['expected_move'] * 100:.2f}%</strong></div>
          <div class="metric"><span>Confidence</span><strong>{result['ensemble']['confidence']:.1f}%</strong></div>
        </div>
        <div class="signal">{result['combined_signal']['label']} · {result['combined_signal']['confidence']:.1f}%</div>
      </div>
      <div class="card">
        <span class="section-kicker">Live Context</span>
        <h2 style="margin:6px 0 14px;">{result.get('live_badge', 'Historical latest row')}</h2>
        <div class="metric-grid">
          <div class="metric"><span>Current Price</span><strong>{result['latest_price']:.2f}</strong></div>
          <div class="metric"><span>Monte Carlo P(Up)</span><strong>{result['monte_carlo']['summary']['p_up'] * 100:.1f}%</strong></div>
          <div class="metric"><span>Expected Price</span><strong>{result['monte_carlo']['summary']['expected_price']:.2f}</strong></div>
          <div class="metric"><span>P5 / P95</span><strong>{result['monte_carlo']['summary']['p5']:.2f} / {result['monte_carlo']['summary']['p95']:.2f}</strong></div>
        </div>
      </div>
    </section>

    <section class="grid-3">{''.join(cards)}</section>

    <section class="section">
      <div class="section-head">
        <div>
          <span class="section-kicker">3D Analytics</span>
          <h2>Probability Surfaces and Model Geometry</h2>
        </div>
      </div>
      <div class="chart-grid-2">
        <div class="card">{figure_html['feature_surface']}</div>
        <div class="card">{figure_html['probability_surface']}</div>
        <div class="card">{figure_html['scatter_cloud']}</div>
        <div class="card">{figure_html['monte_carlo_paths']}</div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <span class="section-kicker">Diagnostics</span>
          <h2>Evaluation, Correlation, and Simulation Range</h2>
        </div>
      </div>
      <div class="chart-grid-2">
        <div class="card">{figure_html['rolling_accuracy']}</div>
        <div class="card">{figure_html['correlation_heatmap']}</div>
        <div class="card">{figure_html['confusion_heatmaps']}</div>
        <div class="card">{figure_html['radar']}</div>
        <div class="card">{figure_html['monte_carlo_density']}</div>
        <div class="card">{figure_html['distribution_hist']}</div>
      </div>
    </section>

    <div class="footer-note">
      For educational purposes only. Not financial advice. Machine-learning probabilities on market direction are uncertain and should be treated as one signal among many, not a trading guarantee.
    </div>
    <script id="dashboard-data" type="application/json">{payload}</script>
  </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path


def _build_figures(result: dict) -> dict[str, go.Figure]:
    models = list(result["models"].keys())
    top_features = result["top_features"]
    importance_matrix = np.array(
        [[result["models"][model]["feature_importance"].get(feature, 0.0) for feature in top_features] for model in models]
    )
    feature_surface = go.Figure(
        data=[
            go.Surface(
                z=importance_matrix,
                x=top_features,
                y=models,
                colorscale="Magma",
            )
        ]
    )
    feature_surface.update_layout(title="3D Feature Importance Surface", template="plotly_dark", height=460)

    walk_forward = pd.DataFrame(result["walk_forward"])
    probability_rows = []
    for model in models:
      probability_rows.append(walk_forward[f"{model}_prob"].fillna(0.5).to_list())
    probability_surface = go.Figure(
        data=[go.Surface(z=np.array(probability_rows), x=walk_forward["date"], y=models, colorscale="Viridis")]
    )
    probability_surface.update_layout(title="3D Probability Evolution", template="plotly_dark", height=460)

    scatter = go.Figure(
        data=[
            go.Scatter3d(
                x=walk_forward[f"{models[0]}_prob"],
                y=walk_forward[f"{models[1]}_prob"],
                z=walk_forward[f"{models[2]}_prob"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=walk_forward["actual"],
                    colorscale=[[0, "#ff6b6b"], [1, "#4fd18b"]],
                    opacity=0.78,
                ),
                text=walk_forward["date"],
            )
        ]
    )
    scatter.update_layout(title="3D Model Probability Scatter", template="plotly_dark", height=460)

    rolling_accuracy = go.Figure()
    for model in models:
        rolling_accuracy.add_trace(
            go.Scatter(x=walk_forward["date"], y=walk_forward[model], mode="lines", name=model)
        )
    rolling_accuracy.update_layout(title="Rolling Walk-Forward Accuracy", template="plotly_dark", height=420, yaxis_tickformat=".0%")

    correlation = pd.DataFrame(result["correlation"])
    correlation_heatmap = go.Figure(
        data=[
            go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.index,
                colorscale="RdBu",
                zmid=0,
            )
        ]
    )
    correlation_heatmap.update_layout(title="Feature Correlation Heatmap", template="plotly_dark", height=420)

    confusion_heatmaps = go.Figure()
    for idx, model in enumerate(models):
        matrix = np.array(result["models"][model]["confusion"])
        confusion_heatmaps.add_trace(
            go.Heatmap(
                z=matrix,
                x=["Pred Down", "Pred Up"],
                y=["Actual Down", "Actual Up"],
                colorscale="Oranges",
                showscale=idx == 0,
                visible=idx == 0,
            )
        )
    buttons = []
    for idx, model in enumerate(models):
        visible = [i == idx for i in range(len(models))]
        buttons.append(dict(label=model, method="update", args=[{"visible": visible}, {"title": f"{model} Confusion Matrix"}]))
    confusion_heatmaps.update_layout(
        title=f"{models[0]} Confusion Matrix",
        template="plotly_dark",
        updatemenus=[dict(type="buttons", buttons=buttons, x=0.0, y=1.15)],
        height=420,
    )

    radar = go.Figure()
    radar_metrics = ["cv_accuracy", "auc", "precision", "recall"]
    labels = ["CV Accuracy", "AUC", "Precision", "Recall"]
    for model in models:
        values = [result["models"][model][metric] for metric in radar_metrics]
        radar.add_trace(go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]], fill="toself", name=model))
    radar.update_layout(title="Model Quality Radar", template="plotly_dark", polar=dict(radialaxis=dict(range=[0, 1])), height=420)

    monte_carlo_paths = go.Figure()
    paths = np.array(result["monte_carlo"]["paths"])
    sample_count = min(500, paths.shape[1])
    sampled = np.linspace(0, paths.shape[1] - 1, sample_count, dtype=int)
    for order, idx in enumerate(sampled):
        monte_carlo_paths.add_trace(
            go.Scatter3d(
                x=np.arange(paths.shape[0]),
                y=np.full(paths.shape[0], order),
                z=paths[:, idx],
                mode="lines",
                line=dict(color="#ff9b2f", width=2),
                showlegend=False,
            )
        )
    monte_carlo_paths.update_layout(title="3D Monte Carlo Paths", template="plotly_dark", height=460)

    density_surface = np.array(result["monte_carlo"]["density_surface"])
    monte_carlo_density = go.Figure(
        data=[
            go.Surface(
                z=density_surface,
                x=np.arange(density_surface.shape[1]),
                y=result["monte_carlo"]["price_axis"],
                colorscale="Plasma",
            )
        ]
    )
    monte_carlo_density.update_layout(title="3D Monte Carlo Density Wave", template="plotly_dark", height=420)

    terminal = paths[-1]
    distribution_hist = go.Figure(data=[go.Histogram(x=terminal, nbinsx=50, marker_color="#4fd18b")])
    distribution_hist.update_layout(title="Final Price Distribution", template="plotly_dark", height=420)

    return {
        "feature_surface": feature_surface,
        "probability_surface": probability_surface,
        "scatter_cloud": scatter,
        "rolling_accuracy": rolling_accuracy,
        "correlation_heatmap": correlation_heatmap,
        "confusion_heatmaps": confusion_heatmaps,
        "radar": radar,
        "monte_carlo_paths": monte_carlo_paths,
        "monte_carlo_density": monte_carlo_density,
        "distribution_hist": distribution_hist,
    }


def _json_default(value):
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")
