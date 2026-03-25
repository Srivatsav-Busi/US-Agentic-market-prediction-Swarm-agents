from __future__ import annotations

import json
from pathlib import Path


def render_dashboard(result: dict, output_path: str | Path | None = None) -> str:
    payload = json.dumps(result)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{result['project_name']}</title>
  <style>
    :root {{
      --bg: #f2eee6;
      --panel: rgba(255,255,255,0.78);
      --panel-strong: rgba(255,255,255,0.92);
      --line: rgba(55, 42, 20, 0.12);
      --ink: #211811;
      --muted: #6c6258;
      --accent: #9d4d26;
      --accent-2: #21584a;
      --good: #237345;
      --bad: #a33030;
      --warn: #9a6a18;
      --shadow: 0 18px 48px rgba(70, 48, 24, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 5% 5%, rgba(157,77,38,0.14), transparent 22%),
        radial-gradient(circle at 95% 10%, rgba(33,88,74,0.14), transparent 18%),
        linear-gradient(180deg, #f7f2ea 0%, #eee3d1 48%, #f8f5ee 100%);
    }}
    .shell {{ max-width: 1480px; margin: 0 auto; padding: 24px; }}
    .card {{
      background: linear-gradient(180deg, var(--panel), var(--panel-strong));
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 20px;
      box-shadow: var(--shadow);
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .kicker {{
      display: block;
      color: var(--muted);
      font-size: 0.77rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      margin-bottom: 8px;
    }}
    h1, h2, h3, p {{ margin-top: 0; }}
    h1 {{ font-size: clamp(2.3rem, 5vw, 4.6rem); margin-bottom: 10px; }}
    h2 {{ font-size: 1.25rem; margin-bottom: 0; }}
    p {{ color: var(--muted); line-height: 1.55; }}
    .badge-row, .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .badge, .chip {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.68);
      font-size: 0.82rem;
    }}
    .health-good {{ color: var(--good); }}
    .health-bad {{ color: var(--bad); }}
    .health-warn {{ color: var(--warn); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-top: 20px;
    }}
    .metric, .mini-card {{
      padding: 14px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.65);
    }}
    .metric span, .mini-card span {{
      display: block;
      color: var(--muted);
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    .metric strong, .mini-card strong {{
      display: block;
      margin-top: 8px;
      font-size: 1.45rem;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
    }}
    .stack {{ display: grid; gap: 18px; }}
    .grid-2, .grid-3, .grid-4 {{
      display: grid;
      gap: 12px;
    }}
    .grid-2 {{ grid-template-columns: repeat(2, 1fr); }}
    .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
    .grid-4 {{ grid-template-columns: repeat(4, 1fr); }}
    .bar {{
      height: 10px;
      border-radius: 999px;
      background: rgba(33, 24, 17, 0.08);
      overflow: hidden;
      margin-top: 10px;
    }}
    .bar > div {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }}
    .list, .timeline, .heatmap, .plain-list {{
      margin: 0;
      padding-left: 18px;
    }}
    .list li, .timeline li, .heatmap li, .plain-list li {{
      margin-bottom: 8px;
      color: var(--muted);
    }}
    .source-card, .report-card, .trace-card {{
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.6);
      padding: 16px;
    }}
    .sources-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .toolbar input, .toolbar select {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.82);
      padding: 12px 14px;
      font: inherit;
      color: var(--ink);
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }}
    button {{
      cursor: pointer;
      font: inherit;
      color: inherit;
    }}
    .tab {{
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.68);
      padding: 8px 12px;
    }}
    .tab.active {{
      background: var(--ink);
      color: #fff;
      border-color: var(--ink);
    }}
    .muted {{ color: var(--muted); }}
    .pill-good {{ color: var(--good); }}
    .pill-bad {{ color: var(--bad); }}
    pre {{
      margin: 12px 0 0;
      max-height: 320px;
      overflow: auto;
      border-radius: 14px;
      padding: 12px;
      background: rgba(36, 28, 19, 0.92);
      color: #efe7db;
      font-size: 0.84rem;
    }}
    .empty {{
      border: 1px dashed var(--line);
      border-radius: 16px;
      padding: 18px;
      text-align: center;
      color: var(--muted);
      background: rgba(255,255,255,0.45);
    }}
    @media (max-width: 1120px) {{
      .hero, .layout, .grid-4, .grid-3, .grid-2, .sources-grid, .metrics {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 780px) {{
      .shell {{ padding: 16px; }}
      .hero, .layout, .grid-4, .grid-3, .grid-2, .sources-grid, .metrics, .toolbar {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="card">
        <span class="kicker">Next Session Outlook</span>
        <h1 id="hero-target"></h1>
        <p id="hero-summary"></p>
        <div class="badge-row">
          <div class="badge" id="health-badge"></div>
          <div class="badge" id="direction-badge"></div>
          <div class="badge" id="score-badge"></div>
          <div class="badge" id="stat-badge"></div>
          <div class="badge" id="backend-badge"></div>
        </div>
        <div class="metrics">
          <div class="metric"><span>Prediction Date</span><strong id="metric-date"></strong></div>
          <div class="metric"><span>Next Session</span><strong id="metric-next-session"></strong></div>
          <div class="metric"><span>Confidence</span><strong id="metric-confidence"></strong></div>
          <div class="metric"><span>Evidence Used</span><strong id="metric-sources"></strong></div>
        </div>
      </div>
      <div class="card">
        <span class="kicker">Run Health</span>
        <h2>Trust and Evidence State</h2>
        <div class="grid-2" style="margin-top:14px;">
          <div class="mini-card"><span>Health</span><strong id="run-health"></strong></div>
          <div class="mini-card"><span>Gate Failures</span><strong id="gate-count"></strong></div>
          <div class="mini-card"><span>Distinct Providers</span><strong id="provider-count"></strong></div>
          <div class="mini-card"><span>Proxy Usage</span><strong id="proxy-count"></strong></div>
          <div class="mini-card"><span>Expected Return</span><strong id="expected-return"></strong></div>
          <div class="mini-card"><span>Expected Volatility</span><strong id="expected-volatility"></strong></div>
        </div>
        <div class="chip-row" id="gate-chip-row"></div>
      </div>
    </section>

    <div class="layout">
      <div class="stack">
        <section class="card">
          <span class="kicker">Market Snapshot</span>
          <h2>Live Context Board</h2>
          <div class="grid-4" id="snapshot-grid" style="margin-top:14px;"></div>
        </section>

        <section class="card">
          <span class="kicker">Confidence Breakdown</span>
          <h2>Why Confidence Landed Here</h2>
          <div class="grid-3" id="confidence-grid" style="margin-top:14px;"></div>
        </section>

        <section class="card">
          <span class="kicker">Category Contributions</span>
          <h2>Weight and Score Contribution</h2>
          <div class="grid-2" id="category-grid" style="margin-top:14px;"></div>
        </section>

        <section class="card">
          <span class="kicker">Decision Trace</span>
          <h2>Raw Evidence to Final Score</h2>
          <div class="grid-2" id="trace-grid" style="margin-top:14px;"></div>
        </section>

        <section class="card">
          <span class="kicker">Challenge Layer</span>
          <h2>Conflict Map and Why Not Another Label</h2>
          <div class="grid-2" style="margin-top:14px;">
            <div class="report-card">
              <span class="kicker">Challenge Summary</span>
              <p id="challenge-summary"></p>
              <ul class="plain-list" id="challenge-list"></ul>
            </div>
            <div class="report-card">
              <span class="kicker">Counterfactuals</span>
              <ul class="plain-list" id="counterfactual-list"></ul>
            </div>
          </div>
        </section>

        <section class="card">
          <span class="kicker">Agent Reports</span>
          <h2>Category Research</h2>
          <div class="chips" id="report-tabs"></div>
          <div class="report-card" id="report-panel"></div>
        </section>

        <section class="card">
          <span class="kicker">Source Subagents</span>
          <h2>Scored Source Reports</h2>
          <div class="grid-2" id="source-agent-grid" style="margin-top:14px;"></div>
        </section>

        <section class="card">
          <span class="kicker">Evidence Explorer</span>
          <h2>Filter and Inspect Sources</h2>
          <div class="toolbar">
            <input id="search-input" type="search" placeholder="Search title, summary, publisher, or feature tags" />
            <select id="sort-select">
              <option value="recent">Most Recent</option>
              <option value="bullish">Most Bullish</option>
              <option value="bearish">Most Bearish</option>
              <option value="quality">Highest Quality</option>
            </select>
          </div>
          <div class="chips" id="category-filters"></div>
          <div class="sources-grid" id="sources-grid"></div>
        </section>
      </div>

      <div class="stack">
        <section class="card">
          <span class="kicker">Data Quality</span>
          <h2>Validation and Penalties</h2>
          <div class="grid-2" id="quality-grid" style="margin-top:14px;"></div>
          <div class="grid-2" id="penalty-grid" style="margin-top:12px;"></div>
        </section>

        <section class="card">
          <span class="kicker">Freshness Heatmap</span>
          <h2>Evidence Freshness</h2>
          <ul class="heatmap" id="freshness-list"></ul>
        </section>

        <section class="card">
          <span class="kicker">Duplicate Narrative Detector</span>
          <h2>Narrative Compression</h2>
          <ul class="plain-list" id="duplicate-list"></ul>
        </section>

        <section class="card">
          <span class="kicker">Proxy Usage</span>
          <h2>Fallback and Proxy Panel</h2>
          <ul class="plain-list" id="proxy-list"></ul>
        </section>

        <section class="card">
          <span class="kicker">Diagnostics</span>
          <h2>Provider and Run Diagnostics</h2>
          <div class="grid-2" id="diagnostic-grid" style="margin-top:14px;"></div>
          <ul class="plain-list" id="fetch-list" style="margin-top:12px;"></ul>
        </section>

        <section class="card">
          <span class="kicker">Warnings</span>
          <ul class="plain-list" id="warning-list"></ul>
        </section>

        <section class="card">
          <span class="kicker">Payload</span>
          <details>
            <summary>Inspect raw JSON</summary>
            <pre id="payload-pre"></pre>
          </details>
        </section>
      </div>
    </div>
  </div>

  <script>
    const data = {payload};
    const state = {{ category: "all", sort: "recent", search: "", activeReport: "economic" }};

    const reportMap = {{
      economic: {{ title: "Economic Research", summary: data.economic_report }},
      political: {{ title: "Political Policy", summary: data.political_report }},
      social: {{ title: "Social Sentiment", summary: data.social_report }},
      market: {{ title: "Market Microstructure", summary: data.market_context_report }},
    }};

    function escapeHtml(value) {{
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }}

    function formatPct(value) {{
      const num = Number(value || 0);
      return `${{num >= 0 ? "+" : ""}}${{num.toFixed(2)}}%`;
    }}

    function formatHealthClass(value) {{
      if (value === "HEALTHY") return "health-good";
      if (value === "LOW_TRUST") return "health-bad";
      return "health-warn";
    }}

    function renderHero() {{
      document.getElementById("hero-target").textContent = data.target;
      document.getElementById("hero-summary").textContent = data.summary;
      document.getElementById("metric-date").textContent = data.prediction_date;
      document.getElementById("metric-next-session").textContent = data.next_session_date || "n/a";
      document.getElementById("metric-confidence").textContent = `${{Number(data.confidence || 0).toFixed(1)}}%`;
      document.getElementById("metric-sources").textContent = String((data.sources || []).length);
      document.getElementById("run-health").innerHTML = `<span class="${{formatHealthClass(data.run_health)}}">${{escapeHtml(data.run_health)}}</span>`;
      document.getElementById("gate-count").textContent = String((data.data_quality_summary?.gate_failures || []).length);
      document.getElementById("provider-count").textContent = String(data.data_quality_summary?.distinct_provider_count || 0);
      document.getElementById("proxy-count").textContent = String((data.used_proxies || []).length);
      document.getElementById("health-badge").innerHTML = `Run Health <strong class="${{formatHealthClass(data.run_health)}}">${{escapeHtml(data.run_health)}}</strong>`;
      document.getElementById("direction-badge").innerHTML = `Direction <strong>${{escapeHtml(data.prediction_label)}}</strong>`;
      document.getElementById("score-badge").innerHTML = `Final Score <strong>${{Number(data.final_score || 0).toFixed(3)}}</strong>`;
      document.getElementById("stat-badge").innerHTML = `Stat Engine <strong>${{escapeHtml(data.statistical_engine_status || "n/a")}}</strong>`;
      document.getElementById("backend-badge").innerHTML = `Backend <strong>${{escapeHtml(data.backend_diagnostics?.provider || "rule_based")}}</strong>`;
      document.getElementById("expected-return").textContent = formatPct((data.expected_return || 0) * 100);
      document.getElementById("expected-volatility").textContent = formatPct((data.expected_volatility || 0) * 100);
      const gates = data.data_quality_summary?.gate_failures || [];
      document.getElementById("gate-chip-row").innerHTML = gates.length
        ? gates.map(item => `<div class="chip">${{escapeHtml(item)}}</div>`).join("")
        : `<div class="chip">No trust gate failures</div>`;
    }}

    function renderSnapshot() {{
      const root = document.getElementById("snapshot-grid");
      const entries = Object.entries(data.market_snapshot?.series || {{}});
      if (!entries.length) {{
        root.innerHTML = `<div class="empty">Live market snapshot data was unavailable for this run.</div>`;
        return;
      }}
      root.innerHTML = entries.map(([name, values]) => `
        <div class="mini-card">
          <span>${{escapeHtml(name)}}</span>
          <strong>${{Number(values.latest || 0).toFixed(2)}}</strong>
          <div class="muted">${{formatPct(values.pct_change)}} · ${{escapeHtml(values.provider || "unknown")}}</div>
        </div>
      `).join("");
    }}

    function renderConfidence() {{
      const breakdown = data.confidence_breakdown || {{}};
      const fields = [
        ["Signal Strength", breakdown.signal_strength],
        ["Source Diversity", breakdown.source_diversity],
        ["Freshness", breakdown.freshness],
        ["Agreement", breakdown.agreement],
        ["Market Completeness", breakdown.market_data_completeness],
        ["Fallback Burden", breakdown.fallback_proxy_burden],
      ];
      document.getElementById("confidence-grid").innerHTML = fields.map(([label, value]) => `
        <div class="mini-card">
          <span>${{escapeHtml(label)}}</span>
          <strong>${{Number(value || 0).toFixed(1)}}%</strong>
          <div class="bar"><div style="width:${{Math.max(0, Math.min(100, Number(value || 0)))}}%"></div></div>
        </div>
      `).join("");
    }}

    function renderCategoryContributions() {{
      const weights = data.category_weights || {{}};
      const categories = [
        ["economic", "Economic"],
        ["political", "Political"],
        ["social", "Social"],
        ["market", "Market Microstructure"],
      ];
      document.getElementById("category-grid").innerHTML = categories.map(([key, label]) => {{
        const report = data[`${{key}}_report`] || "";
        const weight = Number(weights[key] || 0);
        const scoreMatch = (data.decision_trace || []).find(step => step.stage === "category_scoring");
        return `
          <div class="report-card">
            <span class="kicker">${{escapeHtml(label)}}</span>
            <div class="muted">Weight ${{weight.toFixed(2)}}${{scoreMatch ? " · included in weighted trace" : ""}}</div>
            <p>${{escapeHtml(report)}}</p>
          </div>
        `;
      }}).join("");
    }}

    function renderDecisionTrace() {{
      const trace = data.decision_trace || [];
      document.getElementById("trace-grid").innerHTML = trace.length
        ? trace.map(step => `
          <div class="trace-card">
            <span class="kicker">${{escapeHtml(step.stage)}}</span>
            <div><strong>${{escapeHtml(step.value ?? "")}}</strong></div>
            <p>${{escapeHtml(step.summary)}}</p>
            <div class="muted">${{(step.references || []).map(escapeHtml).join(" · ")}}</div>
          </div>
        `).join("")
        : `<div class="empty">No decision trace available.</div>`;
    }}

    function renderChallenge() {{
      const challenge = data.challenge_agent_report || {{}};
      document.getElementById("challenge-summary").textContent = challenge.summary || "No challenge report available.";
      const challengeItems = [
        "Overconfident categories: " + ((challenge.overconfident_categories || []).join(", ") || "none"),
        "Duplicate narratives: " + ((challenge.duplicate_narratives || []).join(", ") || "none"),
        "Proxy risks: " + ((challenge.proxy_risks || []).join(", ") || "none"),
        "Weak confirmation: " + ((challenge.weak_confirmation || []).join(", ") || "none"),
      ];
      document.getElementById("challenge-list").innerHTML = challengeItems.map(item => `<li>${{escapeHtml(item)}}</li>`).join("");
      const counterfactuals = [
        "Why not Neutral: " + ((data.data_quality_summary?.gate_failures || []).length ? "trust gates failed or quality was capped" : "directional separation exceeded neutral band"),
        "Why not Up: " + ((data.prediction_label === "UP") ? "selected as final label" : "penalties, challenge review, or category mix weakened upside conviction"),
        "Why not Down: " + ((data.prediction_label === "DOWN") ? "selected as final label" : "penalties, challenge review, or category mix weakened downside conviction"),
      ];
      document.getElementById("counterfactual-list").innerHTML = counterfactuals.map(item => `<li>${{escapeHtml(item)}}</li>`).join("");
    }}

    function renderReports() {{
      const tabs = document.getElementById("report-tabs");
      tabs.innerHTML = Object.entries(reportMap).map(([key, report]) => `
        <button class="tab ${{state.activeReport === key ? "active" : ""}}" data-report="${{key}}">${{escapeHtml(report.title)}}</button>
      `).join("");
      tabs.querySelectorAll("[data-report]").forEach(button => {{
        button.addEventListener("click", () => {{
          state.activeReport = button.dataset.report;
          renderReports();
        }});
      }});
      const report = reportMap[state.activeReport];
      document.getElementById("report-panel").innerHTML = `
        <span class="kicker">${{escapeHtml(report.title)}}</span>
        <p>${{escapeHtml(report.summary || "No report available.")}}</p>
      `;
    }}

    function renderSourceAgents() {{
      const reports = data.source_agent_reports || [];
      document.getElementById("source-agent-grid").innerHTML = reports.length
        ? reports.map(report => `
          <div class="source-card">
            <span class="kicker">${{escapeHtml(report.source)}}</span>
            <div><strong>${{Number(report.score || 0).toFixed(2)}}</strong> · confidence ${{Number(report.source_confidence || 0).toFixed(2)}}</div>
            <p>${{escapeHtml(report.summary)}}</p>
            <div class="muted">Reliability ${{Number(report.source_reliability || 0).toFixed(2)}} · ${{escapeHtml(report.source_regime_fit || "standard")}}</div>
          </div>
        `).join("")
        : `<div class="empty">No source reports available.</div>`;
    }}

    function renderQuality() {{
      const summary = data.data_quality_summary || {{}};
      const qualityRows = [
        ["Valid Items", summary.valid_item_count],
        ["Rejected Items", summary.rejected_item_count],
        ["Duplicates Removed", summary.duplicate_item_count],
        ["Stale Items", summary.stale_item_count],
        ["Malformed Items", summary.malformed_item_count],
        ["Average Quality", Number(summary.average_quality_score || 0).toFixed(3)],
      ];
      document.getElementById("quality-grid").innerHTML = qualityRows.map(([label, value]) => `
        <div class="mini-card"><span>${{escapeHtml(label)}}</span><strong>${{escapeHtml(value)}}</strong></div>
      `).join("");
      const penalties = Object.entries(data.quality_penalties || {{}});
      document.getElementById("penalty-grid").innerHTML = penalties.length
        ? penalties.map(([label, value]) => `
          <div class="mini-card">
            <span>${{escapeHtml(label.replaceAll("_", " "))}}</span>
            <strong>${{Number(value || 0).toFixed(3)}}</strong>
          </div>
        `).join("")
        : `<div class="empty">No quality penalties recorded.</div>`;
    }}

    function renderFreshness() {{
      const items = [...(data.sources || [])]
        .sort((a, b) => (b.freshness_score || 0) - (a.freshness_score || 0))
        .slice(0, 8);
      document.getElementById("freshness-list").innerHTML = items.length
        ? items.map(item => `<li>${{escapeHtml(item.title)}} · freshness ${{Number((item.freshness_score || 0) * 100).toFixed(0)}}%</li>`).join("")
        : `<li>No evidence items available.</li>`;
    }}

    function renderDuplicates() {{
      const count = data.data_quality_summary?.duplicate_item_count || 0;
      const challenge = data.challenge_agent_report || {{}};
      const items = count
        ? [`${{count}} duplicate items were removed before scoring.`, ...(challenge.duplicate_narratives || [])]
        : ["No duplicate narratives were detected."];
      document.getElementById("duplicate-list").innerHTML = items.map(item => `<li>${{escapeHtml(item)}}</li>`).join("");
    }}

    function renderProxies() {{
      const proxies = data.used_proxies || [];
      const items = proxies.length
        ? proxies.map(item => `Proxy instrument used for ${{item}}`)
        : ["No proxy instruments used in this run."];
      document.getElementById("proxy-list").innerHTML = items.map(item => `<li>${{escapeHtml(item)}}</li>`).join("");
    }}

    function renderDiagnostics() {{
      const diag = data.source_diagnostics || {{}};
      const rows = [
        ["Quote Provider", diag.quote_provider_status || "unknown"],
        ["News Provider", diag.news_provider_status || "unknown"],
        ["TradingAgents Adapter", diag.tradingagents_adapter || "unknown"],
        ["RSS Provider", diag.rss_provider || "unknown"],
        ["Network Reachability", diag.network_error ? "error" : "ok"],
        ["Backend Used", data.backend_diagnostics?.backend_used ? "yes" : "fallback"],
      ];
      document.getElementById("diagnostic-grid").innerHTML = rows.map(([label, value]) => `
        <div class="mini-card"><span>${{escapeHtml(label)}}</span><strong>${{escapeHtml(value)}}</strong></div>
      `).join("");
      const fetchResults = diag.fetch_results || [];
      document.getElementById("fetch-list").innerHTML = fetchResults.length
        ? fetchResults.slice(0, 10).map(item => `
          <li>${{escapeHtml(item.fetch_group)}} · ${{escapeHtml(item.provider_name)}} · status ${{escapeHtml(item.status)}} · latency ${{escapeHtml(item.latency_ms)}}ms</li>
        `).join("")
        : "<li>No fetch diagnostics available.</li>";
    }}

    function renderWarnings() {{
      const warnings = data.warnings?.length ? data.warnings : ["No pipeline warnings."];
      document.getElementById("warning-list").innerHTML = warnings.map(item => `<li>${{escapeHtml(item)}}</li>`).join("");
    }}

    function buildFilters() {{
      const categories = ["all", ...new Set((data.sources || []).map(source => source.category))];
      const root = document.getElementById("category-filters");
      root.innerHTML = categories.map(category => `
        <button class="tab ${{state.category === category ? "active" : ""}}" data-category="${{category}}">
          ${{escapeHtml(category)}}
        </button>
      `).join("");
      root.querySelectorAll("[data-category]").forEach(button => {{
        button.addEventListener("click", () => {{
          state.category = button.dataset.category;
          buildFilters();
          renderSources();
        }});
      }});
    }}

    function filteredSources() {{
      let items = [...(data.sources || [])];
      if (state.category !== "all") {{
        items = items.filter(source => source.category === state.category);
      }}
      if (state.search.trim()) {{
        const search = state.search.trim().toLowerCase();
        items = items.filter(source =>
          [source.title, source.summary, source.source, source.category, source.instrument]
            .some(value => String(value || "").toLowerCase().includes(search))
        );
      }}
      if (state.sort === "bullish") {{
        items.sort((a, b) => (b.impact_score || 0) - (a.impact_score || 0));
      }} else if (state.sort === "bearish") {{
        items.sort((a, b) => (a.impact_score || 0) - (b.impact_score || 0));
      }} else if (state.sort === "quality") {{
        items.sort((a, b) => (b.quality_score || 0) - (a.quality_score || 0));
      }} else {{
        items.sort((a, b) => String(b.published_at || "").localeCompare(String(a.published_at || "")));
      }}
      return items;
    }}

    function renderSources() {{
      const items = filteredSources();
      const root = document.getElementById("sources-grid");
      root.innerHTML = items.length
        ? items.map(source => `
          <div class="source-card">
            <span class="kicker">${{escapeHtml(source.category)}} · ${{escapeHtml(source.impact)}}</span>
            <h3>${{escapeHtml(source.title)}}</h3>
            <div class="muted">${{escapeHtml(source.source)}} · quality ${{Number(source.quality_score || 0).toFixed(2)}} · freshness ${{Number((source.freshness_score || 0) * 100).toFixed(0)}}%</div>
            <p>${{escapeHtml(source.summary || "No summary available.")}}</p>
          </div>
        `).join("")
        : `<div class="empty">No evidence matched the current filters.</div>`;
    }}

    function bindControls() {{
      document.getElementById("search-input").addEventListener("input", event => {{
        state.search = event.target.value;
        renderSources();
      }});
      document.getElementById("sort-select").addEventListener("change", event => {{
        state.sort = event.target.value;
        renderSources();
      }});
      document.getElementById("payload-pre").textContent = JSON.stringify(data, null, 2);
    }}

    renderHero();
    renderSnapshot();
    renderConfidence();
    renderCategoryContributions();
    renderDecisionTrace();
    renderChallenge();
    renderReports();
    renderSourceAgents();
    renderQuality();
    renderFreshness();
    renderDuplicates();
    renderProxies();
    renderDiagnostics();
    renderWarnings();
    buildFilters();
    renderSources();
    bindControls();
  </script>
</body>
</html>
"""
    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")
    return html
