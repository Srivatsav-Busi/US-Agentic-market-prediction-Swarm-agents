import { motion } from "framer-motion";
import { useEffect, useState } from "react";

import { useBrowserRoute } from "./useBrowserRoute";
import { useBackendArtifact } from "./useBackendArtifact";

const WORKFLOW_ROUTES = [
  { path: "/", label: "Overview", eyebrow: "Stage 1", title: "Live Artifact" },
  { path: "/scenario", label: "Scenario Lab", eyebrow: "Stage 2", title: "Stress Test" },
  { path: "/interrogate", label: "Interrogate", eyebrow: "Stage 3", title: "Analyst Q&A" },
  { path: "/archive", label: "Archive", eyebrow: "Stage 4", title: "Reports And Replay" },
];

const INFO_ROUTES = [
  { path: "/about", label: "About", eyebrow: "System", title: "Prediction Backend" },
];

const ROUTES = [...WORKFLOW_ROUTES, ...INFO_ROUTES];

const SCENARIO_PRESETS = [
  { id: "inflation", label: "Inflation Shock", scenario: "CPI reaccelerates and bond yields jump higher." },
  { id: "pivot", label: "Dovish Pivot", scenario: "The Fed signals an earlier and deeper cutting cycle." },
  { id: "growth", label: "Growth Scare", scenario: "Mega-cap earnings disappoint and growth expectations reset lower." },
  { id: "liquidity", label: "Liquidity Relief", scenario: "Credit spreads tighten and risk appetite broadens across sectors." },
];

const ANALYSTS = [
  { id: "economic", label: "Economic Desk", description: "Rates, inflation, macro surprise, growth sensitivity." },
  { id: "political", label: "Policy Desk", description: "Regulation, fiscal posture, election and policy shocks." },
  { id: "social", label: "Sentiment Desk", description: "Narrative flow, retail tone, headline crowding." },
  { id: "market", label: "Market Desk", description: "Breadth, positioning, cross-asset confirmation, tape quality." },
];

const BACKEND_PHASES = [
  {
    eyebrow: "Phase 1",
    title: "Source Collection",
    tags: ["inputs", "quality"],
    summary: "Live quotes, history, sector inputs, and news are gathered and quality-scored before the run can continue.",
    details: [
      "The backend pulls cross-asset market quotes, historical series, optional sector snapshots, TradingAgents feeds, and direct Google News RSS inputs for the target session.",
      "Every item is normalized and checked for freshness, malformed timestamps, duplicate clusters, proxy use, and provider diversity before it can influence the forecast.",
      "This stage produces the market snapshot, historical context, validated evidence items, warnings, and source diagnostics that downstream stages consume.",
    ],
  },
  {
    eyebrow: "Phase 2",
    title: "Graph-First Ingestion",
    tags: ["graph", "features"],
    summary: "Validated evidence is mapped into a knowledge graph so the system reasons over relationships, not just raw text blobs.",
    details: [
      "The pipeline constructs an ingestion graph from evidence, instruments, providers, features, and run context before the final prediction stack executes.",
      "A graph quality layer adjusts evidence trust based on corroboration, contradiction, staleness clusters, and source monoculture risks.",
      "Graph feature vectors and temporal deltas are extracted here, giving the decision stack access to relationship-aware signals and changes versus prior runs.",
    ],
  },
  {
    eyebrow: "Phase 3",
    title: "Swarm Simulation",
    tags: ["reasoning", "simulation"],
    summary: "A synthetic market swarm debates the environment over multiple rounds and emits crowd-dynamics signals.",
    details: [
      "The backend creates multiple personas with different archetypes, biases, and activity configurations, then seeds the environment with initial posts.",
      "Those agents interact over timed rounds, generating consensus, conflict, stance drift, and thematic pressure rather than a single monolithic opinion.",
      "The swarm output becomes a narrative pressure layer that complements the graph and statistical signals instead of replacing them.",
    ],
  },
  {
    eyebrow: "Phase 4",
    title: "Specialist Agents",
    tags: ["agents", "challenge"],
    summary: "Specialist agents refine evidence, reason across domains, and then challenge the emerging thesis before publish.",
    details: [
      "Source agents sharpen and summarize raw evidence into cleaner domain-ready inputs for the rest of the reasoning stack.",
      "Research agents then evaluate the run from market, macro, political, and sentiment perspectives to surface convergences and contradictions.",
      "A dedicated challenge pass attempts to invalidate the draft thesis so confidence can be reduced when the call is not robust.",
    ],
  },
  {
    eyebrow: "Phase 5",
    title: "Forecast Synthesis",
    tags: ["decision", "calibration"],
    summary: "Graph priors, swarm diagnostics, specialist outputs, and quantitative overlays are fused into one forecast artifact.",
    details: [
      "The synthesizer merges live features, graph priors, temporal deltas, specialist reports, challenge output, and forecasting overlays into a final judgment.",
      "This is where the prediction label, confidence, top drivers, market projection, confidence notes, and run-health framing are assembled.",
      "The result is not just a label. It is a structured artifact with diagnostics, traces, evidence context, graph summaries, and simulation state.",
    ],
  },
  {
    eyebrow: "Phase 6",
    title: "Persistence And Graph Build",
    tags: ["publish", "neo4j"],
    summary: "The finished artifact is serialized, stored locally, served to the UI, and can then be projected into Neo4j Aura.",
    details: [
      "The backend writes the completed run to JSON and HTML, persists core data into SQLite, and updates runtime state files the UI consumes.",
      "Graph lifecycle metadata is also tracked locally through graph project, task, and snapshot tables so operators can inspect queue/build history.",
      "A post-run graph build can push the explainability graph into Neo4j Aura, making the run available for retrieval, replay, and external graph inspection.",
    ],
  },
];

const BACKEND_PILLARS = [
  {
    title: "Inputs",
    detail: "Cross-asset quotes, history, macro/news feeds, sector data, and run-time fallbacks.",
  },
  {
    title: "Reasoning",
    detail: "Graph-first feature extraction, swarm simulation, source agents, research agents, challenge pass.",
  },
  {
    title: "Decision",
    detail: "Forecast synthesis merges statistical overlays, graph priors, narrative drivers, and calibration notes.",
  },
  {
    title: "Outputs",
    detail: "JSON artifact, static HTML dashboard, SQLite state, graph lifecycle records, optional Aura graph.",
  },
];

const BACKEND_NOTES = [
  "If markets are closed, the backend takes a faster market-closed path and publishes a neutral artifact instead of running the full live reasoning stack.",
  "Run health reflects evidence quality and pipeline completeness, not just the final label. A prediction can be neutral while still carrying partial or degraded health.",
  "Knowledge graph work is split into two layers: graph-aware forecasting inside the run and a post-run graph build for explainability, retrieval, and Neo4j persistence.",
  "The UI reads the latest artifact and graph lifecycle state from the local web server, so operator pages stay aligned with the most recent completed run.",
];

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function formatPercent(value, digits = 1) {
  if (!isFiniteNumber(value)) {
    return "Pending";
  }
  return `${value.toFixed(digits)}%`;
}

function formatRatioPercent(value, digits = 1) {
  if (!isFiniteNumber(value)) {
    return "Pending";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

function formatCompactNumber(value) {
  if (!isFiniteNumber(value)) {
    return "Pending";
  }
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

function formatSignedPercent(value, digits = 2) {
  if (!isFiniteNumber(value)) {
    return "Pending";
  }
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;
}

function formatDate(value) {
  return value || "Pending";
}

function titleize(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function toneClass(value) {
  const normalized = String(value || "").toLowerCase();
  if (["bullish", "positive", "healthy"].includes(normalized)) {
    return "positive";
  }
  if (["bearish", "negative", "degraded"].includes(normalized)) {
    return "negative";
  }
  if (["warning", "partial", "simulated"].includes(normalized)) {
    return "warning";
  }
  return "neutral";
}

function stripInlineMarkdown(value) {
  return String(value || "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

function buildSummaryBrief(summary) {
  const cleaned = stripInlineMarkdown(summary);
  if (!cleaned) {
    return { lead: "", sections: [] };
  }

  const compact = cleaned
    .replace(/\bConfidence\s*:\s*[^.]+(?:\.)?/i, "")
    .replace(/\bEpistemic Status\s*:\s*[^.]+(?:\.)?/i, "")
    .replace(/\s+/g, " ")
    .trim();

  const normalized = compact
    .replace(/\bJudgment Rationale\s*:/gi, "Judgment Rationale:")
    .replace(/\bExpected Microstructure\b/gi, "|Expected Microstructure|")
    .replace(/\bTactical Implication\b/gi, "|Tactical Implication|")
    .replace(/\bRisk Adjustment\b/gi, "|Risk Adjustment|");

  const [introPart, ...labeledParts] = normalized.split("|");
  const lead = introPart
    .replace(/\bJudgment Rationale\s*:/i, "")
    .trim();

  const numberedMatches = [...lead.matchAll(/(\d+\.\s.*?)(?=\s\d+\.\s|$)/g)].map((match) => match[1].trim());
  const leadWithoutNumbered = lead.replace(/(\d+\.\s.*?)(?=\s\d+\.\s|$)/g, "").trim();

  const sections = [];
  if (leadWithoutNumbered) {
    sections.push({ title: "Executive View", body: leadWithoutNumbered });
  }

  numberedMatches.forEach((item) => {
    const titleMatch = item.match(/^\d+\.\s([^.!?]+)(.*)$/);
    if (!titleMatch) {
      sections.push({ title: "Key Point", body: item });
      return;
    }
    const title = titleMatch[1].trim();
    const body = titleMatch[2].trim() || item.trim();
    sections.push({ title, body });
  });

  labeledParts.forEach((part) => {
    const trimmed = part.trim();
    if (!trimmed) {
      return;
    }
    const separatorIndex = trimmed.indexOf(":");
    if (separatorIndex === -1) {
      sections.push({ title: trimmed, body: "" });
      return;
    }
    sections.push({
      title: trimmed.slice(0, separatorIndex).trim(),
      body: trimmed.slice(separatorIndex + 1).trim(),
    });
  });

  return {
    lead: sections[0]?.title === "Executive View" ? sections[0].body : leadWithoutNumbered || cleaned,
    sections: sections[0]?.title === "Executive View" ? sections.slice(1) : sections,
  };
}

function scoreHealthTone(health) {
  switch ((health || "").toUpperCase()) {
    case "HEALTHY":
      return "positive";
    case "PARTIAL":
    case "SIMULATED":
      return "warning";
    case "DEGRADED":
    case "LOW_TRUST":
      return "negative";
    default:
      return "neutral";
  }
}

function ShellCard({ title, eyebrow, children, className = "" }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      className={`shell-card ${className}`.trim()}
    >
      {eyebrow ? <p className="panel-eyebrow">{eyebrow}</p> : null}
      {title ? <h2 className="panel-title">{title}</h2> : null}
      {children}
    </motion.section>
  );
}

function ProjectionChart({ projection, forecastSummary }) {
  const basePoints = projection?.base || [];
  if (!projection || typeof projection !== "object" || basePoints.length === 0) {
    return (
      <ShellCard title="Market Pressure Projection" eyebrow="30D Path">
        <p className="muted-copy">No projection data available yet.</p>
      </ShellCard>
    );
  }

  const width = 800;
  const height = 320;
  const paddingLeft = 56;
  const paddingRight = 28;
  const paddingTop = 32;
  const paddingBottom = 36;

  const values = [
    ...basePoints.map((point) => point.predicted_price),
    ...(projection.bull || []).map((point) => point.predicted_price),
    ...(projection.bear || []).map((point) => point.predicted_price),
    ...(projection.confidence_band?.lower || []),
    ...(projection.confidence_band?.upper || []),
  ].filter(isFiniteNumber);

  const min = Math.min(...values);
  const max = Math.max(...values);
  const margin = (max - min || 1) * 0.1;
  const yMin = min - margin;
  const yMax = max + margin;
  const maxLength = Math.max(basePoints.length, (projection.bull || []).length, (projection.bear || []).length);

  const scaleX = (index) => paddingLeft + (index / Math.max(maxLength - 1, 1)) * (width - paddingLeft - paddingRight);
  const scaleY = (value) => paddingTop + (1 - (value - yMin) / (yMax - yMin || 1)) * (height - paddingTop - paddingBottom);

  const buildPath = (points, key) => {
    if (!points.length) {
      return "";
    }
    let path = `M ${scaleX(0)} ${scaleY(points[0][key])}`;
    for (let index = 1; index < points.length; index += 1) {
      const x1 = scaleX(index - 1);
      const y1 = scaleY(points[index - 1][key]);
      const x2 = scaleX(index);
      const y2 = scaleY(points[index][key]);
      const cx = (x1 + x2) / 2;
      path += ` C ${cx} ${y1}, ${cx} ${y2}, ${x2} ${y2}`;
    }
    return path;
  };

  const bandPath = [
    ...(projection.confidence_band?.upper || []).map((value, index) =>
      `${index === 0 ? "M" : "L"} ${scaleX(index)} ${scaleY(value)}`,
    ),
    ...[...(projection.confidence_band?.lower || [])]
      .map((value, index) => ({ value, index }))
      .reverse()
      .map(({ value, index }) => `L ${scaleX(index)} ${scaleY(value)}`),
    "Z",
  ].join(" ");

  const yTicks = Array.from({ length: 5 }, (_, index) => yMin + (index * (yMax - yMin)) / 4);

  return (
    <ShellCard title="Market Pressure Projection" eyebrow="30D Path">
      <div className="projection-shell">
        <svg viewBox={`0 0 ${width} ${height}`} className="projection-svg">
          <defs>
            <linearGradient id="bandGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent-strong)" stopOpacity="0.18" />
              <stop offset="100%" stopColor="var(--accent-strong)" stopOpacity="0.03" />
            </linearGradient>
            <linearGradient id="baseGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#7f9cff" />
              <stop offset="100%" stopColor="#f8f7f1" />
            </linearGradient>
          </defs>

          {yTicks.map((tick) => (
            <g key={tick}>
              <line x1={paddingLeft} y1={scaleY(tick)} x2={width - paddingRight} y2={scaleY(tick)} className="chart-gridline" />
              <text x={paddingLeft - 10} y={scaleY(tick) + 4} textAnchor="end" className="chart-axis-label">
                {tick.toFixed(0)}
              </text>
            </g>
          ))}

          <path d={bandPath} fill="url(#bandGradient)" />
          <path d={buildPath(projection.bear || [], "predicted_price")} fill="none" stroke="var(--negative)" strokeWidth="1.5" strokeDasharray="4 4" opacity="0.6" />
          <path d={buildPath(projection.bull || [], "predicted_price")} fill="none" stroke="var(--positive)" strokeWidth="1.5" strokeDasharray="4 4" opacity="0.6" />
          <path d={buildPath(basePoints, "predicted_price")} fill="none" stroke="url(#baseGradient)" strokeWidth="3" strokeLinecap="round" />
          <circle cx={scaleX(0)} cy={scaleY(basePoints[0].predicted_price)} r="4" fill="var(--accent-strong)" />
        </svg>
      </div>
      <div className="metric-grid two-up">
        <div className="metric-tile">
          <span className="metric-label">Expected 30D Return</span>
          <strong>{formatSignedPercent(forecastSummary?.expected_return_30d)}</strong>
        </div>
        <div className="metric-tile">
          <span className="metric-label">Expected 30D Volatility</span>
          <strong>{formatRatioPercent(forecastSummary?.expected_volatility_30d)}</strong>
        </div>
      </div>
    </ShellCard>
  );
}

function StageRail({ currentPath, status, data, historyCount, interviewReady }) {
  const isSimulated = String(data.run_health || "").toUpperCase() === "SIMULATED";
  const stages = WORKFLOW_ROUTES.map((route) => {
    let state = "pending";
    if (route.path === "/") {
      state = status === "ready" ? "complete" : status === "loading" ? "active" : "pending";
    } else if (route.path === "/scenario") {
      state = isSimulated ? "complete" : "pending";
    } else if (route.path === "/interrogate") {
      state = interviewReady ? "complete" : "pending";
    } else if (route.path === "/archive") {
      state = historyCount ? "complete" : "pending";
    }
    if (route.path === currentPath) {
      state = "active";
    }
    return { ...route, state };
  });

  return (
    <ShellCard title="Mission Rail" eyebrow="Workflow">
      <div className="rail-list">
        {stages.map((stage) => (
          <div key={stage.path} className={`rail-item rail-${stage.state}`}>
            <div className="rail-marker">
              <span>{stage.eyebrow}</span>
            </div>
            <div className="rail-copy">
              <strong>{stage.label}</strong>
              <p>{stage.title}</p>
            </div>
          </div>
        ))}
      </div>
    </ShellCard>
  );
}

function ShellSidebar({ data, status, history, currentPath, interviewReady, navigate, onReset }) {
  return (
    <aside className="shell-sidebar">
      <ShellCard title="Market Pressure Atlas" eyebrow="Finance Console" className="brand-card">
        <p className="brand-copy">
          A staged operator console for live forecast review, scenario injection, analyst interrogation, and report replay.
        </p>
        <div className="status-pills">
          <span className={`status-pill ${scoreHealthTone(data.run_health)}`}>{data.run_health}</span>
          <span className="status-pill neutral">{titleize(status)}</span>
        </div>
      </ShellCard>

      <StageRail
        currentPath={currentPath}
        status={status}
        data={data}
        historyCount={history.length}
        interviewReady={interviewReady}
      />

      <ShellCard title="Quick Actions" eyebrow="Operator Controls">
        <div className="metric-grid two-up">
          <div className="metric-tile">
            <span className="metric-label">Target</span>
            <strong>{data.target || "Pending"}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Model</span>
            <strong>{data.ensemble_diagnostics?.mode || "baseline"}</strong>
          </div>
        </div>
        <div className="button-stack">
          <button className="primary-button" onClick={onReset}>
            Reload Live Feed
          </button>
          <button className="secondary-button" onClick={() => navigate("/scenario")}>
            Open Scenario Lab
          </button>
          {data.analysis_report?.html_path ? (
            <a className="text-link" href={data.analysis_report.html_path} target="_blank" rel="noreferrer">
              Open Analysis Report
            </a>
          ) : null}
          {data.reportPath ? (
            <a className="text-link" href={data.reportPath} target="_blank" rel="noreferrer">
              Open Latest Dashboard
            </a>
          ) : null}
        </div>
      </ShellCard>
    </aside>
  );
}

function Header({ currentPath, navigate, data, status }) {
  const activeRoute = ROUTES.find((route) => route.path === currentPath) || ROUTES[0];

  return (
    <header className="app-header">
      <div>
        <p className="header-kicker">{activeRoute.eyebrow}</p>
        <h1>{activeRoute.title}</h1>
      </div>
      <nav className="route-nav" aria-label="Workflow stages">
        {ROUTES.map((route) => (
          <a
            key={route.path}
            href={route.path}
            className={`route-link ${route.path === currentPath ? "active" : ""}`}
            onClick={(event) => {
              event.preventDefault();
              navigate(route.path);
            }}
          >
            {route.label}
          </a>
        ))}
      </nav>
      <div className="header-status">
        <div>
          <span className="metric-label">Live Snapshot</span>
          <strong>{formatDate(data.prediction_date)}</strong>
        </div>
        <div>
          <span className="metric-label">Frontend State</span>
          <strong>{titleize(status)}</strong>
        </div>
      </div>
    </header>
  );
}

function SwarmSetupPanel({ setup, summary, reporting }) {
  const profiles = setup?.profiles || [];
  const activityConfigs = setup?.activity_configs || [];
  const timeConfig = setup?.time_config || {};
  const seedPosts = setup?.seed_posts || [];
  const isLargeSwarm = reporting?.scale_mode === "large" || reporting?.guardrails?.large_swarm;
  const highlightedAgents = reporting?.agent_activity?.highlighted_agents || [];
  const lowPriorityCount = reporting?.agent_activity?.low_priority_count || 0;
  const clusterComposition = reporting?.cluster_composition || [];
  const dominantSignals = reporting?.dominant_signals || [];

  if (isLargeSwarm) {
    return (
      <ShellCard title="Swarm Setup" eyebrow="Large-Swarm Summary">
        <div className="metric-grid four-up">
          <div className="metric-tile">
            <span className="metric-label">Personas</span>
            <strong>{formatCompactNumber(reporting?.totals?.total_personas || profiles.length || summary?.agent_count || 0)}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Avg Active / Round</span>
            <strong>{reporting?.totals?.average_active_agents_per_round ?? "Pending"}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Participation</span>
            <strong>{formatRatioPercent(reporting?.totals?.participation_rate)}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Seed Posts</span>
            <strong>{formatCompactNumber(reporting?.totals?.seed_post_count || seedPosts.length || 0)}</strong>
          </div>
        </div>
        <div className="status-pills" style={{ marginTop: 14 }}>
          {dominantSignals.map((signal) => (
            <span key={`${signal.label}-${signal.value}`} className={`status-pill ${toneClass(signal.tone)}`}>
              {signal.label}: {typeof signal.value === "number" ? formatRatioPercent(signal.value) : titleize(signal.value)}
            </span>
          ))}
        </div>
        <div className="swarm-summary-grid">
          <div className="stack-list compact-list">
            <div className="stack-row">
              <strong>Cluster composition</strong>
              <p>Largest communities are surfaced directly so the long tail does not dominate the panel.</p>
            </div>
            {clusterComposition.slice(0, 4).map((cluster) => (
              <div key={cluster.cluster_id} className="split-row">
                <div>
                  <strong>{cluster.label}</strong>
                  <p>{cluster.top_archetypes?.join(", ") || "Mixed archetypes"}</p>
                </div>
                <span className="status-pill neutral">
                  {cluster.agent_count} · {formatRatioPercent(cluster.share)}
                </span>
              </div>
            ))}
          </div>
          <div className="stack-list compact-list">
            <div className="stack-row">
              <strong>Active leaders</strong>
              <p>Most active agents are shown explicitly. Low-priority agents are collapsed into an aggregate bucket.</p>
            </div>
            {highlightedAgents.slice(0, 5).map((agent) => (
              <div key={agent.agent_id} className="split-row">
                <div>
                  <strong>{agent.name}</strong>
                  <p>{titleize(agent.cluster)} · {agent.action_count} actions across {agent.rounds_active.length} rounds</p>
                </div>
                <span className={`status-pill ${toneClass(agent.stance_bias)}`}>{agent.stance_bias}</span>
              </div>
            ))}
            <div className="stack-row swarm-muted-row">
              <strong>{lowPriorityCount} low-priority agents collapsed</strong>
              <p>These agents had limited participation and are intentionally summarized rather than listed individually.</p>
            </div>
          </div>
        </div>
      </ShellCard>
    );
  }

  return (
    <ShellCard title="Swarm Setup" eyebrow="Prepared Population">
      <div className="metric-grid three-up">
        <div className="metric-tile">
          <span className="metric-label">Profiles</span>
          <strong>{profiles.length || summary?.agent_count || 0}</strong>
        </div>
        <div className="metric-tile">
          <span className="metric-label">Behavior Configs</span>
          <strong>{activityConfigs.length || 0}</strong>
        </div>
        <div className="metric-tile">
          <span className="metric-label">Seed Posts</span>
          <strong>{seedPosts.length || summary?.seed_post_count || 0}</strong>
        </div>
      </div>
      <div className="metric-grid two-up">
        <div className="metric-tile">
          <span className="metric-label">Round Clock</span>
          <strong>
            {timeConfig.start_hour ?? "?"}:00 / {timeConfig.minutes_per_round ?? "?"}m
          </strong>
        </div>
        <div className="metric-tile">
          <span className="metric-label">Peak Hours</span>
          <strong>{(timeConfig.peak_hours || []).slice(0, 4).join(", ") || "Pending"}</strong>
        </div>
      </div>
      <div className="stack-list compact-list">
        {profiles.slice(0, 4).map((profile) => (
          <div key={profile.agent_id} className="split-row">
            <div>
              <strong>{profile.name}</strong>
              <p>{profile.entity_type} · {profile.archetype}</p>
            </div>
            <span className={`status-pill ${profile.stance_bias === "bearish" ? "negative" : profile.stance_bias === "bullish" ? "positive" : "neutral"}`}>
              {profile.stance_bias}
            </span>
          </div>
        ))}
        {!profiles.length ? <p className="muted-copy">Swarm setup will populate after a forecast or simulation run completes.</p> : null}
      </div>
    </ShellCard>
  );
}

function SwarmRoundsPanel({ rounds, summary, reporting }) {
  const isLargeSwarm = reporting?.scale_mode === "large" || reporting?.guardrails?.large_swarm;

  if (isLargeSwarm) {
    return (
      <ShellCard title="Swarm Runtime" eyebrow="Dominant Signals">
        <div className="metric-grid four-up">
          <div className="metric-tile">
            <span className="metric-label">Rounds</span>
            <strong>{reporting?.totals?.round_count || rounds?.length || summary?.round_count || 0}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Dominant Stance</span>
            <strong>{summary?.dominant_stance || "mixed"}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Consensus</span>
            <strong>{isFiniteNumber(summary?.average_consensus) ? summary.average_consensus.toFixed(2) : "Pending"}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Peak Active</span>
            <strong>{formatCompactNumber(reporting?.totals?.max_active_agents_per_round || 0)}</strong>
          </div>
        </div>
        <div className="swarm-summary-grid">
          <div className="stack-list compact-list">
            <div className="stack-row">
              <strong>Top debated themes</strong>
              <p>Theme ranking combines mention count, breadth across agents, and persistence across rounds.</p>
            </div>
            {(reporting?.top_debated_themes || []).map((theme) => (
              <div key={theme.theme} className="split-row">
                <div>
                  <strong>{titleize(theme.theme)}</strong>
                  <p>{theme.mentions} mentions · {theme.distinct_agents} agents · {theme.round_span} rounds</p>
                </div>
                <span className={`status-pill ${toneClass(theme.stance_balance)}`}>{theme.stance_balance}</span>
              </div>
            ))}
            {!reporting?.top_debated_themes?.length ? <p className="muted-copy">No debated themes were extracted from the current swarm trace.</p> : null}
          </div>
          <div className="stack-list compact-list">
            <div className="stack-row">
              <strong>Round highlights</strong>
              <p>Only the key signal shifts are shown for larger simulations to keep the trace readable.</p>
            </div>
            {(reporting?.round_highlights || []).map((round) => (
              <div key={round.round_index} className="stack-row">
                <div className="split-row">
                  <strong>Round {round.round_index + 1}</strong>
                  <span className={`status-pill ${toneClass(round.dominant_stance)}`}>
                    {round.active_agents} active · {formatRatioPercent(round.participation_rate)}
                  </span>
                </div>
                <p>{round.summary}</p>
                <p className="muted-copy">Themes: {round.top_themes?.length ? round.top_themes.map(titleize).join(", ") : "No dominant theme extracted."}</p>
              </div>
            ))}
          </div>
        </div>
      </ShellCard>
    );
  }

  return (
    <ShellCard title="Swarm Runtime" eyebrow="Round Activity">
      <div className="metric-grid three-up">
        <div className="metric-tile">
          <span className="metric-label">Rounds</span>
          <strong>{rounds?.length || summary?.round_count || 0}</strong>
        </div>
        <div className="metric-tile">
          <span className="metric-label">Dominant Stance</span>
          <strong>{summary?.dominant_stance || "mixed"}</strong>
        </div>
        <div className="metric-tile">
          <span className="metric-label">Consensus</span>
          <strong>{isFiniteNumber(summary?.average_consensus) ? summary.average_consensus.toFixed(2) : "Pending"}</strong>
        </div>
      </div>
      <div className="stack-list compact-list">
        {(rounds || []).slice(0, 3).map((round) => (
          <div key={round.round_index} className="stack-row">
            <strong>Round {round.round_index + 1}</strong>
            <p>{round.summary}</p>
          </div>
        ))}
        {!rounds?.length ? <p className="muted-copy">No swarm rounds available in the current artifact.</p> : null}
      </div>
    </ShellCard>
  );
}

function GraphOverviewPanel({ graph, runId, onRefresh }) {
  const [buildState, setBuildState] = useState({ loading: false, error: "", note: "" });

  async function triggerBuild() {
    setBuildState({ loading: true, error: "", note: "" });
    try {
      const response = await fetch("/api/graph/build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || "Graph build failed");
      }
      setBuildState({ loading: false, error: "", note: `Queued ${payload.task_id}` });
      if (onRefresh) {
        window.setTimeout(() => {
          onRefresh();
        }, 600);
      }
    } catch (error) {
      setBuildState({
        loading: false,
        error: error instanceof Error ? error.message : "Graph build failed",
        note: "",
      });
    }
  }

  const highlights = graph?.highlights || {};

  return (
    <ShellCard title="Knowledge Graph" eyebrow="Explainability Map">
      <div className="metric-grid three-up">
        <div className="metric-tile">
          <span className="metric-label">Status</span>
          <strong>{graph?.status || "Not Built"}</strong>
        </div>
        <div className="metric-tile">
          <span className="metric-label">Nodes</span>
          <strong>{graph?.node_count || 0}</strong>
        </div>
        <div className="metric-tile">
          <span className="metric-label">Edges</span>
          <strong>{graph?.edge_count || 0}</strong>
        </div>
      </div>
      <p className="muted-copy">{graph?.ontology?.analysis_summary || "Graph build will connect evidence, features, reports, swarm activity, and forecast outputs for this run."}</p>
      <div className="button-row">
        <button className="primary-button" type="button" onClick={triggerBuild} disabled={buildState.loading || !runId}>
          {buildState.loading ? "Queueing..." : (graph ? "Rebuild Graph" : "Build Graph")}
        </button>
      </div>
      {buildState.error ? <p className="error-copy">{buildState.error}</p> : null}
      {buildState.note ? <p className="muted-copy">{buildState.note}</p> : null}
      <div className="stack-list compact-list">
        {(highlights.top_features || []).slice(0, 3).map((item) => (
          <div key={item.id} className="split-row">
            <div>
              <strong>{item.label}</strong>
              <p>{item.summary || item.type}</p>
            </div>
            <span className="status-pill neutral">{item.type}</span>
          </div>
        ))}
        {(highlights.top_evidence || []).slice(0, 2).map((item) => (
          <div key={item.id} className="split-row">
            <div>
              <strong>{item.label}</strong>
              <p>{item.properties?.source || item.type}</p>
            </div>
            <span className="status-pill neutral">Evidence</span>
          </div>
        ))}
        {!graph ? <p className="muted-copy">No graph snapshot is attached to this run yet.</p> : null}
      </div>
    </ShellCard>
  );
}

function SummaryBrief({ summary }) {
  const brief = buildSummaryBrief(summary);

  if (!brief.lead && !brief.sections.length) {
    return <p className="muted-copy">Narrative brief will appear once the latest run is available.</p>;
  }

  return (
    <div className="summary-brief">
      {brief.lead ? <p className="summary-lead">{brief.lead}</p> : null}
      {brief.sections.length ? (
        <div className="summary-section-list">
          {brief.sections.slice(0, 4).map((section) => (
            <div key={`${section.title}-${section.body.slice(0, 24)}`} className="summary-section">
              <strong>{section.title}</strong>
              {section.body ? <p>{section.body}</p> : null}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function AboutStageAccordion() {
  const [openIndex, setOpenIndex] = useState(0);

  return (
    <div className="about-accordion">
      {BACKEND_PHASES.map((phase, index) => {
        const isOpen = openIndex === index;
        return (
          <article key={phase.title} className={`about-stage-card ${isOpen ? "open" : ""}`}>
            <button
              type="button"
              className="about-stage-toggle"
              onClick={() => setOpenIndex((current) => (current === index ? -1 : index))}
              aria-expanded={isOpen}
            >
              <div className="about-stage-head">
                <div className="about-step-marker">
                  <span>{String(index + 1).padStart(2, "0")}</span>
                </div>
                <div className="about-step-copy">
                  <p className="panel-eyebrow">{phase.eyebrow}</p>
                  <h3>{phase.title}</h3>
                  <div className="about-stage-tags">
                    {(phase.tags || []).map((tag) => (
                      <span key={tag} className="about-stage-tag">
                        {tag}
                      </span>
                    ))}
                  </div>
                  <p>{phase.summary}</p>
                </div>
              </div>
              <span className="about-stage-icon" aria-hidden="true">
                {isOpen ? "−" : "+"}
              </span>
            </button>
            {isOpen ? (
              <div className="about-stage-detail">
                <div className="stack-list compact-list">
                  {phase.details.map((detail) => (
                    <div key={detail} className="stack-row">
                      <p>{detail}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </article>
        );
      })}
    </div>
  );
}

function AnalysisReportPanel({ analysisReport, subjectLabel, onGenerate, canGenerate, generateLabel }) {
  const [reportState, setReportState] = useState({ loading: false, error: "" });
  const completed = analysisReport?.status === "completed";
  const running = analysisReport && ["queued", "running"].includes(analysisReport.status);

  async function handleGenerate(forceRebuild = false) {
    if (!canGenerate || !onGenerate) {
      return;
    }
    setReportState({ loading: true, error: "" });
    try {
      await onGenerate(forceRebuild);
      setReportState({ loading: false, error: "" });
    } catch (error) {
      setReportState({
        loading: false,
        error: error instanceof Error ? error.message : "Report generation failed",
      });
    }
  }

  return (
    <ShellCard title="Analysis Report" eyebrow={subjectLabel}>
      <div className="metric-grid three-up">
        <div className="metric-tile">
          <span className="metric-label">Status</span>
          <strong>{analysisReport?.status || "Not Generated"}</strong>
        </div>
        <div className="metric-tile">
          <span className="metric-label">Stage</span>
          <strong>{analysisReport?.progress_stage || "pending"}</strong>
        </div>
        <div className="metric-tile">
          <span className="metric-label">Sections</span>
          <strong>{analysisReport ? `${analysisReport.completed_sections || 0}/${analysisReport.section_count || 0}` : "0/0"}</strong>
        </div>
      </div>
      <p className="muted-copy">{analysisReport?.summary || "Generate a structured narrative over the current artifact, graph state, and simulation trace when present."}</p>
      <div className="button-row">
        <button className="primary-button" type="button" onClick={() => handleGenerate(false)} disabled={!canGenerate || reportState.loading || running}>
          {reportState.loading || running ? "Working..." : generateLabel}
        </button>
        {analysisReport ? (
          <button className="secondary-button" type="button" onClick={() => handleGenerate(true)} disabled={!canGenerate || reportState.loading}>
            Regenerate
          </button>
        ) : null}
      </div>
      {reportState.error ? <p className="error-copy">{reportState.error}</p> : null}
      {analysisReport?.error_message ? <p className="error-copy">{analysisReport.error_message}</p> : null}
      <div className="button-stack">
        {completed && analysisReport?.html_path ? (
          <a className="text-link" href={analysisReport.html_path} target="_blank" rel="noreferrer">
            Open Analysis HTML
          </a>
        ) : null}
        {completed && analysisReport?.markdown_path ? (
          <a className="text-link" href={analysisReport.markdown_path} target="_blank" rel="noreferrer">
            Open Analysis Markdown
          </a>
        ) : null}
      </div>
    </ShellCard>
  );
}

function OverviewRoute({ data, status, resetLiveFeed, generateReport }) {
  const marketSeries = Object.entries(data.market_snapshot?.series || {});
  const healthTone = scoreHealthTone(data.run_health);

  return (
    <div className="route-grid">
      <ShellCard title="Current Market Call" eyebrow="Live Artifact" className="overview-hero-card">
        <div className="overview-hero-layout">
          <div className="overview-callout">
            <div>
              <p className="signal-label">{data.prediction_label}</p>
              <h3>{formatPercent(data.confidence)} confidence</h3>
              <SummaryBrief summary={data.summary} />
            </div>
            <div className="status-pills">
              <span className={`status-pill ${healthTone}`}>{data.run_health}</span>
              <span className="status-pill neutral">{data.target}</span>
              <span className="status-pill neutral">{status === "ready" ? "Live feed synced" : "Waiting for sync"}</span>
            </div>
            <div className="button-row">
              <button className="primary-button" type="button" onClick={resetLiveFeed}>
                Reload Live Feed
              </button>
              {data.analysis_report?.html_path ? (
                <a className="text-link" href={data.analysis_report.html_path} target="_blank" rel="noreferrer">
                  Open Analysis Report
                </a>
              ) : null}
            </div>
          </div>

          <div className="overview-snapshot-grid">
            <div className="metric-tile">
              <span className="metric-label">Prediction Date</span>
              <strong>{formatDate(data.prediction_date)}</strong>
            </div>
            <div className="metric-tile">
              <span className="metric-label">Next Session</span>
              <strong>{formatDate(data.next_session_date)}</strong>
            </div>
            <div className="metric-tile">
              <span className="metric-label">Expected Return</span>
              <strong>{formatSignedPercent(data.forecast_summary?.expected_return_30d)}</strong>
            </div>
            <div className="metric-tile">
              <span className="metric-label">Expected Volatility</span>
              <strong>{formatRatioPercent(data.forecast_summary?.expected_volatility_30d)}</strong>
            </div>
            <div className="metric-tile">
              <span className="metric-label">Regime</span>
              <strong>{data.forecast_summary?.regime_label || data.regime_label || "Pending"}</strong>
            </div>
            <div className="metric-tile">
              <span className="metric-label">Model Stack</span>
              <strong>{data.model_version || data.ensemble_diagnostics?.mode || "Pending"}</strong>
            </div>
          </div>
        </div>
      </ShellCard>

      <ProjectionChart projection={data.market_projection} forecastSummary={data.forecast_summary} />

      <GraphOverviewPanel graph={data.graph_summary} runId={data.run_id} onRefresh={resetLiveFeed} />

      <AnalysisReportPanel
        analysisReport={data.analysis_report}
        subjectLabel="Run-Level Narrative"
        canGenerate={Boolean(data.run_id)}
        generateLabel="Generate Run Report"
        onGenerate={(forceRebuild) => generateReport({ run_id: data.run_id, force_rebuild: forceRebuild })}
      />

      <ShellCard title="Primary Drivers" eyebrow="Reasoning Stack">
        <div className="stack-list">
          {(data.top_drivers || []).slice(0, 5).map((driver) => (
            <div key={driver.name} className="stack-row">
              <strong>{driver.name}</strong>
              <p>{driver.summary}</p>
            </div>
          ))}
          {!data.top_drivers?.length ? <p className="muted-copy">Driver-level reasoning will appear once a forecast artifact is available.</p> : null}
        </div>
      </ShellCard>

      <ShellCard title="Trust Signal" eyebrow="Integrity Gates">
        <p className="muted-copy">
          {data.data_quality_summary?.gate_failures?.length
            ? `${data.data_quality_summary.gate_failures.length} validation gates are warning on the current artifact.`
            : "All integrity gates passed for the loaded artifact."}
        </p>
        <div className="metric-grid two-up">
          <div className="metric-tile">
            <span className="metric-label">Distinct Providers</span>
            <strong>{data.data_quality_summary?.distinct_provider_count || "Pending"}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Statistical Failures</span>
            <strong>{data.statistical_failures?.length || 0}</strong>
          </div>
        </div>
      </ShellCard>

      <SwarmSetupPanel setup={data.swarm_setup} summary={data.swarm_summary} reporting={data.swarm_reporting} />

      <SwarmRoundsPanel rounds={data.swarm_rounds} summary={data.swarm_summary} reporting={data.swarm_reporting} />

      <ShellCard title="Cross-Asset Pulse" eyebrow="Market Snapshot">
        <div className="snapshot-grid">
          {marketSeries.map(([label, series]) => (
            <div key={label} className="snapshot-card">
              <span className="metric-label">{label}</span>
              <strong>{isFiniteNumber(series.latest) ? series.latest.toLocaleString() : "Pending"}</strong>
              <span className={series.pct_change >= 0 ? "tone-positive" : "tone-negative"}>
                {formatSignedPercent((series.pct_change || 0) / 100)}
              </span>
            </div>
          ))}
          {!marketSeries.length ? <p className="muted-copy">Cross-asset context is not available in the current artifact.</p> : null}
        </div>
      </ShellCard>
    </div>
  );
}

function ScenarioRoute({ data, simulateScenario, resetLiveFeed, loading, generateReport }) {
  const [customScenario, setCustomScenario] = useState("");
  const [feedback, setFeedback] = useState("");
  const [lastScenario, setLastScenario] = useState("");

  async function runScenario(scenario) {
    setFeedback("");
    const result = await simulateScenario(scenario);
    if (result.ok) {
      setLastScenario(scenario);
      return;
    }
    setFeedback(result.error || "Scenario failed.");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const scenario = customScenario.trim();
    if (!scenario) {
      setFeedback("Enter a custom macro or market shock first.");
      return;
    }
    await runScenario(scenario);
  }

  return (
    <div className="route-grid">
      <ShellCard title="Scenario Workflow" eyebrow="Stress Test">
        <p className="muted-copy">
          Make a hypothesis, inject it into the live artifact, and review how the model shifts under explicit pressure.
        </p>
        <div className="metric-grid three-up">
          <div className="metric-tile">
            <span className="metric-label">Live Target</span>
            <strong>{data.target}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Current Call</span>
            <strong>{data.prediction_label}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Run Health</span>
            <strong>{data.run_health}</strong>
          </div>
        </div>
      </ShellCard>

      <ShellCard title="Scenario Presets" eyebrow="Quick Injection">
        <div className="preset-grid">
          {SCENARIO_PRESETS.map((preset) => (
            <button key={preset.id} className="scenario-chip" onClick={() => runScenario(preset.scenario)} disabled={loading}>
              <strong>{preset.label}</strong>
              <span>{preset.scenario}</span>
            </button>
          ))}
        </div>
      </ShellCard>

      <ShellCard title="Custom Scenario" eyebrow="Manual Injection">
        <form className="scenario-form" onSubmit={handleSubmit}>
          <label className="input-label" htmlFor="custom-scenario">
            Describe the shock
          </label>
          <textarea
            id="custom-scenario"
            value={customScenario}
            onChange={(event) => setCustomScenario(event.target.value)}
            placeholder="Example: Treasury yields spike 25 bps after a hot inflation print and the dollar breaks out."
            rows={5}
          />
          <div className="button-row">
            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "Running Scenario..." : "Run Scenario"}
            </button>
            <button className="secondary-button" type="button" onClick={resetLiveFeed}>
              Reset To Live Feed
            </button>
          </div>
          {feedback ? <p className="error-copy">{feedback}</p> : null}
        </form>
      </ShellCard>

      <ShellCard title="Scenario Result" eyebrow="Post-Run State">
        <div className="metric-grid three-up">
          <div className="metric-tile">
            <span className="metric-label">Prediction</span>
            <strong>{data.prediction_label}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Confidence</span>
            <strong>{formatPercent(data.confidence)}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">30D Return</span>
            <strong>{formatSignedPercent(data.forecast_summary?.expected_return_30d)}</strong>
          </div>
        </div>
        {lastScenario ? (
          <div className="callout">
            <span className="metric-label">Latest Injection</span>
            <p>{lastScenario}</p>
          </div>
        ) : null}
        <p className="muted-copy">{data.summary}</p>
      </ShellCard>

      <AnalysisReportPanel
        analysisReport={data.analysis_report}
        subjectLabel="Simulation Narrative"
        canGenerate={Boolean(data.simulation_id)}
        generateLabel="Generate Simulation Report"
        onGenerate={(forceRebuild) => generateReport({ simulation_id: data.simulation_id, force_rebuild: forceRebuild })}
      />

      <ShellCard title="Confidence Notes" eyebrow="What Changed">
        <div className="stack-list">
          {(data.confidence_notes || []).slice(0, 5).map((note) => (
            <div key={note} className="stack-row">
              <p>{note}</p>
            </div>
          ))}
          {!data.confidence_notes?.length ? <p className="muted-copy">The simulated artifact did not include explicit confidence notes.</p> : null}
        </div>
      </ShellCard>

      <SwarmSetupPanel setup={data.swarm_setup} summary={data.swarm_summary} reporting={data.swarm_reporting} />

      <ShellCard title="Sector Readthrough" eyebrow="Tactical Rotation">
        <div className="stack-list">
          {(data.sector_outlook || []).slice(0, 6).map((item) => (
            <div key={item.ticker || item.sector_symbol} className="split-row">
              <div>
                <strong>{item.sector || item.sector_name}</strong>
                <p>{item.ticker || item.sector_symbol}</p>
              </div>
              <span className={`status-pill ${(item.prediction === "bearish" || item.expected_return_30d < 0) ? "negative" : "positive"}`}>
                {item.prediction || item.recommendation_label}
              </span>
            </div>
          ))}
          {!data.sector_outlook?.length ? <p className="muted-copy">Sector-level readthrough is not available yet.</p> : null}
        </div>
      </ShellCard>
    </div>
  );
}

function InterrogateRoute({ interactionState, setInteractionState, runReportChat, runSimulationInterview, runSimulationInterviewBatch, runSimulationInterviewAll, history }) {
  const selectedAnalyst = interactionState.selectedAnalyst;
  const selectedAnalystMeta = ANALYSTS.find((analyst) => analyst.id === selectedAnalyst) || ANALYSTS[0];
  const profiles = interactionState.simulationContext?.profiles || [];
  const selectedSimulationProfile = profiles.find((profile) => profile.agent_id === interactionState.selectedAgentIds[0]) || null;
  const reportRunOptions = [
    { value: "", label: "Latest published run" },
    ...history
      .filter((item) => item.run_id)
      .map((item) => ({
        value: item.run_id,
        label: `${item.run_date || item.run_id} · ${item.prediction_label || "unknown"} · ${item.run_health || "pending"}`,
      })),
  ].filter((option, index, options) => options.findIndex((item) => item.value === option.value) === index);

  function toggleAgent(agentId) {
    setInteractionState((prev) => {
      const nextIds = prev.selectedAgentIds.includes(agentId)
        ? prev.selectedAgentIds.filter((id) => id !== agentId)
        : [...prev.selectedAgentIds, agentId];
      return { ...prev, selectedAgentIds: nextIds };
    });
  }

  async function askReport(event) {
    event.preventDefault();
    if (!interactionState.reportPrompt.trim()) {
      setInteractionState((prev) => ({ ...prev, error: "Enter a report follow-up question." }));
      return;
    }
    setInteractionState((prev) => ({ ...prev, loading: true, error: "" }));
    try {
      const payload = await runReportChat({
        message: interactionState.reportPrompt,
        chat_history: prevHistoryToChat(interactionState.reportHistory),
        run_id: interactionState.reportRunId || dataOrNull(interactionState.reportRunId),
        focus_category: selectedAnalyst,
      });
      setInteractionState((prev) => ({
        ...prev,
        reportResponse: payload.response || "",
        reportCitations: payload.citations || [],
        diagnostics: payload.diagnostics || null,
        reportHistory: [
          ...prev.reportHistory,
          { role: "user", content: prev.reportPrompt },
          { role: "assistant", content: payload.response || "" },
        ],
        loading: false,
        error: "",
      }));
    } catch (error) {
      setInteractionState((prev) => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : "Report chat failed",
      }));
    }
  }

  async function askSimulation(mode) {
    if (!interactionState.simulationId || !interactionState.simulationPrompt.trim()) {
      setInteractionState((prev) => ({ ...prev, error: "Select a simulation and enter a question." }));
      return;
    }
    setInteractionState((prev) => ({ ...prev, loading: true, error: "" }));
    try {
      let items = [];
      if (mode === "all") {
        const payload = await runSimulationInterviewAll({
          simulation_id: interactionState.simulationId,
          message: interactionState.simulationPrompt,
          chat_history: prevHistoryToChat(interactionState.simulationHistory),
        });
        items = payload.items || [];
      } else if (mode === "batch") {
        const payload = await runSimulationInterviewBatch({
          simulation_id: interactionState.simulationId,
          agent_ids: interactionState.selectedAgentIds,
          message: interactionState.simulationPrompt,
          chat_history: prevHistoryToChat(interactionState.simulationHistory),
        });
        items = payload.items || [];
      } else {
        const agentId = interactionState.selectedAgentIds[0];
        const payload = await runSimulationInterview({
          simulation_id: interactionState.simulationId,
          agent_id: agentId,
          message: interactionState.simulationPrompt,
          chat_history: prevHistoryToChat(interactionState.simulationHistory),
        });
        items = [payload];
      }
      setInteractionState((prev) => ({
        ...prev,
        simulationResponses: items,
        simulationHistory: [
          ...prev.simulationHistory,
          { role: "user", content: prev.simulationPrompt },
          { role: "assistant", content: items.map((item) => `${item.agent_id}: ${item.response}`).join("\n\n") },
        ],
        loading: false,
        error: "",
      }));
    } catch (error) {
      setInteractionState((prev) => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : "Simulation interview failed",
      }));
    }
  }

  return (
    <div className="route-grid">
      <ShellCard title="Deep Interaction" eyebrow="Interrogate">
        <p className="muted-copy">
          Switch between report-grounded follow-up analysis and interviews with persisted swarm personas reconstructed from saved runtime artifacts.
        </p>
        <div className="analyst-tabs">
          <button className={`analyst-tab ${interactionState.mode === "report" ? "active" : ""}`} onClick={() => setInteractionState((prev) => ({ ...prev, mode: "report" }))}>
            Report Chat
          </button>
          <button className={`analyst-tab ${interactionState.mode === "simulation" ? "active" : ""}`} onClick={() => setInteractionState((prev) => ({ ...prev, mode: "simulation" }))}>
            Agent Interviews
          </button>
        </div>
      </ShellCard>

      {interactionState.mode === "report" ? (
        <>
          <ShellCard title="Report Lens" eyebrow="Desk Focus">
            <div className="analyst-tabs">
              {ANALYSTS.map((analyst) => (
                <button
                  key={analyst.id}
                  className={`analyst-tab ${selectedAnalyst === analyst.id ? "active" : ""}`}
                  onClick={() => setInteractionState((prev) => ({ ...prev, selectedAnalyst: analyst.id }))}
                >
                  {analyst.label}
                </button>
              ))}
            </div>
            <label className="input-label" htmlFor="report-run-selector">
              Anchor run
            </label>
            <select
              id="report-run-selector"
              value={interactionState.reportRunId}
              onChange={(event) => setInteractionState((prev) => ({ ...prev, reportRunId: event.target.value }))}
            >
              {reportRunOptions.map((option) => (
                <option key={option.value || "latest"} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <p className="muted-copy">{selectedAnalystMeta.description}</p>
          </ShellCard>

          <ShellCard title="Ask The Report" eyebrow="Prompt">
            <form className="scenario-form" onSubmit={askReport}>
              <label className="input-label" htmlFor="report-prompt">
                Question for {selectedAnalystMeta.label}
              </label>
              <textarea
                id="report-prompt"
                value={interactionState.reportPrompt}
                onChange={(event) => setInteractionState((prev) => ({ ...prev, reportPrompt: event.target.value }))}
                rows={5}
                placeholder="Ask why the call landed here, what evidence is strongest, or what would invalidate the stance."
              />
              <div className="button-row">
                <button className="primary-button" type="submit" disabled={interactionState.loading}>
                  {interactionState.loading ? "Querying..." : "Ask Report"}
                </button>
              </div>
            </form>
          </ShellCard>

          <ShellCard title="Response" eyebrow="Artifact Analysis">
            {interactionState.reportResponse ? (
              <div className="response-panel">
                <p>{interactionState.reportResponse}</p>
                <div className="status-pills">
                  <span className="status-pill neutral">{titleize(selectedAnalyst)}</span>
                  {interactionState.diagnostics?.provider ? <span className="status-pill neutral">{interactionState.diagnostics.provider}</span> : null}
                  {interactionState.diagnostics?.backend_used === false ? <span className="status-pill warning">Fallback</span> : null}
                </div>
                <div className="stack-list compact-list">
                  {(interactionState.reportCitations || []).map((item) => (
                    <div key={`${item.type}-${item.ref}`} className="split-row">
                      <strong>{item.label}</strong>
                      <span className="status-pill neutral">{item.type}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="muted-copy">No report answer yet. The latest reply stays here while you move through the console.</p>
            )}
            {interactionState.error ? <p className="error-copy">{interactionState.error}</p> : null}
          </ShellCard>
        </>
      ) : (
        <>
          <ShellCard title="Simulation Context" eyebrow="Persisted Runtime">
            <div className="metric-grid three-up">
              <div className="metric-tile">
                <span className="metric-label">Simulation ID</span>
                <strong>{interactionState.simulationContext?.simulation_id || "Pending"}</strong>
              </div>
              <div className="metric-tile">
                <span className="metric-label">Profiles</span>
                <strong>{profiles.length || 0}</strong>
              </div>
              <div className="metric-tile">
                <span className="metric-label">Ready</span>
                <strong>{interactionState.simulationContext?.ready_for_interview ? "Yes" : "No"}</strong>
              </div>
            </div>
            <p className="muted-copy">
              {interactionState.runtimeInspection?.run_state?.status
                ? `Runtime status: ${interactionState.runtimeInspection.run_state.status}.`
                : "Runtime inspection has not loaded yet."}
            </p>
          </ShellCard>

          <ShellCard title="Interview Agents" eyebrow="Selection">
            <label className="input-label" htmlFor="simulation-prompt">Question for the swarm</label>
            <textarea
              id="simulation-prompt"
              value={interactionState.simulationPrompt}
              onChange={(event) => setInteractionState((prev) => ({ ...prev, simulationPrompt: event.target.value }))}
              rows={5}
              placeholder="Ask how a persona interpreted the setup, what shifted in later rounds, or which evidence changed their stance."
            />
            <div className="stack-list compact-list">
              {profiles.map((profile) => (
                <button
                  key={profile.agent_id}
                  className={`analyst-tab ${interactionState.selectedAgentIds.includes(profile.agent_id) ? "active" : ""}`}
                  onClick={() => toggleAgent(profile.agent_id)}
                >
                  {profile.name}
                </button>
              ))}
              {!profiles.length ? <p className="muted-copy">No persisted simulation profiles found.</p> : null}
            </div>
            <div className="button-row">
              <button className="primary-button" type="button" disabled={interactionState.loading || !interactionState.selectedAgentIds.length} onClick={() => askSimulation("single")}>
                Interview One
              </button>
              <button className="secondary-button" type="button" disabled={interactionState.loading || interactionState.selectedAgentIds.length < 2} onClick={() => askSimulation("batch")}>
                Interview Selected
              </button>
              <button className="secondary-button" type="button" disabled={interactionState.loading || !profiles.length} onClick={() => askSimulation("all")}>
                Interview All
              </button>
            </div>
            {interactionState.error ? <p className="error-copy">{interactionState.error}</p> : null}
          </ShellCard>

          <ShellCard title="Runtime Inspection" eyebrow="Timeline">
            <div className="stack-list compact-list">
              {(interactionState.runtimeInspection?.timeline || []).slice(0, 5).map((round) => (
                <div key={round.round_index} className="stack-row">
                  <strong>Round {round.round_index + 1}</strong>
                  <p>{round.summary || "No summary available."}</p>
                </div>
              ))}
              {!interactionState.runtimeInspection?.timeline?.length ? <p className="muted-copy">No round timeline is available for this simulation.</p> : null}
            </div>
          </ShellCard>

          <ShellCard title="Interview Responses" eyebrow="Persona Output">
            <div className="stack-list">
              {(interactionState.simulationResponses || []).map((item) => (
                <div key={item.agent_id} className="stack-row">
                  <strong>{profiles.find((profile) => profile.agent_id === item.agent_id)?.name || item.agent_id}</strong>
                  <p>{item.response}</p>
                </div>
              ))}
              {!interactionState.simulationResponses?.length ? (
                <p className="muted-copy">
                  {selectedSimulationProfile ? `No reply yet from ${selectedSimulationProfile.name}.` : "No simulation interview responses yet."}
                </p>
              ) : null}
            </div>
          </ShellCard>
        </>
      )}
    </div>
  );
}

function prevHistoryToChat(history) {
  return (history || []).slice(-6);
}

function dataOrNull(value) {
  return value || null;
}

function ArchiveRoute({ history, graphHistory, data }) {
  return (
    <div className="route-grid">
      <ShellCard title="Archive Context" eyebrow="Replay Surface">
        <div className="metric-grid three-up">
          <div className="metric-tile">
            <span className="metric-label">Historical Runs</span>
            <strong>{history.length}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Current Target</span>
            <strong>{data.target}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Latest Snapshot</span>
            <strong>{formatDate(data.prediction_date)}</strong>
          </div>
        </div>
      </ShellCard>

      <ShellCard title="Historical Runs" eyebrow="Audit Trail">
        <div className="stack-list">
          {history.slice().reverse().map((item) => (
            <div key={`${item.run_date}-${item.prediction_label}-${item.source_file}`} className="split-row">
              <div>
                <strong>{item.run_date}</strong>
                <p>{item.prediction_label} · {item.regime_label || "regime pending"} · {item.run_health}</p>
              </div>
              <div className="inline-actions">
                <span className="status-pill neutral">{formatPercent(item.confidence)}</span>
                {item.reportPath ? (
                  <a href={item.reportPath} target="_blank" rel="noreferrer" className="text-link">
                    Dashboard
                  </a>
                ) : null}
                {item.analysis_report?.html_path ? (
                  <a href={item.analysis_report.html_path} target="_blank" rel="noreferrer" className="text-link">
                    Analysis
                  </a>
                ) : null}
              </div>
            </div>
          ))}
          {!history.length ? <p className="muted-copy">Historical runs will appear here after the pipeline publishes multiple artifacts.</p> : null}
        </div>
      </ShellCard>

      <ShellCard title="Graph Builds" eyebrow="Graph Archive">
        <div className="stack-list">
          {(graphHistory || []).map((item) => (
            <div key={item.project_id} className="split-row">
              <div>
                <strong>{item.run_id || item.project_id}</strong>
                <p>{item.status} · {item.node_count || 0} nodes · {item.edge_count || 0} edges</p>
              </div>
              <span className="status-pill neutral">{item.progress_stage || "pending"}</span>
            </div>
          ))}
          {!graphHistory?.length ? <p className="muted-copy">Graph build history will appear after knowledge graph jobs are queued.</p> : null}
        </div>
      </ShellCard>
    </div>
  );
}

function AboutRoute() {
  return (
    <div className="route-grid about-grid">
      <ShellCard title="Why This Exists" eyebrow="Prediction Backend" className="about-hero-card">
        <div className="about-hero-layout">
          <div>
            <p className="muted-copy about-lead">
              This console sits on top of a graph-first market prediction backend. It does not produce a call from a single model invocation. It collects structured evidence, simulates market debate, runs specialist agents, challenges its own thesis, and only then publishes the artifact the UI reads.
            </p>
            <div className="status-pills">
              <span className="status-pill neutral">Graph First</span>
              <span className="status-pill neutral">Swarm Assisted</span>
              <span className="status-pill neutral">SQLite Persisted</span>
              <span className="status-pill neutral">Neo4j Ready</span>
            </div>
          </div>
          <div className="about-pillar-grid">
            {BACKEND_PILLARS.map((pillar) => (
              <div key={pillar.title} className="about-pillar">
                <span className="metric-label">{pillar.title}</span>
                <strong>{pillar.title}</strong>
                <p>{pillar.detail}</p>
              </div>
            ))}
          </div>
        </div>
      </ShellCard>

      <ShellCard title="Execution Order" eyebrow="Pipeline Timeline" className="about-timeline-card">
        <p className="muted-copy about-stage-intro">
          The page stays high level by default. Open any stage to inspect what that part of the backend actually does and what it contributes to the final prediction artifact.
        </p>
        <AboutStageAccordion />
      </ShellCard>

      <ShellCard title="What Shapes The Call" eyebrow="Prediction Stack">
        <div className="stack-list">
          <div className="stack-row">
            <strong>Live Evidence Quality</strong>
            <p>Provider diversity, stale-item rejection, freshness scoring, and proxy usage determine how trustworthy the run should feel before any forecast logic begins.</p>
          </div>
          <div className="stack-row">
            <strong>Graph Priors</strong>
            <p>The pipeline turns evidence into entities, relationships, and graph features so contradictions, corroboration, thematic clusters, and temporal change all affect the call.</p>
          </div>
          <div className="stack-row">
            <strong>Swarm Diagnostics</strong>
            <p>Consensus, conflict, stance skew, and round summaries from the simulated swarm act as a narrative pressure layer on top of raw market data.</p>
          </div>
          <div className="stack-row">
            <strong>Challenge Pass</strong>
            <p>Before publish, the backend runs a final adversarial pass to test whether the thesis survives conflicting evidence and whether confidence should be reduced.</p>
          </div>
        </div>
      </ShellCard>

      <ShellCard title="Operational Notes" eyebrow="How To Read It">
        <div className="stack-list">
          {BACKEND_NOTES.map((note) => (
            <div key={note} className="stack-row">
              <p>{note}</p>
            </div>
          ))}
        </div>
      </ShellCard>
    </div>
  );
}

export default function App() {
  const { pathname, navigate } = useBrowserRoute(ROUTES.map((route) => route.path));
  const {
    data,
    history,
    graphHistory,
    status,
    error,
    simulateScenario,
    resetLiveFeed,
    generateReport,
    runReportChat,
    loadSimulationInteractionContext,
    loadRuntimeInspection,
    runSimulationInterview,
    runSimulationInterviewBatch,
    runSimulationInterviewAll,
  } = useBackendArtifact();
  const [interactionState, setInteractionState] = useState({
    mode: "report",
    selectedAnalyst: ANALYSTS[0].id,
    reportPrompt: "What is the strongest evidence behind the current call?",
    reportResponse: "",
    reportHistory: [],
    reportCitations: [],
    reportRunId: "",
    simulationPrompt: "What changed your stance across the simulation rounds?",
    simulationContext: null,
    runtimeInspection: null,
    selectedAgentIds: [],
    simulationId: "",
    simulationResponses: [],
    simulationHistory: [],
    diagnostics: null,
    loading: false,
    error: "",
  });

  useEffect(() => {
    let cancelled = false;
    async function loadInteractionContext() {
      try {
        const context = await loadSimulationInteractionContext();
        if (cancelled) {
          return;
        }
        const simulationId = context.simulation_id || "";
        const runtimeInspection = simulationId ? await loadRuntimeInspection(simulationId).catch(() => null) : null;
        if (cancelled) {
          return;
        }
        setInteractionState((prev) => ({
          ...prev,
          simulationContext: context,
          runtimeInspection,
          simulationId,
          selectedAgentIds: prev.selectedAgentIds.length ? prev.selectedAgentIds : context.profiles?.[0]?.agent_id ? [context.profiles[0].agent_id] : [],
        }));
      } catch (_error) {
        if (!cancelled) {
          setInteractionState((prev) => ({ ...prev, simulationContext: null, runtimeInspection: null }));
        }
      }
    }
    loadInteractionContext();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!data.run_id) {
      return;
    }
    setInteractionState((prev) => (prev.reportRunId ? prev : { ...prev, reportRunId: data.run_id }));
  }, [data.run_id]);

  let routeContent = (
    <OverviewRoute
      data={data}
      status={status}
      resetLiveFeed={resetLiveFeed}
      generateReport={generateReport}
    />
  );
  if (pathname === "/scenario") {
    routeContent = (
      <ScenarioRoute
        data={data}
        simulateScenario={simulateScenario}
        resetLiveFeed={resetLiveFeed}
        loading={status === "loading"}
        generateReport={generateReport}
      />
    );
  } else if (pathname === "/interrogate") {
    routeContent = (
      <InterrogateRoute
        interactionState={interactionState}
        setInteractionState={setInteractionState}
        runReportChat={runReportChat}
        runSimulationInterview={runSimulationInterview}
        runSimulationInterviewBatch={runSimulationInterviewBatch}
        runSimulationInterviewAll={runSimulationInterviewAll}
        history={history}
      />
    );
  } else if (pathname === "/archive") {
    routeContent = <ArchiveRoute history={history} graphHistory={graphHistory} data={data} />;
  } else if (pathname === "/about") {
    routeContent = <AboutRoute />;
  }

  return (
    <div className="app-shell">
      <Header currentPath={pathname} navigate={navigate} data={data} status={status} />
      {error ? <div className="global-banner">{error}</div> : null}
      <div className="shell-body">
        <ShellSidebar
          data={data}
          status={status}
          history={history}
          currentPath={pathname}
          interviewReady={Boolean(interactionState.reportResponse || interactionState.simulationResponses.length)}
          navigate={navigate}
          onReset={resetLiveFeed}
        />
        <main className="route-stage">{routeContent}</main>
      </div>
    </div>
  );
}
