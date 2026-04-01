import { useEffect, useState } from "react";

export const DEFAULT_DATA = {
  prediction_label: "Awaiting Run",
  confidence: null,
  run_health: "NO DATA",
  summary: "The next market brief will appear here once the latest run is available.",
  prediction_date: null,
  next_session_date: null,
  run_id: null,
  expected_return: null,
  expected_volatility: null,
  reportPath: null,
  target: "S&P 500",
  confidence_breakdown: {},
  data_quality_summary: {},
  category_weights: {},
  quality_penalties: {},
  challenge_agent_report: {},
  decision_trace: [],
  signal_features: [],
  source_diagnostics: {},
  posterior_probabilities: {},
  neutral_band: {},
  statistical_failures: [],
  source_agent_reports: [],
  sources: [],
  market_snapshot: {},
  economic_report: "",
  political_report: "",
  social_report: "",
  market_context_report: "",
  market_projection: {},
  sector_outlook: [],
  forecast_summary: {},
  top_drivers: [],
  confidence_notes: [],
  ensemble_diagnostics: {},
  regime_probabilities: {},
  history_coverage: {},
  swarm_setup: {},
  swarm_reporting: {},
  graph_summary: null,
  simulation_id: null,
  analysis_report: null,
};

export function useBackendArtifact() {
  const [state, setState] = useState({ data: DEFAULT_DATA, history: [], graphHistory: [], status: "loading", error: "" });

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const bundle = await loadLatestBundle();

        if (!cancelled) {
          setState(bundle);
        }
      } catch (error) {
        if (!cancelled) {
          setState({
            data: DEFAULT_DATA,
            history: [],
            graphHistory: [],
            status: "empty",
            error: error instanceof Error ? error.message : "Failed to load latest artifact",
          });
        }
      }
    }

    load();

    // Auto-refresh the artifact every 5 minutes so when the new 8 AM report drops,
    // the UI updates without requiring a manual page reload.
    const timer = setInterval(() => {
      load();
    }, 300000);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const activeReport = state.data.analysis_report;
    if (!activeReport || !["queued", "running"].includes(activeReport.status)) {
      return undefined;
    }
    const timer = window.setTimeout(async () => {
      const query = activeReport.report_id
        ? `report_id=${encodeURIComponent(activeReport.report_id)}`
        : state.data.simulation_id
          ? `simulation_id=${encodeURIComponent(state.data.simulation_id)}`
          : state.data.run_id
            ? `run_id=${encodeURIComponent(state.data.run_id)}`
            : "";
      if (!query) {
        return;
      }
      try {
        const response = await fetch(`/api/report/status?${query}`, { cache: "no-store" });
        if (!response.ok) {
          return;
        }
        const payload = await response.json();
        setState((prev) => ({
          ...prev,
          data: {
            ...prev.data,
            analysis_report: summarizeReportPayload(payload, prev.data.analysis_report),
          },
        }));
      } catch (_error) {
        return;
      }
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [state.data.analysis_report, state.data.run_id, state.data.simulation_id]);

  async function simulateScenario(scenario) {
    setState((prev) => ({ ...prev, status: "loading", error: "" }));
    try {
      const response = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario }),
      });
      if (!response.ok) {
        throw new Error("Scenario simulation failed");
      }
      const payload = await response.json();
      setState((prev) => ({
        ...prev,
        data: { ...DEFAULT_DATA, ...payload, analysis_report: payload.analysis_report || null },
        status: "ready",
        error: "",
      }));
      return { ok: true };
    } catch (error) {
      setState((prev) => ({
        ...prev,
        status: prev.data.prediction_label === "Awaiting Run" ? "empty" : "ready",
        error: error instanceof Error ? error.message : "Scenario simulation failed",
      }));
      return { ok: false, error: error instanceof Error ? error.message : "Scenario simulation failed" };
    }
  }

  async function resetLiveFeed() {
    setState((prev) => ({ ...prev, status: "loading", error: "" }));
    try {
      setState(await loadLatestBundle());
      return { ok: true };
    } catch (error) {
      setState((prev) => ({
        ...prev,
        status: prev.data.prediction_label === "Awaiting Run" ? "empty" : "ready",
        error: error instanceof Error ? error.message : "Failed to reload latest artifact",
      }));
      return { ok: false, error: error instanceof Error ? error.message : "Failed to reload latest artifact" };
    }
  }

  async function generateReport(options) {
    const response = await fetch("/api/report/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || "Report generation failed");
    }
    const summary = summarizeReportPayload(payload, null);
    setState((prev) => ({
      ...prev,
      data: { ...prev.data, analysis_report: summary },
    }));
    return summary;
  }

  async function runReportChat({ message, chat_history = [], run_id = null, focus_category = null }) {
    const response = await fetch("/api/report/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, chat_history, run_id, focus_category }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || "Report chat failed");
    }
    return payload;
  }

  async function loadSimulationInteractionContext(simulationId = null) {
    const query = simulationId ? `?simulation_id=${encodeURIComponent(simulationId)}` : "";
    const response = await fetch(`/api/simulations/interaction-context${query}`, { cache: "no-store" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || "Failed to load simulation context");
    }
    return payload;
  }

  async function loadRuntimeInspection(simulationId) {
    const response = await fetch(`/api/simulations/runtime-inspection?simulation_id=${encodeURIComponent(simulationId)}`, { cache: "no-store" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || "Failed to load runtime inspection");
    }
    return payload;
  }

  async function runSimulationInterview({ simulation_id, agent_id, message, chat_history = [] }) {
    const response = await fetch("/api/simulations/interview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ simulation_id, agent_id, message, chat_history }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || "Simulation interview failed");
    }
    return payload;
  }

  async function runSimulationInterviewBatch({ simulation_id, agent_ids, message, chat_history = [] }) {
    const response = await fetch("/api/simulations/interview/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ simulation_id, agent_ids, message, chat_history }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || "Batch simulation interview failed");
    }
    return payload;
  }

  async function runSimulationInterviewAll({ simulation_id, message, chat_history = [] }) {
    const response = await fetch("/api/simulations/interview/all", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ simulation_id, message, chat_history }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || "Interview all agents failed");
    }
    return payload;
  }

  return {
    ...state,
    simulateScenario,
    resetLiveFeed,
    generateReport,
    runReportChat,
    loadSimulationInteractionContext,
    loadRuntimeInspection,
    runSimulationInterview,
    runSimulationInterviewBatch,
    runSimulationInterviewAll,
  };
}

function summarizeReportPayload(payload, previous) {
  if (!payload || typeof payload !== "object") {
    return previous || null;
  }
  const progress = payload.progress || {};
  return {
    report_id: payload.report_id,
    status: payload.status,
    subject_type: payload.subject_type,
    subject_id: payload.subject_id,
    task_id: payload.task_id,
    progress_stage: payload.progress_stage,
    title: payload.title || previous?.title || "",
    summary: payload.summary || previous?.summary || "",
    section_count: payload.section_count ?? progress.total_sections ?? previous?.section_count ?? 0,
    completed_sections: payload.completed_sections ?? progress.completed_sections ?? previous?.completed_sections ?? 0,
    markdown_path: payload.markdown_path || previous?.markdown_path || null,
    html_path: payload.html_path || previous?.html_path || null,
    updated_at: payload.updated_at || previous?.updated_at || null,
    error_message: payload.error_message || null,
  };
}

async function loadLatestBundle() {
  const [latestResponse, historyResponse] = await Promise.all([
    fetch("/api/latest-result", { cache: "no-store" }),
    fetch("/api/forecast-history", { cache: "no-store" }),
  ]);
  if (!latestResponse.ok) {
    throw new Error("No backend artifact found");
  }

  const latestPayload = await latestResponse.json();
  const historyPayload = historyResponse.ok ? await historyResponse.json() : { items: [] };
  const graphPayload = await fetch("/api/graph/latest", { cache: "no-store" })
    .then((response) => (response.ok ? response.json() : { latest: null, items: [] }))
    .catch(() => ({ latest: null, items: [] }));

  return {
    data: {
      ...DEFAULT_DATA,
      ...latestPayload,
      analysis_report: latestPayload.analysis_report || null,
      graph_summary: graphPayload.latest,
    },
    history: historyPayload.items || [],
    graphHistory: graphPayload.items || [],
    status: "ready",
    error: "",
  };
}
