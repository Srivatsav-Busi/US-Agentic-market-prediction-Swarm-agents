from __future__ import annotations

from typing import Any


def project_simulation_graph_deltas(
    store,
    *,
    simulation_id: str,
    base_run_stem: str,
    target: str,
    graph_deltas: list[dict[str, Any]],
    round_batch_size: int = 1,
) -> None:
    batch_size = max(1, int(round_batch_size or 1))
    for start in range(0, len(graph_deltas), batch_size):
        round_batch = graph_deltas[start : start + batch_size]
        store.execute_write(
            _project_graph_delta_batch,
            {
                "simulation_id": simulation_id,
                "base_run_stem": base_run_stem,
                "target": target,
                "graph_deltas": round_batch,
            },
        )


def _project_graph_delta_batch(tx: Any, parameters: dict[str, Any]) -> None:
    for delta in list(parameters.get("graph_deltas") or []):
        tx.run(
            "MERGE (s:SimulationRun {simulation_id: $simulation_id}) "
            "SET s.base_run_stem = $base_run_stem, s.target = $target "
            "MERGE (r:SimulationRound {simulation_id: $simulation_id, round_index: $round_index}) "
            "SET r.timestamp = $timestamp, "
            "    r.consensus_score = $consensus_score, "
            "    r.conflict_score = $conflict_score",
            simulation_id=parameters["simulation_id"],
            base_run_stem=parameters.get("base_run_stem"),
            target=parameters.get("target"),
            round_index=int(delta.get("round_index", 0) or 0),
            timestamp=delta.get("timestamp"),
            consensus_score=float(((delta.get("consensus_shift") or {}).get("consensus_score", 0.0)) or 0.0),
            conflict_score=float(((delta.get("consensus_shift") or {}).get("conflict_score", 0.0)) or 0.0),
        )
        for node in list(delta.get("created_action_nodes") or []):
            tx.run(
                "MERGE (a:SimulationAction {action_node_id: $action_node_id}) "
                "SET a.simulation_id = $simulation_id, "
                "    a.round_index = $round_index, "
                "    a.agent_id = $agent_id, "
                "    a.action_type = $action_type, "
                "    a.direction = $direction, "
                "    a.target_agent_id = $target_agent_id",
                action_node_id=node.get("action_node_id"),
                simulation_id=parameters["simulation_id"],
                round_index=int(delta.get("round_index", 0) or 0),
                agent_id=node.get("agent_id"),
                action_type=node.get("action_type"),
                direction=node.get("direction"),
                target_agent_id=node.get("target_agent_id"),
            )
