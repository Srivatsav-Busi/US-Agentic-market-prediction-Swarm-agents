from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable
from urllib import error, request


class BaseLLMClient:
    def complete(self, system_prompt: str, user_prompt: str, fallback: str) -> str:
        return fallback

    def summarize(self, system_prompt: str, user_prompt: str, fallback: str) -> str:
        return self.complete(system_prompt, user_prompt, fallback)

    def diagnostics(self) -> dict:
        return {"provider": "rule_based", "backend_used": False}


class RuleBasedLLMClient(BaseLLMClient):
    pass


@dataclass
class LLMProgressTracker:
    lock: threading.Lock = field(default_factory=threading.Lock)
    requests_started: int = 0
    requests_completed: int = 0
    requests_failed: int = 0
    attempts_started: int = 0
    inflight: int = 0
    last_error: str | None = None
    groups: dict[str, dict[str, int | str | None]] = field(default_factory=dict)
    on_update: Callable[[dict], None] | None = None

    def start(self, group: str) -> None:
        with self.lock:
            self.requests_started += 1
            self.inflight += 1
            group_state = self.groups.setdefault(group, self._new_group_state())
            group_state["started"] += 1
            group_state["inflight"] += 1
            snapshot = self._snapshot_locked()
        self._emit(snapshot)

    def attempt(self, group: str) -> None:
        with self.lock:
            self.attempts_started += 1
            group_state = self.groups.setdefault(group, self._new_group_state())
            group_state["attempts"] += 1
            snapshot = self._snapshot_locked()
        self._emit(snapshot)

    def complete(self, group: str) -> None:
        with self.lock:
            self.requests_completed += 1
            self.inflight = max(0, self.inflight - 1)
            group_state = self.groups.setdefault(group, self._new_group_state())
            group_state["completed"] += 1
            group_state["inflight"] = max(0, int(group_state["inflight"]) - 1)
            snapshot = self._snapshot_locked()
        self._emit(snapshot)

    def fail(self, group: str, message: str) -> None:
        with self.lock:
            self.requests_failed += 1
            self.inflight = max(0, self.inflight - 1)
            self.last_error = message
            group_state = self.groups.setdefault(group, self._new_group_state())
            group_state["failed"] += 1
            group_state["inflight"] = max(0, int(group_state["inflight"]) - 1)
            group_state["last_error"] = message
            snapshot = self._snapshot_locked()
        self._emit(snapshot)

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict:
        groups = {name: dict(values) for name, values in self.groups.items()}
        for values in groups.values():
            started = int(values["started"])
            completed = int(values["completed"])
            failed = int(values["failed"])
            inflight = int(values["inflight"])
            values["all_completed"] = inflight == 0 and started == completed + failed
        return {
            "requests_started": self.requests_started,
            "requests_completed": self.requests_completed,
            "requests_failed": self.requests_failed,
            "attempts_started": self.attempts_started,
            "inflight": self.inflight,
            "all_completed": self.inflight == 0 and self.requests_started == self.requests_completed + self.requests_failed,
            "last_error": self.last_error,
            "groups": groups,
        }

    @staticmethod
    def _new_group_state() -> dict[str, int | str | None]:
        return {
            "started": 0,
            "completed": 0,
            "failed": 0,
            "attempts": 0,
            "inflight": 0,
            "last_error": None,
        }

    def _emit(self, snapshot: dict) -> None:
        if self.on_update is not None:
            self.on_update(snapshot)


@dataclass
class OpenRouterClient(BaseLLMClient):
    api_key: str
    model: str
    base_url: str
    timeout_seconds: int = 30
    max_attempts: int = 3
    retry_backoff_seconds: float = 2.0
    app_name: str = "market-prediction"
    referrer: str = "http://localhost"
    max_completion_tokens: int | None = None
    last_error: str | None = None
    request_group: str = "general"
    progress_tracker: LLMProgressTracker | None = None

    def complete(self, system_prompt: str, user_prompt: str, fallback: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        if self.max_completion_tokens is not None:
            payload["max_completion_tokens"] = max(1, int(self.max_completion_tokens))
        req = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.referrer,
                "X-Title": self.app_name,
            },
            method="POST",
        )
        tracker = self.progress_tracker
        if tracker is not None:
            tracker.start(self.request_group)

        response_body = ""
        last_error = ""
        for attempt in range(1, max(1, self.max_attempts) + 1):
            if tracker is not None:
                tracker.attempt(self.request_group)
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    response_body = response.read().decode("utf-8")
                break
            except error.HTTPError as exc:
                try:
                    details = exc.read().decode("utf-8")
                except Exception:
                    details = ""
                last_error = details or f"http_status={exc.code}"
            except (OSError, error.URLError, TimeoutError) as exc:
                last_error = str(exc)

            if attempt >= max(1, self.max_attempts):
                self.last_error = last_error
                if tracker is not None:
                    tracker.fail(self.request_group, last_error)
                return fallback
            time.sleep(min(30.0, max(0.5, self.retry_backoff_seconds) * attempt))

        try:
            parsed = json.loads(response_body)
            choice = (parsed.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = "\n".join(part for part in text_parts if part)
            output = str(content).strip()
            self.last_error = None
            if tracker is not None:
                tracker.complete(self.request_group)
            return output or fallback
        except Exception as exc:
            self.last_error = f"response_parse_error: {exc}"
            if tracker is not None:
                tracker.fail(self.request_group, self.last_error)
            return fallback

    def diagnostics(self) -> dict:
        return {
            "provider": "openrouter",
            "backend_used": self.last_error is None,
            "model": self.model,
            "base_url": self.base_url,
            "last_error": self.last_error,
            "request_group": self.request_group,
            "timeout_seconds": self.timeout_seconds,
            "max_attempts": self.max_attempts,
            "max_completion_tokens": self.max_completion_tokens,
        }


def create_llm_client(config: dict) -> BaseLLMClient:
    provider = (config.get("llm_provider") or "openrouter").lower()
    if provider == "rule_based":
        return RuleBasedLLMClient()

    api_key = (config.get("llm_api_key") or "").strip()
    base_url = (config.get("llm_api_base_url") or "").strip()
    model = (config.get("llm_model") or "").strip()
    if not (api_key and base_url and model):
        return RuleBasedLLMClient()

    return OpenRouterClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_seconds=int(config.get("llm_api_timeout_seconds", 30) or 30),
        max_attempts=int(config.get("llm_api_max_attempts", 3) or 3),
        retry_backoff_seconds=float(config.get("llm_api_retry_backoff_seconds", 2.0) or 2.0),
        app_name=str(config.get("project_name", "market-prediction")),
        referrer=str(config.get("llm_api_referrer", "http://localhost")),
    )


def is_real_llm_client(llm_client: BaseLLMClient) -> bool:
    return isinstance(llm_client, OpenRouterClient)


def with_timeout(llm_client: BaseLLMClient, timeout_seconds: int) -> BaseLLMClient:
    if isinstance(llm_client, OpenRouterClient):
        return replace(llm_client, timeout_seconds=max(1, int(timeout_seconds)))
    return llm_client


def clone_llm_client(llm_client: BaseLLMClient) -> BaseLLMClient:
    if isinstance(llm_client, OpenRouterClient):
        return replace(llm_client)
    return llm_client


def with_progress_tracker(llm_client: BaseLLMClient, tracker: LLMProgressTracker | None) -> BaseLLMClient:
    if isinstance(llm_client, OpenRouterClient):
        return replace(llm_client, progress_tracker=tracker)
    return llm_client


def with_max_completion_tokens(llm_client: BaseLLMClient, max_completion_tokens: int | None) -> BaseLLMClient:
    if isinstance(llm_client, OpenRouterClient):
        return replace(
            llm_client,
            max_completion_tokens=max(1, int(max_completion_tokens)) if max_completion_tokens is not None else None,
        )
    return llm_client


def with_request_group(llm_client: BaseLLMClient, request_group: str) -> BaseLLMClient:
    if isinstance(llm_client, OpenRouterClient):
        return replace(llm_client, request_group=request_group)
    return llm_client
