"""
Experiment tracing via Opik Cloud SDK.

OpikTracer: sends traces to Opik Cloud for dashboard inspection.
LocalTraceRecorder: writes trace JSON locally (fallback when Opik is unavailable).

Each pipeline run creates one trace with spans for each stage:
  load_example → select_ontology → build_prompts → extract_entities
  → normalize_entities → link_entities → materialize_graph → evaluate
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Optional
from uuid import uuid4


# ---------------------------------------------------------------------------
# OpikTracer — Opik Cloud SDK
# ---------------------------------------------------------------------------

class OpikTracer:
    """Send experiment traces to Opik Cloud for observability."""

    def __init__(self, *, project_name: str) -> None:
        import opik as opik_sdk

        self.project_name = project_name
        self._client = opik_sdk.Opik(
            project_name=project_name,
            workspace=os.getenv("OPIK_WORKSPACE", "default"),
        )
        self._trace: Optional[Any] = None
        self._trace_id: str = ""
        self._spans: list[Any] = []
        self._scores: list[dict[str, Any]] = []
        self.backend = "opik_cloud"

    def begin_trace(
        self,
        *,
        name: str,
        metadata: dict[str, Any],
        tags: Optional[list[str]] = None,
    ) -> str:
        """Start a new trace. Returns trace_id."""
        self._trace_metadata = metadata
        trace_kwargs: dict[str, Any] = {"name": name, "metadata": metadata}
        if tags:
            trace_kwargs["tags"] = tags
        self._trace = self._client.trace(**trace_kwargs)
        self._trace_id = self._trace.id
        self._spans = []
        self._scores = []
        return self._trace_id

    @contextmanager
    def span(
        self,
        name: str,
        *,
        inputs: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Iterator[dict[str, Any]]:
        """Context manager that creates an Opik span within the current trace.

        Yields a mutable dict where the caller can set 'outputs'.
        """
        result: dict[str, Any] = {"outputs": {}}
        started = perf_counter()
        creation_metadata = metadata or {}

        span_obj = self._trace.span(
            name=name,
            input=inputs or {},
            metadata=creation_metadata,
        )
        try:
            yield result
        except Exception as exc:
            span_obj.end(
                output={"error": str(exc)},
                metadata={**creation_metadata, "duration_ms": int((perf_counter() - started) * 1000)},
            )
            raise
        else:
            span_obj.end(
                output=result["outputs"],
                metadata={**creation_metadata, "duration_ms": int((perf_counter() - started) * 1000)},
            )
        self._spans.append(span_obj)

    def log_score(self, name: str, value: float, reason: str = "") -> None:
        """Record a feedback score on the current trace."""
        self._trace.log_feedback_score(name=name, value=value, reason=reason)
        self._scores.append({"name": name, "value": value, "reason": reason})

    def end_trace(
        self,
        *,
        output: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Finalize the current trace and return its ID."""
        merged_metadata = {**self._trace_metadata, **(metadata or {})}
        self._trace.end(
            output=output or {},
            metadata=merged_metadata,
        )
        trace_id = self._trace_id
        self._trace = None
        return trace_id


# ---------------------------------------------------------------------------
# LocalTraceRecorder — JSON file fallback (no Opik dependency)
# ---------------------------------------------------------------------------

@dataclass
class TraceSpanRecord:
    name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[dict[str, Any]] = None
    duration_ms: Optional[int] = None


class LocalTraceRecorder:
    """Write trace spans to local JSON files (Opik-free fallback)."""

    def __init__(self, *, project_name: str, log_dir: str = "artifacts/opik_local") -> None:
        self.project_name = project_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trace_id = str(uuid4())
        self._spans: list[TraceSpanRecord] = []
        self._scores: list[dict[str, Any]] = []
        self.backend = "local_json"

    def begin_trace(
        self,
        *,
        name: str,
        metadata: dict[str, Any],
        tags: Optional[list[str]] = None,
    ) -> str:
        self.trace_id = str(uuid4())
        self._trace_name = name
        self._trace_metadata = metadata
        self._trace_tags = tags or []
        self._spans = []
        self._scores = []
        return self.trace_id

    @contextmanager
    def span(
        self,
        name: str,
        *,
        inputs: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Iterator[dict[str, Any]]:
        result: dict[str, Any] = {"outputs": {}}
        started = perf_counter()
        record = TraceSpanRecord(
            name=name,
            inputs=inputs or {},
            metadata=metadata or {},
        )
        try:
            yield result
        except Exception as exc:
            record.error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            record.outputs = result["outputs"]
            record.duration_ms = int((perf_counter() - started) * 1000)
            self._spans.append(record)

    def log_score(self, name: str, value: float, reason: str = "") -> None:
        self._scores.append({"name": name, "value": value, "reason": reason})

    def end_trace(
        self,
        *,
        output: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        merged_metadata = {**getattr(self, "_trace_metadata", {}), **(metadata or {})}
        payload = {
            "trace_id": self.trace_id,
            "project_name": self.project_name,
            "trace_name": getattr(self, "_trace_name", ""),
            "metadata": merged_metadata,
            "tags": getattr(self, "_trace_tags", []),
            "output": output or {},
            "spans": [asdict(s) for s in self._spans],
            "scores": self._scores,
        }
        path = self.log_dir / f"{self.trace_id}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return self.trace_id


def make_tracer(*, project_name: str, use_cloud: bool = True) -> OpikTracer | LocalTraceRecorder:
    """Factory: try Opik Cloud first, fall back to local JSON."""
    if use_cloud and os.getenv("OPIK_API_KEY"):
        try:
            return OpikTracer(project_name=project_name)
        except Exception:
            pass
    return LocalTraceRecorder(project_name=project_name)
