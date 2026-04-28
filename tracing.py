# tracing.py
"""
Lightweight observability layer for the LangGraph agent.

Each node execution emits a JSON record to traces/{run_id}.jsonl. Records
include node name, input state summary, output state summary, timing, and
any errors. Designed to be inspected with:

    cat traces/{run_id}.jsonl | jq .
    grep '"node":"validate_output"' traces/*.jsonl

The same pattern (structured records appended to a sink) maps directly to
production audit logs — just swap the file for a Postgres append-only table.
"""

import json
import os
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ---- Configuration ----
TRACE_DIR = Path("traces")
TRACE_ENABLED = os.getenv("AGENT_TRACE_ENABLED", "true").lower() == "true"

# Per-run identifier — set by start_run() at the top of extract_remittance,
# read by node decorators. ContextVar makes this thread-safe per call.
_run_id: ContextVar[str] = ContextVar("_run_id", default="")


def start_run() -> str:
    """Begin a new run. Returns a unique run_id. Call once at the top of extract_remittance."""
    if not TRACE_ENABLED:
        return ""
    run_id = uuid.uuid4().hex[:12]
    _run_id.set(run_id)
    TRACE_DIR.mkdir(exist_ok=True)
    _emit({
        "event": "run_start",
        "run_id": run_id,
        "timestamp": _now(),
    })
    return run_id


def end_run(status: str = "ok", **extra: Any) -> None:
    """Mark a run complete. Call after graph.invoke()."""
    if not TRACE_ENABLED:
        return
    _emit({
        "event": "run_end",
        "run_id": _run_id.get(),
        "status": status,
        "timestamp": _now(),
        **extra,
    })


def trace_node(fn: Callable) -> Callable:
    """
    Decorator: wrap a node function with start/end/error logging.

    Usage:
        @trace_node
        def call_llm_node(state): ...
    """
    def wrapped(state):
        if not TRACE_ENABLED:
            return fn(state)

        node_name = fn.__name__.replace("_node", "")
        run_id = _run_id.get()
        t0 = time.perf_counter()

        _emit({
            "event": "node_start",
            "run_id": run_id,
            "node": node_name,
            "timestamp": _now(),
            "input_summary": _summarize_state(state),
        })

        try:
            result = fn(state)
        except Exception as e:
            _emit({
                "event": "node_error",
                "run_id": run_id,
                "node": node_name,
                "timestamp": _now(),
                "error": str(e),
                "duration_ms": _ms_since(t0),
            })
            raise

        _emit({
            "event": "node_end",
            "run_id": run_id,
            "node": node_name,
            "timestamp": _now(),
            "duration_ms": _ms_since(t0),
            "output_keys": list(result.keys()) if isinstance(result, dict) else [],
            "output_summary": _summarize_update(result),
        })
        return result

    wrapped.__name__ = fn.__name__
    return wrapped


# ---- Helpers ----

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _emit(record: dict) -> None:
    """Append one JSON record to the run's trace file."""
    run_id = record.get("run_id") or _run_id.get()
    if not run_id:
        return
    path = TRACE_DIR / f"{run_id}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _summarize_state(state) -> dict:
    """Compact summary of state — full state is too noisy for traces."""
    if not hasattr(state, "model_dump"):
        return {"type": type(state).__name__}

    d = state.model_dump()
    return {
        "remittance_text_length": len(d.get("remittance_text") or ""),
        "messages_count": len(d.get("messages") or []),
        "validation_retries": d.get("validation_retries", 0),
        "has_advice": d.get("advice") is not None,
        "has_validation_error": d.get("validation_error") is not None,
        "routing_decision": d.get("routing_decision"),
    }


def _summarize_update(update) -> dict:
    """Compact summary of a node's return value."""
    if not isinstance(update, dict):
        return {"type": type(update).__name__}

    summary = {}
    for k, v in update.items():
        if k == "messages" and isinstance(v, list):
            summary[k] = f"+{len(v)} message(s)"
        elif k == "validation_error" and v:
            summary[k] = f"set: {v[:80]}..."
        elif k == "advice" and v is not None:
            summary[k] = "set"
        elif k == "routing_decision" and v is not None:
            summary[k] = str(v)
        elif k == "action_result" and v is not None:
            summary[k] = v.get("action", "set")
        else:
            summary[k] = v
    return summary
