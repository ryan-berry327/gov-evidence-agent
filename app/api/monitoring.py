"""
Monitoring via Azure Application Insights, using the Azure Monitor OpenTelemetry
distro to export logs, traces and custom metrics.

If no connection string is set (local dev, CI) everything here is a no-op — the
service must boot regardless of whether monitoring is configured.

The custom metrics are agent-specific (steps per request, evidence count): the
signals that show the agent misbehaving, which generic HTTP metrics miss.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("gov-evidence-agent.monitoring")

_ENABLED = False


def setup_monitoring(app) -> bool:
    """Configure App Insights if a connection string is present. Returns enabled?"""
    global _ENABLED
    conn = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not conn:
        logger.info("App Insights not configured; running without cloud telemetry")
        return False

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        # Exports logs, traces and metrics to App Insights via OpenTelemetry.
        configure_azure_monitor(connection_string=conn)
        # Auto-instrument FastAPI: every request becomes a traced span with timing.
        FastAPIInstrumentor.instrument_app(app)
        _ENABLED = True
        logger.info("App Insights monitoring enabled")
        return True
    except Exception:
        # Never let telemetry setup crash the service.
        logger.exception("failed to configure App Insights; continuing without it")
        return False


# --- custom metric emitters (no-ops if monitoring disabled) ----------------
def _meter():
    if not _ENABLED:
        return None
    try:
        from opentelemetry import metrics
        return metrics.get_meter("gov-evidence-agent")
    except Exception:
        return None


_metrics_cache: dict = {}


def record_request_metrics(latency_ms: float, steps: int, evidence: int) -> None:
    """Emit agent-specific custom metrics. Safe to call when disabled."""
    meter = _meter()
    if meter is None:
        return
    try:
        if "latency" not in _metrics_cache:
            _metrics_cache["latency"] = meter.create_histogram(
                "agent.request.latency_ms", unit="ms",
                description="End-to-end agent latency")
            _metrics_cache["steps"] = meter.create_histogram(
                "agent.request.steps", description="Agent loop steps per request")
            _metrics_cache["evidence"] = meter.create_histogram(
                "agent.request.evidence_count", description="Chunks retrieved")
        _metrics_cache["latency"].record(latency_ms)
        _metrics_cache["steps"].record(steps)
        _metrics_cache["evidence"].record(evidence)
    except Exception:
        logger.exception("failed to record custom metrics")
