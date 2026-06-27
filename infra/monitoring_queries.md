# Application Insights monitoring queries (KQL)

These are the queries you'd pin to a dashboard and wire to alert rules once the
app is emitting telemetry to Application Insights. KQL (Kusto Query Language) is
the query language App Insights / Log Analytics use.

The point for interview: monitoring isn't "I turned on App Insights." It's
knowing *which signals matter for an AI agent specifically* and turning them into
alerts. Generic web metrics (5xx rate, latency) matter, but the agent-specific
ones below are what catch an AI system misbehaving.

---

## 1. Request latency (p50 / p95 / p99) over time

p95 latency is the honest user-experience number; a rising p95 with a flat p50
means a subset of requests is degrading (often the multi-search ones).

```kql
requests
| where timestamp > ago(24h)
| summarize
    p50 = percentile(duration, 50),
    p95 = percentile(duration, 95),
    p99 = percentile(duration, 99)
  by bin(timestamp, 15m)
| render timechart
```

## 2. Failure rate (the 503s from our error boundary)

```kql
requests
| where timestamp > ago(24h)
| summarize total = count(), failed = countif(success == false) by bin(timestamp, 15m)
| extend failure_rate = todouble(failed) / total
| render timechart
```

## 3. Agent steps per request (AGENT-SPECIFIC)

Our custom metric. A creeping average step count means the agent is searching
more times to answer — a sign retrieval quality has dropped or questions got
harder. This is a leading indicator a generic dashboard would miss.

```kql
customMetrics
| where name == "agent.request.steps"
| where timestamp > ago(24h)
| summarize avg_steps = avg(value), max_steps = max(value) by bin(timestamp, 30m)
| render timechart
```

## 4. Exceptions, most frequent first (triage view)

```kql
exceptions
| where timestamp > ago(24h)
| summarize count() by type, outerMessage
| order by count_ desc
```

## 5. Slowest individual requests (find the pathological cases)

```kql
requests
| where timestamp > ago(6h)
| top 20 by duration desc
| project timestamp, name, duration, resultCode
```

---

## Alert rules to create (thresholds tune with real traffic)

- **High failure rate**: failure_rate > 0.05 over 15m  -> warns the backend or
  provider is failing.
- **Latency regression**: p95 duration > 8000 ms over 15m -> degraded UX.
- **Agent thrashing**: avg agent.request.steps > 4 over 30m -> retrieval likely
  degraded; investigate the semantic layer.

Each alert maps to a likely root cause, so an on-call engineer knows where to look
rather than just that "something is wrong".
