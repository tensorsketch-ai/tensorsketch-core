"""Drift detection over the emitted result stream — alert when live behavior shifts.

`OnlineMonitor` scores production runs and emits one `TrialResult` record per run. `DriftMonitor`
watches that same stream and raises a `DriftAlert` when aggregate behavior drifts from a baseline:

- a **drop in pass rate** (overall or for one grader) via a two-proportion z-test against the
  baseline — the natural test for binary pass/fail outcomes;
- a **change-point in cost or latency** via the Page-Hinkley test — the canonical O(1) streaming
  detector for a shift in a numeric mean.

`DriftMonitor` is itself a `Reporter`, so it chains straight after the online monitor. Persist the
raw results *and* watch for drift in one hand-off with `MultiReporter`:

    baseline = Baseline.from_report(offline_report)
    drift = DriftMonitor(baseline, reporter=JsonlReporter("drift.jsonl"))
    monitor = OnlineMonitor(graders, reporter=MultiReporter(JsonlReporter("online.jsonl"), drift))

The rolling window lives in *this process's* memory — TensorSketch persists nothing here. Kill the
process
and the detector's state is gone; the results and alerts themselves live in whatever store your
reporters write to. TensorSketch emits (results and alerts); it never owns a drift database.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from .reporters import Reporter, deliver
from .runner import Report


def two_proportion_z(passes_a: int, n_a: int, passes_b: int, n_b: int) -> float:
    """Signed z-statistic for H0: p_a == p_b (pooled two-proportion test).

    Returns `(p_b - p_a) / se`, so with `a` = baseline and `b` = the current window a *negative*
    value means the window's pass rate has fallen below baseline. `0.0` if either sample is empty
    or the standard error degenerates.
    """
    if n_a == 0 or n_b == 0:
        return 0.0
    p_a = passes_a / n_a
    p_b = passes_b / n_b
    pool = (passes_a + passes_b) / (n_a + n_b)
    se = math.sqrt(pool * (1.0 - pool) * (1.0 / n_a + 1.0 / n_b))
    if se == 0.0:
        return 0.0
    return (p_b - p_a) / se


@dataclass
class PageHinkley:
    """Online change-point detector for the mean of a numeric stream (Page-Hinkley test).

    O(1) per sample, no window. `delta` is the tolerated drift (slack that a healthy stream may
    wander within); `lam` (lambda) is the alarm threshold. `direction` selects which shift to flag:
    ``"up"`` (a rise — e.g. cost/latency regressions), ``"down"`` (a fall — e.g. a score), or
    ``"both"``. `update` returns True at a change-point but does **not** reset, so the caller can
    read `statistic` before calling `reset` to re-arm for the next shift.
    """

    delta: float = 0.05
    lam: float = 1.0
    direction: str = "up"
    _n: int = field(default=0, repr=False)
    _mean: float = field(default=0.0, repr=False)
    _cum_up: float = field(default=0.0, repr=False)
    _min_up: float = field(default=0.0, repr=False)
    _cum_down: float = field(default=0.0, repr=False)
    _max_down: float = field(default=0.0, repr=False)

    def update(self, x: float) -> bool:
        """Feed one sample; return True if a change-point in the chosen direction has fired."""
        self._n += 1
        self._mean += (x - self._mean) / self._n
        # Upward test: accumulate (x - running_mean - delta); alarm when the cumulative sum rises
        # far enough (lam) above its own running minimum.
        self._cum_up += x - self._mean - self.delta
        self._min_up = min(self._min_up, self._cum_up)
        # Downward test is the mirror image (+ delta, compared against the running maximum).
        self._cum_down += x - self._mean + self.delta
        self._max_down = max(self._max_down, self._cum_down)
        rise = self._cum_up - self._min_up
        fall = self._max_down - self._cum_down
        up = self.direction in ("up", "both") and rise > self.lam
        down = self.direction in ("down", "both") and fall > self.lam
        return up or down

    @property
    def mean(self) -> float:
        """The current running mean of the stream."""
        return self._mean

    @property
    def statistic(self) -> float:
        """How far the accumulator has moved in the monitored direction (compare to `lam`)."""
        rise = self._cum_up - self._min_up
        fall = self._max_down - self._cum_down
        if self.direction == "down":
            return fall
        if self.direction == "both":
            return max(rise, fall)
        return rise

    def reset(self) -> None:
        """Clear all state — call after handling an alarm to detect the next shift afresh."""
        self._n = 0
        self._mean = 0.0
        self._cum_up = 0.0
        self._min_up = 0.0
        self._cum_down = 0.0
        self._max_down = 0.0


@dataclass
class Baseline:
    """Reference statistics that drift is measured against — usually your last green offline eval.

    Build it from a `Report` with `Baseline.from_report(report)`, or set the fields directly. The
    per-grader rates let a drop in one specific check (e.g. a safety grader) trip on its own.
    """

    pass_rate: float = 1.0
    n: int = 0
    mean_cost: float = 0.0
    mean_latency: float = 0.0
    grader_pass_rates: dict[str, float] = field(default_factory=dict)
    grader_n: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_report(cls, report: Report) -> Baseline:
        """Derive a baseline from an offline `Report`: pass rate, cost/latency, per-grader rates."""
        trials = [t for c in report.cases for t in c.trials]
        grader_flags: dict[str, list[bool]] = {}
        for trial in trials:
            for grade in trial.grades:
                grader_flags.setdefault(grade.name, []).append(grade.passed)
        return cls(
            pass_rate=report.completion_rate,
            n=len(trials),
            mean_cost=report.mean_cost,
            mean_latency=report.mean_latency,
            grader_pass_rates={
                name: fmean(1.0 if f else 0.0 for f in flags)
                for name, flags in grader_flags.items()
            },
            grader_n={name: len(flags) for name, flags in grader_flags.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_rate": round(self.pass_rate, 4),
            "n": self.n,
            "mean_cost_usd": round(self.mean_cost, 6),
            "mean_latency_ms": round(self.mean_latency, 1),
            "grader_pass_rates": {k: round(v, 4) for k, v in self.grader_pass_rates.items()},
        }


@dataclass
class DriftAlert:
    """One detected drift — emitted through the monitor's `Reporter` and returned from `emit`."""

    metric: str  # "pass_rate", "cost_usd", "latency_ms", or "grader:<name>"
    kind: str  # "proportion" | "change_point"
    direction: str  # "down" (regression in quality) | "up" (rise in cost/latency)
    baseline: float
    current: float
    statistic: float
    threshold: float
    n: int
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift": True,
            "metric": self.metric,
            "kind": self.kind,
            "direction": self.direction,
            "baseline": round(self.baseline, 6),
            "current": round(self.current, 6),
            "statistic": round(self.statistic, 4),
            "threshold": round(self.threshold, 4),
            "n": self.n,
            "reason": self.reason,
        }


class DriftMonitor:
    """Watch the online result stream and alert on drift.

    Implements the `Reporter` protocol, so hand it to `OnlineMonitor(reporter=...)` — on its own to
    watch only, or inside a `MultiReporter` alongside a store to persist and watch at once. Holds a
    bounded rolling window **in memory only**; nothing is persisted by the monitor itself.
    """

    def __init__(
        self,
        baseline: Baseline | None = None,
        *,
        reporter: Reporter | None = None,
        window: int = 50,
        min_samples: int = 20,
        z_threshold: float = 3.0,
        ph_delta: float = 0.05,
        ph_lambda: float = 1.0,
    ) -> None:
        self.baseline = baseline
        self._reporter = reporter
        self._window: deque[dict[str, Any]] = deque(maxlen=window)
        self._min_samples = min_samples
        self._z_threshold = z_threshold
        self._cost_ph = PageHinkley(delta=ph_delta, lam=ph_lambda, direction="up")
        self._latency_ph = PageHinkley(delta=ph_delta, lam=ph_lambda, direction="up")
        # Per-metric latch: hold a proportion metric "in alert" until it recovers, so a sustained
        # regression fires once rather than on every subsequent record.
        self._alerting: set[str] = set()
        self.alerts: list[DriftAlert] = []

    async def emit(self, record: dict[str, Any]) -> list[DriftAlert]:
        """Ingest one result record (`TrialResult.to_dict()`), test for drift, emit any alerts.

        Returns the alerts raised by *this* record (empty if none), and forwards each to the
        monitor's `reporter`. Async because it may await the downstream sink.
        """
        self._window.append(record)
        alerts: list[DriftAlert] = []

        cost = _num(record.get("cost_usd"))
        if cost is not None:
            base_cost = self.baseline.mean_cost if self.baseline else 0.0
            hit = self._numeric_alert("cost_usd", cost, self._cost_ph, base_cost)
            if hit is not None:
                alerts.append(hit)
        latency = _num(record.get("latency_ms"))
        if latency is not None:
            base_lat = self.baseline.mean_latency if self.baseline else 0.0
            hit = self._numeric_alert("latency_ms", latency, self._latency_ph, base_lat)
            if hit is not None:
                alerts.append(hit)

        if self.baseline is not None and len(self._window) >= self._min_samples:
            alerts.extend(self._proportion_alerts(self.baseline))

        for alert in alerts:
            self.alerts.append(alert)
            if self._reporter is not None:
                await deliver(self._reporter, alert.to_dict())
        return alerts

    def _numeric_alert(
        self, metric: str, value: float, ph: PageHinkley, baseline_mean: float
    ) -> DriftAlert | None:
        # Feed the detector the value *relative to* the baseline mean when we have one, so a single
        # (delta, lambda) works across metrics of wildly different scale (dollars vs milliseconds).
        signal = value / baseline_mean if baseline_mean > 0 else value
        if not ph.update(signal):
            return None
        statistic = ph.statistic
        ph.reset()
        return DriftAlert(
            metric=metric,
            kind="change_point",
            direction="up",
            baseline=baseline_mean,
            current=value,
            statistic=statistic,
            threshold=ph.lam,
            n=len(self._window),
            reason=f"{metric} change-point: {value:.4g} vs baseline mean {baseline_mean:.4g}",
        )

    def _proportion_alerts(self, base: Baseline) -> list[DriftAlert]:
        out: list[DriftAlert] = []
        n = len(self._window)
        passes = sum(1 for r in self._window if r.get("passed"))
        out.extend(self._proportion_check("pass_rate", passes, n, base.pass_rate, base.n))
        for name, base_rate in base.grader_pass_rates.items():
            flags = [f for r in self._window if (f := _grade_passed(r, name)) is not None]
            if len(flags) < self._min_samples:
                continue
            g_passes = sum(1 for f in flags if f)
            base_n = base.grader_n.get(name, base.n)
            out.extend(
                self._proportion_check(f"grader:{name}", g_passes, len(flags), base_rate, base_n)
            )
        return out

    def _proportion_check(
        self, metric: str, passes: int, n: int, base_rate: float, base_n: int
    ) -> list[DriftAlert]:
        z = two_proportion_z(round(base_rate * base_n), base_n, passes, n)
        current = passes / n if n else 0.0
        # Alert only on a *drop* (a quality regression), and only once until it recovers.
        if z <= -self._z_threshold and metric not in self._alerting:
            self._alerting.add(metric)
            return [
                DriftAlert(
                    metric=metric,
                    kind="proportion",
                    direction="down",
                    baseline=base_rate,
                    current=current,
                    statistic=z,
                    threshold=-self._z_threshold,
                    n=n,
                    reason=(
                        f"{metric} dropped to {current:.0%} from baseline "
                        f"{base_rate:.0%} (z={z:.2f})"
                    ),
                )
            ]
        if z > -self._z_threshold:
            self._alerting.discard(metric)
        return []


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _grade_passed(record: dict[str, Any], name: str) -> bool | None:
    for grade in record.get("grades", []):
        if grade.get("name") == name:
            return bool(grade.get("passed"))
    return None
