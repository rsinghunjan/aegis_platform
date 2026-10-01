"""Minimal ETL/batch processing orchestration.

A dependency-free sequential (or thread-parallel) pipeline runner for data
processing steps. Not a replacement for Airflow/Dagster in large
deployments, but sufficient for in-process batch jobs and testable without
any external scheduler.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("aegis.data.etl")


@dataclass
class ETLStep:
    name: str
    fn: Callable[[Dict[str, Any]], Any]
    depends_on: tuple = field(default_factory=tuple)


@dataclass
class StepResult:
    name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


class ETLPipeline:
    """Runs a DAG of :class:`ETLStep` objects in dependency order.

    Steps with no unmet dependencies at the same "wave" run concurrently
    via a thread pool; results of upstream steps are made available to
    downstream steps through the shared ``context`` dict.
    """

    def __init__(self, steps: List[ETLStep], max_workers: int = 4):
        self.steps = {step.name: step for step in steps}
        self._validate_dag()
        self.max_workers = max_workers

    def _validate_dag(self) -> None:
        for step in self.steps.values():
            for dep in step.depends_on:
                if dep not in self.steps:
                    raise ValueError(f"step '{step.name}' depends on unknown step '{dep}'")

    def _ready_steps(self, completed: set) -> List[ETLStep]:
        return [
            step
            for name, step in self.steps.items()
            if name not in completed and set(step.depends_on) <= completed
        ]

    def run(self, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, StepResult]:
        context: Dict[str, Any] = dict(initial_context or {})
        results: Dict[str, StepResult] = {}
        completed: set = set()

        while len(completed) < len(self.steps):
            batch = self._ready_steps(completed)
            if not batch:
                remaining = set(self.steps) - completed
                raise RuntimeError(f"circular dependency detected among steps: {remaining}")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_step = {
                    executor.submit(self._run_step, step, context): step for step in batch
                }
                for future in as_completed(future_to_step):
                    step = future_to_step[future]
                    result = future.result()
                    results[step.name] = result
                    completed.add(step.name)
                    if result.success:
                        context[step.name] = result.output
                    else:
                        logger.error("ETL step '%s' failed: %s", step.name, result.error)

        return results

    @staticmethod
    def _run_step(step: ETLStep, context: Dict[str, Any]) -> StepResult:
        start = time.monotonic()
        try:
            output = step.fn(context)
            duration_ms = (time.monotonic() - start) * 1000
            return StepResult(name=step.name, success=True, output=output, duration_ms=duration_ms)
        except Exception as exc:  # pragma: no cover - defensive
            duration_ms = (time.monotonic() - start) * 1000
            return StepResult(
                name=step.name, success=False, error=str(exc), duration_ms=duration_ms
            )
