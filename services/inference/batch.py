"""Batch inference queueing with bounded concurrency."""
from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from .providers import InferenceRequest, InferenceResult
from .router import ModelRouter


@dataclass
class BatchJob:
    job_id: str
    request: InferenceRequest
    result: Optional[InferenceResult] = None
    error: Optional[str] = None
    done: threading.Event = field(default_factory=threading.Event)


class BatchInferenceQueue:
    """A bounded worker-pool queue for batching inference requests.

    Submissions return a job id immediately; callers can poll
    :meth:`status` or block with :meth:`wait`. This keeps request handling
    decoupled from the (potentially slow) model call, similar to how the
    existing agentic runtime decouples run creation from execution.
    """

    def __init__(self, router: ModelRouter, workers: int = 2, max_queue_size: int = 1000):
        self._router = router
        self._queue: "queue.Queue[BatchJob]" = queue.Queue(maxsize=max_queue_size)
        self._jobs: Dict[str, BatchJob] = {}
        self._lock = threading.Lock()
        self._workers = [
            threading.Thread(target=self._worker_loop, daemon=True) for _ in range(workers)
        ]
        self._stopped = False
        for worker in self._workers:
            worker.start()

    def submit(self, request: InferenceRequest) -> str:
        job = BatchJob(job_id=str(uuid.uuid4()), request=request)
        with self._lock:
            self._jobs[job.job_id] = job
        self._queue.put(job)
        return job.job_id

    def _worker_loop(self) -> None:
        while not self._stopped:
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                job.result = self._router.generate(job.request)
            except Exception as exc:  # pragma: no cover - defensive
                job.error = str(exc)
            finally:
                job.done.set()
                self._queue.task_done()

    def status(self, job_id: str) -> str:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if not job.done.is_set():
            return "pending"
        return "failed" if job.error else "succeeded"

    def wait(self, job_id: str, timeout: Optional[float] = None) -> BatchJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.done.wait(timeout=timeout)
        return job

    def shutdown(self) -> None:
        self._stopped = True
        for worker in self._workers:
            worker.join(timeout=1.0)
