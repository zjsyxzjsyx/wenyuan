"""Celery app. Queues keep orchestration, CPU work and GPU work from blocking each other:

  orchestrate: build_library (extract + de-duplicate a whole library, then fan out pages)
  cpu:         per-page sectioning, chunking, MySQL writes
  gpu:         embedding + Milvus writes, one task at a time so batches don't fight over VRAM

Run N single-process CPU workers rather than one thread pool: extraction is pure-Python
parsing, so threads serialize on the GIL (the first full build used one core of 16).
prefork is unavailable on Windows, hence separate solo processes:

  celery -A app.workers.celery_app worker -Q orchestrate,cpu --pool=solo -n cpu1@%h   (repeat cpu2, cpu3, ...)
  celery -A app.workers.celery_app worker -Q gpu --pool=solo -n gpu@%h

Listening on orchestrate *and* cpu lets page tasks start while other libraries are
still being extracted, instead of queueing behind every build_library task.
"""
from __future__ import annotations

from celery import Celery

from app.config import get_settings

settings = get_settings()
app = Celery("wenyuan", broker=settings.redis_url, backend=settings.redis_url, include=["app.workers.tasks"])
app.conf.update(
    task_routes={
        "app.workers.tasks.build_library": {"queue": "orchestrate"},
        "app.workers.tasks.embed_and_index": {"queue": "gpu"},
        "app.workers.tasks.*": {"queue": "cpu"},
    },
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # A task is acknowledged only after it finishes, and a worker that dies mid-task hands it
    # back to the queue; with prefetch 1 a slow page does not hold others hostage.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    result_expires=24 * 3600,
)
