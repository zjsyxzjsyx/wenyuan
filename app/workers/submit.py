"""Submit a build to the Celery workers and follow it to the end.

`python -m app.workers.submit python django --variant struct_merge [--rebuild]`
Creates the collection (dropping it with --rebuild), queues one build_library task
per library, then prints progress until every page chain has finished.
"""
from __future__ import annotations

import argparse
import time

from celery.result import GroupResult

from app.db.models import engine
from app.ingest import pipeline
from app.ingest.sites import SITES
from app.retrieval import milvus_store
from app.tracing import spans
from app.workers.tasks import build_library


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("libraries", nargs="+", choices=sorted(SITES))
    parser.add_argument("--variant", choices=pipeline.VARIANTS, default="struct_merge")
    parser.add_argument("--rebuild", action="store_true", help="drop and recreate the collection first")
    args = parser.parse_args()

    spans.configure(engine())
    collection = f"chunks_{args.variant}"
    milvus_store.create_collection(milvus_store.client(), collection, drop=args.rebuild)
    began = time.perf_counter()
    with spans.span("ingest", "submit", libraries=args.libraries, variant=args.variant) as root:
        builds = [build_library.delay(lib, args.variant, spans.inject()) for lib in args.libraries]
        for build in builds:
            info = build.get()
            jobs = GroupResult.restore(info["group_id"])
            while not jobs.ready():
                print(f"  {info['library']}: {jobs.completed_count()}/{info['pages']} pages", flush=True)
                time.sleep(10)
            failed = sum(r.failed() for r in jobs.results)
            chunks = sum(r.result for r in jobs.results if r.successful())
            print(f"{info['library']}: pages={info['pages']} near_dup={info['near_dup']} chunks={chunks} failed={failed}")
        root.set(seconds=round(time.perf_counter() - began))
    print(f"done in {time.perf_counter() - began:.0f}s, trace {root.trace_id}")


if __name__ == "__main__":
    main()
