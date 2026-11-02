"""Copy a collection's chunks into a new collection with another embedding model (E7).

`python -m app.ingest.reembed --model Qwen/Qwen3-Embedding-0.6B --target chunks_struct_merge_qwen3e`
Same chunks, same text, same fields (so BM25 is identical); only the dense vectors
change, which makes an embedding comparison a single-variable experiment. Private
notes are not copied. Documents are embedded without a prompt; queries use the
model's own query prompt at search time (eval.run --query-prompt).
"""
from __future__ import annotations

import argparse
import time

from app.models.local import ModelServer
from app.retrieval import milvus_store

FIELDS = milvus_store.OUTPUT_FIELDS


def reembed(source: str, target: str, model: str, batch: int = 128, drop: bool = False) -> dict:
    mc, models = milvus_store.client(), ModelServer(timeout=1800)
    dim = len(models.embed(["dimension probe"], model=model)[0])
    milvus_store.create_collection(mc, target, dim=dim, drop=drop)
    done, began = 0, time.perf_counter()
    iterator = mc.query_iterator(source, batch_size=batch, filter=f'library != "{milvus_store.PRIVATE}"',
                                 output_fields=FIELDS)
    while rows := iterator.next():
        vectors = models.embed([r["text"] for r in rows], model=model, batch=batch)
        milvus_store.insert_rows(mc, target, [{**{k: r[k] for k in FIELDS}, "dense": v}
                                              for r, v in zip(rows, vectors, strict=True)])
        done += len(rows)
        if done % (batch * 40) < batch:
            print(f"{done} chunks, {time.perf_counter() - began:.0f}s", flush=True)
    iterator.close()
    return {"target": target, "model": model, "dim": dim, "chunks": done,
            "seconds": round(time.perf_counter() - began)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source", default="chunks_struct_merge")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--drop", action="store_true", help="rebuild the target from scratch")
    args = parser.parse_args()
    print(reembed(args.source, args.target, args.model, args.batch, args.drop))


if __name__ == "__main__":
    main()
