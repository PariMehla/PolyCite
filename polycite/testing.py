"""Deterministic fake Cohere transport — used by tests AND by
`scripts/run_pipeline.py --mode dry-run` to prove the pipeline's wiring end
to end without spending API budget or needing network. Never smart, never a
substitute for real numbers: it approximates rerank/generation with cheap
token-overlap heuristics so results are directionally sane, not because
they're meant to be reported anywhere.
"""
from __future__ import annotations

import hashlib


def _pseudo_vector(text: str, dims: int = 16) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:dims]]


def _overlap_score(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def fake_transport(endpoint: str, model: str, params: dict) -> dict:
    if endpoint == "embed":
        return {"embeddings": {"float": [_pseudo_vector(t) for t in params["texts"]]}}

    if endpoint == "rerank":
        query = params["query"]
        docs = params["documents"]
        scored = sorted(
            range(len(docs)), key=lambda i: _overlap_score(query, docs[i]), reverse=True
        )
        top_n = params.get("top_n", 5)
        return {
            "results": [
                {"index": i, "relevance_score": _overlap_score(query, docs[i])} for i in scored[:top_n]
            ]
        }

    if endpoint == "chat":
        documents = params.get("documents") or []
        prompt = params["messages"][-1]["content"]
        if not documents:
            return {"message": {"content": [{"type": "text", "text": "NO_ANSWER"}], "citations": []}}
        best = max(documents, key=lambda d: _overlap_score(prompt, d["data"]["text"]))
        snippet = best["data"]["text"].split(".")[0]
        return {
            "message": {
                "content": [{"type": "text", "text": snippet}],
                "citations": [
                    {
                        "start": 0,
                        "end": len(snippet),
                        "text": snippet,
                        "sources": [{"type": "document", "id": best["id"]}],
                    }
                ],
            }
        }

    if endpoint == "tokenize":
        return {"tokens": list(range(max(1, len(params["text"].split()))))}

    raise ValueError(f"fake_transport: unknown endpoint {endpoint}")
