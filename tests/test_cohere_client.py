import pytest

from polycite.cohere_client import (
    BatchNotConfirmedError,
    BudgetExceededError,
    CohereClient,
    estimate_and_confirm,
)


def _counting_transport():
    calls = {"n": 0}

    def transport(endpoint, model, params):
        calls["n"] += 1
        if endpoint == "embed":
            return {"embeddings": {"float": [[0.1, 0.2] for _ in params["texts"]]}}
        if endpoint == "rerank":
            return {"results": [{"index": 0, "relevance_score": 0.9}]}
        if endpoint == "chat":
            return {"message": {"content": [{"type": "text", "text": "ok"}], "citations": []}}
        raise ValueError(endpoint)

    return transport, calls


def test_embed_hits_transport_once_and_caches(tmp_path):
    transport, calls = _counting_transport()
    client = CohereClient(cache_dir=tmp_path, transport=transport)
    client.embed(model="embed-v4.0", input_type="search_document", texts=["hello"])
    client.embed(model="embed-v4.0", input_type="search_document", texts=["hello"])
    assert calls["n"] == 1  # second call served from disk cache
    assert client.counter.count == 1


def test_different_params_are_not_conflated(tmp_path):
    transport, calls = _counting_transport()
    client = CohereClient(cache_dir=tmp_path, transport=transport)
    client.embed(model="embed-v4.0", input_type="search_document", texts=["a"])
    client.embed(model="embed-v4.0", input_type="search_document", texts=["b"])
    assert calls["n"] == 2
    assert client.counter.count == 2


def test_budget_exceeded_raises(tmp_path):
    transport, _ = _counting_transport()
    client = CohereClient(cache_dir=tmp_path, transport=transport, budget=1)
    client.embed(model="embed-v4.0", input_type="search_document", texts=["a"])
    with pytest.raises(BudgetExceededError):
        client.embed(model="embed-v4.0", input_type="search_document", texts=["b"])


def test_cached_call_does_not_count_against_budget(tmp_path):
    transport, _ = _counting_transport()
    client = CohereClient(cache_dir=tmp_path, transport=transport, budget=1)
    client.embed(model="embed-v4.0", input_type="search_document", texts=["a"])
    # same call again: should be served from cache, not blocked by budget=1
    client.embed(model="embed-v4.0", input_type="search_document", texts=["a"])
    assert client.counter.count == 1


def test_embed_rejects_batches_over_max_texts(tmp_path):
    transport, _ = _counting_transport()
    client = CohereClient(cache_dir=tmp_path, transport=transport)
    with pytest.raises(ValueError):
        client.embed(model="embed-v4.0", input_type="search_document", texts=["x"] * 97)


def test_call_counter_persists_across_client_instances(tmp_path):
    transport, _ = _counting_transport()
    client1 = CohereClient(cache_dir=tmp_path, transport=transport)
    client1.embed(model="embed-v4.0", input_type="search_document", texts=["a"])
    client2 = CohereClient(cache_dir=tmp_path, transport=transport)
    assert client2.counter.count == 1


def test_rerank_and_chat_round_trip(tmp_path):
    transport, _ = _counting_transport()
    client = CohereClient(cache_dir=tmp_path, transport=transport)
    rerank_resp = client.rerank(model="rerank-v3.5", query="q", documents=["d1", "d2"], top_n=1)
    assert rerank_resp["results"][0]["index"] == 0
    chat_resp = client.chat(model="command-a", messages=[{"role": "user", "content": "hi"}])
    assert chat_resp["message"]["content"][0]["text"] == "ok"


def test_estimate_and_confirm_auto_confirm_true_does_not_prompt():
    estimate_and_confirm(100, "test batch", auto_confirm=True)  # must not raise/hang


def test_estimate_and_confirm_raises_when_not_confirmable(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(BatchNotConfirmedError):
        estimate_and_confirm(100, "test batch", auto_confirm=False)


class _FlakySDK:
    """Fails with a 429 for the first `fail_times` calls, then succeeds.
    Used to test the real-SDK retry path, which transport-based tests
    (everything else in this file) never touch -- `transport=` bypasses the
    SDK call and its surrounding retry loop entirely."""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def embed(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            err = RuntimeError("trial token rate limit exceeded")
            err.status_code = 429
            raise err

        class _Resp:
            def dict(self):
                return {"embeddings": {"float": [[0.1, 0.2]]}}

        return _Resp()


def test_retries_on_429_and_eventually_succeeds(tmp_path, monkeypatch):
    # Regression test: a real live run hit exactly this (embed's token-volume
    # trial limit) and the backoff needs to be strong enough to survive it.
    monkeypatch.setattr("time.sleep", lambda s: None)  # don't actually wait in tests
    client = CohereClient(cache_dir=tmp_path)
    fake_sdk = _FlakySDK(fail_times=2)
    client._client = fake_sdk  # bypasses _sdk_client()'s API-key requirement
    resp = client.embed(model="embed-v", input_type="search_document", texts=["hello"])
    assert resp["embeddings"]["float"] == [[0.1, 0.2]]
    assert fake_sdk.calls == 3  # 2 failures + 1 success


def test_raises_after_exhausting_all_retries_on_persistent_429(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = CohereClient(cache_dir=tmp_path)
    fake_sdk = _FlakySDK(fail_times=1000)  # never succeeds
    client._client = fake_sdk
    with pytest.raises(RuntimeError):
        client.embed(model="embed-v", input_type="search_document", texts=["hello"])
    assert fake_sdk.calls == 8  # max_attempts=7 retries + the initial try
