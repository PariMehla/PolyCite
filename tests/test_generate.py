from polycite.cohere_client import CohereClient
from polycite.generate.cohere_chat import build_documents, generate_answer


def test_build_documents_maps_id_and_text():
    docs = build_documents([("p1", "hello"), ("p2", "world")])
    assert docs == [{"id": "p1", "data": {"text": "hello"}}, {"id": "p2", "data": {"text": "world"}}]


def _fake_chat_transport(cited_id: str, text: str, abstain: bool = False):
    def transport(endpoint, model, params):
        assert endpoint == "chat"
        if abstain:
            return {"message": {"content": [{"type": "text", "text": "NO_ANSWER"}], "citations": []}}
        return {
            "message": {
                "content": [{"type": "text", "text": text}],
                "citations": [{"start": 0, "end": len(text), "text": text, "sources": [{"type": "document", "id": cited_id}]}],
            }
        }

    return transport


def test_generate_answer_parses_citations(tmp_path):
    client = CohereClient(cache_dir=tmp_path, transport=_fake_chat_transport("p1", "nine in the morning"))
    result = generate_answer(client, "What time?", "eng_Latn", [("p1", "the library opens at nine")])
    assert result["text"] == "nine in the morning"
    assert result["cited_passage_ids"] == {"p1"}
    assert result["abstained"] is False


def test_generate_answer_detects_abstention(tmp_path):
    client = CohereClient(cache_dir=tmp_path, transport=_fake_chat_transport("p1", "", abstain=True))
    result = generate_answer(client, "What time?", "eng_Latn", [("p1", "unrelated text")])
    assert result["abstained"] is True
    assert result["text"] == "NO_ANSWER"
