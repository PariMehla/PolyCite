from polycite.cohere_client import CohereClient
from polycite.generate.cohere_chat import build_documents, generate_answer


def test_build_documents_assigns_short_positional_ids():
    docs, id_map = build_documents([("p1", "hello"), ("p2", "world")])
    assert docs == [{"id": "d0", "data": {"text": "hello"}}, {"id": "d1", "data": {"text": "world"}}]
    assert id_map == {"d0": "p1", "d1": "p2"}


def test_build_documents_never_sends_a_long_passage_id_to_cohere():
    # Real bug found against the live API: Cohere rejects document ids over
    # 100 chars, and our passage_ids are URL-based (Belebele `link` column)
    # and routinely exceed that.
    long_passage_id = "eng_Latn_" + "https://en.wikibooks.org/wiki/" + ("x" * 100)
    assert len(long_passage_id) > 100
    docs, id_map = build_documents([(long_passage_id, "some text")])
    assert len(docs[0]["id"]) < 100
    assert id_map[docs[0]["id"]] == long_passage_id


def _fake_chat_transport(echoed_doc_id: str, text: str, abstain: bool = False):
    """echoed_doc_id: the positional id ("d0", ...) the fake API echoes back
    in citations, exactly like the real API echoes back whatever id we sent."""

    def transport(endpoint, model, params):
        assert endpoint == "chat"
        if abstain:
            return {"message": {"content": [{"type": "text", "text": "NO_ANSWER"}], "citations": []}}
        return {
            "message": {
                "content": [{"type": "text", "text": text}],
                "citations": [
                    {"start": 0, "end": len(text), "text": text, "sources": [{"type": "document", "id": echoed_doc_id}]}
                ],
            }
        }

    return transport


def test_generate_answer_parses_citations(tmp_path):
    client = CohereClient(cache_dir=tmp_path, transport=_fake_chat_transport("d0", "nine in the morning"))
    result = generate_answer(client, "What time?", "eng_Latn", [("p1", "the library opens at nine")])
    assert result["text"] == "nine in the morning"
    assert result["cited_passage_ids"] == {"p1"}  # mapped back from "d0" to the real passage_id
    assert result["abstained"] is False


def test_generate_answer_detects_abstention(tmp_path):
    client = CohereClient(cache_dir=tmp_path, transport=_fake_chat_transport("d0", "", abstain=True))
    result = generate_answer(client, "What time?", "eng_Latn", [("p1", "unrelated text")])
    assert result["abstained"] is True
    assert result["text"] == "NO_ANSWER"
