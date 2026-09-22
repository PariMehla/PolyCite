"""Grounded generation via Cohere's native `documents` parameter.

Using `documents=` (rather than stuffing passages into the prompt as text)
gets structured citations back in `message.citations[i].sources[*].id`,
which we map to passage IDs for deterministic citation scoring — no string
matching, no judge needed for attribution.

Cohere rejects document ids over 100 characters (confirmed against the live
API: `invalid request: document id length must be less than 100`). Our own
passage IDs are built from full Belebele/Wikipedia URLs and routinely
exceed that, so we never send them to Cohere directly — build_documents()
assigns short positional ids ("d0", "d1", ...) and returns the map back to
real passage IDs, which generate_answer uses to translate citations.
"""
from __future__ import annotations

from pathlib import Path

from polycite.cohere_client import CohereClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "answer_with_citations.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()

NO_ANSWER_TOKEN = "NO_ANSWER"

# Belebele language codes -> a name to drop into the prompt so the model
# knows what "answer in {answer_language}" means. Extend as needed.
LANGUAGE_NAMES = {
    "eng_Latn": "English",
    "fra_Latn": "French",
    "zho_Hans": "Simplified Chinese",
    "arb_Arab": "Arabic",
    "hin_Deva": "Hindi",
    "ben_Beng": "Bengali",
    "swh_Latn": "Swahili",
    "yor_Latn": "Yoruba",
}


def build_documents(candidates: list[tuple[str, str]]) -> tuple[list[dict], dict[str, str]]:
    """candidates: list of (passage_id, text) -> (Cohere Document objects with
    short positional ids, {positional_id: real_passage_id})."""
    id_map: dict[str, str] = {}
    documents: list[dict] = []
    for i, (passage_id, text) in enumerate(candidates):
        doc_id = f"d{i}"
        id_map[doc_id] = passage_id
        documents.append({"id": doc_id, "data": {"text": text}})
    return documents, id_map


def generate_answer(
    client: CohereClient,
    question: str,
    query_language: str,
    candidates: list[tuple[str, str]],
    model: str = "command-a-03-2025",
    temperature: float = 0.0,
    prompt_template: str | None = None,
) -> dict:
    """Returns {"text": str, "cited_passage_ids": set[str], "abstained": bool}.

    prompt_template overrides the default prompt (must have the same
    {answer_language}/{question} placeholders) -- used by
    scripts/test_anti_abstention_prompt.py to A/B a prompt variant without
    touching the validated default."""
    answer_language = LANGUAGE_NAMES.get(query_language, query_language)
    template = prompt_template if prompt_template is not None else _PROMPT_TEMPLATE
    prompt = template.format(answer_language=answer_language, question=question)
    documents, id_map = build_documents(candidates)
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        documents=documents,
        temperature=temperature,
    )
    message = resp["message"]
    text_parts = [c["text"] for c in message.get("content") or [] if c.get("type") == "text"]
    text = "".join(text_parts).strip()

    cited_ids: set[str] = set()
    for citation in message.get("citations") or []:
        for source in citation.get("sources") or []:
            raw_id = source.get("id")
            if raw_id in id_map:
                cited_ids.add(id_map[raw_id])

    abstained = text.strip().upper() == NO_ANSWER_TOKEN
    return {"text": text, "cited_passage_ids": cited_ids, "abstained": abstained, "raw": resp}
