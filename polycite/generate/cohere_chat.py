"""Grounded generation via Cohere's native `documents` parameter.

Using `documents=` (rather than stuffing passages into the prompt as text)
gets structured citations back in `message.citations[i].sources[*].id`,
which we map straight to passage IDs for deterministic citation scoring —
no string matching, no judge needed for attribution.
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


def build_documents(candidates: list[tuple[str, str]]) -> list[dict]:
    """candidates: list of (passage_id, text) -> Cohere Document objects."""
    return [{"id": pid, "data": {"text": text}} for pid, text in candidates]


def generate_answer(
    client: CohereClient,
    question: str,
    query_language: str,
    candidates: list[tuple[str, str]],
    model: str = "command-a-03-2025",
    temperature: float = 0.0,
) -> dict:
    """Returns {"text": str, "cited_passage_ids": set[str], "abstained": bool}."""
    answer_language = LANGUAGE_NAMES.get(query_language, query_language)
    prompt = _PROMPT_TEMPLATE.format(answer_language=answer_language, question=question)
    documents = build_documents(candidates)
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
            if source.get("id"):
                cited_ids.add(source["id"])

    abstained = text.strip().upper() == NO_ANSWER_TOKEN
    return {"text": text, "cited_passage_ids": cited_ids, "abstained": abstained, "raw": resp}
