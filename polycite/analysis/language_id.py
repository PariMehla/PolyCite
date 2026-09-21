"""Script-based language-fidelity check — a v1 stand-in for GlotLID/fastText.

Real language ID (GlotLID, fastText's lid.176) needs a model download from
huggingface.co, blocked in this sandbox. Unicode script detection is a real,
zero-dependency fallback that works perfectly for the 4 non-Latin languages
in this benchmark (zho_Hans, arb_Arab, hin_Deva, ben_Beng) since their
scripts are unique in the language set. It CANNOT distinguish the 4
Latin-script languages (eng_Latn, fra_Latn, swh_Latn, yor_Latn) from each
other — for those, `detect_script` returns "Latn" and the caller should
either treat language-fidelity checks as inconclusive or swap in a real
langid model before trusting results on those languages.
"""
from __future__ import annotations

import unicodedata

_SCRIPT_TO_LANGUAGE = {
    "Han": "zho_Hans",
    "Arabic": "arb_Arab",
    "Devanagari": "hin_Deva",
    "Bengali": "ben_Beng",
}


def _char_script(ch: str) -> str | None:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    if "CJK" in name or name.startswith("HANGZHOU") or name.startswith("KANGXI"):
        return "Han"
    for script in ("ARABIC", "DEVANAGARI", "BENGALI"):
        if script in name:
            return script.title()
    if ch.isalpha():
        return "Latn"
    return None


def detect_script(text: str) -> str | None:
    counts: dict[str, int] = {}
    for ch in text:
        script = _char_script(ch)
        if script:
            counts[script] = counts.get(script, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def detect_language_coarse(text: str) -> str | None:
    """Returns a Belebele language code for unique-script languages, "Latn"
    (ambiguous) for Latin-script text, or None if undetectable."""
    script = detect_script(text)
    if script is None:
        return None
    return _SCRIPT_TO_LANGUAGE.get(script, script)
