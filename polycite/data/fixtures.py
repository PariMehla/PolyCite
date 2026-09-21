"""Synthetic 8-language mini dataset used for tests and offline dry runs.

This repo was built in a sandbox with no egress to huggingface.co, so the
real Belebele download (data/belebele.py) has never been exercised end to
end. These fixtures give every downstream module (retrieval, rerank,
generation, scoring, attribution) something real to run against, so the
pipeline's *logic* is proven correct even before anyone has internet + a
Cohere key. They deliberately mirror Belebele's shape: one passage, one
question, four options, per language, with parallel content across languages
(shorter and much smaller, so tests run instantly).
"""
from __future__ import annotations

from dataclasses import dataclass

LANGUAGES = ["eng_Latn", "fra_Latn", "zho_Hans", "arb_Arab", "hin_Deva", "ben_Beng", "swh_Latn", "yor_Latn"]


@dataclass(frozen=True)
class Passage:
    passage_id: str
    language: str
    text: str


@dataclass(frozen=True)
class Question:
    question_id: str
    language: str
    passage_id: str
    question: str
    options: list[str]
    answer_index: int  # 0-3, into options


# Five short parallel "articles" (English original + placeholder translations).
# Not real translations — this is a structural fixture, not a linguistic one.
_ARTICLES = {
    "p1": {
        "eng_Latn": "The city built a new library downtown. It opens every day at nine in the morning.",
        "fra_Latn": "[fr] La ville a construit une nouvelle bibliotheque au centre-ville. Elle ouvre tous les jours a neuf heures.",
        "zho_Hans": "[zh] 这座城市在市中心建了一座新图书馆。它每天早上九点开放。",
        "arb_Arab": "[ar] بنت المدينة مكتبة جديدة في وسط المدينة. تفتح كل يوم الساعة التاسعة صباحا.",
        "hin_Deva": "[hi] शहर ने डाउनटाउन में एक नया पुस्तकालय बनाया। यह हर दिन सुबह नौ बजे खुलता है।",
        "ben_Beng": "[bn] শহরটি ডাউনটাউনে একটি নতুন গ্রন্থাগার তৈরি করেছে। এটি প্রতিদিন সকাল নয়টায় খোলে।",
        "swh_Latn": "[sw] Jiji lilijenga maktaba mpya katikati ya mji. Inafunguliwa kila siku saa tatu asubuhi.",
        "yor_Latn": "[yo] Ilu naa ko ile-ikawe tuntun ni aarin ilu. O n si ni gbogbo owuro ago mesan.",
        "answer": "nine in the morning",
        "distractors": ["six in the evening", "noon", "midnight"],
        "question": "What time does the new library open?",
    },
    "p2": {
        "eng_Latn": "Farmers in the valley plant rice in early spring and harvest it in late summer.",
        "fra_Latn": "[fr] Les agriculteurs de la vallee plantent du riz au debut du printemps et le recoltent a la fin de l'ete.",
        "zho_Hans": "[zh] 山谷里的农民在初春种植水稻，在夏末收获。",
        "arb_Arab": "[ar] يزرع المزارعون في الوادي الأرز في أوائل الربيع ويحصدونه في أواخر الصيف.",
        "hin_Deva": "[hi] घाटी के किसान शुरुआती वसंत में चावल बोते हैं और देर से गर्मियों में इसकी कटाई करते हैं।",
        "ben_Beng": "[bn] উপত্যকার কৃষকরা বসন্তের শুরুতে ধান রোপণ করে এবং গ্রীষ্মের শেষে ফসল কাটে।",
        "swh_Latn": "[sw] Wakulima katika bonde hupanda mchele mwanzoni mwa masika na kuuvuna mwishoni mwa kiangazi.",
        "yor_Latn": "[yo] Awon agbe ni afonifoji gbin iresi ni ibere orun ati ki won ka a ni opin ogbele.",
        "answer": "late summer",
        "distractors": ["early winter", "mid spring", "late autumn"],
        "question": "When do the farmers harvest the rice?",
    },
    "p3": {
        "eng_Latn": "The museum's new exhibit features paintings donated by a local collector last year.",
        "fra_Latn": "[fr] La nouvelle exposition du musee presente des peintures donnees par un collectionneur local l'annee derniere.",
        "zho_Hans": "[zh] 博物馆的新展览展出了去年一位当地收藏家捐赠的画作。",
        "arb_Arab": "[ar] يعرض المعرض الجديد للمتحف لوحات تبرع بها جامع محلي العام الماضي.",
        "hin_Deva": "[hi] संग्रहालय की नई प्रदर्शनी में पिछले साल एक स्थानीय संग्रहकर्ता द्वारा दान की गई पेंटिंग्स हैं।",
        "ben_Beng": "[bn] জাদুঘরের নতুন প্রদর্শনীতে গত বছর একজন স্থানীয় সংগ্রাহকের দান করা চিত্রকর্ম রয়েছে।",
        "swh_Latn": "[sw] Onyesho jipya la makumbusho lina michoro iliyotolewa na mkusanyaji wa eneo hilo mwaka jana.",
        "yor_Latn": "[yo] Ifihan tuntun ile-ipamo naa ni awon aworan ti eni ti n gba aworan agbegbe fi funni ni odun to koja.",
        "answer": "a local collector",
        "distractors": ["the museum's founder", "a foreign government", "the city council"],
        "question": "Who donated the paintings in the new exhibit?",
    },
}


def build_fixture_dataset() -> tuple[list[Passage], list[Question]]:
    passages: list[Passage] = []
    questions: list[Question] = []
    for pid, article in _ARTICLES.items():
        for lang in LANGUAGES:
            passages.append(Passage(passage_id=f"{pid}_{lang}", language=lang, text=article[lang]))
            options = [article["answer"], *article["distractors"]]
            # deterministic shuffle so answer isn't always index 0
            order = [(i + hash(f"{pid}{lang}") ) % 4 for i in range(4)]
            seen = []
            for o in order:
                if o not in seen:
                    seen.append(o)
            for i in range(4):
                if i not in seen:
                    seen.append(i)
            shuffled = [options[i] for i in seen[:4]]
            answer_index = shuffled.index(article["answer"])
            questions.append(
                Question(
                    question_id=f"{pid}_{lang}_q",
                    language=lang,
                    passage_id=f"{pid}_{lang}",
                    question=article["question"] if lang == "eng_Latn" else f"[{lang}] {article['question']}",
                    options=shuffled,
                    answer_index=answer_index,
                )
            )
    return passages, questions
