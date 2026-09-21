from polycite.analysis.language_id import detect_language_coarse


def test_detects_chinese():
    assert detect_language_coarse("这座城市在市中心建了一座新图书馆") == "zho_Hans"


def test_detects_arabic():
    assert detect_language_coarse("بنت المدينة مكتبة جديدة") == "arb_Arab"


def test_detects_hindi():
    assert detect_language_coarse("शहर ने डाउनटाउन में एक नया पुस्तकालय बनाया") == "hin_Deva"


def test_detects_bengali():
    assert detect_language_coarse("শহরটি ডাউনটাউনে একটি নতুন গ্রন্থাগার তৈরি করেছে") == "ben_Beng"


def test_latin_script_is_ambiguous():
    assert detect_language_coarse("The city built a new library") == "Latn"


def test_empty_text_is_none():
    assert detect_language_coarse("") is None
