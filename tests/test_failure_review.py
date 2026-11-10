import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.analysis.failure_review import term_overlap


def test_term_overlap_uses_whole_words_and_stopwords() -> None:
    found, missing = term_overlap(
        "The cats chase mice",
        "A cat",
        "Dogs chase the mouse.",
    )
    assert found == ["chase"]
    assert missing == ["cats", "mice"]
