import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.chunking import chunk_text


def test_short_text_is_one_chunk() -> None:
    assert list(chunk_text("one two three", chunk_size=4, overlap=1)) == [
        "one two three"
    ]


def test_chunks_have_expected_overlap() -> None:
    chunks = list(
        chunk_text("one two three four five six", chunk_size=4, overlap=2)
    )
    assert chunks == ["one two three four", "three four five six"]


def test_final_chunk_can_be_shorter() -> None:
    chunks = list(
        chunk_text(
            "one two three four five six seven",
            chunk_size=4,
            overlap=2,
        )
    )
    assert chunks[-1] == "five six seven"


def test_empty_text_has_no_chunks() -> None:
    assert list(chunk_text("")) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (4, -1), (4, 4), (4, 5)],
)
def test_invalid_window_configuration(
    chunk_size: int, overlap: int
) -> None:
    with pytest.raises(ValueError):
        list(chunk_text("text", chunk_size, overlap))
