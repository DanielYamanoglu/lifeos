from lifeos.ingest import chunk_text


def test_empty():
    assert chunk_text("  \n ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_paragraphs_are_packed_and_bounded():
    text = "\n\n".join(f"Paragraph number {i} with some filler words." for i in range(30))
    chunks = chunk_text(text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert "Paragraph number 0" in chunks[0] and "Paragraph number 29" in chunks[-1]


def test_long_sentence_is_hard_split_with_overlap():
    chunks = chunk_text("x" * 1000, max_chars=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert sum(len(c) for c in chunks) >= 1000


def test_deterministic():
    text = "A. " * 400
    assert chunk_text(text) == chunk_text(text)
