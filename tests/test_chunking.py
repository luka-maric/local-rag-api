import pytest

from app.services.chunking import (
    ChunkingConfig,
    ChunkingService,
    ChunkStrategy,
    TextChunk,
)


def make_service(
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: int = 100,
    chunk_overlap: int = 20,
    min_chunk_len: int = 5,
) -> ChunkingService:
    return ChunkingService(ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
        min_chunk_len=min_chunk_len,
    ))


def test_empty_string_returns_empty_list():
    service = make_service()
    assert service.chunk("") == []


def test_whitespace_only_returns_empty_list():
    service = make_service()
    assert service.chunk("   \n\n\t  ") == []


def test_short_text_becomes_single_chunk():
    service = make_service(chunk_size=500)
    text = "This is a short document."
    chunks = service.chunk(text)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_index == 0


def test_char_strategy_produces_multiple_chunks():
    service = make_service(strategy=ChunkStrategy.CHAR, chunk_size=20, chunk_overlap=5, min_chunk_len=1)
    text = "A" * 60
    chunks = service.chunk(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 20


def test_char_strategy_overlap_content_repeated():
    text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 52 chars
    service = make_service(
        strategy=ChunkStrategy.CHAR,
        chunk_size=20,
        chunk_overlap=5,
        min_chunk_len=1,
    )
    chunks = service.chunk(text)

    assert len(chunks) >= 2

    # The last 5 chars of chunk[0] must appear at the start of chunk[1]
    tail_of_first = chunks[0].text[-5:]
    assert chunks[1].text.startswith(tail_of_first), (
        f"Overlap not found: '{tail_of_first}' not at start of '{chunks[1].text[:10]}'"
    )


def test_word_strategy_no_mid_word_cuts():
    service = make_service(strategy=ChunkStrategy.WORD, chunk_size=5, chunk_overlap=1, min_chunk_len=1)
    text = "The quick brown fox jumped over the lazy dog near the river bank"
    original_words = set(text.split())

    chunks = service.chunk(text)

    assert len(chunks) > 1
    for chunk in chunks:
        for word in chunk.text.split():
            assert word in original_words, f"'{word}' is not a word from the original text"


def test_word_strategy_overlap_words_repeated():
    text = "one two three four five six seven eight nine ten"
    service = make_service(
        strategy=ChunkStrategy.WORD,
        chunk_size=4,
        chunk_overlap=2,
        min_chunk_len=1,
    )
    chunks = service.chunk(text)

    assert len(chunks) >= 2

    words_0 = chunks[0].text.split()
    tail_words = words_0[-2:]

    words_1 = chunks[1].text.split()
    start_words = words_1[:2]
    assert tail_words == start_words, (
        f"Word overlap not found: {tail_words} vs {start_words}"
    )


def test_sentence_strategy_does_not_cut_mid_sentence():
    service = make_service(
        strategy=ChunkStrategy.SENTENCE,
        chunk_size=2,
        chunk_overlap=0,
        min_chunk_len=1,
    )
    text = (
        "The quick brown fox jumped. "
        "It was a sunny day. "
        "The dog watched from afar. "
        "Nobody expected what happened next. "
        "The fox disappeared into the forest."
    )
    chunks = service.chunk(text)

    assert len(chunks) >= 2

    for chunk in chunks[:-1]:
        stripped = chunk.text.rstrip()
        assert stripped[-1] in ".!?", (
            f"Chunk does not end at sentence boundary: '{stripped[-20:]}'"
        )


def test_sentence_strategy_overlap_sentence_repeated():
    service = make_service(
        strategy=ChunkStrategy.SENTENCE,
        chunk_size=2,
        chunk_overlap=1,
        min_chunk_len=1,
    )
    text = (
        "First sentence here. "
        "Second sentence here. "
        "Third sentence here. "
        "Fourth sentence here."
    )
    chunks = service.chunk(text)

    assert len(chunks) >= 2

    assert "Second sentence here" in chunks[1].text, (
        f"Overlap sentence not found in chunk[1]: '{chunks[1].text}'"
    )


def test_recursive_strategy_prefers_paragraph_boundaries():
    para1 = "This is paragraph one. It has two sentences."
    para2 = "This is paragraph two. It also has two sentences."
    text = f"{para1}\n\n{para2}"

    service = make_service(
        strategy=ChunkStrategy.RECURSIVE,
        chunk_size=200,
        chunk_overlap=0,
        min_chunk_len=5,
    )
    chunks = service.chunk(text)

    texts = [c.text for c in chunks]
    assert any(para1 in t for t in texts), "Para1 was split when it should fit whole"
    assert any(para2 in t for t in texts), "Para2 was split when it should fit whole"


def test_recursive_strategy_on_realistic_text():
    text = """
    Artificial intelligence has transformed many industries over the past decade.
    Machine learning models can now perform tasks that were once considered exclusively human.

    Natural language processing allows computers to understand and generate human language.
    This includes translation, summarization, and question answering.

    Retrieval-augmented generation combines the strengths of search and language models.
    Instead of relying on the model's training data alone, RAG retrieves relevant documents first.
    This makes answers more accurate and grounded in specific knowledge bases.

    Vector databases store embeddings — numerical representations of text.
    Similarity search finds the closest vectors to a query, returning the most relevant chunks.
    The quality of chunking directly affects retrieval precision.
    """.strip()

    service = make_service(
        strategy=ChunkStrategy.RECURSIVE,
        chunk_size=300,
        chunk_overlap=50,
        min_chunk_len=20,
    )
    chunks = service.chunk(text)

    assert len(chunks) >= 1

    # No chunk should be dramatically oversized (allow 50% tolerance for edge cases)
    for chunk in chunks:
        assert len(chunk.text) <= service.config.chunk_size * 1.5, (
            f"Chunk is too large: {len(chunk.text)} chars"
        )

    for expected_i, chunk in enumerate(chunks):
        assert chunk.chunk_index == expected_i


def test_text_chunk_metadata_positions():
    text = "Hello world. This is a test document. It has multiple sentences."
    service = make_service(
        strategy=ChunkStrategy.SENTENCE,
        chunk_size=1,
        chunk_overlap=0,
        min_chunk_len=1,
    )
    chunks = service.chunk(text)

    assert len(chunks) >= 1

    for chunk in chunks:
        assert isinstance(chunk, TextChunk)
        assert chunk.char_start >= 0
        assert chunk.char_end <= len(text)
        assert chunk.char_start < chunk.char_end
        extracted = text[chunk.char_start:chunk.char_end]
        assert chunk.text in extracted or extracted in chunk.text, (
            f"Position mismatch: extracted='{extracted}', chunk='{chunk.text}'"
        )


def test_min_chunk_len_discards_short_chunks():
    text = "Hello world."
    service = ChunkingService(ChunkingConfig(
        strategy=ChunkStrategy.CHAR,
        chunk_size=10,
        chunk_overlap=0,
        min_chunk_len=10,
    ))
    chunks = service.chunk(text)

    for chunk in chunks:
        assert len(chunk.text) >= 10, f"Chunk shorter than min_chunk_len: '{chunk.text}'"


def test_chunk_indices_are_unique_and_sequential():
    service = make_service(
        strategy=ChunkStrategy.RECURSIVE,
        chunk_size=50,
        chunk_overlap=10,
        min_chunk_len=5,
    )
    text = " ".join([f"Sentence number {i} is here." for i in range(20)])
    chunks = service.chunk(text)

    assert len(chunks) >= 2

    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks))), (
        f"Indices not sequential: {indices}"
    )
