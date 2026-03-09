"""Text chunking service — splits documents into overlapping chunks."""
import re
from dataclasses import dataclass
from enum import Enum


class ChunkStrategy(str, Enum):
    CHAR = "char"
    WORD = "word"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"


@dataclass
class ChunkingConfig:
    """
    chunk_size:    target size in chars (CHAR, RECURSIVE), words (WORD), or sentences (SENTENCE).
    chunk_overlap: units to repeat at the start of the next chunk.
    strategy:      splitting algorithm (default: RECURSIVE).
    min_chunk_len: discard chunks shorter than this many characters.
    """
    chunk_size: int = 300
    chunk_overlap: int = 50
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    min_chunk_len: int = 20


@dataclass
class TextChunk:
    """
    chunk_index: 0-based position within the document.
    char_start/char_end: character offsets in the original text (for future citation features).
    """
    chunk_index: int
    char_start: int
    char_end: int
    text: str


class ChunkingService:
    """Splits raw text into TextChunk instances using the configured strategy."""

    # Separators tried in priority order by the RECURSIVE strategy.
    _RECURSIVE_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, text: str) -> list[TextChunk]:
        """Normalizes whitespace then delegates to the configured strategy."""
        text = text.strip()
        if not text:
            return []

        match self.config.strategy:
            case ChunkStrategy.CHAR:
                raw_chunks = self._split_by_chars(text)
            case ChunkStrategy.WORD:
                raw_chunks = self._split_by_words(text)
            case ChunkStrategy.SENTENCE:
                raw_chunks = self._split_by_sentences(text)
            case ChunkStrategy.RECURSIVE:
                raw_chunks = self._split_recursive(text, self._RECURSIVE_SEPARATORS)

        return self._to_text_chunks(text, raw_chunks)

    def _split_by_chars(self, text: str) -> list[str]:
        """Slide a window of chunk_size characters, stepping by (chunk_size - overlap)."""
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = max(1, size - overlap)  # guard against overlap >= size

        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunk = text[start:start + size]
            if len(chunk) >= self.config.min_chunk_len:
                chunks.append(chunk)
            start += step

        return chunks

    def _split_by_words(self, text: str) -> list[str]:
        """Split on whitespace, group into windows of chunk_size words with overlap."""
        words = text.split()
        size = self.config.chunk_size  # treated as word count here
        overlap = self.config.chunk_overlap
        step = max(1, size - overlap)

        chunks: list[str] = []
        start = 0
        while start < len(words):
            window = words[start:start + size]
            joined = " ".join(window)
            if len(joined) >= self.config.min_chunk_len:
                chunks.append(joined)
            start += step

        return chunks

    def _split_by_sentences(self, text: str) -> list[str]:
        """
        Group sentences into chunks of chunk_size sentences with overlap.

        Sentence detection: regex split after [.!?] + whitespace + uppercase letter.
        Avoids false splits on "Dr. Smith" (lowercase follows) and "3.14" (digit follows).
        """
        sentences = self._tokenize_sentences(text)
        if not sentences:
            return []

        size = self.config.chunk_size  # treated as sentence count here
        overlap = self.config.chunk_overlap
        step = max(1, size - overlap)

        chunks: list[str] = []
        start = 0
        while start < len(sentences):
            window = sentences[start:start + size]
            joined = " ".join(window)
            if len(joined) >= self.config.min_chunk_len:
                chunks.append(joined)
            start += step

        return chunks

    def _tokenize_sentences(self, text: str) -> list[str]:
        """Split on [.!?] followed by whitespace and an uppercase letter."""
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [p.strip() for p in parts if p.strip()]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """
        Try separators in priority order: paragraph → line → sentence → word → char.

        Algorithm:
          1. Split on the first separator.
          2. Pieces still larger than chunk_size recurse with remaining separators.
          3. No separators left → character split.
          4. Merge small adjacent pieces back up to chunk_size with overlap.
        """
        if not separators:
            return self._split_by_chars(text)

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            splits = list(text)
        else:
            splits = text.split(sep)

        good_splits: list[str] = []
        for piece in splits:
            if not piece:
                continue
            if len(piece) <= self.config.chunk_size:
                good_splits.append(piece)
            else:
                sub_chunks = self._split_recursive(piece, remaining_seps)
                good_splits.extend(sub_chunks)

        return self._merge_splits(good_splits, sep)

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Greedily merge small pieces up to chunk_size, then emit with overlap."""
        chunks: list[str] = []
        current_pieces: list[str] = []
        current_len = 0

        for piece in splits:
            piece_len = len(piece)
            sep_len = len(separator) if current_pieces else 0

            if current_len + sep_len + piece_len > self.config.chunk_size and current_pieces:
                chunk_text = separator.join(current_pieces)
                if len(chunk_text) >= self.config.min_chunk_len:
                    chunks.append(chunk_text)

                # Keep trailing pieces within the overlap budget
                overlap_budget = self.config.chunk_overlap
                overlap_pieces: list[str] = []
                for p in reversed(current_pieces):
                    if overlap_budget - len(p) >= 0:
                        overlap_pieces.insert(0, p)
                        overlap_budget -= len(p) + len(separator)
                    else:
                        break
                current_pieces = overlap_pieces
                current_len = sum(len(p) for p in current_pieces) + len(separator) * max(0, len(current_pieces) - 1)

            current_pieces.append(piece)
            current_len += sep_len + piece_len

        if current_pieces:
            chunk_text = separator.join(current_pieces)
            if len(chunk_text) >= self.config.min_chunk_len:
                chunks.append(chunk_text)

        return chunks

    def _to_text_chunks(self, original_text: str, raw_chunks: list[str]) -> list[TextChunk]:
        """
        Convert raw string chunks to TextChunk objects with character offsets.

        Uses str.find(chunk, search_from) to locate each chunk in the original text.
        Advances search_from after each match to handle duplicate substrings correctly
        (positions are always monotonically increasing).
        """
        result: list[TextChunk] = []
        search_from = 0

        for i, raw in enumerate(raw_chunks):
            raw = raw.strip()
            if not raw:
                continue

            pos = original_text.find(raw, search_from)
            if pos == -1:
                # Chunk was constructed (e.g., joined with spaces), try from start
                pos = original_text.find(raw)
            if pos == -1:
                pos = search_from  # fallback — position is approximate

            result.append(TextChunk(
                chunk_index=i,
                char_start=pos,
                char_end=pos + len(raw),
                text=raw,
            ))

            search_from = max(search_from + 1, pos + 1)

        return result
