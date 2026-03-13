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
    chunk_size: int = 300
    chunk_overlap: int = 50
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    min_chunk_len: int = 20


@dataclass
class TextChunk:
    chunk_index: int
    char_start: int
    char_end: int
    text: str


class ChunkingService:
    _RECURSIVE_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, text: str) -> list[TextChunk]:
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
        size = self.config.chunk_size
        step = max(1, size - self.config.chunk_overlap)
        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunk = text[start:start + size]
            if len(chunk) >= self.config.min_chunk_len:
                chunks.append(chunk)
            start += step
        return chunks

    def _split_by_words(self, text: str) -> list[str]:
        words = text.split()
        size = self.config.chunk_size
        step = max(1, size - self.config.chunk_overlap)
        chunks: list[str] = []
        start = 0
        while start < len(words):
            joined = " ".join(words[start:start + size])
            if len(joined) >= self.config.min_chunk_len:
                chunks.append(joined)
            start += step
        return chunks

    def _split_by_sentences(self, text: str) -> list[str]:
        # Regex avoids false splits on "Dr. Smith" (lowercase follows) and "3.14" (digit follows)
        sentences = self._tokenize_sentences(text)
        if not sentences:
            return []
        size = self.config.chunk_size
        step = max(1, size - self.config.chunk_overlap)
        chunks: list[str] = []
        start = 0
        while start < len(sentences):
            joined = " ".join(sentences[start:start + size])
            if len(joined) >= self.config.min_chunk_len:
                chunks.append(joined)
            start += step
        return chunks

    def _tokenize_sentences(self, text: str) -> list[str]:
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [p.strip() for p in parts if p.strip()]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return self._split_by_chars(text)

        sep = separators[0]
        splits = list(text) if sep == "" else text.split(sep)

        good_splits: list[str] = []
        for piece in splits:
            if not piece:
                continue
            if len(piece) <= self.config.chunk_size:
                good_splits.append(piece)
            else:
                good_splits.extend(self._split_recursive(piece, separators[1:]))

        return self._merge_splits(good_splits, sep)

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
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
        result: list[TextChunk] = []
        search_from = 0

        for i, raw in enumerate(raw_chunks):
            raw = raw.strip()
            if not raw:
                continue

            pos = original_text.find(raw, search_from)
            if pos == -1:
                pos = original_text.find(raw)  # joined chunk — search from start
            if pos == -1:
                pos = search_from  # approximate fallback

            result.append(TextChunk(chunk_index=i, char_start=pos, char_end=pos + len(raw), text=raw))
            search_from = max(search_from + 1, pos + 1)

        return result
