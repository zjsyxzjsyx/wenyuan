import pytest

from app.ingest.chunking.fixed_token import ChunkTokenLimitExceededError, chunk_by_token_size


class CharTokenizer:
    """One token per character, so window boundaries are easy to reason about."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


tok = CharTokenizer()


def test_windows_overlap():
    chunks = chunk_by_token_size(tok, "abcdefghij", chunk_token_size=4, chunk_overlap_token_size=1)
    assert [c["content"] for c in chunks] == ["abcd", "defg", "ghij", "j"]
    assert [c["chunk_order_index"] for c in chunks] == [0, 1, 2, 3]
    assert chunks[-1]["tokens"] == 1


def test_split_by_character_keeps_small_pieces_whole():
    chunks = chunk_by_token_size(tok, "ab\n\ncdefgh", chunk_token_size=4, chunk_overlap_token_size=0,
                                 split_by_character="\n\n")
    assert [c["content"] for c in chunks] == ["ab", "cdef", "gh"]


def test_split_by_character_only_rejects_oversized_piece():
    with pytest.raises(ChunkTokenLimitExceededError):
        chunk_by_token_size(tok, "ab\n\ncdefgh", chunk_token_size=4, split_by_character="\n\n",
                            split_by_character_only=True)


@pytest.mark.parametrize("overlap", [-1, 4, 5])
def test_invalid_overlap(overlap):
    with pytest.raises(ValueError):
        chunk_by_token_size(tok, "abcdefgh", chunk_token_size=4, chunk_overlap_token_size=overlap)
