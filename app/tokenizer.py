"""Token counting shared by chunking and context budgeting."""
from __future__ import annotations

from typing import Protocol

import tiktoken


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


class TiktokenTokenizer:
    def __init__(self, encoding: str = "cl100k_base") -> None:
        self._enc = tiktoken.get_encoding(encoding)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)
