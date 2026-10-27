"""Chat-message source adapters. Only the interface and an offline fake live here;
a real Telegram/Slack adapter would implement `fetch` against the provider's API."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    id: str
    author: str
    text: str


class ChatAdapter(ABC):
    name = "chat"

    @abstractmethod
    def fetch(self) -> Iterable[ChatMessage]:
        """Return new messages. Re-delivery is safe: ingestion is idempotent."""


class FakeChatAdapter(ChatAdapter):
    name = "fake-chat"

    def __init__(self, messages: Iterable[ChatMessage]) -> None:
        self._messages = list(messages)

    def fetch(self) -> Iterable[ChatMessage]:
        return list(self._messages)
