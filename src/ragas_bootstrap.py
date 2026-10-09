"""Patch missing langchain_community.chat_models.vertexai before importing ragas."""

from __future__ import annotations

import sys
import types


def ensure_ragas_importable() -> None:
    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name in sys.modules:
        return
    pkg_name = "langchain_community.chat_models"
    if pkg_name not in sys.modules:
        sys.modules[pkg_name] = types.ModuleType(pkg_name)
    stub = types.ModuleType(mod_name)

    class ChatVertexAI:  # noqa: D106 — ragas imports this symbol; we do not use Vertex.
        pass

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[mod_name] = stub
