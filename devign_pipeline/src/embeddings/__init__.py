"""Embedding utilities for vulnerability detection."""

from .word2vec import (
    train_word2vec,
    create_embedding_matrix,
    save_word2vec_outputs,
    tokens_from_input_ids,
    DEFAULT_W2V_CONFIG,
)

__all__ = [
    "train_word2vec",
    "create_embedding_matrix",
    "save_word2vec_outputs",
    "tokens_from_input_ids",
    "DEFAULT_W2V_CONFIG",
]
