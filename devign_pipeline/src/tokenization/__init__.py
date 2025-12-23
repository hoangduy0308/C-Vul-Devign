"""Tokenization utilities for C code"""

from .tokenizer import Tokenizer, tokenize_code
from .normalization import Normalizer, normalize_code
from .vocab import Vocabulary, VocabBuilder
from .vectorizer import (
    CodeVectorizer,
    VectorizerConfig,
    EncodedSample,
    vectorize_chunk,
    vectorize_dataset_parallel,
    StreamingVectorizer,
)
from .preserve_tokenizer import (
    PreserveIdentifierTokenizer,
    build_preserve_vocab,
    vectorize_preserve,
    vectorize_batch_preserve,
    FORCE_KEEP_IDENTIFIERS,
    PRESERVED_NUMBERS,
    PRESERVED_HEX,
    PRESERVED_OCTAL,
    DEFAULT_CONFIG as PRESERVE_TOKENIZER_CONFIG,
)

__all__ = [
    "Tokenizer",
    "tokenize_code",
    "Normalizer",
    "normalize_code",
    "Vocabulary",
    "VocabBuilder",
    "CodeVectorizer",
    "VectorizerConfig",
    "EncodedSample",
    "vectorize_chunk",
    "vectorize_dataset_parallel",
    "StreamingVectorizer",
    # Preserve tokenizer
    "PreserveIdentifierTokenizer",
    "build_preserve_vocab",
    "vectorize_preserve",
    "vectorize_batch_preserve",
    "FORCE_KEEP_IDENTIFIERS",
    "PRESERVED_NUMBERS",
    "PRESERVED_HEX",
    "PRESERVED_OCTAL",
    "PRESERVE_TOKENIZER_CONFIG",
]
