"""
Extended Token Type System for C Vulnerability Detection

This module provides vulnerability-relevant token type classification.
Token types help the model understand the semantic role of each token.
"""

from enum import IntEnum
from typing import List, Optional
from .tokenizer import TokenType


class ExtendedTokenType(IntEnum):
    """
    Extended token type IDs for vulnerability detection.
    Maps to embedding lookup indices (0-15 range).
    """
    PAD = 0              # Padding token
    UNKNOWN = 1          # Unknown/default type
    KEYWORD = 2          # C/C++ keywords
    IDENTIFIER = 3       # General identifiers
    TYPE = 4             # Type names (int, char, etc.)
    OPERATOR = 5         # Operators (+, -, *, etc.)
    LITERAL = 6          # Numeric/string/char literals
    PUNCTUATION = 7      # Punctuation (;, {, }, etc.)
    DANGEROUS_CALL = 8   # Dangerous APIs (strcpy, sprintf, gets, etc.)
    SAFE_CALL = 9        # Safe APIs (strncpy, snprintf, etc.)
    ALLOC = 10           # Memory allocation (malloc, calloc, realloc)
    FREE = 11            # Memory deallocation (free)
    DEREF = 12           # Pointer dereference (*, ->)
    ARRAY_ACCESS = 13    # Array access ([])
    NULL_CHECK = 14      # NULL comparisons
    BOUNDS_CHECK = 15    # Size/length checks


# Total number of extended token types
NUM_EXTENDED_TOKEN_TYPES = 16


# ============================================================
# API Lists for Vulnerability Detection
# ============================================================

DANGEROUS_APIS = {
    # Buffer overflow prone
    'strcpy', 'strcat', 'sprintf', 'vsprintf', 'gets', 'scanf', 'fscanf',
    'sscanf', 'vscanf', 'vfscanf', 'vsscanf',
    # Format string vulnerabilities
    'printf', 'fprintf', 'vprintf', 'vfprintf',
    # Memory operations without size
    'memcpy', 'memmove', 'memset',
    # Deprecated/unsafe
    'atoi', 'atol', 'atof', 'strtok',
    # File operations
    'mktemp', 'tmpnam', 'tempnam',
    # Exec family
    'system', 'popen', 'execl', 'execle', 'execlp', 'execv', 'execve', 'execvp',
}

SAFE_APIS = {
    # Safe string functions
    'strncpy', 'strncat', 'snprintf', 'vsnprintf',
    # Safe memory functions
    'memcpy_s', 'memmove_s', 'memset_s',
    # Safe string functions (Windows)
    'strcpy_s', 'strcat_s', 'sprintf_s',
    # Safe input
    'fgets', 'getline',
    # Bounds-checked
    'strlcpy', 'strlcat',
}

ALLOC_APIS = {
    'malloc', 'calloc', 'realloc', 'aligned_alloc',
    'valloc', 'pvalloc', 'memalign', 'posix_memalign',
    # C++ allocation
    'new', 'new[]',
    # QEMU/FFmpeg specific
    'g_malloc', 'g_malloc0', 'g_new', 'g_new0',
    'av_malloc', 'av_mallocz', 'av_realloc',
}

FREE_APIS = {
    'free',
    # C++ deallocation
    'delete', 'delete[]',
    # QEMU/FFmpeg specific
    'g_free', 'av_free', 'av_freep',
}

# Null check related tokens
NULL_CHECK_TOKENS = {
    'NULL', 'null', 'nullptr', '0',
}

# Bounds check related functions
BOUNDS_CHECK_APIS = {
    'strlen', 'sizeof', 'size', 'length', 'len', 'count',
    'strnlen', 'wcslen', 'mbslen',
}

# Dereference operators
DEREF_OPERATORS = {'*', '->'}

# Array access operator
ARRAY_ACCESS_OPERATORS = {'[', ']'}

# Error handling keywords
ERROR_HANDLING_KEYWORDS = {'goto', 'error', 'err', 'fail', 'failed', 'failure'}


def get_extended_token_type(
    token_text: str,
    base_type: Optional[TokenType] = None,
    prev_token: Optional[str] = None,
    next_token: Optional[str] = None,
) -> int:
    """
    Get extended token type ID for a token.
    
    Args:
        token_text: The token text
        base_type: The base TokenType from the tokenizer (optional)
        prev_token: Previous token text for context (optional)
        next_token: Next token text for context (optional)
    
    Returns:
        Integer type ID (0-15) for embedding lookup
    """
    if not token_text:
        return ExtendedTokenType.PAD
    
    text_lower = token_text.lower()
    
    # Check for dangerous APIs first (highest priority for vulnerability detection)
    if text_lower in DANGEROUS_APIS or token_text in DANGEROUS_APIS:
        return ExtendedTokenType.DANGEROUS_CALL
    
    # Check for safe APIs
    if text_lower in SAFE_APIS or token_text in SAFE_APIS:
        return ExtendedTokenType.SAFE_CALL
    
    # Check for allocation APIs
    if text_lower in ALLOC_APIS or token_text in ALLOC_APIS:
        return ExtendedTokenType.ALLOC
    
    # Check for free APIs
    if text_lower in FREE_APIS or token_text in FREE_APIS:
        return ExtendedTokenType.FREE
    
    # Check for bounds check related tokens
    if text_lower in BOUNDS_CHECK_APIS or token_text in BOUNDS_CHECK_APIS:
        return ExtendedTokenType.BOUNDS_CHECK
    
    # Check for NULL check tokens
    if token_text in NULL_CHECK_TOKENS or text_lower == 'null':
        return ExtendedTokenType.NULL_CHECK
    
    # Check for dereference operators
    if token_text in DEREF_OPERATORS:
        return ExtendedTokenType.DEREF
    
    # Check for array access
    if token_text in ARRAY_ACCESS_OPERATORS:
        return ExtendedTokenType.ARRAY_ACCESS
    
    # Map base TokenType to ExtendedTokenType
    if base_type is not None:
        if base_type == TokenType.KEYWORD:
            return ExtendedTokenType.KEYWORD
        elif base_type == TokenType.IDENTIFIER:
            return ExtendedTokenType.IDENTIFIER
        elif base_type == TokenType.TYPE:
            return ExtendedTokenType.TYPE
        elif base_type == TokenType.OPERATOR:
            return ExtendedTokenType.OPERATOR
        elif base_type in (TokenType.LITERAL_NUM, TokenType.LITERAL_STR, TokenType.LITERAL_CHAR):
            return ExtendedTokenType.LITERAL
        elif base_type == TokenType.PUNCTUATION:
            return ExtendedTokenType.PUNCTUATION
    
    # Default classification based on text
    if token_text.isdigit() or (token_text.startswith('0x') and len(token_text) > 2):
        return ExtendedTokenType.LITERAL
    
    if token_text in {';', ',', '{', '}', '(', ')', ':', '.'}:
        return ExtendedTokenType.PUNCTUATION
    
    if token_text in {'+', '-', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=',
                      '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '++', '--'}:
        return ExtendedTokenType.OPERATOR
    
    return ExtendedTokenType.IDENTIFIER


def get_token_type_ids_for_sequence(
    tokens: List[str],
    base_types: Optional[List[TokenType]] = None,
) -> List[int]:
    """
    Get token type IDs for a sequence of tokens.
    
    Args:
        tokens: List of token texts
        base_types: Optional list of base TokenTypes (same length as tokens)
    
    Returns:
        List of integer type IDs
    """
    if not tokens:
        return []
    
    type_ids = []
    n = len(tokens)
    
    for i, token in enumerate(tokens):
        base_type = base_types[i] if base_types and i < len(base_types) else None
        prev_token = tokens[i - 1] if i > 0 else None
        next_token = tokens[i + 1] if i < n - 1 else None
        
        type_id = get_extended_token_type(token, base_type, prev_token, next_token)
        type_ids.append(type_id)
    
    return type_ids


def pad_token_type_ids(
    type_ids: List[int],
    max_length: int,
    pad_value: int = 0,
) -> List[int]:
    """
    Pad token type IDs to max_length.
    
    Args:
        type_ids: List of token type IDs
        max_length: Target length
        pad_value: Value to use for padding (default: 0 = PAD)
    
    Returns:
        Padded list of type IDs
    """
    if len(type_ids) >= max_length:
        return type_ids[:max_length]
    
    return type_ids + [pad_value] * (max_length - len(type_ids))


# ============================================================
# Statistics and Debugging
# ============================================================

def get_token_type_stats(type_ids: List[int]) -> dict:
    """Get statistics about token type distribution."""
    from collections import Counter
    
    counter = Counter(type_ids)
    total = len(type_ids)
    
    stats = {}
    for type_enum in ExtendedTokenType:
        count = counter.get(type_enum.value, 0)
        stats[type_enum.name] = {
            'count': count,
            'ratio': count / total if total > 0 else 0.0
        }
    
    return stats


def type_id_to_name(type_id: int) -> str:
    """Convert type ID to human-readable name."""
    try:
        return ExtendedTokenType(type_id).name
    except ValueError:
        return f"UNKNOWN_{type_id}"
