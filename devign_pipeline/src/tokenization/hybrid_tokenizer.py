"""Hybrid tokenizer: preserve dangerous APIs, normalize other identifiers"""

import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
from multiprocessing import Pool
from functools import partial


DANGEROUS_APIS = {
    'alloca', 'calloc', 'close', 'fclose', 'fgetc', 'fgets', 'fopen',
    'fprintf', 'fputs', 'fread', 'free', 'fscanf', 'fwrite', 'getc',
    'getchar', 'gets', 'malloc', 'memchr', 'memcmp', 'memcpy', 'memmove',
    'memset', 'open', 'printf', 'puts', 'read', 'realloc', 'scanf',
    'snprintf', 'sprintf', 'sscanf', 'strcat', 'strcmp', 'strcpy',
    'strlen', 'strncat', 'strncmp', 'strncpy', 'vprintf', 'vsnprintf',
    'vsprintf', 'write'
}

C_KEYWORDS = {
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
    'inline', 'int', 'long', 'register', 'restrict', 'return', 'short',
    'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union',
    'unsigned', 'void', 'volatile', 'while', '_Bool', '_Complex', '_Imaginary',
    'NULL', 'true', 'false', 'nullptr'
}

NORMALIZED_TOKENS = {
    'FUNC': 'FUNC',
    'ID': 'ID',
    'NUM': 'NUM',
    'FLOAT': 'FLOAT',
    'STR': 'STR',
    'CHAR': 'CHAR',
}

SPECIAL_TOKENS = {
    'PAD': 0,
    'UNK': 1,
    'BOS': 2,
    'EOS': 3,
    'SEP': 4,
}

TOKEN_PATTERNS = [
    ('STRING', r'"(?:[^"\\]|\\.)*"'),
    ('CHAR', r"'(?:[^'\\]|\\.)*'"),
    ('COMMENT_MULTI', r'/\*[\s\S]*?\*/'),
    ('COMMENT_SINGLE', r'//[^\n]*'),
    ('HEX', r'0[xX][0-9a-fA-F]+[uUlL]*'),
    ('FLOAT', r'\d+\.\d*[fFlL]?|\.\d+[fFlL]?|\d+[eE][+-]?\d+[fFlL]?'),
    ('NUMBER', r'\d+[uUlL]*'),
    ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('OPERATOR', r'->|<<|>>|<=|>=|==|!=|&&|\|\||[+\-*/%&|^~<>=!]'),
    ('PUNCTUATION', r'[(){}\[\];,.:?#]'),
    ('WHITESPACE', r'\s+'),
]

COMBINED_PATTERN = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_PATTERNS)
TOKEN_REGEX = re.compile(COMBINED_PATTERN)


class HybridTokenizer:
    """Tokenizer that preserves dangerous APIs while normalizing other tokens"""
    
    def __init__(self, preserve_dangerous_apis: bool = True, preserve_keywords: bool = True):
        self.preserve_dangerous_apis = preserve_dangerous_apis
        self.preserve_keywords = preserve_keywords
        self.regex = TOKEN_REGEX
    
    def tokenize(self, code: str) -> List[str]:
        """
        Tokenize code with hybrid strategy:
        - Keep: dangerous APIs, C keywords, operators, punctuation
        - Normalize: function names -> FUNC, identifiers -> ID, literals -> NUM/STR/CHAR
        """
        tokens = []
        pos = 0
        code_len = len(code)
        
        raw_tokens = []
        for match in self.regex.finditer(code):
            token_type = match.lastgroup
            value = match.group()
            raw_tokens.append((value, token_type))
        
        for i, (value, token_type) in enumerate(raw_tokens):
            if token_type in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                continue
            
            if token_type == 'STRING':
                tokens.append('STR')
            elif token_type == 'CHAR':
                tokens.append('CHAR')
            elif token_type in ('NUMBER', 'HEX'):
                tokens.append('NUM')
            elif token_type == 'FLOAT':
                tokens.append('FLOAT')
            elif token_type == 'IDENTIFIER':
                tokens.append(self._normalize_identifier(value, raw_tokens, i))
            elif token_type in ('OPERATOR', 'PUNCTUATION'):
                tokens.append(value)
        
        return tokens
    
    def _normalize_identifier(self, value: str, raw_tokens: List[Tuple[str, str]], 
                               current_idx: int) -> str:
        """Normalize identifier based on context"""
        if self.preserve_keywords and value in C_KEYWORDS:
            return value
        
        if self.preserve_dangerous_apis and value in DANGEROUS_APIS:
            return value
        
        next_idx = current_idx + 1
        while next_idx < len(raw_tokens):
            next_val, next_type = raw_tokens[next_idx]
            if next_type not in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                break
            next_idx += 1
        
        if next_idx < len(raw_tokens):
            next_val, _ = raw_tokens[next_idx]
            if next_val == '(':
                return 'FUNC'
        
        return 'ID'
    
    def tokenize_batch(self, codes: List[str], n_jobs: int = 1) -> List[List[str]]:
        """Batch tokenization"""
        if n_jobs == 1:
            return [self.tokenize(code) for code in codes]
        
        with Pool(n_jobs) as pool:
            results = pool.map(self.tokenize, codes)
        return results


def build_hybrid_vocab(codes: List[str], min_freq: int = 2, 
                       max_size: int = 500) -> Dict[str, int]:
    """
    Build vocabulary from tokenized codes.
    
    Args:
        codes: List of raw code strings
        min_freq: Minimum frequency for token inclusion
        max_size: Maximum vocabulary size
        
    Returns:
        Dict mapping token to id
    """
    tokenizer = HybridTokenizer()
    
    token_counts = Counter()
    for code in codes:
        tokens = tokenizer.tokenize(code)
        token_counts.update(tokens)
    
    vocab = {tok: idx for tok, idx in SPECIAL_TOKENS.items()}
    
    sorted_tokens = sorted(
        [(tok, count) for tok, count in token_counts.items() if count >= min_freq],
        key=lambda x: (-x[1], x[0])
    )
    
    remaining_size = max_size - len(vocab)
    for tok, _ in sorted_tokens[:remaining_size]:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    
    return vocab


def vectorize(tokens: List[str], vocab: Dict[str, int], 
              max_len: int = 512) -> Tuple[List[int], List[int]]:
    """
    Convert tokens to input_ids and attention_mask.
    
    Args:
        tokens: List of token strings
        vocab: Token to id mapping
        max_len: Maximum sequence length
        
    Returns:
        Tuple of (input_ids, attention_mask)
    """
    unk_id = vocab.get('UNK', 1)
    pad_id = vocab.get('PAD', 0)
    
    input_ids = [vocab.get(tok, unk_id) for tok in tokens[:max_len]]
    
    actual_len = len(input_ids)
    
    if actual_len < max_len:
        padding_len = max_len - actual_len
        input_ids.extend([pad_id] * padding_len)
    
    attention_mask = [1] * actual_len + [0] * (max_len - actual_len)
    
    return input_ids, attention_mask


if __name__ == "__main__":
    test_code = '''
    void process_request(char *user_input) {
        char buf[256];
        int len = strlen(user_input);
        
        if (len > 0) {
            strcpy(buf, user_input);
            printf("Input: %s", buf);
        }
        
        char *ptr = malloc(1024);
        if (ptr != NULL) {
            memcpy(ptr, buf, len);
            free(ptr);
        }
    }
    '''
    
    print("=" * 60)
    print("HYBRID TOKENIZER TEST")
    print("=" * 60)
    
    tokenizer = HybridTokenizer()
    tokens = tokenizer.tokenize(test_code)
    
    print("\n[Tokenized Output]")
    print(tokens)
    print(f"\nTotal tokens: {len(tokens)}")
    
    print("\n[Token Analysis]")
    preserved_apis = [t for t in tokens if t in DANGEROUS_APIS]
    print(f"Preserved dangerous APIs: {preserved_apis}")
    
    normalized_funcs = [t for t in tokens if t == 'FUNC']
    print(f"Normalized functions (FUNC): {len(normalized_funcs)}")
    
    normalized_ids = [t for t in tokens if t == 'ID']
    print(f"Normalized identifiers (ID): {len(normalized_ids)}")
    
    print("\n[Building Vocabulary]")
    vocab = build_hybrid_vocab([test_code], min_freq=1, max_size=100)
    print(f"Vocab size: {len(vocab)}")
    print(f"Sample vocab entries: {dict(list(vocab.items())[:15])}")
    
    print("\n[Vectorization]")
    input_ids, attention_mask = vectorize(tokens, vocab, max_len=64)
    print(f"Input IDs (first 20): {input_ids[:20]}")
    print(f"Attention mask (first 20): {attention_mask[:20]}")
    print(f"Non-padded length: {sum(attention_mask)}")
    
    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)
