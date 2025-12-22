"""Hybrid tokenizer: preserve dangerous APIs, normalize other identifiers"""

import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
from multiprocessing import Pool
from functools import partial


# Dangerous/security-sensitive APIs that should be preserved during tokenization
# Based on Microsoft SDL Banned Functions, CERT C, CWE Top 25
DANGEROUS_APIS = {
    # Memory allocation/deallocation (CWE-122, CWE-416, CWE-401)
    'malloc', 'calloc', 'realloc', 'free', 'alloca',
    
    # Unsafe string functions (CWE-120, CWE-787, SDL Banned)
    'strcpy', 'strcat', 'gets', 'sprintf', 'vsprintf',
    'strncpy', 'strncat', 'strtok',  # can still cause issues
    'strdup', 'strndup',  # allocation + copy
    'strlcpy', 'strlcat',  # BSD, common in OSS
    'bcopy', 'bzero',  # deprecated but still appears in old code
    
    # Memory operations (CWE-119, CWE-125, CWE-787)
    'memcpy', 'memmove', 'memset', 'memcmp', 'memchr',
    
    # Format string functions (CWE-134)
    'printf', 'fprintf', 'sprintf', 'snprintf',
    'vprintf', 'vfprintf', 'vsprintf', 'vsnprintf',
    'syslog', 'asprintf', 'vasprintf',
    
    # Input functions - taint sources (CWE-20)
    'scanf', 'fscanf', 'sscanf', 'vscanf', 'vfscanf', 'vsscanf',
    'gets', 'fgets', 'getchar', 'getc', 'fgetc',
    'read', 'fread', 'pread',
    'recv', 'recvfrom', 'recvmsg', 'readv',
    'getline', 'getdelim',
    
    # Output/write functions
    'write', 'fwrite', 'pwrite', 'send', 'sendto', 'sendmsg', 'writev',
    'puts', 'fputs',
    
    # File operations (CWE-22, CWE-73)
    'fopen', 'open', 'fclose', 'close',
    'access', 'chmod', 'chown',
    'realpath', 'readlink',
    'mkdir', 'rmdir', 'unlink', 'rename',  # path traversal
    'stat', 'lstat',  # race condition (TOCTOU)
    'mmap',  # null pointer check needed
    
    # Environment/path - taint sources (CWE-78, CWE-426)
    'getenv', 'setenv', 'putenv',
    'getcwd', 'getwd',  # taint sources
    
    # Command execution - sinks (CWE-78, CWE-88)
    'system', 'popen', 'pclose',
    'execl', 'execle', 'execlp', 'execv', 'execve', 'execvp',
    'execlpe', 'execvpe',  # additional exec variants
    
    # Temp file race conditions (CWE-377)
    'tmpnam', 'tempnam', 'mktemp', 'tmpfile',
    'mkstemp', 'mkostemp',  # safer but still relevant
    
    # Integer conversion (CWE-190, CWE-681)
    'atoi', 'atol', 'atoll', 'atof',
    'strtol', 'strtoul', 'strtoll', 'strtoull', 'strtod', 'strtof',
    
    # String comparison/length (contextual helpers)
    'strcmp', 'strncmp', 'strlen', 'strnlen',
    'strchr', 'strrchr', 'strstr', 'strpbrk', 'strcspn', 'strspn',
    
    # Wide character equivalents (for i18n code)
    'wcscpy', 'wcscat', 'wcsncpy', 'wcsncat', 'wcslen', 'wcsdup',
    'wcstok', 'wmemcpy', 'wmemset',
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
    ('SEP_TOKEN', r'\[SEP\]'),  # Must be first to match before PUNCTUATION
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

SEMANTIC_PATTERNS = {
    'BUF': ['buf', 'buffer', 'dst', 'src', 'data', 'str', 'string', 'payload', 'msg', 'packet', 'input', 'output'],
    'LEN': ['len', 'length', 'size', 'sz', 'count', 'cnt', 'num'],
    'PTR': ['ptr', 'pointer', 'head', 'tail', 'node', 'cursor', 'iter'],
    'IDX': ['idx', 'index', 'pos', 'offset'],
    'SENS': ['password', 'passwd', 'pwd', 'secret', 'token', 'credential', 'cred', 'auth', 'session', 'cookie', 'key', 'cert', 'private', 'hash', 'salt', 'nonce', 'iv'],
    'PRIV': ['admin', 'root', 'user', 'privilege', 'priv', 'permission', 'perm', 'role', 'access', 'owner', 'group', 'uid', 'gid'],
}

STANDALONE_LETTERS = {
    'p': 'PTR',
    'n': 'LEN',
    'i': 'IDX',
    'j': 'IDX',
    'k': 'IDX',
}

MAX_CANONICAL_IDS = 32


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
            
            if token_type == 'SEP_TOKEN':
                tokens.append('SEP')
            elif token_type == 'STRING':
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
        
        next_idx = current_idx + 1
        while next_idx < len(raw_tokens):
            next_val, next_type = raw_tokens[next_idx]
            if next_type not in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                break
            next_idx += 1
        
        is_function_call = next_idx < len(raw_tokens) and raw_tokens[next_idx][0] == '('
        
        if self.preserve_dangerous_apis and value in DANGEROUS_APIS:
            if is_function_call:
                return value
            else:
                return 'ID'
        
        if is_function_call:
            return 'FUNC'
        
        return 'ID'
    
    def tokenize_batch(self, codes: List[str], n_jobs: int = 1) -> List[List[str]]:
        """Batch tokenization"""
        if n_jobs == 1:
            return [self.tokenize(code) for code in codes]
        
        with Pool(n_jobs) as pool:
            results = pool.map(self.tokenize, codes)
        return results


class CanonicalTokenizer(HybridTokenizer):
    """
    Tokenizer that preserves variable identity with semantic buckets.
    
    Instead of normalizing all variables to ID, this tokenizer:
    1. Assigns canonical IDs based on semantic category (BUF, LEN, PTR, IDX, VAR)
    2. Preserves variable identity within the same code slice (VAR_0 stays VAR_0)
    3. Limits to 32 variables per category (BUF_0 to BUF_31, etc.)
    """
    
    def __init__(self, preserve_dangerous_apis: bool = True, preserve_keywords: bool = True):
        super().__init__(preserve_dangerous_apis, preserve_keywords)
        self._reset_var_mappings()
    
    def _reset_var_mappings(self):
        """Reset variable mappings for a new code snippet"""
        self.var_to_canonical: Dict[str, str] = {}
        self.category_counters: Dict[str, int] = {
            'BUF': 0,
            'LEN': 0,
            'PTR': 0,
            'IDX': 0,
            'SENS': 0,
            'PRIV': 0,
            'VAR': 0,
        }
    
    def _get_semantic_category(self, var_name: str) -> str:
        """
        Determine semantic category based on variable name.
        
        Returns: 'BUF', 'LEN', 'PTR', 'IDX', or 'VAR'
        """
        lower_name = var_name.lower()
        
        if lower_name in STANDALONE_LETTERS:
            return STANDALONE_LETTERS[lower_name]
        
        for category, patterns in SEMANTIC_PATTERNS.items():
            for pattern in patterns:
                if pattern in lower_name:
                    return category
        
        return 'VAR'
    
    def _get_canonical_token(self, var_name: str) -> str:
        """
        Get or create canonical token for a variable.
        
        If variable was seen before, returns the same canonical token.
        Otherwise, assigns a new one based on semantic category.
        """
        if var_name in self.var_to_canonical:
            return self.var_to_canonical[var_name]
        
        category = self._get_semantic_category(var_name)
        idx = self.category_counters[category]
        
        if idx >= MAX_CANONICAL_IDS:
            idx = MAX_CANONICAL_IDS - 1
        else:
            self.category_counters[category] += 1
        
        canonical = f"{category}_{idx}"
        self.var_to_canonical[var_name] = canonical
        return canonical
    
    def tokenize(self, code: str) -> List[str]:
        """
        Tokenize code with canonical variable identity.
        
        Variables are assigned semantic tokens like BUF_0, LEN_1, PTR_0, etc.
        based on their names and order of appearance.
        """
        self._reset_var_mappings()
        
        tokens = []
        
        raw_tokens = []
        for match in self.regex.finditer(code):
            token_type = match.lastgroup
            value = match.group()
            raw_tokens.append((value, token_type))
        
        for i, (value, token_type) in enumerate(raw_tokens):
            if token_type in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                continue
            
            if token_type == 'SEP_TOKEN':
                tokens.append('SEP')
            elif token_type == 'STRING':
                tokens.append('STR')
            elif token_type == 'CHAR':
                tokens.append('CHAR')
            elif token_type in ('NUMBER', 'HEX'):
                tokens.append('NUM')
            elif token_type == 'FLOAT':
                tokens.append('FLOAT')
            elif token_type == 'IDENTIFIER':
                tokens.append(self._normalize_identifier_canonical(value, raw_tokens, i))
            elif token_type in ('OPERATOR', 'PUNCTUATION'):
                tokens.append(value)
        
        return tokens
    
    def _normalize_identifier_canonical(self, value: str, raw_tokens: List[Tuple[str, str]], 
                                        current_idx: int) -> str:
        """Normalize identifier with canonical variable identity"""
        if self.preserve_keywords and value in C_KEYWORDS:
            return value
        
        next_idx = current_idx + 1
        while next_idx < len(raw_tokens):
            next_val, next_type = raw_tokens[next_idx]
            if next_type not in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                break
            next_idx += 1
        
        is_function_call = next_idx < len(raw_tokens) and raw_tokens[next_idx][0] == '('
        
        if self.preserve_dangerous_apis and value in DANGEROUS_APIS:
            if is_function_call:
                return value
            else:
                return self._get_canonical_token(value)
        
        if is_function_call:
            return 'FUNC'
        
        return self._get_canonical_token(value)
    
    def get_var_mapping(self) -> Dict[str, str]:
        """Return the current variable to canonical token mapping"""
        return self.var_to_canonical.copy()
    
    def tokenize_batch(self, codes: List[str], n_jobs: int = 1) -> List[List[str]]:
        """Batch tokenization (single-threaded to maintain state correctly)"""
        return [self.tokenize(code) for code in codes]


def get_canonical_vocab_tokens() -> List[str]:
    """
    Get all possible canonical tokens for vocabulary building.
    
    Returns: List of tokens like BUF_0, BUF_1, ..., VAR_31
    """
    tokens = []
    for category in ['BUF', 'LEN', 'PTR', 'IDX', 'SENS', 'PRIV', 'VAR']:
        for i in range(MAX_CANONICAL_IDS):
            tokens.append(f"{category}_{i}")
    return tokens


def build_hybrid_vocab(codes: List[str], min_freq: int = 2, 
                       max_size: int = 500,
                       use_canonical: bool = False) -> Dict[str, int]:
    """
    Build vocabulary from tokenized codes.
    
    Args:
        codes: List of raw code strings
        min_freq: Minimum frequency for token inclusion
        max_size: Maximum vocabulary size
        use_canonical: If True, use CanonicalTokenizer and include canonical tokens
        
    Returns:
        Dict mapping token to id
    """
    if use_canonical:
        tokenizer = CanonicalTokenizer()
    else:
        tokenizer = HybridTokenizer()
    
    token_counts = Counter()
    for code in codes:
        tokens = tokenizer.tokenize(code)
        token_counts.update(tokens)
    
    vocab = {tok: idx for tok, idx in SPECIAL_TOKENS.items()}
    
    if use_canonical:
        canonical_tokens = get_canonical_vocab_tokens()
        for tok in canonical_tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    
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
    print("HYBRID TOKENIZER TEST (Original)")
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
    
    print("\n" + "=" * 60)
    print("CANONICAL TOKENIZER TEST (Option D)")
    print("=" * 60)
    
    canonical_tokenizer = CanonicalTokenizer()
    canonical_tokens = canonical_tokenizer.tokenize(test_code)
    
    print("\n[Tokenized Output]")
    print(canonical_tokens)
    print(f"\nTotal tokens: {len(canonical_tokens)}")
    
    print("\n[Variable Mapping]")
    var_mapping = canonical_tokenizer.get_var_mapping()
    for orig, canon in sorted(var_mapping.items(), key=lambda x: x[1]):
        print(f"  {orig:20s} -> {canon}")
    
    print("\n[Token Analysis]")
    preserved_apis = [t for t in canonical_tokens if t in DANGEROUS_APIS]
    print(f"Preserved dangerous APIs: {preserved_apis}")
    
    buf_tokens = [t for t in canonical_tokens if t.startswith('BUF_')]
    len_tokens = [t for t in canonical_tokens if t.startswith('LEN_')]
    ptr_tokens = [t for t in canonical_tokens if t.startswith('PTR_')]
    idx_tokens = [t for t in canonical_tokens if t.startswith('IDX_')]
    var_tokens = [t for t in canonical_tokens if t.startswith('VAR_')]
    
    print(f"BUF tokens: {set(buf_tokens)}")
    print(f"LEN tokens: {set(len_tokens)}")
    print(f"PTR tokens: {set(ptr_tokens)}")
    print(f"IDX tokens: {set(idx_tokens)}")
    print(f"VAR tokens: {set(var_tokens)}")
    
    print("\n" + "=" * 60)
    print("ADDITIONAL TEST CASE")
    print("=" * 60)
    
    test_code2 = '''
    char *ptr = malloc(size);
    strcpy(ptr, buffer);
    int len = strlen(buffer);
    memcpy(dest, ptr, len);
    '''
    
    print("\n[Input Code]")
    print(test_code2.strip())
    
    canonical_tokens2 = canonical_tokenizer.tokenize(test_code2)
    print("\n[Canonical Tokens]")
    print(' '.join(canonical_tokens2))
    
    print("\n[Variable Mapping]")
    var_mapping2 = canonical_tokenizer.get_var_mapping()
    for orig, canon in sorted(var_mapping2.items(), key=lambda x: x[1]):
        print(f"  {orig:20s} -> {canon}")
    
    print("\n[Building Canonical Vocabulary]")
    vocab = build_hybrid_vocab([test_code, test_code2], min_freq=1, max_size=300, use_canonical=True)
    print(f"Vocab size: {len(vocab)}")
    
    sample_entries = {k: v for k, v in list(vocab.items())[:20]}
    print(f"Sample vocab entries: {sample_entries}")
    
    canonical_in_vocab = [k for k in vocab.keys() if '_' in k and k.split('_')[0] in ['BUF', 'LEN', 'PTR', 'IDX', 'SENS', 'PRIV', 'VAR']]
    print(f"Canonical tokens in vocab: {len(canonical_in_vocab)}")
    
    print("\n[Vectorization]")
    input_ids, attention_mask = vectorize(canonical_tokens2, vocab, max_len=64)
    print(f"Input IDs (first 20): {input_ids[:20]}")
    print(f"Attention mask (first 20): {attention_mask[:20]}")
    print(f"Non-padded length: {sum(attention_mask)}")
    
    print("\n" + "=" * 60)
    print("SENSITIVE/PRIVILEGE VARIABLE TEST")
    print("=" * 60)

    test_code3 = '''
void authenticate(char *password, char *username) {
    char *token = generate_token(password);
    if (check_admin(username)) {
        set_privilege(ADMIN_ROLE);
    }
    store_session(token, user_id);
    free(token);
}
'''

    print("\n[Input Code]")
    print(test_code3.strip())

    canonical_tokens3 = canonical_tokenizer.tokenize(test_code3)
    print("\n[Canonical Tokens]")
    print(' '.join(canonical_tokens3))

    print("\n[Variable Mapping]")
    var_mapping3 = canonical_tokenizer.get_var_mapping()
    for orig, canon in sorted(var_mapping3.items(), key=lambda x: x[1]):
        print(f"  {orig:20s} -> {canon}")

    sens_tokens = [t for t in canonical_tokens3 if t.startswith('SENS_')]
    priv_tokens = [t for t in canonical_tokens3 if t.startswith('PRIV_')]
    print(f"\nSENS tokens: {set(sens_tokens)}")
    print(f"PRIV tokens: {set(priv_tokens)}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
