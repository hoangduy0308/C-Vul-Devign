"""
Optimized Hybrid Tokenizer for C Vulnerability Detection

Features:
- API family mapping (project-specific APIs -> canonical families)
- Defense pattern detection  
- Semantic bucket canonicalization for identifiers
- Smart number handling

Based on Oracle analysis for Devign dataset optimization.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from tqdm import tqdm

from .hybrid_tokenizer import C_KEYWORDS, TOKEN_REGEX, SPECIAL_TOKENS

logger = logging.getLogger(__name__)


# =============================================================================
# API FAMILIES - Map project-specific APIs to canonical tokens
# =============================================================================

# Each family can have:
# - 'funcs': set of all function names in this family
# - 'preserve_exact': set of functions to keep as exact tokens (most common/important)
API_FAMILIES = {
    'API_ALLOC': {
        'funcs': {
            'malloc', 'calloc', 'realloc', 'alloca',
            'av_malloc', 'av_mallocz', 'av_calloc', 'av_realloc', 'av_reallocp',
            'av_realloc_f', 'av_fast_realloc', 'av_buffer_alloc', 'av_buffer_allocz',
            'g_malloc', 'g_malloc0', 'g_malloc_n', 'g_malloc0_n', 'g_new', 'g_new0',
            'g_renew', 'g_try_malloc', 'g_try_malloc0', 'g_try_realloc',
            'qemu_malloc', 'qemu_mallocz', 'qemu_memalign', 'qemu_blockalign',
            'kmalloc', 'kzalloc', 'kcalloc', 'krealloc', 'vmalloc', 'vzalloc', 'kvmalloc',
        },
        'preserve_exact': {'malloc', 'calloc', 'realloc'},
    },
    'API_FREE': {
        'funcs': {
            'free', 'av_free', 'av_freep', 'g_free', 'qemu_free',
            'kfree', 'vfree', 'kvfree',
        },
        'preserve_exact': {'free'},
    },
    'API_UNREF': {
        'funcs': {
            'av_buffer_unref', 'av_frame_unref', 'av_packet_unref',
            'av_frame_free', 'av_packet_free', 'av_dict_free',
            'avformat_close_input', 'avcodec_close', 'avcodec_free_context',
            'sws_freeContext', 'swr_free', 'avfilter_graph_free',
            'object_unref', 'g_object_unref', 'qobject_unref', 'blk_unref', 'bdrv_unref',
        },
        'preserve_exact': set(),
    },
    'API_REF': {
        'funcs': {
            'av_buffer_ref', 'av_frame_ref', 'av_packet_ref',
            'object_ref', 'g_object_ref', 'qobject_ref',
        },
        'preserve_exact': set(),
    },
    'API_STRDUP': {
        'funcs': {
            'strdup', 'strndup', 'av_strdup', 'av_strndup', 'av_asprintf',
            'g_strdup', 'g_strndup', 'g_strdup_printf', 'g_strconcat',
            'qemu_strdup', 'wcsdup',
        },
        'preserve_exact': {'strdup', 'strndup'},
    },
    'API_COPY': {
        'funcs': {
            'memcpy', 'memmove', 'strcpy', 'strncpy', 'strlcpy', 'bcopy',
            'wmemcpy', 'wcscpy', 'wcsncpy',
        },
        'preserve_exact': {'memcpy', 'memmove', 'strcpy', 'strncpy'},
    },
    'API_CONCAT': {
        'funcs': {
            'strcat', 'strncat', 'strlcat', 'wcscat', 'wcsncat',
        },
        'preserve_exact': {'strcat', 'strncat'},
    },
    'API_SET': {
        'funcs': {
            'memset', 'bzero', 'wmemset',
        },
        'preserve_exact': {'memset'},
    },
    'API_FORMAT': {
        'funcs': {
            'sprintf', 'snprintf', 'vsprintf', 'vsnprintf', 'asprintf', 'vasprintf',
            'printf', 'fprintf', 'vprintf', 'vfprintf', 'syslog',
        },
        'preserve_exact': {'sprintf', 'snprintf', 'printf', 'fprintf'},
    },
    'API_IO_READ': {
        'funcs': {
            'read', 'fread', 'pread', 'recv', 'recvfrom', 'recvmsg', 'readv',
            'fgets', 'gets', 'getchar', 'getc', 'fgetc', 'getline', 'getdelim',
            'scanf', 'fscanf', 'sscanf', 'vscanf', 'vfscanf', 'vsscanf',
        },
        'preserve_exact': {'read', 'fread', 'recv', 'recvfrom', 'gets', 'fgets', 'getchar', 'scanf', 'fscanf', 'sscanf'},
    },
    'API_IO_WRITE': {
        'funcs': {
            'write', 'fwrite', 'pwrite', 'send', 'sendto', 'sendmsg', 'writev',
            'puts', 'fputs',
        },
        'preserve_exact': {'write', 'fwrite', 'send', 'sendto'},
    },
    'API_FILE': {
        'funcs': {
            'fopen', 'open', 'fclose', 'close', 'mmap',
            'access', 'chmod', 'chown', 'stat', 'lstat',
            'mkdir', 'rmdir', 'unlink', 'rename',
            'realpath', 'readlink',
        },
        'preserve_exact': {'fopen', 'fclose', 'open', 'close', 'mmap'},
    },
    'API_EXEC': {
        'funcs': {
            'system', 'popen', 'pclose',
            'execl', 'execle', 'execlp', 'execv', 'execve', 'execvp',
            'execlpe', 'execvpe',
        },
        'preserve_exact': {'system', 'popen', 'execve'},
    },
    'API_CONVERT': {
        'funcs': {
            'atoi', 'atol', 'atoll', 'atof',
            'strtol', 'strtoul', 'strtoll', 'strtoull', 'strtod', 'strtof',
        },
        'preserve_exact': {'atoi', 'atol', 'strtol', 'strtoul'},
    },
    'API_STRLEN': {
        'funcs': {
            'strlen', 'strnlen', 'wcslen',
        },
        'preserve_exact': {'strlen'},
    },
    'API_STRCMP': {
        'funcs': {
            'strcmp', 'strncmp', 'strcasecmp', 'strncasecmp', 'memcmp',
        },
        'preserve_exact': {'strcmp', 'strncmp', 'memcmp'},
    },
    'API_STRCHR': {
        'funcs': {
            'strchr', 'strrchr', 'strstr', 'strpbrk', 'strcspn', 'strspn', 'memchr',
            'strtok', 'wcstok',
        },
        'preserve_exact': {'strchr', 'strstr'},
    },
    'API_ENV': {
        'funcs': {
            'getenv', 'setenv', 'putenv', 'getcwd', 'getwd',
        },
        'preserve_exact': set(),
    },
    'API_TEMP': {
        'funcs': {
            'tmpnam', 'tempnam', 'mktemp', 'tmpfile', 'mkstemp', 'mkostemp',
        },
        'preserve_exact': set(),
    },
}

# Build reverse lookup: function_name -> API_FAMILY
_API_LOOKUP: Dict[str, str] = {}
for family, data in API_FAMILIES.items():
    for func in data['funcs']:
        _API_LOOKUP[func] = family

# Generate UNIVERSAL_DANGEROUS from preserve_exact sets (ensures consistency)
UNIVERSAL_DANGEROUS: set = set()
for family, data in API_FAMILIES.items():
    UNIVERSAL_DANGEROUS.update(data.get('preserve_exact', set()))


# =============================================================================
# DEFENSE FAMILIES
# =============================================================================

DEFENSE_FAMILIES = {
    'DEF_ASSERT': {'assert', 'static_assert', 'ASSERT', 'BUG_ON', 'WARN_ON'},
    'DEF_BOUNDS': {'bounds_check', 'range_check', 'size_check', 'length_check'},
    'DEF_SAFE': {'snprintf', 'strlcpy', 'strlcat', 'memcpy_s', 'memmove_s', 'memset_s'},
    'DEF_ERROR': {'perror', 'strerror'},
}

# Build reverse lookup
_DEFENSE_LOOKUP: Dict[str, str] = {}
for family, funcs in DEFENSE_FAMILIES.items():
    for func in funcs:
        _DEFENSE_LOOKUP[func] = family

# Patterns for dynamic defense detection
DEFENSE_PATTERNS = [
    ('DEF_CHECK', ['check', 'valid', 'verify']),
    ('DEF_SANITIZE', ['sanit', 'escape', 'filter']),
    ('DEF_BOUNDS', ['bound', 'range', 'limit']),
    ('DEF_SAFE', ['safe_', '_safe']),
]


# =============================================================================
# SEMANTIC BUCKETS for Identifiers
# =============================================================================

SEMANTIC_BUCKETS = {
    'BUF': ['buf', 'buffer', 'dst', 'src', 'data', 'str', 'string', 'payload', 
            'msg', 'packet', 'input', 'output', 'dest', 'source', 'out', 
            'tmp', 'scratch', 'line', 'name', 'text', 'content'],
    'LEN': ['len', 'length', 'size', 'sz', 'count', 'cnt', 'num'],
    'CAP': ['cap', 'capacity', 'limit', 'max', 'min', 'avail', 'remaining', 
            'rem', 'bound', 'alloc'],
    'PTR': ['ptr', 'pointer', 'head', 'tail', 'node', 'cursor', 'iter', 
            'next', 'prev', 'current', 'cur'],
    'IDX': ['idx', 'index', 'pos', 'offset', 'off', 'ofs', 'start', 'end'],
    'RET': ['ret', 'result', 'rv', 'rc', 'status', 'code'],
    'ERR': ['err', 'error', 'errno', 'fail', 'failure'],
    'FD': ['fd', 'fp', 'file', 'path', 'filename', 'filepath', 'dir', 
           'directory', 'handle', 'stream'],
    'CMD': ['cmd', 'command', 'arg', 'args', 'argv', 'argc', 'env', 'param'],
    'FLAGS': ['flags', 'flag', 'mode', 'mask', 'opt', 'options', 'attr', 'type'],
    'SENS': ['password', 'passwd', 'pwd', 'secret', 'token', 'credential', 'cred',
             'auth', 'session', 'cookie', 'key', 'cert', 'private', 'hash', 
             'salt', 'nonce', 'iv', 'encrypt', 'decrypt'],
    'PRIV': ['admin', 'root', 'user', 'privilege', 'priv', 'permission', 'perm',
             'role', 'access', 'owner', 'group', 'uid', 'gid', 'sudo'],
    'MEM': ['mem', 'memory', 'heap', 'stack', 'block', 'chunk', 'pool', 'cache'],
    'CTX': ['ctx', 'context', 'state', 'info', 'config', 'cfg', 'setting'],
}

# Single letter variable mappings
SINGLE_LETTER_MAP = {
    'p': 'PTR',
    'n': 'LEN', 
    'i': 'IDX',
    'j': 'IDX',
    'k': 'IDX',
    's': 'BUF',
    'c': 'VAR',
    'x': 'VAR',
    'y': 'VAR',
    'm': 'VAR',
}

MAX_CANONICAL_IDS = 12  # 12 is a safer middle ground: 8 was too small for complex PDG slices with many buffers; functions with >12 unique buffers will use BUF_OVF

# Truly dangerous APIs - most critical security-relevant functions
# (distinct from UNIVERSAL_DANGEROUS which is auto-generated from preserve_exact)
TRULY_DANGEROUS_APIS = {
    'gets', 'strcpy', 'strcat', 'sprintf', 'vsprintf',
    'scanf', 'sscanf', 'fscanf',
    'system', 'popen', 'execve', 'execl', 'execv',
    'mktemp', 'tmpnam', 'tempnam',
    'memcpy', 'memmove', 'wcscpy', 'wcscat',
    'realpath', 'getwd', 'getcwd',
}

# Safe versions of dangerous APIs - preserve exact to distinguish from dangerous
SAFE_APIS = {
    'snprintf', 'vsnprintf',
    'strncpy', 'strlcpy', 'strncat', 'strlcat',
    'fgets', 'getline',
    'memcpy_s', 'strcpy_s', 'strncpy_s', 'strcat_s',
}


# =============================================================================
# NUMBER HANDLING
# =============================================================================

PRESERVED_SMALL_INTS = set(str(i) for i in range(17))  # 0-16

PRESERVED_POWERS_OF_TWO = {
    '32', '64', '128', '256', '512', '1024', '2048', '4096',
    '8192', '16384', '32768', '65536',
}

PRESERVED_BOUNDARIES = {
    '255', '1023', '4095', '65535', '2147483647', '4294967295',
    '100', '200', '500', '1000',
}

PRESERVED_HEX = {
    '0x0', '0x1', '0xff', '0xffff', '0xffffffff', '0x7fffffff',
    '0x80', '0x7f', '0x20', '0x00',
}

PRESERVED_OCTAL = {
    '0644', '0600', '0755', '0777', '0666', '0700', '0400',
}

# Tokens that indicate the following '-1' is a unary negative (not subtraction)
UNARY_CONTEXT_TOKENS = {
    '=', '==', '!=', '<', '>', '<=', '>=',
    '(', ',', 'return', '?', ':', 'case',
    '+', '-', '*', '/', '%', '&', '|', '^',
    '&&', '||', '<<', '>>', '[', '{', ';'
}


# =============================================================================
# TOKENIZER CLASS
# =============================================================================

class OptimizedHybridTokenizer:
    """
    Optimized tokenizer for C vulnerability detection.
    
    Key improvements over PreserveTokenizer:
    1. API family mapping reduces vocab from 130+ to ~20 API tokens
    2. Semantic buckets for identifiers reduce noise
    3. Defense pattern detection helps reduce false positives
    4. Smart number handling preserves vulnerability-relevant constants
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.regex = TOKEN_REGEX
        self.max_canonical_ids = self.config.get('max_canonical_ids', MAX_CANONICAL_IDS)
        self.use_indexed_buckets = self.config.get('use_indexed_buckets', True)
        self._saturation_warned: set = set()
        self._reset_var_mappings()
    
    def _reset_var_mappings(self):
        """Reset variable mappings for a new code snippet."""
        self.var_to_canonical: Dict[str, str] = {}
        self.bucket_counters: Dict[str, int] = {
            bucket: 0 for bucket in list(SEMANTIC_BUCKETS.keys()) + ['VAR']
        }
    
    def _get_api_family(self, func_name: str) -> Optional[str]:
        """
        Get API family token for a function.
        
        Returns:
            - Exact token if in UNIVERSAL_DANGEROUS, TRULY_DANGEROUS_APIS, or SAFE_APIS
            - API family token if in families
            - None otherwise
        """
        if func_name in UNIVERSAL_DANGEROUS:
            return func_name
        
        if func_name in TRULY_DANGEROUS_APIS:
            return func_name
        
        if func_name in SAFE_APIS:
            return func_name
        
        if func_name in _API_LOOKUP:
            return _API_LOOKUP[func_name]
        
        return None
    
    def _get_defense_token(self, func_name: str) -> Optional[str]:
        """
        Get defense token for a function.
        
        Returns defense family token or None.
        """
        if func_name in _DEFENSE_LOOKUP:
            return _DEFENSE_LOOKUP[func_name]
        
        lower_name = func_name.lower()
        for defense_token, patterns in DEFENSE_PATTERNS:
            for pattern in patterns:
                if pattern in lower_name:
                    return defense_token
        
        return None
    
    def _split_identifier(self, identifier: str) -> List[str]:
        """
        Split identifier into components for semantic matching.
        
        Handles:
        - snake_case: buf_len -> ['buf', 'len']
        - camelCase: bufLen -> ['buf', 'len']
        - Mixed: myBufLen -> ['my', 'buf', 'len']
        """
        # First split on underscores
        parts = identifier.split('_')
        
        result = []
        for part in parts:
            if not part:
                continue
            # Split camelCase
            camel_split = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', part)
            if camel_split:
                result.extend([s.lower() for s in camel_split])
            else:
                result.append(part.lower())
        
        return result
    
    def _get_semantic_bucket(self, identifier: str) -> str:
        """
        Determine semantic bucket for an identifier.
        
        Returns bucket like 'BUF', 'LEN', 'PTR', etc. or 'VAR' for generic.
        """
        lower_id = identifier.lower()
        
        # Check single letter first
        if lower_id in SINGLE_LETTER_MAP:
            return SINGLE_LETTER_MAP[lower_id]
        
        # Split and check components
        components = self._split_identifier(identifier)
        
        # Check each component against buckets
        for bucket, patterns in SEMANTIC_BUCKETS.items():
            for component in components:
                if component in patterns:
                    return bucket
                # Also check if pattern is substring of component
                for pattern in patterns:
                    if len(pattern) >= 3 and pattern in component:
                        return bucket
        
        return 'VAR'
    
    def _get_canonical_token(self, identifier: str) -> str:
        """
        Get or create canonical token for an identifier.
        
        Returns tokens like BUF_0, LEN_1, PTR_0, VAR_5, etc.
        Or if use_indexed_buckets=False, returns just BUF, LEN, PTR, etc.
        """
        if identifier in self.var_to_canonical:
            return self.var_to_canonical[identifier]
        
        bucket = self._get_semantic_bucket(identifier)
        
        # UNINDEXED mode: return bucket name directly without index
        if not self.use_indexed_buckets:
            self.var_to_canonical[identifier] = bucket
            return bucket
        
        # INDEXED mode: return bucket with index (BUF_0, BUF_1, etc.)
        idx = self.bucket_counters[bucket]
        
        if idx >= self.max_canonical_ids:
            # Use overflow token to prevent false aliasing between distinct variables
            canonical = f"{bucket}_OVF"
            if bucket not in self._saturation_warned:
                logger.warning(f"Bucket '{bucket}' saturated at {self.max_canonical_ids} IDs. "
                             f"Additional variables will use {canonical}. "
                             f"Consider increasing max_canonical_ids if this occurs frequently.")
                self._saturation_warned.add(bucket)
        else:
            self.bucket_counters[bucket] += 1
            canonical = f"{bucket}_{idx}"
        self.var_to_canonical[identifier] = canonical
        return canonical
    
    def _should_keep_number(self, value: str, token_type: str) -> Tuple[bool, str]:
        """
        Determine if a number should be kept as-is or normalized.
        
        Returns: (should_keep, token_to_use)
        """
        # Handle hex numbers
        if token_type == 'HEX':
            clean_hex = re.sub(r'[uUlL]+$', '', value.lower())
            if clean_hex in PRESERVED_HEX:
                return True, clean_hex
            return False, 'NUM_HEX'
        
        # Regular numbers - remove suffix
        clean_num = re.sub(r'[uUlL]+$', '', value)
        
        # Check octal (starts with 0 but not 0x)
        if clean_num.startswith('0') and len(clean_num) > 1 and not clean_num.startswith('0x'):
            if clean_num in PRESERVED_OCTAL:
                return True, f'PERM_{clean_num}'
            return False, 'NUM_OCT'
        
        # Small integers (0-16)
        if clean_num in PRESERVED_SMALL_INTS:
            return True, clean_num
        
        # Powers of two
        if clean_num in PRESERVED_POWERS_OF_TWO:
            return True, clean_num
        
        # Boundary values
        if clean_num in PRESERVED_BOUNDARIES:
            return True, clean_num
        
        return False, 'NUM'
    
    def _looks_like_type_cast(self, tokens: List[str]) -> bool:
        """Check if previous tokens look like a type cast before unary minus."""
        if len(tokens) < 2 or tokens[-1] != ')':
            return False
        
        typeish_keywords = {
            # C primitive types
            'void', 'char', 'short', 'int', 'long', 'signed', 'unsigned',
            'float', 'double',
            # Fixed-width integer types
            'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
            'int8_t', 'int16_t', 'int32_t', 'int64_t',
            # Size/pointer types
            'size_t', 'ssize_t', 'ptrdiff_t', 'intptr_t', 'uintptr_t',
            # POSIX types
            'off_t', 'time_t', 'pid_t', 'uid_t', 'gid_t',
            'socklen_t', 'mode_t', 'dev_t', 'ino_t', 'nlink_t',
            'blksize_t', 'blkcnt_t', 'clock_t', 'useconds_t', 'suseconds_t',
        }
        
        # Find matching open paren
        depth = 1
        for i in range(len(tokens) - 2, max(0, len(tokens) - 10) - 1, -1):
            if tokens[i] == ')':
                depth += 1
            elif tokens[i] == '(':
                depth -= 1
                if depth == 0:
                    # Check inside for type-like tokens
                    inside = tokens[i+1:-1]
                    for tok in inside:
                        if tok in typeish_keywords or tok.endswith('_t'):
                            return True
                    break
        return False
    
    def tokenize(self, code: str) -> List[str]:
        """
        Tokenize code with optimized hybrid strategy.
        
        Returns: List of tokens
        """
        self._reset_var_mappings()
        tokens = []
        
        # Extract raw tokens
        raw_tokens = []
        for match in self.regex.finditer(code):
            token_type = match.lastgroup
            value = match.group()
            raw_tokens.append((value, token_type))
        
        i = 0
        while i < len(raw_tokens):
            value, token_type = raw_tokens[i]
            
            # Skip whitespace and comments
            if token_type in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                i += 1
                continue
            
            # Handle -1 pattern
            if token_type == 'OPERATOR' and value == '-':
                next_i = i + 1
                while next_i < len(raw_tokens) and raw_tokens[next_i][1] in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                    next_i += 1
                
                if next_i < len(raw_tokens):
                    next_val, next_type = raw_tokens[next_i]
                    if next_type == 'NUMBER' and next_val == '1':
                        if len(tokens) == 0 or tokens[-1] in UNARY_CONTEXT_TOKENS or self._looks_like_type_cast(tokens):
                            tokens.append('NEG_1')
                            i = next_i + 1
                            continue
            
            # Process by token type
            if token_type == 'SEP_TOKEN':
                tokens.append('SEP')
            
            elif token_type == 'STRING':
                tokens.append('STR')
            
            elif token_type == 'CHAR':
                tokens.append('CHAR')
            
            elif token_type in ('NUMBER', 'HEX'):
                _, token_to_use = self._should_keep_number(value, token_type)
                tokens.append(token_to_use)
            
            elif token_type == 'FLOAT':
                tokens.append('FLOAT')
            
            elif token_type == 'IDENTIFIER':
                tokens.append(self._normalize_identifier(value, raw_tokens, i))
            
            elif token_type in ('OPERATOR', 'PUNCTUATION'):
                tokens.append(value)
            
            i += 1
        
        return tokens
    
    def _normalize_identifier(self, value: str, raw_tokens: List, current_idx: int) -> str:
        """
        Normalize an identifier based on context.
        
        Priority:
        1. C keywords -> keep exact
        2. Function calls:
           a. API family -> family token or exact if universal
           b. Defense pattern -> defense token
           c. Other -> FUNC
        3. Variables -> semantic bucket (BUF_0, LEN_1, etc.)
        """
        # C keywords - keep exact
        if value in C_KEYWORDS:
            return value
        
        # Check if this is a function call (followed by '(')
        next_idx = current_idx + 1
        while next_idx < len(raw_tokens):
            next_val, next_type = raw_tokens[next_idx]
            if next_type not in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                break
            next_idx += 1
        
        is_function_call = next_idx < len(raw_tokens) and raw_tokens[next_idx][0] == '('
        
        # Check for member access (obj.func or obj->func)
        prev_idx = current_idx - 1
        while prev_idx >= 0:
            prev_val, prev_type = raw_tokens[prev_idx]
            if prev_type not in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                break
            prev_idx -= 1
        
        is_member_access = prev_idx >= 0 and raw_tokens[prev_idx][0] in ('.', '->')
        
        if is_function_call and not is_member_access:
            # Check API family first
            api_token = self._get_api_family(value)
            if api_token:
                return api_token
            
            # Check defense pattern
            defense_token = self._get_defense_token(value)
            if defense_token:
                return defense_token
            
            # Generic function
            return 'FUNC'
        
        # Variable - use semantic bucket
        return self._get_canonical_token(value)
    
    def tokenize_batch(self, codes: List[str], show_progress: bool = True) -> List[List[str]]:
        """Batch tokenization."""
        if show_progress:
            return [self.tokenize(code) for code in tqdm(codes, desc="Tokenizing")]
        return [self.tokenize(code) for code in codes]


# =============================================================================
# VOCABULARY BUILDING
# =============================================================================

def get_all_vocab_tokens(
    max_canonical_ids: int = MAX_CANONICAL_IDS,
    use_indexed_buckets: bool = True
) -> List[str]:
    """
    Get all predefined vocabulary tokens.
    
    Args:
        max_canonical_ids: Maximum number of IDs per semantic bucket (default: 8)
        use_indexed_buckets: If True, generate BUF_0, BUF_1, etc. 
                            If False, generate just BUF, LEN, etc.
    
    Returns list of all tokens that should be in vocab.
    """
    tokens = []
    
    # Special tokens
    tokens.extend(SPECIAL_TOKENS.keys())
    
    # API family tokens
    tokens.extend(API_FAMILIES.keys())
    
    # Universal dangerous APIs
    tokens.extend(UNIVERSAL_DANGEROUS)
    
    # Truly dangerous APIs (may overlap with universal, but ensures they're included)
    tokens.extend(TRULY_DANGEROUS_APIS)
    
    # Defense tokens
    tokens.extend(DEFENSE_FAMILIES.keys())
    for token, _ in DEFENSE_PATTERNS:
        if token not in tokens:
            tokens.append(token)
    
    # Semantic bucket tokens
    all_buckets = list(SEMANTIC_BUCKETS.keys()) + ['VAR']
    if use_indexed_buckets:
        # Indexed mode: BUF_0 to BUF_{max_canonical_ids-1}, etc.
        for bucket in all_buckets:
            for i in range(max_canonical_ids):
                tokens.append(f"{bucket}_{i}")
            # Add overflow token for each bucket
            tokens.append(f"{bucket}_OVF")
    else:
        # Unindexed mode: just BUF, LEN, PTR, etc.
        tokens.extend(all_buckets)
    
    # C keywords
    tokens.extend(C_KEYWORDS)
    
    # Number tokens
    tokens.extend(PRESERVED_SMALL_INTS)
    tokens.append('NEG_1')
    tokens.extend(PRESERVED_POWERS_OF_TWO)
    tokens.extend(PRESERVED_BOUNDARIES)
    tokens.extend(PRESERVED_HEX)
    tokens.extend(['NUM', 'NUM_HEX', 'NUM_OCT'])
    for perm in PRESERVED_OCTAL:
        tokens.append(f'PERM_{perm}')
    
    # Literal tokens
    tokens.extend(['STR', 'CHAR', 'FLOAT', 'FUNC'])
    
    return tokens


def build_optimized_vocab(
    tokens_list: List[List[str]] = None,
    min_freq: int = 2,
    max_size: int = 2000,
    config: Dict = None,
) -> Tuple[Dict[str, int], Dict]:
    """
    Build compact vocabulary for optimized tokenizer.
    
    The vocab is much smaller than PreserveTokenizer (2k vs 30k) because:
    1. API families instead of individual APIs
    2. Semantic buckets instead of raw identifiers
    3. Normalized numbers
    
    Args:
        tokens_list: Optional list of tokenized samples (for frequency filtering)
        min_freq: Minimum frequency for additional tokens
        max_size: Maximum vocabulary size
        config: Optional config dict with 'max_canonical_ids' and 'use_indexed_buckets'
    
    Returns:
        vocab: {token: id}
        debug_info: Statistics
    """
    config = config or {}
    max_canonical_ids = config.get('max_canonical_ids', MAX_CANONICAL_IDS)
    use_indexed_buckets = config.get('use_indexed_buckets', True)
    
    # Start with special tokens
    vocab = {tok: idx for tok, idx in SPECIAL_TOKENS.items()}
    
    # Add all predefined tokens
    predefined = get_all_vocab_tokens(
        max_canonical_ids=max_canonical_ids,
        use_indexed_buckets=use_indexed_buckets
    )
    for tok in predefined:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    
    # Common operators and punctuation
    operators = [
        '+', '-', '*', '/', '%', '&', '|', '^', '~', '!',
        '<', '>', '=', '?', ':', ';', ',', '.', 
        '(', ')', '[', ']', '{', '}', '#',
        '==', '!=', '<=', '>=', '&&', '||', '<<', '>>', '->', '++', '--',
    ]
    for op in operators:
        if op not in vocab:
            vocab[op] = len(vocab)
    
    # If tokens provided, add frequent ones up to max_size
    if tokens_list:
        token_counts = Counter()
        for tokens in tokens_list:
            token_counts.update(tokens)
        
        total_tokens = sum(token_counts.values())
        
        # Add frequent tokens not already in vocab
        sorted_tokens = sorted(
            [(tok, count) for tok, count in token_counts.items() 
             if count >= min_freq and tok not in vocab],
            key=lambda x: (-x[1], x[0])
        )
        
        remaining = max(0, max_size - len(vocab))
        for tok, _ in sorted_tokens[:remaining]:
            vocab[tok] = len(vocab)
        
        # Calculate coverage
        covered = sum(token_counts.get(tok, 0) for tok in vocab)
        coverage = covered / total_tokens if total_tokens > 0 else 0
        
        debug_info = {
            'vocab_size': len(vocab),
            'total_unique_tokens': len(token_counts),
            'total_token_count': total_tokens,
            'coverage': coverage,
            'predefined_tokens': len(predefined),
            'added_from_data': len(vocab) - len(predefined) - len(SPECIAL_TOKENS) - len(operators),
        }
    else:
        debug_info = {
            'vocab_size': len(vocab),
            'predefined_tokens': len(predefined),
            'note': 'No token data provided, using predefined vocab only',
        }
    
    return vocab, debug_info


def vectorize_optimized(
    tokens: List[str],
    vocab: Dict[str, int],
    max_len: int = 512,
    truncation_strategy: str = 'head_tail',
    head_tokens: int = 192,
    tail_tokens: int = 319,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Vectorize tokens to input IDs.
    
    Args:
        tokens: List of token strings
        vocab: Token to ID mapping
        max_len: Maximum sequence length
        truncation_strategy: 'back', 'front', or 'head_tail'
        head_tokens: Tokens from start (for head_tail)
        tail_tokens: Tokens from end (for head_tail)
    
    Returns:
        input_ids: List of token IDs
        attention_mask: 1 for real tokens, 0 for padding
        unk_positions: Positions where UNK was used
    
    .. deprecated::
        Use :class:`HeadTailVectorizationStrategy` from 
        ``src.tokenization.vectorization_strategy`` instead.
    """
    import warnings
    warnings.warn(
        "vectorize_optimized is deprecated. Use HeadTailVectorizationStrategy from "
        "src.tokenization.vectorization_strategy instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    unk_id = vocab.get('UNK', 1)
    pad_id = vocab.get('PAD', 0)
    
    # Apply truncation
    if len(tokens) <= max_len:
        truncated = tokens
    elif truncation_strategy == 'front':
        truncated = tokens[-max_len:]
    elif truncation_strategy == 'head_tail':
        effective_tail = min(tail_tokens, max_len - head_tokens - 1)
        effective_head = min(head_tokens, max_len - effective_tail - 1)
        
        if len(tokens) <= effective_head + effective_tail:
            truncated = tokens
        else:
            head = tokens[:effective_head]
            tail = tokens[-effective_tail:] if effective_tail > 0 else []
            truncated = head + ['SEP'] + tail
    else:  # 'back'
        truncated = tokens[:max_len]
    
    # Convert to IDs
    input_ids = []
    unk_positions = []
    
    for i, tok in enumerate(truncated):
        if tok in vocab:
            input_ids.append(vocab[tok])
        else:
            input_ids.append(unk_id)
            unk_positions.append(i)
    
    # Padding
    actual_len = len(input_ids)
    if actual_len < max_len:
        input_ids.extend([pad_id] * (max_len - actual_len))
    
    attention_mask = [1] * actual_len + [0] * (max_len - actual_len)
    
    return input_ids, attention_mask, unk_positions
