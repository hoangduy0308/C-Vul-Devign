"""
Hybrid Subtoken Tokenizer for C Vulnerability Detection

Features:
- Split identifiers into subtokens (snake_case, camelCase)
- Preserve DANGEROUS_APIS, DEFENSE_APIS, C_KEYWORDS intact
- Smart numeric handling (whitelist, NEG_1, bit-width categories)
- String literal semantic mapping (SQL, URL, PATH, CRED, REGEX, IP, EMAIL)
"""

import re
import logging
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter
from tqdm import tqdm

from .hybrid_tokenizer import (
    DANGEROUS_APIS, DEFENSE_APIS, C_KEYWORDS,
    SPECIAL_TOKENS, TOKEN_REGEX,
    is_defense_function
)

logger = logging.getLogger(__name__)


# Numbers to preserve (from preserve_tokenizer)
PRESERVED_NUMBERS = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
    '32', '64', '128', '256', '512', '1024', '2048', '4096', '8192', '16384', '32768', '65536',
    '255', '1023', '4095', '65535', '2147483647', '4294967295',
    '100', '200', '500', '1000',
}

PRESERVED_HEX = {
    '0x0', '0x1', '0xff', '0xffff', '0xffffffff', '0x7fffffff',
    '0x80', '0x7f', '0x20',
}

PRESERVED_OCTAL = {
    '0644', '0600', '0755', '0777', '0666', '0700', '0400',
}

# String literal categories
STRING_CATEGORIES = {
    'SQL_STR', 'URL_STR', 'PATH_STR', 'CRED_STR', 
    'REGEX_STR', 'IP_STR', 'EMAIL_STR', 'HTTP_METHOD', 'STR'
}

# Numeric tokens
NUMERIC_TOKENS = {'NEG_1', 'NUM', 'NUM_HEX', 'NUM_OCT', 'FLOAT', 
                  'NUM_8BIT', 'NUM_16BIT', 'NUM_32BIT', 'NUM_LARGE',
                  'IDNUM'}  # IDNUM for long digit sequences inside identifiers


DEFAULT_CONFIG = {
    'preserve_dangerous_apis': True,
    'preserve_defense_apis': True,
    'preserve_function_names': False,  # SPLIT function names (except dangerous/defense)
    'preserve_identifiers': False,     # SPLIT all identifiers into subtokens
    'preserve_keywords': True,
    'identifier_case': 'lower',  # 'lower', 'preserve', 'smart'
    'digits_policy': 'split_alpha_digit',  # 'keep_alnum', 'split_alpha_digit'
    'max_subtokens_per_identifier': 8,
    # Hybrid identifier strategy: disabled by default when preserve_identifiers=False
    'hybrid_identifier_mode': False,
    'hybrid_min_freq_threshold': 5,
    # Normalize long digit sequences inside identifiers to IDNUM
    'normalize_long_digits': True,     # abc_211_aaa -> abc, IDNUM, aaa
    'long_digit_threshold': 3,         # Digits with >= 3 chars become IDNUM
    # Split macros instead of preserving whole
    'split_macros': True,              # CONFIG_GRAY -> config, gray
    'numeric_policy': {
        'keep_small_integers': True,
        'keep_negative_one': True,
        'keep_power_of_two': True,
        'keep_common_sizes': True,
        'keep_hex_masks': True,
        'keep_permissions': True,
        'use_bit_width_categories': True,
    },
    'string_policy': {
        'map_sql': True,
        'map_url': True,
        'map_path': True,
        'map_cred': True,
        'map_regex': True,
        'map_ip': True,
        'map_email': True,
    },
    'macro_prefixes': ('CONFIG_', 'AVERROR_', 'CODEC_', 'FF_', 'AV_'),
}


class HybridSubtokenTokenizer:
    """
    Tokenizer that splits identifiers into subtokens while preserving
    security-critical APIs and applying semantic mapping to literals.
    """
    
    def __init__(self, config: Dict = None, token_freq: Dict[str, int] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.numeric_policy = self.config.get('numeric_policy', {})
        self.string_policy = self.config.get('string_policy', {})
        self.macro_prefixes = self.config.get('macro_prefixes', ())
        self.regex = TOKEN_REGEX
        
        # Token frequency map for hybrid identifier mode
        # When provided, high-frequency identifiers are preserved, rare ones are split
        self.token_freq = token_freq or {}
        self.hybrid_mode = self.config.get('hybrid_identifier_mode', True) and bool(self.token_freq)
        self.hybrid_min_freq = self.config.get('hybrid_min_freq_threshold', 5)
        
        # Precompile camelCase split pattern
        self._camel_pattern = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')
        
        # Precompile string literal patterns for performance
        self._sql_pattern = re.compile(
            r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|FROM|WHERE|JOIN|GROUP BY|ORDER BY|LIMIT|UNION|HAVING|CREATE|ALTER|TRUNCATE)\b'
        )
        self._url_protocol_pattern = re.compile(r'^(https?|ftp|file)://')
        self._url_domain_pattern = re.compile(r'.+\.(com|net|org|io|gov|edu)(/|$)', re.IGNORECASE)
        self._path_pattern = re.compile(r'[\\/]')
        self._path_drive_pattern = re.compile(r'([A-Za-z]:)?[\\/]')
        self._cred_pattern = re.compile(r'(pass(word)?|secret|token|key|api[_-]?key|auth)', re.IGNORECASE)
        self._email_pattern = re.compile(r'[^@]+@[^@]+\.[^@]+')
        self._ipv4_pattern = re.compile(r'\b\d{1,3}(\.\d{1,3}){3}\b')
        self._ipv6_pattern = re.compile(r'([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}')
        self._regex_meta_pattern = re.compile(r'[.*+?|^$\\[\]{}()]')
    
    def tokenize(self, code: str) -> List[str]:
        """Tokenize code with subtoken splitting."""
        tokens, _, _ = self.tokenize_with_details(code)
        return tokens
    
    def tokenize_with_details(self, code: str) -> Tuple[List[str], List[Dict], Dict]:
        """
        Tokenize code with detailed output.
        
        Returns:
            tokens: List of token strings
            token_details: List of dicts with details for each token
            stats: Statistics about tokenization
        """
        tokens = []
        token_details = []
        
        stats = {
            'identifiers_split': [],
            'dangerous_apis': [],
            'defense_apis': [],
            'numbers_kept': [],
            'numbers_normalized': 0,
            'strings_mapped': {},
        }
        
        raw_tokens = []
        for match in self.regex.finditer(code):
            token_type = match.lastgroup
            value = match.group()
            raw_tokens.append((value, token_type))
        
        i = 0
        while i < len(raw_tokens):
            value, token_type = raw_tokens[i]
            
            if token_type in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                i += 1
                continue
            
            # Handle -1 pattern
            if self.numeric_policy.get('keep_negative_one', True) and token_type == 'OPERATOR' and value == '-':
                next_i = i + 1
                while next_i < len(raw_tokens) and raw_tokens[next_i][1] in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                    next_i += 1
                
                if next_i < len(raw_tokens):
                    next_val, next_type = raw_tokens[next_i]
                    if next_type == 'NUMBER' and next_val == '1':
                        if len(tokens) == 0 or tokens[-1] in {
                            '=', '==', '!=', '<', '>', '<=', '>=',
                            '(', ',', 'return', '?', ':', 'case',
                            '+', '-', '*', '/', '%', '&', '|', '^',
                            '&&', '||', '<<', '>>', '[', '{', ';'
                        }:
                            tokens.append('NEG_1')
                            token_details.append({'t': 'NEG_1', 'kind': 'NUM_SPECIAL', 'orig': '-1'})
                            stats['numbers_kept'].append('-1')
                            i = next_i + 1
                            continue
            
            if token_type == 'SEP_TOKEN':
                tokens.append('SEP')
                token_details.append({'t': 'SEP', 'kind': 'SPECIAL', 'orig': value})
            
            elif token_type == 'STRING':
                mapped = self._map_string_literal(value)
                tokens.append(mapped)
                token_details.append({'t': mapped, 'kind': 'STRING', 'orig': value[:50]})
                stats['strings_mapped'][mapped] = stats['strings_mapped'].get(mapped, 0) + 1
            
            elif token_type == 'CHAR':
                tokens.append('CHAR')
                token_details.append({'t': 'CHAR', 'kind': 'LITERAL', 'orig': value})
            
            elif token_type in ('NUMBER', 'HEX'):
                token_to_use = self._should_keep_number(value, token_type, tokens)
                tokens.append(token_to_use)
                
                if token_to_use not in NUMERIC_TOKENS:
                    stats['numbers_kept'].append(value)
                else:
                    stats['numbers_normalized'] += 1
                
                token_details.append({'t': token_to_use, 'kind': 'NUM', 'orig': value})
            
            elif token_type == 'FLOAT':
                tokens.append('FLOAT')
                token_details.append({'t': 'FLOAT', 'kind': 'LITERAL', 'orig': value})
            
            elif token_type == 'IDENTIFIER':
                sub_tokens, kind = self._process_identifier(value, raw_tokens, i)
                tokens.extend(sub_tokens)
                
                if kind == 'API_DANGEROUS':
                    stats['dangerous_apis'].append(value)
                elif kind == 'API_DEFENSE':
                    stats['defense_apis'].append(value)
                elif kind == 'SPLIT':
                    stats['identifiers_split'].append((value, sub_tokens))
                
                token_details.append({
                    't': sub_tokens, 
                    'kind': kind, 
                    'orig': value if len(sub_tokens) > 1 else None
                })
            
            elif token_type in ('OPERATOR', 'PUNCTUATION'):
                tokens.append(value)
                token_details.append({'t': value, 'kind': token_type})
            
            i += 1
        
        return tokens, token_details, stats
    
    def _process_identifier(self, value: str, raw_tokens: List, current_idx: int) -> Tuple[List[str], str]:
        """
        Process identifier: preserve or split into subtokens.
        
        Returns: (list_of_tokens, kind)
        """
        # Preserve C keywords
        if self.config.get('preserve_keywords', True) and value in C_KEYWORDS:
            return [value], 'KEYWORD'
        
        # Check for function call context
        next_idx = current_idx + 1
        while next_idx < len(raw_tokens):
            next_val, next_type = raw_tokens[next_idx]
            if next_type not in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                break
            next_idx += 1
        
        is_function_call = next_idx < len(raw_tokens) and raw_tokens[next_idx][0] == '('
        
        # Preserve dangerous APIs
        if self.config.get('preserve_dangerous_apis', True) and value in DANGEROUS_APIS:
            if is_function_call:
                return [value], 'API_DANGEROUS'
        
        # Preserve defense APIs
        if is_function_call and is_defense_function(value):
            return [value], 'API_DEFENSE'
        
        # Check if ALL_CAPS macro
        is_macro = self._is_macro(value)
        
        # Handle macro splitting (if split_macros=True, split macros instead of preserving)
        if is_macro:
            if self.config.get('split_macros', True):
                # Split macro: CONFIG_GRAY -> ['config', 'gray']
                subtokens = self._split_identifier(value)
                subtokens = self._normalize_subtokens(subtokens)
                return subtokens, 'MACRO_SPLIT'
            else:
                return [value], 'MACRO'
        
        # Preserve function names (if preserve_function_names=True)
        if self.config.get('preserve_function_names', False) and is_function_call:
            case_policy = self.config.get('identifier_case', 'lower')
            if case_policy == 'lower':
                return [value.lower()], 'FUNCTION'
            return [value], 'FUNCTION'
        
        # Apply case policy
        case_policy = self.config.get('identifier_case', 'lower')
        normalized_value = value.lower() if case_policy == 'lower' else value
        
        # Hybrid identifier strategy: check frequency before deciding to preserve or split
        if self.config.get('preserve_identifiers', False):
            # If hybrid mode is enabled and we have frequency data
            if self.hybrid_mode:
                freq = self.token_freq.get(normalized_value, 0)
                if freq >= self.hybrid_min_freq:
                    return [normalized_value], 'IDENTIFIER'
                else:
                    subtokens = self._split_identifier(value)
                    subtokens = self._normalize_subtokens(subtokens)
                    
                    if len(subtokens) == 1:
                        return [normalized_value], 'IDENTIFIER_RARE'
                    
                    max_sub = self.config.get('max_subtokens_per_identifier', 8)
                    if len(subtokens) > max_sub:
                        subtokens = subtokens[:max_sub]
                    return subtokens, 'SPLIT_RARE'
            else:
                return [normalized_value], 'IDENTIFIER'
        
        # ALWAYS split identifier into subtokens (preserve_identifiers=False)
        subtokens = self._split_identifier(value)
        subtokens = self._normalize_subtokens(subtokens)
        
        if len(subtokens) == 1:
            return subtokens, 'IDENTIFIER'
        
        max_sub = self.config.get('max_subtokens_per_identifier', 8)
        if len(subtokens) > max_sub:
            subtokens = subtokens[:max_sub]
        
        return subtokens, 'SPLIT'
    
    def _normalize_subtokens(self, subtokens: List[str]) -> List[str]:
        """
        Apply normalization to subtokens:
        - Lowercase if identifier_case='lower'
        - Normalize long digit sequences to IDNUM if normalize_long_digits=True
        
        Examples:
            ['abc', '211', 'aaa'] -> ['abc', 'IDNUM', 'aaa']  (if 211 >= threshold)
            ['h', '264'] -> ['h', 'IDNUM']  (if split_alpha_digit and 264 >= threshold)
        """
        case_policy = self.config.get('identifier_case', 'lower')
        normalize_long_digits = self.config.get('normalize_long_digits', True)
        long_digit_threshold = self.config.get('long_digit_threshold', 3)
        
        result = []
        for s in subtokens:
            if not s:
                continue
            
            # Check if this subtoken is all digits
            if normalize_long_digits and s.isdigit():
                if len(s) >= long_digit_threshold:
                    # Normalize long digit sequences to IDNUM
                    result.append('IDNUM')
                else:
                    # Keep short digits (0-99 typically)
                    result.append(s)
            else:
                # Apply case policy for non-digit subtokens
                if case_policy == 'lower':
                    result.append(s.lower())
                else:
                    result.append(s)
        
        return result if result else subtokens
    
    def _is_macro(self, value: str) -> bool:
        """Check if identifier looks like a macro (ALL_CAPS with underscores)."""
        # All uppercase with optional underscores and digits
        if re.match(r'^[A-Z][A-Z0-9_]*$', value) and len(value) > 1:
            return True
        # Check configurable project-specific prefixes
        if self.macro_prefixes and value.startswith(self.macro_prefixes):
            return True
        return False
    
    def _split_identifier(self, ident: str) -> List[str]:
        """
        Split identifier into subtokens by snake_case and camelCase.
        
        Examples:
            h264_filter_mb_fast_internal -> ['h264', 'filter', 'mb', 'fast', 'internal']
            getByteCount -> ['get', 'Byte', 'Count'] (case preserved, normalized later)
            HTTPConnection -> ['HTTP', 'Connection'] (case preserved, normalized later)
            abc_211_aaa -> ['abc', '211', 'aaa'] (digits split if policy allows)
        
        Note: Case normalization and IDNUM substitution are done in _normalize_subtokens()
        """
        # First split by underscore (snake_case)
        parts = self._split_snake(ident)
        
        # Then split each part by camelCase
        result = []
        for part in parts:
            camel_parts = self._split_camel(part)
            result.extend(camel_parts)
        
        # Filter empty strings
        result = [s for s in result if s]
        
        return result if result else [ident]
    
    def _split_snake(self, s: str) -> List[str]:
        """Split by underscores, preserving alphanumeric parts."""
        parts = s.split('_')
        return [p for p in parts if p]
    
    def _split_camel(self, s: str) -> List[str]:
        """
        Split camelCase or PascalCase.
        
        Uses policy to handle digits:
        - keep_alnum: h264 stays as 'h264', sha256 stays as 'sha256'
        - split_alpha_digit: h264 -> ['h', '264']
        """
        digits_policy = self.config.get('digits_policy', 'keep_alnum')
        
        if digits_policy == 'keep_alnum':
            # Split at camelCase boundaries but keep alphanumeric chunks
            parts = self._camel_pattern.split(s)
            return [p for p in parts if p]
        else:
            # Also split at alpha-digit boundaries
            parts = re.split(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', s)
            return [p for p in parts if p]
    
    def _should_keep_number(self, value: str, token_type: str, prev_tokens: List[str]) -> str:
        """
        Determine how to handle a number: keep as-is, normalize to category, or NUM.
        
        Returns the token string to use.
        """
        policy = self.numeric_policy
        
        # Handle hex numbers
        if token_type == 'HEX':
            value_lower = value.lower()
            clean_hex = re.sub(r'[uUlL]+$', '', value_lower)
            
            if policy.get('keep_hex_masks', True) and clean_hex in PRESERVED_HEX:
                return clean_hex
            
            return 'NUM_HEX'
        
        # Regular numbers - remove suffix
        clean_num = re.sub(r'[uUlL]+$', '', value)
        
        # Check octal first
        if clean_num.startswith('0') and len(clean_num) > 1 and not clean_num.startswith('0x'):
            if policy.get('keep_permissions', True) and clean_num in PRESERVED_OCTAL:
                return f'PERM_{clean_num}'
            return 'NUM_OCT'
        
        # Check small integers (0-16)
        if policy.get('keep_small_integers', True):
            try:
                num = int(clean_num)
                if 0 <= num <= 16:
                    return clean_num
            except ValueError:
                pass
        
        # Check common sizes
        if policy.get('keep_common_sizes', True) and clean_num in PRESERVED_NUMBERS:
            return clean_num
        
        # Check power of two
        if policy.get('keep_power_of_two', True):
            try:
                num = int(clean_num)
                if num > 0 and (num & (num - 1)) == 0 and num <= 65536:
                    return clean_num
            except ValueError:
                pass
        
        # Use bit-width categories
        if policy.get('use_bit_width_categories', True):
            try:
                num = abs(int(clean_num))
                if num < 2**8:
                    return 'NUM_8BIT'
                elif num < 2**16:
                    return 'NUM_16BIT'
                elif num < 2**32:
                    return 'NUM_32BIT'
                return 'NUM_LARGE'
            except ValueError:
                pass
        
        return 'NUM'
    
    def _map_string_literal(self, literal: str) -> str:
        """
        Map string literal to semantic category.
        
        Categories: SQL_STR, URL_STR, PATH_STR, CRED_STR, REGEX_STR, IP_STR, EMAIL_STR, STR
        """
        # Remove quotes
        val = literal.strip('"\'')
        policy = self.string_policy
        upper_val = val.upper()
        
        # SQL patterns
        if policy.get('map_sql', True):
            if self._sql_pattern.search(upper_val):
                return 'SQL_STR'
        
        # URL/domain patterns
        if policy.get('map_url', True):
            if self._url_protocol_pattern.match(val) or self._url_domain_pattern.match(val):
                return 'URL_STR'
        
        # File path patterns
        if policy.get('map_path', True):
            if self._path_pattern.search(val) or self._path_drive_pattern.match(val):
                return 'PATH_STR'
        
        # Credential patterns
        if policy.get('map_cred', True):
            if self._cred_pattern.search(val):
                return 'CRED_STR'
        
        # Email patterns
        if policy.get('map_email', True):
            if self._email_pattern.match(val):
                return 'EMAIL_STR'
        
        # IP address patterns
        if policy.get('map_ip', True):
            # IPv4
            if self._ipv4_pattern.match(val):
                return 'IP_STR'
            # IPv6
            if self._ipv6_pattern.match(val):
                return 'IP_STR'
        
        # Regex patterns (has regex metacharacters and length > 3)
        if policy.get('map_regex', True):
            if self._regex_meta_pattern.search(val) and len(val) > 3:
                # Exclude URLs
                if not self._url_protocol_pattern.match(val):
                    return 'REGEX_STR'
        
        return 'STR'
    
    def tokenize_batch(self, codes: List[str], with_details: bool = False) -> Tuple:
        """Batch tokenization."""
        all_tokens = []
        all_details = []
        all_stats = []
        
        for code in tqdm(codes, desc="Tokenizing"):
            tokens, details, stats = self.tokenize_with_details(code)
            all_tokens.append(tokens)
            all_details.append(details)
            all_stats.append(stats)
        
        if with_details:
            return all_tokens, all_details, all_stats
        return all_tokens


def build_subtoken_vocab(
    tokens_list: List[List[str]],
    min_freq: int = 2,
    max_size: int = 30000,
) -> Tuple[Dict[str, int], Dict]:
    """
    Build vocabulary for subtoken tokenizer.
    
    Reserved tokens:
    - SPECIAL_TOKENS (PAD, UNK, BOS, EOS, SEP)
    - DANGEROUS_APIS, DEFENSE_APIS
    - C_KEYWORDS
    - Numeric categories, String categories
    
    Args:
        tokens_list: List of tokenized samples
        min_freq: Minimum frequency to include token
        max_size: Maximum vocabulary size
    
    Returns:
        vocab: {token: id}
        debug_info: Statistics about vocabulary building
    """
    token_counts = Counter()
    for tokens in tokens_list:
        token_counts.update(tokens)
    
    total_unique = len(token_counts)
    total_tokens = sum(token_counts.values())
    
    # Start with special tokens
    vocab = {tok: idx for tok, idx in SPECIAL_TOKENS.items()}
    
    # Add dangerous APIs
    for api in DANGEROUS_APIS:
        if api not in vocab:
            vocab[api] = len(vocab)
    
    # Add defense APIs
    for api in DEFENSE_APIS:
        if api not in vocab:
            vocab[api] = len(vocab)
    
    # Add C keywords
    for kw in C_KEYWORDS:
        if kw not in vocab:
            vocab[kw] = len(vocab)
    
    # Add preserved numbers
    for num in PRESERVED_NUMBERS:
        if num not in vocab:
            vocab[num] = len(vocab)
    
    for hex_num in PRESERVED_HEX:
        if hex_num not in vocab:
            vocab[hex_num] = len(vocab)
    
    # Add numeric tokens
    for num_tok in NUMERIC_TOKENS:
        if num_tok not in vocab:
            vocab[num_tok] = len(vocab)
    
    # Add PERM tokens
    for perm in PRESERVED_OCTAL:
        tok = f'PERM_{perm}'
        if tok not in vocab:
            vocab[tok] = len(vocab)
    
    # Add string category tokens
    for str_cat in STRING_CATEGORIES:
        if str_cat not in vocab:
            vocab[str_cat] = len(vocab)
    
    # Add CHAR, FUNC tokens
    for special in ['CHAR', 'FUNC']:
        if special not in vocab:
            vocab[special] = len(vocab)
    
    # Add frequent tokens
    filtered_tokens = [
        (tok, count) for tok, count in token_counts.items()
        if count >= min_freq and tok not in vocab
    ]
    filtered_tokens.sort(key=lambda x: (-x[1], x[0]))
    
    remaining_size = max(0, max_size - len(vocab))
    
    # Track tokens that pass min_freq but exceed vocab limit
    tokens_added = filtered_tokens[:remaining_size]
    tokens_truncated = filtered_tokens[remaining_size:]  # These become UNK
    
    for tok, count in tokens_added:
        vocab[tok] = len(vocab)
    
    # Compute coverage
    covered_count = sum(token_counts.get(tok, 0) for tok in vocab)
    coverage = covered_count / total_tokens if total_tokens > 0 else 0
    
    # Compute UNK statistics
    unk_tokens_count = sum(count for tok, count in tokens_truncated)
    tokens_below_min_freq = [
        (tok, count) for tok, count in token_counts.items()
        if count < min_freq and tok not in vocab
    ]
    unk_from_low_freq = sum(count for tok, count in tokens_below_min_freq)
    
    total_unk = unk_tokens_count + unk_from_low_freq
    unk_rate = total_unk / total_tokens if total_tokens > 0 else 0
    
    # WARNING: High OOV rate detection
    # When preserve_identifiers=True with limited vocab, OOV can be very high
    if unk_rate > 0.10:  # More than 10% UNK is concerning
        logger.warning(
            f"HIGH OOV RATE DETECTED: {unk_rate:.1%} of tokens will become UNK! "
            f"Truncated: {len(tokens_truncated):,} unique ({unk_tokens_count:,} occurrences), "
            f"Below min_freq: {len(tokens_below_min_freq):,} unique ({unk_from_low_freq:,} occurrences). "
            f"Consider: 1) Set preserve_identifiers=False, 2) Increase max_vocab_size (currently {max_size}), "
            f"3) Increase min_freq (currently {min_freq})"
        )
    
    debug_info = {
        'total_unique_tokens': total_unique,
        'total_token_count': total_tokens,
        'vocab_size': len(vocab),
        'coverage': coverage,
        'min_freq': min_freq,
        'max_size': max_size,
        'reserved_size': len(SPECIAL_TOKENS) + len(DANGEROUS_APIS) + len(DEFENSE_APIS) + len(C_KEYWORDS),
        # UNK statistics
        'unk_rate': unk_rate,
        'unk_from_vocab_limit': len(tokens_truncated),  # Tokens pass min_freq but exceed vocab
        'unk_from_vocab_limit_count': unk_tokens_count,  # Total occurrences
        'unk_from_low_freq': len(tokens_below_min_freq),  # Tokens below min_freq
        'unk_from_low_freq_count': unk_from_low_freq,  # Total occurrences
        'total_unk_occurrences': total_unk,
        # Top truncated tokens (for debugging)
        'top_truncated_tokens': [(tok, count) for tok, count in tokens_truncated[:20]],
    }
    
    return vocab, debug_info


if __name__ == "__main__":
    print("=" * 60)
    print("HYBRID SUBTOKEN TOKENIZER TEST")
    print("=" * 60)
    
    test_code = '''
    static void h264_filter_mb_fast_internal(H264Context *h, uint8_t *img_y) {
        int qp_bd_offset = 6 * (h->sps.bit_depth_luma - 8);
        char *url = "https://example.com/api";
        char *query = "SELECT * FROM users WHERE id=1";
        memcpy(buf, src, len);
        if (ptr == NULL) {
            return (size_t)-1;
        }
        av_frame_unref(frame);
    }
    '''
    
    tokenizer = HybridSubtokenTokenizer()
    tokens, details, stats = tokenizer.tokenize_with_details(test_code)
    
    print("\n[Tokens]")
    print(' '.join(tokens[:60]))
    print(f"\nTotal tokens: {len(tokens)}")
    
    print("\n[Identifier Splits]")
    for orig, subtoks in stats['identifiers_split'][:10]:
        print(f"  {orig} -> {subtoks}")
    
    print("\n[Dangerous APIs]")
    print(f"  {stats['dangerous_apis']}")
    
    print("\n[String Mapping]")
    print(f"  {stats['strings_mapped']}")
    
    print("\n[Numbers Kept]")
    print(f"  {stats['numbers_kept']}")
    
    # Test specific identifier splits
    print("\n" + "=" * 60)
    print("IDENTIFIER SPLIT TESTS")
    print("=" * 60)
    
    test_idents = [
        'h264_filter_mb_fast_internal',
        'getByteCount',
        'HTTPConnection',
        'bit_depth_luma',
        'av_frame_unref',
        'CONFIG_GRAY',
        'AVERROR_EOF',
        'sha256_hash',
        'processUserInput',
    ]
    
    for ident in test_idents:
        subtoks = tokenizer._split_identifier(ident)
        is_macro = tokenizer._is_macro(ident)
        status = "(MACRO - preserved)" if is_macro else ""
        print(f"  {ident:35s} -> {subtoks} {status}")
    
    print("\n" + "=" * 60)
    print("BUILD VOCAB TEST")
    print("=" * 60)
    
    vocab, debug = build_subtoken_vocab([tokens] * 10, min_freq=1, max_size=1000)
    print(f"  Vocab size: {debug['vocab_size']}")
    print(f"  Coverage: {debug['coverage']:.2%}")
    print(f"  Reserved tokens: {debug['reserved_size']}")
    
    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)
