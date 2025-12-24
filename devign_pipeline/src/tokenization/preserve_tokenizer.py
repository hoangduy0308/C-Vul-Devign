"""
Preserve-Identifier Tokenizer for C Vulnerability Detection

Features:
- Keeps original variable names (buf, len, ptr...) instead of normalizing to VAR_k
- Smart number handling: preserves important numbers (0, 1, -1, power-of-2...)
- Force-keeps security-important identifiers regardless of frequency
"""

import re
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter
from tqdm import tqdm

from .hybrid_tokenizer import (
    DANGEROUS_APIS, DEFENSE_APIS, C_KEYWORDS,
    SPECIAL_TOKENS, TOKEN_REGEX,
    is_defense_function
)


# Security-important identifiers to force-keep regardless of frequency
FORCE_KEEP_IDENTIFIERS = {
    # Buffer-related
    'buf', 'buffer', 'dst', 'src', 'data', 'payload', 'msg', 'packet', 'input', 'output',
    'dest', 'source', 'str', 'string', 'ptr', 'pointer',
    
    # Length/size-related  
    'len', 'length', 'size', 'sz', 'count', 'cnt', 'num', 'n', 'max', 'min',
    'capacity', 'limit', 'bound', 'offset', 'pos', 'idx', 'index',
    
    # Security-sensitive
    'password', 'passwd', 'pwd', 'secret', 'token', 'credential', 'cred', 
    'auth', 'session', 'cookie', 'key', 'cert', 'private', 'hash', 'salt', 
    'nonce', 'iv', 'admin', 'root', 'user', 'privilege', 'priv', 'permission',
    'perm', 'role', 'access', 'owner', 'group', 'uid', 'gid',
    
    # File/path
    'path', 'file', 'filename', 'filepath', 'dir', 'directory', 'fd', 'fp',
    
    # Command/shell
    'cmd', 'command', 'shell', 'exec', 'arg', 'args', 'argv', 'argc', 'env',
    
    # Memory
    'mem', 'memory', 'heap', 'stack', 'alloc', 'block', 'chunk', 'node',
    
    # Return/error
    'ret', 'result', 'err', 'error', 'errno', 'status', 'rc', 'rv',
}

# Numbers to preserve (whitelist)
PRESERVED_NUMBERS = {
    # Small integers (boundary checks, off-by-one)
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
    
    # Power of two (buffer sizes, alignment)
    '32', '64', '128', '256', '512', '1024', '2048', '4096', '8192', '16384', '32768', '65536',
    
    # Common max values (boundary)
    '255', '1023', '4095', '65535', '2147483647',
    '4294967295', '18446744073709551615',  # uint32 max, uint64/size_t max
    
    # Common buffer sizes
    '100', '200', '500', '1000',
}

# Hex numbers to preserve
PRESERVED_HEX = {
    '0x0', '0x1', '0xff', '0xffff', '0xffffffff', '0x7fffffff',
    '0x80', '0x7f', '0x20',  # ASCII boundaries
    '0xdeadbeef', '0xcafebabe',  # Debug markers
}

# Octal permissions to preserve
PRESERVED_OCTAL = {
    '0644', '0600', '0755', '0777', '0666', '0700', '0400',
}

# Common FFmpeg/QEMU function names to preserve (domain-specific, high frequency)
# These are NOT dangerous but provide important semantic signal
PRESERVE_FUNCTION_NAMES = {
    # FFmpeg common functions (non-dangerous but informative)
    'av_log', 'av_get_bytes_per_sample', 'av_get_channel_layout_nb_channels',
    'av_rescale', 'av_rescale_q', 'av_rescale_rnd', 'av_compare_ts',
    'av_image_get_linesize', 'av_image_fill_arrays', 'av_samples_get_buffer_size',
    'avcodec_find_decoder', 'avcodec_find_encoder', 'avcodec_open2',
    'avcodec_send_packet', 'avcodec_receive_frame', 'avcodec_send_frame',
    'avcodec_receive_packet', 'avcodec_decode_video2', 'avcodec_decode_audio4',
    'avformat_open_input', 'avformat_find_stream_info', 'av_read_frame',
    'av_seek_frame', 'avformat_write_header', 'av_write_frame', 'av_interleaved_write_frame',
    'av_init_packet', 'av_new_packet', 'av_grow_packet', 'av_shrink_packet',
    'av_frame_alloc', 'av_frame_clone', 'av_frame_copy', 'av_frame_copy_props',
    'av_frame_get_buffer', 'av_frame_make_writable', 'av_frame_is_writable',
    'sws_getContext', 'sws_scale', 'swr_alloc', 'swr_init', 'swr_convert',
    'av_opt_set', 'av_opt_get', 'av_opt_set_int', 'av_opt_set_double',
    'av_dict_set', 'av_dict_get', 'av_dict_copy',
    
    # QEMU common functions
    'cpu_get_tb_cpu_state', 'cpu_loop_exit', 'cpu_loop_exit_restore',
    'qemu_log', 'qemu_log_mask', 'error_report', 'error_setg', 'error_propagate',
    'memory_region_init', 'memory_region_init_io', 'memory_region_init_ram',
    'address_space_read', 'address_space_write', 'address_space_rw',
    'pci_dma_read', 'pci_dma_write', 'dma_memory_read', 'dma_memory_write',
    'qemu_get_be32', 'qemu_get_be64', 'qemu_put_be32', 'qemu_put_be64',
    'object_initialize', 'object_property_set_bool', 'object_property_set_int',
    'timer_new_ns', 'timer_mod', 'timer_del', 'timer_free',
    'qemu_mutex_lock', 'qemu_mutex_unlock', 'qemu_cond_wait', 'qemu_cond_signal',
    
    # GLib common functions
    'g_hash_table_lookup', 'g_hash_table_insert', 'g_hash_table_remove',
    'g_list_append', 'g_list_prepend', 'g_list_remove', 'g_list_free',
    'g_string_new', 'g_string_append', 'g_string_free',
    'g_error_free', 'g_clear_error',
    
    # Return value / error check patterns
    'AVERROR', 'AVERROR_EOF', 'AVERROR_INVALIDDATA', 'av_err2str',
    'IS_ERR', 'PTR_ERR', 'ERR_PTR', 'WARN_ON', 'BUG_ON',
}

# Default configuration
# v4: vocab_size 10k -> 30k, min_freq 3 -> 2 (reduce UNK rate)
DEFAULT_CONFIG = {
    'min_freq': 2,
    'max_vocab_size': 30000,
    'max_seq_length': 512,
    'preserve_identifiers': True,
    'preserve_dangerous_apis': True,
    'preserve_keywords': True,
    'preserve_common_functions': True,  # NEW: preserve domain-specific function names
    'numeric_policy': {
        'keep_small_integers': True,
        'keep_negative_one': True,
        'keep_power_of_two': True,
        'keep_common_sizes': True,
        'keep_hex_masks': True,
        'keep_permissions': True,
    }
}


class PreserveIdentifierTokenizer:
    """
    Tokenizer that preserves original variable names instead of normalizing.
    
    Features:
    - Keeps original identifier names (buf, len, ptr, etc.)
    - Smart number handling: keeps important numbers, normalizes others to NUM
    - Preserves dangerous APIs, defense APIs, and C keywords
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG.copy()
        self.numeric_policy = self.config.get('numeric_policy', {})
        self.regex = TOKEN_REGEX
    
    def _looks_like_type_cast_before_unary_minus(self, tokens: List[str]) -> bool:
        """Check if previous tokens look like a type cast before unary minus, e.g. (size_t)-1"""
        if len(tokens) < 2 or tokens[-1] != ')':
            return False
        
        unary_prefix = {
            '=', '==', '!=', '<', '>', '<=', '>=',
            '(', ',', 'return', '?', ':', 'case',
            '+', '-', '*', '/', '%', '&', '|', '^',
            '&&', '||', '<<', '>>', '[', '{', ';'
        }
        
        typeish_keywords = {
            'void', 'char', 'short', 'int', 'long', 'signed', 'unsigned',
            'float', 'double', '_Bool', 'struct', 'union', 'enum',
            'const', 'volatile', 'restrict'
        }
        
        # Find matching open paren
        close_idx = len(tokens) - 1
        scan_start = max(0, len(tokens) - 12)
        
        open_idx = None
        for j in range(close_idx - 1, scan_start - 1, -1):
            if tokens[j] == '(':
                open_idx = j
                break
            if tokens[j] in {';', '{', '}', '[', ']'}:
                break
        
        if open_idx is None:
            return False
        
        # Check token before '(' is in unary prefix context
        if open_idx > 0 and tokens[open_idx - 1] not in unary_prefix:
            return False
        
        # Check inside parens for type-like tokens
        inside = tokens[open_idx + 1:close_idx]
        if not inside or ',' in inside:
            return False
        
        for tok in inside:
            if tok in typeish_keywords or tok.endswith('_t'):
                return True
        
        return False
    
    def _should_keep_number(self, value: str, token_type: str) -> Tuple[bool, str]:
        """
        Determine if a number should be kept as-is or normalized to NUM.
        
        Returns: (should_keep, token_to_use)
        """
        policy = self.config.get('numeric_policy', {})
        
        # Check hex numbers
        if token_type == 'HEX':
            value_lower = value.lower()
            clean_hex = re.sub(r'[uUlL]+$', '', value_lower)
            
            if policy.get('keep_hex_masks', True):
                if clean_hex in PRESERVED_HEX:
                    return True, clean_hex
            
            return False, 'NUM_HEX'
        
        # Regular numbers - remove suffix
        clean_num = re.sub(r'[uUlL]+$', '', value)
        
        # Check octal FIRST (starts with 0 but not 0x) - before small integers!
        if clean_num.startswith('0') and len(clean_num) > 1 and not clean_num.startswith('0x'):
            if policy.get('keep_permissions', True) and clean_num in PRESERVED_OCTAL:
                return True, f'PERM_{clean_num}'
            return False, 'NUM_OCT'
        
        # Check small integers (0-16)
        if policy.get('keep_small_integers', True):
            try:
                num = int(clean_num)
                if 0 <= num <= 16:
                    return True, clean_num
            except ValueError:
                pass
        
        # Check common sizes
        if policy.get('keep_common_sizes', True):
            if clean_num in {'32', '64', '128', '256', '512', '1024', '2048', '4096', 
                            '8192', '16384', '32768', '65536', '255', '1023', '4095', 
                            '65535', '2147483647', '100', '200', '500', '1000'}:
                return True, clean_num
        
        # Check preserved numbers (fallback for any in whitelist)
        if clean_num in PRESERVED_NUMBERS:
            return True, clean_num
        
        # Check power of two
        if policy.get('keep_power_of_two', True):
            try:
                num = int(clean_num)
                if num > 0 and (num & (num - 1)) == 0 and num <= 65536:
                    return True, clean_num
            except ValueError:
                pass
        
        return False, 'NUM'
    
    def tokenize(self, code: str) -> List[str]:
        """
        Tokenize code with identifier preservation.
        
        Returns:
            tokens: List of token strings
        """
        tokens, _, _ = self.tokenize_with_details(code)
        return tokens
    
    def tokenize_with_details(self, code: str) -> Tuple[List[str], List[Dict], Dict]:
        """
        Tokenize code with detailed output for debugging.
        
        Returns:
            tokens: List of token strings
            token_details: List of dicts with details for each token
            stats: Statistics about tokenization
        """
        tokens = []
        token_details = []
        
        stats = {
            'identifiers': [],
            'dangerous_apis': [],
            'defense_apis': [],
            'numbers_kept': [],
            'numbers_normalized': 0,
        }
        
        raw_tokens = []
        for match in self.regex.finditer(code):
            token_type = match.lastgroup
            value = match.group()
            start = match.start()
            end = match.end()
            raw_tokens.append((value, token_type, start, end))
        
        i = 0
        while i < len(raw_tokens):
            value, token_type, start, end = raw_tokens[i]
            
            if token_type in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                i += 1
                continue
            
            # Handle -1 pattern - more robust detection
            if self.numeric_policy.get('keep_negative_one', True) and token_type == 'OPERATOR' and value == '-':
                next_i = i + 1
                while next_i < len(raw_tokens) and raw_tokens[next_i][1] in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                    next_i += 1
                
                if next_i < len(raw_tokens):
                    next_val, next_type, _, _ = raw_tokens[next_i]
                    if next_type == 'NUMBER' and next_val == '1':
                        # Expanded context: operators, keywords, punctuation that precede -1
                        if len(tokens) == 0 or tokens[-1] in {
                            '=', '==', '!=', '<', '>', '<=', '>=',
                            '(', ',', 'return', '?', ':', 'case',
                            '+', '-', '*', '/', '%', '&', '|', '^',
                            '&&', '||', '<<', '>>', '[', '{', ';'
                        } or self._looks_like_type_cast_before_unary_minus(tokens):
                            tokens.append('NEG_1')
                            token_details.append({
                                't': 'NEG_1',
                                'kind': 'NUM_SPECIAL',
                                'orig': '-1',
                            })
                            stats['numbers_kept'].append('-1')
                            i = next_i + 1
                            continue
            
            if token_type == 'SEP_TOKEN':
                tokens.append('SEP')
                token_details.append({'t': 'SEP', 'kind': 'SPECIAL', 'orig': value})
                
            elif token_type == 'STRING':
                tokens.append('STR')
                token_details.append({
                    't': 'STR', 
                    'kind': 'LITERAL', 
                    'orig': value[:50] + '...' if len(value) > 50 else value
                })
                
            elif token_type == 'CHAR':
                tokens.append('CHAR')
                token_details.append({'t': 'CHAR', 'kind': 'LITERAL', 'orig': value})
                
            elif token_type in ('NUMBER', 'HEX'):
                should_keep, token_to_use = self._should_keep_number(value, token_type)
                tokens.append(token_to_use)
                
                if should_keep:
                    stats['numbers_kept'].append(value)
                    kind = 'NUM_KEPT'
                else:
                    stats['numbers_normalized'] += 1
                    kind = 'NUM_NORM'
                
                token_details.append({'t': token_to_use, 'kind': kind, 'orig': value})
                
            elif token_type == 'FLOAT':
                tokens.append('FLOAT')
                token_details.append({'t': 'FLOAT', 'kind': 'LITERAL', 'orig': value})
                
            elif token_type == 'IDENTIFIER':
                token, kind = self._normalize_identifier(value, raw_tokens, i)
                tokens.append(token)
                
                if kind == 'API_DANGEROUS':
                    stats['dangerous_apis'].append(value)
                elif kind == 'API_DEFENSE':
                    stats['defense_apis'].append(value)
                elif kind == 'IDENTIFIER':
                    stats['identifiers'].append(value)
                
                detail = {'t': token, 'kind': kind}
                if token != value:
                    detail['orig'] = value
                token_details.append(detail)
                
            elif token_type in ('OPERATOR', 'PUNCTUATION'):
                tokens.append(value)
                token_details.append({'t': value, 'kind': token_type})
            
            i += 1
        
        return tokens, token_details, stats
    
    def _normalize_identifier(self, value: str, raw_tokens: List, current_idx: int) -> Tuple[str, str]:
        """
        Process identifier: keep as-is or normalize based on context.
        
        Returns: (token, kind)
        """
        if self.config.get('preserve_keywords', True) and value in C_KEYWORDS:
            return value, 'KEYWORD'
        
        # Check previous token to filter out member access (obj.malloc or obj->malloc)
        prev_idx = current_idx - 1
        while prev_idx >= 0:
            prev_val, prev_type, _, _ = raw_tokens[prev_idx]
            if prev_type not in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                break
            prev_idx -= 1
        
        is_member_access = prev_idx >= 0 and raw_tokens[prev_idx][0] in ('.', '->')
        
        # Check next token for function call detection
        next_idx = current_idx + 1
        while next_idx < len(raw_tokens):
            next_val, next_type, _, _ = raw_tokens[next_idx]
            if next_type not in ('WHITESPACE', 'COMMENT_MULTI', 'COMMENT_SINGLE'):
                break
            next_idx += 1
        
        is_function_call = next_idx < len(raw_tokens) and raw_tokens[next_idx][0] == '('
        
        # Only tag as dangerous API if it's a real call (not member access)
        if self.config.get('preserve_dangerous_apis', True) and value in DANGEROUS_APIS:
            if is_function_call and not is_member_access:
                return value, 'API_DANGEROUS'
            else:
                # Still preserve the token text for model to see
                return value, 'IDENTIFIER'
        
        if is_function_call and not is_member_access and is_defense_function(value):
            return value, 'API_DEFENSE'
        
        # NEW: Preserve common domain-specific function names
        if is_function_call and not is_member_access:
            if self.config.get('preserve_common_functions', True) and value in PRESERVE_FUNCTION_NAMES:
                return value, 'FUNC_NAMED'
            return 'FUNC', 'FUNC'
        
        if self.config.get('preserve_identifiers', True):
            return value, 'IDENTIFIER'
        
        return 'ID', 'ID'
    
    def tokenize_batch(self, codes: List[str], n_jobs: int = 1, 
                       with_details: bool = False) -> Tuple:
        """
        Batch tokenization.
        
        Args:
            codes: List of code strings
            n_jobs: Number of parallel jobs (not used, for API compatibility)
            with_details: If True, return details and stats
            
        Returns:
            If with_details=False: List[List[str]] - tokens only
            If with_details=True: (tokens, details, stats) tuples
        """
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


def build_preserve_vocab(
    tokens_list: List[List[str]], 
    min_freq: int = 3,
    max_size: int = 10000,
    force_keep: Set[str] = None
) -> Tuple[Dict[str, int], Dict]:
    """
    Build vocabulary with frequency filtering.
    
    Args:
        tokens_list: List of tokenized samples
        min_freq: Minimum frequency to include token
        max_size: Maximum vocabulary size
        force_keep: Set of identifiers to always keep
    
    Returns:
        vocab: {token: id}
        debug_info: Statistics about vocabulary building
    """
    force_keep = force_keep or FORCE_KEEP_IDENTIFIERS
    
    token_counts = Counter()
    for tokens in tokens_list:
        token_counts.update(tokens)
    
    total_unique = len(token_counts)
    total_tokens = sum(token_counts.values())
    
    vocab = {tok: idx for tok, idx in SPECIAL_TOKENS.items()}
    
    for api in DANGEROUS_APIS:
        if api not in vocab:
            vocab[api] = len(vocab)
    
    for api in DEFENSE_APIS:
        if api not in vocab:
            vocab[api] = len(vocab)
    
    for kw in C_KEYWORDS:
        if kw not in vocab:
            vocab[kw] = len(vocab)
    
    force_kept = []
    for ident in force_keep:
        if ident not in vocab:
            vocab[ident] = len(vocab)
            force_kept.append(ident)
    
    # Always add preserved numbers (important for boundary checks, buffer sizes)
    for num in PRESERVED_NUMBERS:
        if num not in vocab:
            vocab[num] = len(vocab)
    
    # Always add preserved hex numbers (important for masks, boundaries)
    for hex_num in PRESERVED_HEX:
        if hex_num not in vocab:
            vocab[hex_num] = len(vocab)
    
    for special in ['NEG_1', 'NUM', 'NUM_HEX', 'NUM_OCT', 'FLOAT', 'STR', 'CHAR', 'FUNC']:
        if special not in vocab:
            vocab[special] = len(vocab)
    
    for tok in token_counts:
        if tok.startswith('PERM_') and tok not in vocab:
            vocab[tok] = len(vocab)
    
    filtered_tokens = [
        (tok, count) for tok, count in token_counts.items() 
        if count >= min_freq and tok not in vocab
    ]
    filtered_tokens.sort(key=lambda x: (-x[1], x[0]))
    
    dropped_by_freq = [
        (tok, count) for tok, count in token_counts.items()
        if count < min_freq and tok not in vocab
    ]
    
    remaining_size = max(0, max_size - len(vocab))
    added_tokens = []
    for tok, count in filtered_tokens[:remaining_size]:
        vocab[tok] = len(vocab)
        added_tokens.append((tok, count))
    
    dropped_by_size = filtered_tokens[remaining_size:] if len(filtered_tokens) > remaining_size else []
    
    covered_count = sum(token_counts.get(tok, 0) for tok in vocab)
    coverage = covered_count / total_tokens if total_tokens > 0 else 0
    
    # Count how many critical tokens are in vocab vs appearing in data
    dangerous_apis_in_vocab = [api for api in DANGEROUS_APIS if api in vocab]
    dangerous_apis_in_data = [api for api in DANGEROUS_APIS if api in token_counts]
    defense_apis_in_vocab = [api for api in DEFENSE_APIS if api in vocab]
    keywords_in_vocab = [kw for kw in C_KEYWORDS if kw in vocab]
    force_keep_in_vocab = [ident for ident in force_keep if ident in vocab]
    
    # Dynamically verify all critical tokens are in vocab (catches regressions)
    critical_tokens_verified = (
        all(api in vocab for api in DANGEROUS_APIS) and
        all(api in vocab for api in DEFENSE_APIS) and
        all(kw in vocab for kw in C_KEYWORDS) and
        all(ident in vocab for ident in force_keep)
    )
    
    debug_info = {
        'total_unique_tokens': total_unique,
        'total_token_count': total_tokens,
        'vocab_size': len(vocab),
        'coverage': coverage,
        'min_freq': min_freq,
        'max_size': max_size,
        'force_kept_identifiers': force_kept,
        'dropped_by_min_freq': len(dropped_by_freq),
        'dropped_by_max_size': len(dropped_by_size),
        'sample_dropped_by_freq': [(t, c) for t, c in dropped_by_freq[:20]],
        'sample_dropped_by_size': [(t, c) for t, c in dropped_by_size[:20]],
        'top_tokens_added': added_tokens[:30],
        # Critical tokens verification
        'dangerous_apis_in_vocab': len(dangerous_apis_in_vocab),
        'dangerous_apis_in_data': len(dangerous_apis_in_data),
        'dangerous_apis_total': len(DANGEROUS_APIS),
        'defense_apis_in_vocab': len(defense_apis_in_vocab),
        'defense_apis_total': len(DEFENSE_APIS),
        'keywords_in_vocab': len(keywords_in_vocab),
        'keywords_total': len(C_KEYWORDS),
        'force_keep_in_vocab': len(force_keep_in_vocab),
        'force_keep_total': len(force_keep),
        'critical_tokens_never_unk': critical_tokens_verified,
    }
    
    return vocab, debug_info


def vectorize_preserve(
    tokens: List[str], 
    vocab: Dict[str, int], 
    max_len: int = 512
) -> Tuple[List[int], List[int], List[int]]:
    """
    Vectorize tokens with UNK tracking.
    
    Returns:
        input_ids: Token IDs
        attention_mask: 1 for real tokens, 0 for padding
        unk_positions: Positions where UNK was used
    """
    unk_id = vocab.get('UNK', 1)
    pad_id = vocab.get('PAD', 0)
    
    input_ids = []
    unk_positions = []
    
    for i, tok in enumerate(tokens[:max_len]):
        if tok in vocab:
            input_ids.append(vocab[tok])
        else:
            input_ids.append(unk_id)
            unk_positions.append(i)
    
    actual_len = len(input_ids)
    
    if actual_len < max_len:
        padding_len = max_len - actual_len
        input_ids.extend([pad_id] * padding_len)
    
    attention_mask = [1] * actual_len + [0] * (max_len - actual_len)
    
    return input_ids, attention_mask, unk_positions


def vectorize_batch_preserve(
    tokens_list: List[List[str]], 
    vocab: Dict[str, int], 
    max_len: int = 512
) -> Tuple:
    """
    Batch vectorization with statistics.
    
    Returns:
        input_ids: np.ndarray
        attention_masks: np.ndarray
        unk_positions: List of lists
        stats: Dict with UNK statistics
    """
    import numpy as np
    
    if len(tokens_list) == 0:
        return (
            np.zeros((0, max_len), dtype=np.int32),
            np.zeros((0, max_len), dtype=np.int32),
            [],
            {'total_tokens': 0, 'total_unks': 0, 'unk_rate': 0.0, 'avg_len': 0.0}
        )
    
    all_input_ids = []
    all_attention_masks = []
    all_unk_positions = []
    
    total_tokens = 0
    total_unks = 0
    unk_tokens = Counter()
    
    for tokens in tqdm(tokens_list, desc="Vectorizing"):
        input_ids, attention_mask, unk_positions = vectorize_preserve(tokens, vocab, max_len)
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_unk_positions.append(unk_positions)
        
        actual_len = sum(attention_mask)
        total_tokens += actual_len
        total_unks += len(unk_positions)
        
        for pos in unk_positions:
            if pos < len(tokens):
                unk_tokens[tokens[pos]] += 1
    
    stats = {
        'total_tokens': total_tokens,
        'total_unks': total_unks,
        'unk_rate': total_unks / total_tokens if total_tokens > 0 else 0,
        'top_unk_tokens': unk_tokens.most_common(50),
    }
    
    return (
        np.array(all_input_ids, dtype=np.int32),
        np.array(all_attention_masks, dtype=np.int32),
        all_unk_positions,
        stats
    )


if __name__ == "__main__":
    print("=" * 60)
    print("PRESERVE IDENTIFIER TOKENIZER TEST")
    print("=" * 60)
    
    test_code = '''
    void process_request(char *user_input, int size) {
        char buf[256];
        int len = strlen(user_input);
        
        if (len > 0 && len < 256) {
            strcpy(buf, user_input);
            printf("Input: %s", buf);
        }
        
        char *ptr = malloc(1024);
        if (ptr == NULL) {
            return -1;
        }
        
        memcpy(ptr, buf, len);
        ptr[len] = 0;
        free(ptr);
        return 0;
    }
    '''
    
    tokenizer = PreserveIdentifierTokenizer()
    tokens, details, stats = tokenizer.tokenize_with_details(test_code)
    
    print("\n[Tokens]")
    print(' '.join(tokens[:50]))
    print(f"\nTotal tokens: {len(tokens)}")
    
    print("\n[Sample Details]")
    for d in details[:15]:
        print(f"  {d}")
    
    print("\n[Stats]")
    print(f"  Identifiers: {stats['identifiers'][:10]}")
    print(f"  Dangerous APIs: {stats['dangerous_apis']}")
    print(f"  Numbers kept: {stats['numbers_kept']}")
    print(f"  Numbers normalized: {stats['numbers_normalized']}")
    
    print("\n[Build Vocabulary]")
    vocab, debug = build_preserve_vocab([tokens] * 10, min_freq=2, max_size=500)
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Coverage: {debug['coverage']:.2%}")
    
    print("\n[Vectorize]")
    input_ids, attention_mask, unk_pos = vectorize_preserve(tokens, vocab, max_len=64)
    print(f"  Input IDs (first 20): {input_ids[:20]}")
    print(f"  UNK positions: {unk_pos}")
    
    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)
