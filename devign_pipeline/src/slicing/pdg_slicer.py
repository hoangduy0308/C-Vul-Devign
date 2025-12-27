"""PDG-based Code Slicer for vulnerability detection.

Uses Program Dependence Graph (PDG) for accurate slicing instead of
window-based approach. Follows actual data/control dependencies.

Key differences from window-based slicer:
1. Slices based on dependencies, not fixed window
2. Stops when no more dependencies exist
3. Produces shorter, more focused slices (~30-50 tokens)
4. Better for vulnerability pattern detection
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from enum import Enum

import sys
from pathlib import Path
# Add devign_pipeline root to path (relative to this file: src/slicing/pdg_slicer.py)
_PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

# Try to import tree-sitter for accurate parsing (fallback to regex if unavailable)
try:
    import tree_sitter_c as tsc
    from tree_sitter import Language, Parser
    _TREE_SITTER_AVAILABLE = True
except ImportError:
    _TREE_SITTER_AVAILABLE = False

from src.ast.parser import CFamilyParser, ParseResult
from src.graphs.cfg import CFG, build_cfg
from src.graphs.dfg import DFG, build_dfg
from src.graphs.pdg import PDG, PDGBuilder, build_pdg
from src.slicing.utils import DEFENSE_TOKENS

logger = logging.getLogger(__name__)


class PDGSliceType(Enum):
    BACKWARD = 'backward'
    FORWARD = 'forward'
    BIDIRECTIONAL = 'bidirectional'


@dataclass
class PDGSliceConfig:
    """Configuration for PDG-based slicing"""
    slice_type: PDGSliceType = PDGSliceType.BACKWARD
    
    # Dependency settings
    backward_depth: int = 2          # Max hops backward (reduced from 5)
    forward_depth: int = 1           # Max hops forward (reduced from 3)
    include_data_deps: bool = True   # Include data dependencies
    include_control_deps: bool = True # Include control dependencies
    
    # Control dependency mode
    control_predicate_only: bool = False  # Include control block bodies, not just predicates
    control_block_lines: int = 5          # Max lines to include from controlled block
    
    # Defense semantics preservation
    defense_tokens: Set[str] = field(default_factory=lambda: DEFENSE_TOKENS.copy())
    preserve_defense_statements: bool = True  # Always keep lines with defense tokens
    min_slice_tokens: int = 40               # Minimum tokens for a valid slice
    
    # Output limits
    max_lines: int = 15              # Hard cap on output lines
    max_tokens: int = 100            # Hard cap on tokens (estimated)
    
    # Fallback settings
    fallback_window: int = 3         # Fallback window if PDG fails (reduced from 15)
    
    # Criterion control
    max_criteria: int = 3                 # Max number of criteria to use
    criterion_cluster_gap: int = 5        # Lines within gap are same cluster

    # Separator settings  
    insert_separators: bool = True        # Insert [SEP] between non-contiguous segments
    separator_token: str = "[SEP]"        # Token to insert
    separator_gap: int = 2                # Gap > this triggers separator
    
    # Preprocessing
    remove_comments: bool = True
    normalize_output: bool = True
    
    # SEP normalization settings
    normalize_separators: bool = True
    min_tokens_between_sep: int = 6   # Rate-limit: min tokens between SEPs
    max_sep_ratio: float = 0.08       # Max SEP tokens as ratio of total (8%)
    
    # Deduplication settings
    deduplicate_statements: bool = True
    max_duplicate_calls: int = 2  # Keep at most 2 instances of same call pattern
    
    # Guard detection settings
    guard_scan_max_lines: int = 50  # How far back to scan for guards
    guard_body_max_lines: int = 20  # Max lines to search for guard block end (e.g., matching brace)


@dataclass
class PDGSliceResult:
    """Result of PDG-based slicing"""
    code: str                        # Sliced code
    original_code: str               # Original code
    included_lines: Set[int]         # Lines included in slice
    criterion_lines: List[int]       # Original criterion
    slice_type: PDGSliceType
    statements: List[str]            # Individual statements
    data_dep_lines: Set[int]         # Lines from data deps
    control_dep_lines: Set[int]      # Lines from control deps
    used_fallback: bool              # True if PDG failed and used fallback
    
    def __post_init__(self):
        if isinstance(self.included_lines, list):
            self.included_lines = set(self.included_lines)
        if isinstance(self.data_dep_lines, list):
            self.data_dep_lines = set(self.data_dep_lines)
        if isinstance(self.control_dep_lines, list):
            self.control_dep_lines = set(self.control_dep_lines)


class PDGSlicer:
    """PDG-based code slicer"""
    
    def __init__(self, config: PDGSliceConfig = None):
        self.config = config or PDGSliceConfig()
        self.parser = CFamilyParser()
    
    def _is_defense_line(self, line: str) -> bool:
        """Check if a line contains defense tokens (error handling, cleanup, etc.)."""
        line_lower = line.lower()
        for token in self.config.defense_tokens:
            if token.lower() in line_lower:
                return True
        return False
    
    def _normalize_separators(self, code: str, config: PDGSliceConfig) -> str:
        """Normalize SEP tokens to reduce fragmentation.
        
        1. Collapse consecutive SEPs: "SEP SEP SEP" -> "SEP"
        2. Rate-limit SEPs: ensure min_tokens_between_sep tokens between each SEP
        3. Enforce max_sep_ratio
        """
        if not config.normalize_separators:
            return code
        
        sep_token = config.separator_token  # "[SEP]"
        lines = code.split('\n')
        
        # Step 1: Collapse consecutive SEPs
        collapsed = []
        prev_was_sep = False
        for line in lines:
            is_sep = line.strip() == sep_token
            if is_sep:
                if not prev_was_sep:
                    collapsed.append(line)
                prev_was_sep = True
            else:
                collapsed.append(line)
                prev_was_sep = False
        
        # Step 2: Rate-limit - merge segments that are too short
        # Build segments separated by SEP
        segments = []
        current_segment = []
        for line in collapsed:
            if line.strip() == sep_token:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            else:
                current_segment.append(line)
        if current_segment:
            segments.append(current_segment)
        
        # Merge short segments
        merged_segments = []
        pending_segment = []
        pending_tokens = 0
        
        for segment in segments:
            segment_tokens = sum(len(line.split()) for line in segment if line.strip())
            
            if pending_segment:
                # Check if pending + current meets threshold
                combined_tokens = pending_tokens + segment_tokens
                if combined_tokens < config.min_tokens_between_sep:
                    # Merge: join with space instead of SEP
                    pending_segment.extend(segment)
                    pending_tokens = combined_tokens
                else:
                    # Emit pending, start new
                    merged_segments.append(pending_segment)
                    pending_segment = segment
                    pending_tokens = segment_tokens
            else:
                pending_segment = segment
                pending_tokens = segment_tokens
        
        if pending_segment:
            merged_segments.append(pending_segment)
        
        # Step 3: Enforce max ratio by counting and removing excess SEPs
        total_tokens = sum(
            len(line.split()) 
            for seg in merged_segments 
            for line in seg if line.strip()
        )
        num_seps = max(0, len(merged_segments) - 1)
        max_seps = int(total_tokens * config.max_sep_ratio)
        
        if num_seps > max_seps and num_seps > 0:
            # Need to remove some SEPs - merge smallest adjacent segments
            while len(merged_segments) > max_seps + 1 and len(merged_segments) > 1:
                # Find smallest adjacent pair
                min_size = float('inf')
                merge_idx = 0
                for i in range(len(merged_segments) - 1):
                    size = (sum(len(line.split()) for line in merged_segments[i]) +
                            sum(len(line.split()) for line in merged_segments[i + 1]))
                    if size < min_size:
                        min_size = size
                        merge_idx = i
                
                # Merge at merge_idx
                merged_segments[merge_idx].extend(merged_segments[merge_idx + 1])
                merged_segments.pop(merge_idx + 1)
        
        # Rebuild code with SEPs between segments
        result_lines = []
        for i, segment in enumerate(merged_segments):
            if i > 0:
                result_lines.append(sep_token)
            result_lines.extend(segment)
        
        return '\n'.join(result_lines)
    
    def _get_controlled_lines(self, code: str, control_line: int, pdg: PDG) -> Set[int]:
        """Get lines controlled by a control structure (if/while/for body).
        
        Returns the predicate line plus up to control_block_lines lines from the body,
        prioritizing lines with defense tokens.
        """
        lines = code.split('\n')
        max_line = len(lines)
        result = {control_line}
        
        if control_line < 1 or control_line > max_line:
            return result
        
        control_text = lines[control_line - 1]
        
        if not any(kw in control_text for kw in ['if', 'while', 'for', 'switch']):
            return result
        
        block_start = control_line + 1
        block_end = min(max_line, control_line + self.config.control_block_lines + 10)
        
        brace_count = control_text.count('{') - control_text.count('}')
        block_started = brace_count > 0 or '{' not in control_text
        
        defense_lines = []
        other_lines = []
        
        for i in range(block_start, block_end + 1):
            if i > max_line:
                break
            
            line_text = lines[i - 1]
            brace_count += line_text.count('{') - line_text.count('}')
            
            if not block_started and '{' in line_text:
                block_started = True
                continue
            
            if block_started:
                if brace_count <= 0 and '}' in line_text:
                    break
                
                if line_text.strip():
                    if self._is_defense_line(line_text):
                        defense_lines.append(i)
                    else:
                        other_lines.append(i)
        
        for line_num in defense_lines:
            if len(result) < self.config.control_block_lines + 1:
                result.add(line_num)
        
        for line_num in other_lines:
            if len(result) < self.config.control_block_lines + 1:
                result.add(line_num)
        
        return result
    
    def _check_slice_quality(self, lines: List[str], config: PDGSliceConfig) -> bool:
        """Check if slice meets minimum quality requirements."""
        total_tokens = sum(len(line.split()) for line in lines if line.strip())
        has_defense = any(self._is_defense_line(line) for line in lines)
        has_call = any('(' in line for line in lines)
        return total_tokens >= config.min_slice_tokens or (has_defense and has_call)
    
    def _find_defense_lines(self, code: str) -> Set[int]:
        """Find all lines containing defense tokens."""
        lines = code.split('\n')
        defense_lines = set()
        for i, line in enumerate(lines, 1):
            if self._is_defense_line(line):
                defense_lines.add(i)
        return defense_lines
    
    def _find_defense_lines_near(self, code: str, criterion_lines: List[int], radius: int = 10) -> Set[int]:
        """Find defense lines within radius of criterion lines (fallback when guard detection fails).
        
        Only includes return/break/goto/continue statements that are near criterion lines,
        avoiding flooding the slice with unrelated defense statements from elsewhere in the function.
        """
        code_lines = code.split('\n')
        defense_lines = set()
        defense_pattern = re.compile(r'\b(return|goto|break|continue)\b')
        
        for criterion in criterion_lines:
            start = max(1, criterion - radius)
            end = min(len(code_lines), criterion + radius)
            for line_num in range(start, end + 1):
                if line_num <= len(code_lines):
                    if defense_pattern.search(code_lines[line_num - 1]):
                        defense_lines.add(line_num)
        return defense_lines
    
    # === Guard detection helpers ===
    
    # C and C++ keywords/types to filter out when extracting variable names
    _C_KEYWORDS = frozenset({
        # C keywords
        'if', 'else', 'return', 'sizeof', 'int', 'char', 'void', 'NULL',
        'true', 'false', 'while', 'for', 'switch', 'case', 'break', 'continue',
        'goto', 'struct', 'union', 'enum', 'typedef', 'const', 'static',
        'unsigned', 'signed', 'long', 'short', 'double', 'float', 'extern',
        'volatile', 'register', 'auto', 'inline', 'restrict', 'bool',
        'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'int8_t', 'int16_t',
        'int32_t', 'int64_t', 'size_t', 'ssize_t', 'ptrdiff_t', 'uintptr_t',
        'assert', 'exit', 'abort', 'default', 'do',
        # C++ keywords (Devign dataset may contain C++ code)
        'class', 'template', 'typename', 'try', 'catch', 'throw', 'namespace',
        'using', 'new', 'delete', 'public', 'private', 'protected', 'virtual',
        'override', 'final', 'explicit', 'friend', 'operator', 'this',
        'nullptr', 'constexpr', 'decltype', 'noexcept', 'static_cast',
        'dynamic_cast', 'const_cast', 'reinterpret_cast', 'mutable',
        # Common macros/functions
        'define', 'ifdef', 'ifndef', 'endif', 'include', 'pragma',
        'printf', 'fprintf', 'sprintf', 'snprintf', 'malloc', 'calloc',
        'realloc', 'free', 'memcpy', 'memset', 'memmove', 'strlen', 'strcpy',
        'strncpy', 'strcmp', 'strncmp', 'strcat', 'strncat'
    })
    
    # Tree-sitter parser (initialized lazily)
    _ts_parser: Optional['Parser'] = None
    
    @classmethod
    def _get_ts_parser(cls) -> Optional['Parser']:
        """Get or create tree-sitter parser for C."""
        if not _TREE_SITTER_AVAILABLE:
            return None
        if cls._ts_parser is None:
            try:
                cls._ts_parser = Parser(Language(tsc.language()))
            except Exception:
                return None
        return cls._ts_parser
    
    def _extract_vars_with_tree_sitter(self, line_text: str) -> Optional[Set[str]]:
        """Extract variables using tree-sitter (more accurate than regex).
        
        Returns None if parsing fails, caller should fallback to regex.
        """
        parser = self._get_ts_parser()
        if parser is None:
            return None
        
        # Preprocessor directives (#define, #include, #pragma, etc.) cannot be
        # wrapped in a function body - skip tree-sitter for these lines
        stripped = line_text.strip()
        if stripped.startswith('#'):
            logger.debug(f"Preprocessor directive detected, skipping tree-sitter: {stripped[:50]}")
            return None
        
        try:
            # Wrap in a function to make it parseable
            # Note: This may fail for snippets containing function definitions or complex macros
            wrapper_prefix = "void __wrapper__() { "
            wrapped = f"{wrapper_prefix}{line_text} }}"
            tree = parser.parse(bytes(wrapped, 'utf8'))
            
            if tree.root_node.has_error:
                # Log at debug level to help tune wrapper strategy if needed
                logger.debug(
                    f"Tree-sitter parse error for line (falling back to regex): "
                    f"{line_text[:80]}{'...' if len(line_text) > 80 else ''}"
                )
                return None  # Parse error, fallback to regex
            
            vars_set: Set[str] = set()
            # Track positions to exclude wrapper function identifiers
            wrapper_end_byte = len(wrapper_prefix.encode('utf8'))
            
            def extract_identifiers(node):
                """Recursively extract identifier nodes."""
                # Skip nodes that are part of the wrapper (before our actual code)
                if node.end_byte <= wrapper_end_byte:
                    return
                
                # identifier nodes are variable references
                if node.type == 'identifier':
                    name = node.text.decode('utf8')
                    # Exclude wrapper function name and keywords
                    if name not in self._C_KEYWORDS and name != '__wrapper__':
                        vars_set.add(name)
                # For pointer/array access, get the base
                elif node.type in ('field_expression', 'subscript_expression', 
                                   'pointer_expression', 'call_expression'):
                    # Get the leftmost identifier (base variable)
                    for child in node.children:
                        if child.type == 'identifier':
                            name = child.text.decode('utf8')
                            if name not in self._C_KEYWORDS and name != '__wrapper__':
                                vars_set.add(name)
                            break
                        elif child.type in ('field_expression', 'subscript_expression',
                                            'pointer_expression'):
                            extract_identifiers(child)
                            break
                
                for child in node.children:
                    extract_identifiers(child)
            
            extract_identifiers(tree.root_node)
            return vars_set
            
        except Exception as e:
            logger.debug(
                f"Tree-sitter exception for line (falling back to regex): "
                f"{line_text[:60]}... - {type(e).__name__}: {e}"
            )
            return None  # Any error, fallback to regex
    
    def _extract_vars_from_line_regex(self, line_text: str) -> Set[str]:
        """Extract variables using regex (fallback method)."""
        # Strip comments and string literals first
        clean_text = self._strip_string_literals(line_text)
        # Remove single-line comments
        clean_text = re.sub(r'//.*$', '', clean_text)
        
        # Extract all identifier-like tokens
        all_matches = re.findall(r'\b[A-Za-z_]\w*\b', clean_text)
        
        # Filter out C keywords/types
        vars_set = {m for m in all_matches if m not in self._C_KEYWORDS}
        
        # Also extract base from pointer/array patterns
        # p->member, p.member → extract p
        arrow_dot = re.findall(r'(\b[A-Za-z_]\w*)\s*(?:->|\.)', clean_text)
        vars_set.update(v for v in arrow_dot if v not in self._C_KEYWORDS)
        
        # p[expr] → extract p and vars in expr
        bracket_matches = re.findall(r'(\b[A-Za-z_]\w*)\s*\[', clean_text)
        vars_set.update(v for v in bracket_matches if v not in self._C_KEYWORDS)
        
        # *p, &p → extract p
        deref_addr = re.findall(r'[*&]\s*(\b[A-Za-z_]\w*)\b', clean_text)
        vars_set.update(v for v in deref_addr if v not in self._C_KEYWORDS)
        
        return vars_set
    
    def _extract_vars_from_line(self, line_text: str) -> Set[str]:
        """Extract variable identifiers from a line.
        
        Uses tree-sitter for accurate parsing when available,
        falls back to regex-based extraction otherwise.
        
        Handles patterns like p->x → p, p[i] → p,i, *p → p
        """
        # Try tree-sitter first (more accurate)
        ts_result = self._extract_vars_with_tree_sitter(line_text)
        if ts_result is not None:
            return ts_result
        
        # Fallback to regex
        return self._extract_vars_from_line_regex(line_text)
    
    def _is_guard_predicate(self, line_text: str, vars_of_interest: Set[str]) -> bool:
        """Return True if line is a guard checking vars_of_interest.
        
        Uses tree-sitter for accurate condition parsing when available,
        falls back to regex patterns otherwise.
        
        Patterns detected:
        - Null checks: if (!p), if (p == NULL), if (p == 0), if (NULL == p)
        - Bounds checks: if (n > max), if (n >= max), if (idx >= len), if (n < 0)
        - Assertions: assert(p != NULL), assert(n <= max)
        """
        stripped = line_text.strip()
        
        # Must be an if or assert
        if not (stripped.startswith('if') or stripped.startswith('assert')):
            return False
        
        # Try tree-sitter first for accurate parsing
        result = self._is_guard_predicate_tree_sitter(line_text, vars_of_interest)
        if result is not None:
            return result
        
        # Fallback to regex-based detection
        return self._is_guard_predicate_regex(line_text, vars_of_interest)
    
    def _is_guard_predicate_tree_sitter(self, line_text: str, vars_of_interest: Set[str]) -> Optional[bool]:
        """Use tree-sitter to analyze guard predicate. Returns None if parsing fails."""
        parser = self._get_ts_parser()
        if parser is None:
            return None
        
        # Skip preprocessor directives
        if line_text.strip().startswith('#'):
            return None
        
        try:
            wrapper_prefix = "void __wrapper__() { "
            wrapped = f"{wrapper_prefix}{line_text} }}"
            tree = parser.parse(bytes(wrapped, 'utf8'))
            
            if tree.root_node.has_error:
                return None
            
            # Find if_statement or expression_statement (for assert)
            def find_condition_node(node):
                """Find the condition expression in if/assert statements."""
                if node.type == 'if_statement':
                    # The condition is the parenthesized expression
                    for child in node.children:
                        if child.type == 'parenthesized_expression':
                            return child
                elif node.type == 'expression_statement':
                    # Look for assert call
                    for child in node.children:
                        if child.type == 'call_expression':
                            func_node = child.child_by_field_name('function')
                            if func_node and func_node.text.decode('utf8') == 'assert':
                                args = child.child_by_field_name('arguments')
                                if args:
                                    return args
                
                for child in node.children:
                    result = find_condition_node(child)
                    if result:
                        return result
                return None
            
            cond_node = find_condition_node(tree.root_node)
            if not cond_node:
                return None
            
            # Check if condition contains guard patterns
            def has_guard_pattern(node) -> bool:
                """Check if node represents a guard pattern."""
                if node.type == 'unary_expression':
                    # !p pattern
                    op = node.child_by_field_name('operator')
                    if op and op.text.decode('utf8') == '!':
                        return True
                        
                elif node.type == 'binary_expression':
                    op = node.child_by_field_name('operator')
                    if op:
                        op_text = op.text.decode('utf8')
                        left = node.child_by_field_name('left')
                        right = node.child_by_field_name('right')
                        
                        # Null checks: == NULL, != NULL, == 0
                        if op_text in ('==', '!='):
                            left_text = left.text.decode('utf8') if left else ''
                            right_text = right.text.decode('utf8') if right else ''
                            if 'NULL' in (left_text, right_text) or '0' in (left_text, right_text):
                                return True
                        
                        # Bounds checks: <, >, <=, >=
                        if op_text in ('<', '>', '<=', '>='):
                            return True
                
                # Recurse for compound conditions (&&, ||)
                for child in node.children:
                    if has_guard_pattern(child):
                        return True
                return False
            
            # Check if condition mentions vars of interest and has guard pattern
            cond_vars = self._extract_vars_from_line(cond_node.text.decode('utf8'))
            if not (cond_vars & vars_of_interest):
                return False
            
            return has_guard_pattern(cond_node)
            
        except Exception as e:
            logger.debug(f"Tree-sitter guard detection failed: {type(e).__name__}: {e}")
            return None
    
    def _is_guard_predicate_regex(self, line_text: str, vars_of_interest: Set[str]) -> bool:
        """Fallback regex-based guard predicate detection."""
        # Extract the condition part
        paren_match = re.search(r'\(\s*(.+?)\s*\)', line_text)
        if not paren_match:
            return False
        
        condition = paren_match.group(1)
        
        # Extract vars mentioned in condition
        cond_vars = self._extract_vars_from_line(condition)
        
        # Check if any var of interest is in the condition
        if not (cond_vars & vars_of_interest):
            return False
        
        # Check for guard patterns
        # Null checks: !p, p == NULL, p == 0, NULL == p, p != NULL, !p
        null_patterns = [
            r'!\s*\w+',                      # !p
            r'\w+\s*==\s*NULL',              # p == NULL
            r'NULL\s*==\s*\w+',              # NULL == p
            r'\w+\s*==\s*0\b',               # p == 0
            r'0\s*==\s*\w+',                 # 0 == p
            r'\w+\s*!=\s*NULL',              # p != NULL (assertion style)
            r'NULL\s*!=\s*\w+',              # NULL != p
        ]
        
        # Bounds checks: n > max, n >= max, n < 0, idx >= len
        bounds_patterns = [
            r'\w+\s*>\s*\w+',                # n > max
            r'\w+\s*>=\s*\w+',               # n >= max
            r'\w+\s*<\s*\w+',                # n < max
            r'\w+\s*<=\s*\w+',               # n <= max
            r'\w+\s*<\s*0\b',                # n < 0
            r'\w+\s*>=\s*0\b',               # n >= 0
        ]
        
        all_patterns = null_patterns + bounds_patterns
        
        for pattern in all_patterns:
            if re.search(pattern, condition):
                return True
        
        return False
    
    def _strip_string_literals(self, line_text: str) -> str:
        """Remove string literals from line to avoid counting braces inside strings.
        
        Handles: "string with { braces }" and 'char literals'
        """
        # Remove string literals (handles escaped quotes)
        result = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line_text)
        # Remove char literals
        result = re.sub(r"'(?:[^'\\]|\\.)*'", "''", result)
        return result
    
    def _count_braces(self, line_text: str) -> int:
        """Count net braces (open - close) excluding those in string literals."""
        stripped = self._strip_string_literals(line_text)
        return stripped.count('{') - stripped.count('}')
    
    def _extract_guard_body_lines(self, code_lines: List[str], predicate_line: int) -> Set[int]:
        """From an if-line, find the consequence lines (early-exit actions).
        
        Handles:
        - Single-line: if (!p) return;
        - Braced blocks: if (!p) { ... }
        - Unbraced single statement: if (!p)\n    return;
        
        Returns set of line numbers (1-indexed).
        """
        result: Set[int] = set()
        if predicate_line < 1 or predicate_line > len(code_lines):
            return result
        
        line_text = code_lines[predicate_line - 1]
        
        # Check for single-line form: if (!p) return;
        # Look for statement after the closing paren
        after_paren = re.search(r'\)\s*(.+?)\s*;?\s*$', line_text)
        if after_paren:
            rest = after_paren.group(1).strip()
            if rest and not rest.startswith('{') and rest != '{':
                # Single-line form
                result.add(predicate_line)
                return result
        
        # Check for brace on same line or next line (strip string literals first)
        stripped_line = self._strip_string_literals(line_text)
        has_open_brace = '{' in stripped_line
        
        if has_open_brace:
            # Find matching close brace (using _count_braces to handle string literals)
            brace_count = self._count_braces(line_text)
            max_search = self.config.guard_body_max_lines
            for i in range(predicate_line, min(len(code_lines), predicate_line + max_search)):
                if i > predicate_line:
                    brace_count += self._count_braces(code_lines[i - 1])
                if brace_count <= 0:
                    # Include lines from predicate+1 to i
                    for ln in range(predicate_line + 1, i + 1):
                        result.add(ln)
                    break
                elif i > predicate_line:
                    result.add(i)
        else:
            # Unbraced - next non-empty line is the body
            for i in range(predicate_line + 1, min(len(code_lines) + 1, predicate_line + 5)):
                if i <= len(code_lines) and code_lines[i - 1].strip():
                    result.add(i)
                    break
        
        return result
    
    def _is_early_exit_line(self, line_text: str) -> bool:
        """True if line contains early-exit statement."""
        stripped = line_text.strip()
        early_exit_patterns = [
            r'\breturn\b',
            r'\bgoto\b',
            r'\bbreak\b',
            r'\bcontinue\b',
            r'\bexit\s*\(',
            r'\babort\s*\(',
        ]
        for pattern in early_exit_patterns:
            if re.search(pattern, stripped):
                return True
        return False
    
    def _find_guard_lines_for_criteria(self, code: str, criterion_lines: List[int],
                                        pdg: Optional[PDG] = None) -> Tuple[Set[int], Set[int]]:
        """Find guard lines (null/bounds checks) that protect criterion lines.
        
        For each criterion line:
        - Extract variables used in that line
        - Scan backward up to guard_scan_max_lines
        - Find if-statements that check those variables AND have early-exit consequence
        
        Returns: (guard_lines, guard_defense_lines)
            guard_lines: the if-predicate lines
            guard_defense_lines: the return/goto/break lines inside guards
        """
        guard_lines: Set[int] = set()
        guard_defense_lines: Set[int] = set()
        
        code_lines = code.split('\n')
        max_line = len(code_lines)
        
        for crit_line in criterion_lines:
            if crit_line < 1 or crit_line > max_line:
                continue
            
            # Extract variables from criterion line
            crit_text = code_lines[crit_line - 1]
            vars_of_interest = self._extract_vars_from_line(crit_text)
            
            if not vars_of_interest:
                continue
            
            # Scan backward for guards
            scan_start = max(1, crit_line - self.config.guard_scan_max_lines)
            
            for line_num in range(crit_line - 1, scan_start - 1, -1):
                line_text = code_lines[line_num - 1]
                
                if self._is_guard_predicate(line_text, vars_of_interest):
                    # Check if the body has early exit
                    body_lines = self._extract_guard_body_lines(code_lines, line_num)
                    
                    has_early_exit = False
                    for body_ln in body_lines:
                        if body_ln <= max_line and self._is_early_exit_line(code_lines[body_ln - 1]):
                            has_early_exit = True
                            guard_defense_lines.add(body_ln)
                    
                    # Also check if predicate line itself has early exit (single-line form)
                    if self._is_early_exit_line(line_text):
                        has_early_exit = True
                    
                    if has_early_exit:
                        guard_lines.add(line_num)
                        guard_defense_lines.update(body_lines)
        
        return guard_lines, guard_defense_lines
    
    def _expand_slice_for_quality(self, code: str, included_lines: Set[int], 
                                   criterion_lines: List[int]) -> Set[int]:
        """Expand slice if it doesn't meet quality requirements."""
        lines = code.split('\n')
        current_statements = [lines[l - 1] for l in included_lines if 1 <= l <= len(lines)]
        
        if self._check_slice_quality(current_statements, self.config):
            return included_lines
        
        expanded = set(included_lines)
        
        if self.config.preserve_defense_statements:
            # Only include defense lines near criterion (not all defense lines)
            nearby_defense = self._find_defense_lines_near(code, criterion_lines, radius=8)
            for dl in nearby_defense:
                if len(expanded) < self.config.max_lines:
                    expanded.add(dl)
        
        current_statements = [lines[l - 1] for l in expanded if 1 <= l <= len(lines)]
        if self._check_slice_quality(current_statements, self.config):
            return expanded
        
        criterion_set = set(criterion_lines)
        expansion_radius = 1
        max_radius = 5
        
        while not self._check_slice_quality(current_statements, self.config) and expansion_radius <= max_radius:
            for crit in criterion_lines:
                for offset in range(-expansion_radius, expansion_radius + 1):
                    new_line = crit + offset
                    if 1 <= new_line <= len(lines) and new_line not in expanded:
                        if len(expanded) < self.config.max_lines:
                            expanded.add(new_line)
            
            current_statements = [lines[l - 1] for l in expanded if 1 <= l <= len(lines)]
            expansion_radius += 1
        
        return expanded
    
    def _cluster_and_limit_criteria(self, criterion_lines: List[int]) -> List[int]:
        """Cluster nearby criteria and limit to max_criteria."""
        if not criterion_lines:
            return criterion_lines
        
        # Sort and deduplicate
        criteria = sorted(set(criterion_lines))
        
        if len(criteria) <= self.config.max_criteria:
            return criteria
        
        # Cluster criteria by gap
        clusters = []
        current_cluster = [criteria[0]]
        
        for line in criteria[1:]:
            if line - current_cluster[-1] <= self.config.criterion_cluster_gap:
                current_cluster.append(line)
            else:
                clusters.append(current_cluster)
                current_cluster = [line]
        clusters.append(current_cluster)
        
        # Score clusters by size (more criteria = more important)
        clusters.sort(key=len, reverse=True)
        
        # Take representatives from top clusters
        result = []
        for cluster in clusters:
            if len(result) >= self.config.max_criteria:
                break
            # Take median of cluster as representative
            median_idx = len(cluster) // 2
            result.append(cluster[median_idx])
        
        return sorted(result)
    
    def slice(self, code: str, criterion_lines: List[int],
              pdg: Optional[PDG] = None) -> PDGSliceResult:
        """
        Slice code using PDG-based approach.
        
        Args:
            code: Source code to slice
            criterion_lines: Lines to use as slicing criterion
            pdg: Optional precomputed PDG
            
        Returns:
            PDGSliceResult with sliced code
        """
        if not code or not code.strip():
            return self._empty_result(code, criterion_lines)
        
        # Validate criterion lines
        lines = code.split('\n')
        max_line = len(lines)
        valid_criterion = [l for l in criterion_lines if 1 <= l <= max_line]
        
        if not valid_criterion:
            valid_criterion = [1]
        
        # Cluster and limit criteria to avoid scattered slices
        criterion_lines = self._cluster_and_limit_criteria(valid_criterion)
        
        # Remove comments if configured
        processed_code = code
        if self.config.remove_comments:
            processed_code = self._remove_comments(code)
        
        # Try PDG-based slicing
        try:
            if pdg is None or pdg.is_empty():
                parse_result = self.parser.parse_with_fallback(processed_code)
                if parse_result is None or not parse_result.nodes:
                    return self._fallback_slice(processed_code, valid_criterion)
                
                pdg = build_pdg(parse_result)
            
            if pdg is None or pdg.is_empty():
                return self._fallback_slice(processed_code, valid_criterion)
            
            # Perform slicing based on type
            if self.config.slice_type == PDGSliceType.BACKWARD:
                return self._backward_slice(processed_code, valid_criterion, pdg)
            elif self.config.slice_type == PDGSliceType.FORWARD:
                return self._forward_slice(processed_code, valid_criterion, pdg)
            else:  # BIDIRECTIONAL
                return self._bidirectional_slice(processed_code, valid_criterion, pdg)
        
        except Exception:
            return self._fallback_slice(processed_code, valid_criterion)
    
    def _backward_slice(self, code: str, criterion_lines: List[int],
                        pdg: PDG) -> PDGSliceResult:
        """Backward slicing using PDG"""
        # Get data dependencies
        data_lines: Set[int] = set()
        if self.config.include_data_deps:
            data_lines = pdg.backward_slice(
                criterion_lines,
                max_depth=self.config.backward_depth,
                include_data=True,
                include_control=False
            )
        
        # Get control dependencies
        control_lines: Set[int] = set()
        if self.config.include_control_deps:
            raw_control_lines = pdg.backward_slice(
                criterion_lines,
                max_depth=self.config.backward_depth,
                include_data=False,
                include_control=True
            )
            
            # If predicate-only mode, filter to only condition lines
            if self.config.control_predicate_only:
                control_lines = {
                    line for line in raw_control_lines
                    if line in pdg.nodes and 
                    pdg.nodes[line].node_type == 'condition'
                }
            else:
                # Include control block bodies, not just predicates
                for line in raw_control_lines:
                    if line in pdg.nodes and pdg.nodes[line].node_type == 'condition':
                        controlled = self._get_controlled_lines(code, line, pdg)
                        control_lines.update(controlled)
                    else:
                        control_lines.add(line)
        
        # Combine lines
        all_lines = data_lines | control_lines | set(criterion_lines)
        
        # Find guards for criterion variables
        guard_lines, guard_defense_lines = self._find_guard_lines_for_criteria(code, criterion_lines, pdg)
        all_lines.update(guard_lines)
        all_lines.update(guard_defense_lines)
        
        # Add defense lines if configured - only near criterion, not all
        if self.config.preserve_defense_statements:
            nearby_defense = self._find_defense_lines_near(code, criterion_lines, radius=10)
            all_lines.update(nearby_defense)
        
        # Apply limits with guard priority
        limited_lines = self._apply_limits(code, all_lines, criterion_lines, guard_lines)
        
        # Expand if quality check fails
        limited_lines = self._expand_slice_for_quality(code, limited_lines, criterion_lines)
        
        return self._build_result(
            code, limited_lines, criterion_lines,
            PDGSliceType.BACKWARD, data_lines, control_lines, False
        )
    
    def _forward_slice(self, code: str, criterion_lines: List[int],
                       pdg: PDG) -> PDGSliceResult:
        """Forward slicing using PDG"""
        # For forward, usually only data deps matter
        data_lines: Set[int] = set()
        if self.config.include_data_deps:
            data_lines = pdg.forward_slice(
                criterion_lines,
                max_depth=self.config.forward_depth,
                include_data=True,
                include_control=False
            )
        
        control_lines: Set[int] = set()
        if self.config.include_control_deps and not self.config.control_predicate_only:
            control_lines = pdg.forward_slice(
                criterion_lines,
                max_depth=self.config.forward_depth,
                include_data=False,
                include_control=True
            )
        
        all_lines = data_lines | control_lines | set(criterion_lines)
        
        # Find guards for criterion variables
        guard_lines, guard_defense_lines = self._find_guard_lines_for_criteria(code, criterion_lines, pdg)
        all_lines.update(guard_lines)
        all_lines.update(guard_defense_lines)
        
        # Add defense lines if configured - only near criterion, not all
        if self.config.preserve_defense_statements:
            nearby_defense = self._find_defense_lines_near(code, criterion_lines, radius=10)
            all_lines.update(nearby_defense)
        
        limited_lines = self._apply_limits(code, all_lines, criterion_lines, guard_lines)
        
        # Expand if quality check fails
        limited_lines = self._expand_slice_for_quality(code, limited_lines, criterion_lines)
        
        return self._build_result(
            code, limited_lines, criterion_lines,
            PDGSliceType.FORWARD, data_lines, control_lines, False
        )
    
    def _bidirectional_slice(self, code: str, criterion_lines: List[int],
                             pdg: PDG) -> PDGSliceResult:
        """Bidirectional slicing (backward + forward)"""
        # Backward slice
        backward_data = pdg.backward_slice(
            criterion_lines,
            max_depth=self.config.backward_depth,
            include_data=self.config.include_data_deps,
            include_control=False
        )
        
        backward_control: Set[int] = set()
        if self.config.include_control_deps:
            raw_backward_control = pdg.backward_slice(
                criterion_lines,
                max_depth=self.config.backward_depth,
                include_data=False,
                include_control=True
            )
            if self.config.control_predicate_only:
                backward_control = {
                    line for line in raw_backward_control
                    if line in pdg.nodes and 
                    pdg.nodes[line].node_type == 'condition'
                }
            else:
                # Include control block bodies, not just predicates
                for line in raw_backward_control:
                    if line in pdg.nodes and pdg.nodes[line].node_type == 'condition':
                        controlled = self._get_controlled_lines(code, line, pdg)
                        backward_control.update(controlled)
                    else:
                        backward_control.add(line)
        
        # Forward slice (usually shorter)
        forward_data = pdg.forward_slice(
            criterion_lines,
            max_depth=self.config.forward_depth,
            include_data=self.config.include_data_deps,
            include_control=False
        )
        
        # Combine
        all_data = backward_data | forward_data
        all_control = backward_control
        all_lines = all_data | all_control | set(criterion_lines)
        
        # Find guards for criterion variables
        guard_lines, guard_defense_lines = self._find_guard_lines_for_criteria(code, criterion_lines, pdg)
        all_lines.update(guard_lines)
        all_lines.update(guard_defense_lines)
        
        # Add defense lines if configured - only near criterion, not all
        if self.config.preserve_defense_statements:
            nearby_defense = self._find_defense_lines_near(code, criterion_lines, radius=10)
            all_lines.update(nearby_defense)
        
        limited_lines = self._apply_limits(code, all_lines, criterion_lines, guard_lines)
        
        # Expand if quality check fails
        limited_lines = self._expand_slice_for_quality(code, limited_lines, criterion_lines)
        
        return self._build_result(
            code, limited_lines, criterion_lines,
            PDGSliceType.BIDIRECTIONAL, all_data, all_control, False
        )
    
    def _fallback_slice(self, code: str, criterion_lines: List[int]) -> PDGSliceResult:
        """Fallback to simple window-based slice when PDG fails"""
        lines = code.split('\n')
        max_line = len(lines)
        
        included: Set[int] = set()
        window = self.config.fallback_window
        
        for crit_line in criterion_lines:
            start = max(1, crit_line - window)
            end = min(max_line, crit_line + window)
            included.update(range(start, end + 1))
        
        # Apply limits to fallback slice too
        included = self._apply_limits(code, included, criterion_lines)
        
        return self._build_result(
            code, included, criterion_lines,
            self.config.slice_type, set(), set(), True
        )
    
    def _apply_limits(self, code: str, lines: Set[int],
                      criterion_lines: List[int],
                      guard_lines: Optional[Set[int]] = None) -> Set[int]:
        """Apply max_lines and max_tokens limits.
        
        Priority order for keeping lines when budget is tight:
        1. Criterion lines (always kept)
        2. Guard lines (security-critical checks)
        3. Other lines (by distance from criterion)
        """
        if not lines:
            return lines
        
        code_lines = code.split('\n')
        criterion_set = set(criterion_lines)
        guard_set = guard_lines or set()
        
        # Sort by priority: criterion (0) > guard (1) > other (2), then by distance
        def line_priority(line: int) -> Tuple[int, int]:
            if line in criterion_set:
                priority = 0
            elif line in guard_set:
                priority = 1
            else:
                priority = 2
            distance = min(abs(line - c) for c in criterion_lines) if criterion_lines else 0
            return (priority, distance)
        
        sorted_lines = sorted(lines, key=line_priority)
        
        # Greedy add lines respecting both max_lines and max_tokens
        result: Set[int] = set()
        total_tokens = 0
        
        # Always include criterion lines first (priority 0)
        for line in sorted_lines:
            if line in criterion_set and 1 <= line <= len(code_lines):
                line_text = code_lines[line - 1]
                line_tokens = len(line_text.split())
                result.add(line)
                total_tokens += line_tokens
        
        # Add guard lines next (priority 1), respecting token budget
        for line in sorted_lines:
            if line in result or line not in guard_set:
                continue
            if len(result) >= self.config.max_lines:
                break
            if 1 <= line <= len(code_lines):
                line_text = code_lines[line - 1]
                line_tokens = len(line_text.split())
                
                if total_tokens + line_tokens > self.config.max_tokens:
                    continue
                
                result.add(line)
                total_tokens += line_tokens
        
        # Add remaining lines by distance (priority 2)
        for line in sorted_lines:
            if line in result:
                continue
            if len(result) >= self.config.max_lines:
                break
            if 1 <= line <= len(code_lines):
                line_text = code_lines[line - 1]
                line_tokens = len(line_text.split())
                
                # Check token budget
                if total_tokens + line_tokens > self.config.max_tokens:
                    continue  # Skip this line, try next
                
                result.add(line)
                total_tokens += line_tokens
        
        return result
    
    def _build_result(self, code: str, included_lines: Set[int],
                      criterion_lines: List[int], slice_type: PDGSliceType,
                      data_lines: Set[int], control_lines: Set[int],
                      used_fallback: bool) -> PDGSliceResult:
        """Build PDGSliceResult from included lines"""
        if not included_lines:
            return self._empty_result(code, criterion_lines)
        
        lines = code.split('\n')
        max_line = len(lines)
        
        # Filter valid lines
        valid_lines = {l for l in included_lines if 1 <= l <= max_line}
        
        if not valid_lines:
            return self._empty_result(code, criterion_lines)
        
        # Extract statements with separator insertion
        statements = []
        prev_line_num = None

        for line_num in sorted(valid_lines):
            line = lines[line_num - 1]
            if not line.strip():
                continue
            
            # Insert separator if gap is large enough
            if (self.config.insert_separators and 
                prev_line_num is not None and 
                line_num - prev_line_num > self.config.separator_gap):
                statements.append(self.config.separator_token)
            
            statements.append(line)
            prev_line_num = line_num
        
        # Build code
        if self.config.normalize_output:
            sliced_code = self._normalize_code(statements)
        else:
            sliced_code = '\n'.join(statements)
        
        # Normalize SEP tokens to reduce fragmentation
        sliced_code = self._normalize_separators(sliced_code, self.config)
        
        # Apply deduplication after SEP normalization
        sliced_code = self._deduplicate_statements(sliced_code, self.config)
        
        return PDGSliceResult(
            code=sliced_code,
            original_code=code,
            included_lines=valid_lines,
            criterion_lines=criterion_lines,
            slice_type=slice_type,
            statements=statements,
            data_dep_lines=data_lines & valid_lines,
            control_dep_lines=control_lines & valid_lines,
            used_fallback=used_fallback
        )
    
    def _normalize_code(self, statements: List[str]) -> str:
        """Normalize indentation"""
        if not statements:
            return ""
        
        # Find minimum indentation
        min_indent = float('inf')
        for stmt in statements:
            if stmt.strip():
                indent = len(stmt) - len(stmt.lstrip())
                min_indent = min(min_indent, indent)
        
        if min_indent == float('inf'):
            min_indent = 0
        
        # Dedent
        result = []
        for stmt in statements:
            if len(stmt) >= min_indent:
                result.append(stmt[int(min_indent):])
            else:
                result.append(stmt.lstrip())
        
        return '\n'.join(result)
    
    def _extract_call_pattern(self, segment: str) -> Optional[str]:
        """Extract function call pattern from a code segment.
        
        Examples:
        - "fprintf ( stderr , STR )" -> "fprintf_stderr"
        - "memcpy ( dst , src , len )" -> "memcpy_dst"
        - "free ( ptr )" -> "free_ptr"
        - "if ( x > 0 )" -> None (not a call)
        """
        match = re.match(r'^\s*(\w+)\s*\(\s*(\w+)?', segment)
        if match:
            func_name = match.group(1)
            first_arg = match.group(2) or ''
            if func_name in {'if', 'while', 'for', 'switch', 'return'}:
                return None
            return f"{func_name}_{first_arg}"
        return None
    
    def _deduplicate_statements(self, code: str, config: PDGSliceConfig) -> str:
        """Remove duplicate statement patterns, keeping at most max_duplicate_calls instances.
        
        A "pattern" is the function name + first arg, e.g.:
        - "fprintf ( stderr" 
        - "memcpy ( dst"
        - "free ( ptr"
        """
        if not config.deduplicate_statements:
            return code
        
        sep_token = config.separator_token
        segments = code.split(sep_token)
        
        pattern_counts = {}
        result_segments = []
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            pattern = self._extract_call_pattern(segment)
            
            if pattern is None:
                result_segments.append(segment)
            elif pattern_counts.get(pattern, 0) < config.max_duplicate_calls:
                result_segments.append(segment)
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        return f' {sep_token} '.join(result_segments)
    
    def _remove_comments(self, code: str) -> str:
        """Remove C/C++ comments while preserving line numbers"""
        import re
        
        # Replace block comments with spaces (preserve newlines)
        def replace_block(match):
            return re.sub(r'[^\n]', ' ', match.group(0))
        
        result = re.sub(r'/\*.*?\*/', replace_block, code, flags=re.DOTALL)
        
        # Replace line comments
        result = re.sub(r'//[^\n]*', lambda m: ' ' * len(m.group(0)), result)
        
        return result
    
    def _empty_result(self, code: str, criterion_lines: List[int]) -> PDGSliceResult:
        """Return empty result"""
        return PDGSliceResult(
            code="",
            original_code=code,
            included_lines=set(),
            criterion_lines=criterion_lines,
            slice_type=self.config.slice_type,
            statements=[],
            data_dep_lines=set(),
            control_dep_lines=set(),
            used_fallback=True
        )


def pdg_slice(code: str, criterion_lines: List[int],
              config: PDGSliceConfig = None) -> PDGSliceResult:
    """Convenience function for PDG-based slicing"""
    slicer = PDGSlicer(config)
    return slicer.slice(code, criterion_lines)


if __name__ == "__main__":
    # Test PDG slicer vs window slicer
    test_code = '''static void scsi_read_data(SCSIDiskReq *r) {
    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);
    uint32_t n;
    
    if (r->sector_count == (uint32_t)-1) {
        r->sector_count = 0;
        scsi_req_data(&r->req, r->iov.iov_len);
        return;
    }
    
    if (r->sector_count == 0) {
        scsi_command_complete(r, GOOD, NO_SENSE);
        // BUG: missing return here!
    }
    
    assert(r->req.aiocb == NULL);
    n = r->sector_count;
    if (n > SCSI_DMA_BUF_SIZE / 512)
        n = SCSI_DMA_BUF_SIZE / 512;
    
    r->iov.iov_len = n * 512;
    r->req.aiocb = bdrv_aio_readv(s->bs, r->sector, &r->qiov, n, scsi_read_complete, r);
}'''
    
    print("=== PDG Slicer Test ===\n")
    print("Original code:")
    print(test_code)
    print(f"\nTotal lines: {len(test_code.split(chr(10)))}")
    
    # Test with criterion at the bug location (missing return)
    criterion = [12]  # Line with scsi_command_complete
    
    print(f"\n=== Slicing from criterion line {criterion} ===")
    
    # PDG-based slice
    config = PDGSliceConfig(
        slice_type=PDGSliceType.BIDIRECTIONAL,
        backward_depth=2,
        forward_depth=1,
        control_predicate_only=True,
        max_lines=10
    )
    
    result = pdg_slice(test_code, criterion, config)
    
    print(f"\nPDG Slice:")
    print(f"  Lines: {sorted(result.included_lines)}")
    print(f"  Count: {len(result.included_lines)} lines")
    print(f"  Data deps: {sorted(result.data_dep_lines)}")
    print(f"  Control deps: {sorted(result.control_dep_lines)}")
    print(f"  Used fallback: {result.used_fallback}")
    print(f"\nSliced code:\n{result.code}")
    
    # Compare with window-based
    print("\n=== Comparison ===")
    print(f"PDG slice: {len(result.included_lines)} lines")
    print(f"Window-15 would be: ~30 lines (entire function)")
    print(f"Reduction: ~{100 - (len(result.included_lines) / 22 * 100):.0f}%")
