"""
Enhanced Vulnerability Features v3

Key improvements over v1 (raw counts) and v2 (ratios):
1. RATIO-BASED FEATURES: bounds_per_pointer_op, null_checks_per_malloc
2. PATTERN-BASED BINARY FEATURES: unbounded_strcpy, unchecked_malloc, etc.
3. DANGER SCORE: weighted combination of risky patterns

These features capture WHETHER protective patterns are missing,
not just HOW MANY risky constructs exist.
"""

import re
from typing import Dict, Set, List, Tuple, Optional, Any
from dataclasses import dataclass


# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================

# Lookahead window for finding NULL checks after malloc/alloc calls
# Increased to 15 to handle multi-line error handling, comments, etc.
NULL_CHECK_LOOKAHEAD_LINES = 15

# Lookahead window for detecting double-free patterns
DOUBLE_FREE_LOOKAHEAD_LINES = 50


# ============================================================
# DANGEROUS PATTERN DEFINITIONS
# ============================================================

@dataclass
class DangerousPattern:
    """A pattern that indicates potential vulnerability if unguarded."""
    name: str
    pattern: re.Pattern
    category: str  # buffer_overflow, null_deref, memory_leak, etc.
    severity: float  # 0.0 - 1.0
    guard_pattern: Optional[re.Pattern] = None  # Pattern that mitigates this danger


# Buffer overflow patterns
UNBOUNDED_COPY_PATTERNS = [
    # strcpy without bounds
    DangerousPattern(
        name="unbounded_strcpy",
        pattern=re.compile(r'\bstrcpy\s*\(\s*([^,]+),\s*([^)]+)\)', re.MULTILINE),
        category="buffer_overflow",
        severity=0.9,
        guard_pattern=re.compile(r'\bstrlen\s*\([^)]*\)|sizeof\s*\([^)]*\)', re.MULTILINE)
    ),
    # sprintf without bounds (not snprintf)
    DangerousPattern(
        name="unbounded_sprintf",
        pattern=re.compile(r'\bsprintf\s*\(\s*([^,]+),', re.MULTILINE),
        category="buffer_overflow",
        severity=0.85,
        guard_pattern=None  # Should use snprintf instead
    ),
    # gets() - always dangerous
    DangerousPattern(
        name="dangerous_gets",
        pattern=re.compile(r'\bgets\s*\(\s*\w+\s*\)', re.MULTILINE),
        category="buffer_overflow",
        severity=1.0,
        guard_pattern=None
    ),
    # strcat without checking destination size
    DangerousPattern(
        name="unbounded_strcat",
        pattern=re.compile(r'\bstrcat\s*\(\s*([^,]+),\s*([^)]+)\)', re.MULTILINE),
        category="buffer_overflow",
        severity=0.8,
        guard_pattern=re.compile(r'\bstrlen\s*\([^)]*\)|sizeof\s*\([^)]*\)', re.MULTILINE)
    ),
]

# Null pointer patterns
NULL_DEREF_PATTERNS = [
    # malloc without NULL check
    DangerousPattern(
        name="unchecked_malloc",
        pattern=re.compile(r'(\w+)\s*=\s*(?:malloc|calloc|realloc)\s*\([^)]*\)', re.MULTILINE),
        category="null_deref",
        severity=0.7,
        guard_pattern=re.compile(r'if\s*\(\s*\w+\s*(?:==|!=)\s*NULL|if\s*\(\s*!\s*\w+\s*\)', re.MULTILINE)
    ),
    # Pointer dereference without NULL check
    DangerousPattern(
        name="unchecked_deref",
        pattern=re.compile(r'\*\s*(\w+)|(\w+)\s*->\s*\w+', re.MULTILINE),
        category="null_deref",
        severity=0.5,
        guard_pattern=re.compile(r'if\s*\([^)]*\b\w+\b[^)]*(?:!=|==)\s*NULL', re.MULTILINE)
    ),
]

# Memory management patterns
MEMORY_PATTERNS = [
    # free without NULL check
    DangerousPattern(
        name="unchecked_free",
        pattern=re.compile(r'\bfree\s*\(\s*(\w+)\s*\)', re.MULTILINE),
        category="memory",
        severity=0.4,
        guard_pattern=re.compile(r'if\s*\(\s*\w+\s*(?:!=|==)\s*NULL', re.MULTILINE)
    ),
    # Double free potential (same var freed multiple times pattern)
    DangerousPattern(
        name="potential_double_free",
        pattern=re.compile(r'\bfree\s*\(\s*(\w+)\s*\)', re.MULTILINE),
        category="memory",
        severity=0.8,
        guard_pattern=re.compile(r'(\w+)\s*=\s*NULL\s*;', re.MULTILINE)  # Set to NULL after free
    ),
]

# Array/buffer access patterns
ARRAY_PATTERNS = [
    # Array access with variable index (no bounds check nearby)
    DangerousPattern(
        name="unbounded_array_access",
        pattern=re.compile(r'(\w+)\s*\[\s*([a-zA-Z_]\w*)\s*\]', re.MULTILINE),
        category="buffer_overflow",
        severity=0.6,
        guard_pattern=re.compile(r'\b(?:if|while|for)\s*\([^)]*(?:<|<=|>|>=)\s*(?:sizeof|strlen|size|len|count|num|max)', re.MULTILINE | re.IGNORECASE)
    ),
]

# Integer overflow patterns
INTEGER_PATTERNS = [
    # Multiplication without overflow check
    DangerousPattern(
        name="unchecked_multiplication",
        pattern=re.compile(r'(?:malloc|calloc|realloc)\s*\(\s*[^)]*\*[^)]*\)', re.MULTILINE),
        category="integer_overflow",
        severity=0.6,
        guard_pattern=re.compile(r'(?:SIZE_MAX|UINT_MAX|INT_MAX)\s*/|overflow|check', re.MULTILINE | re.IGNORECASE)
    ),
]

# Format string patterns
FORMAT_PATTERNS = [
    # printf with non-literal format string
    DangerousPattern(
        name="format_string_vuln",
        pattern=re.compile(r'\b(?:printf|fprintf|sprintf|snprintf)\s*\(\s*(?:[^",]+,\s*)?([a-zA-Z_]\w*)\s*\)', re.MULTILINE),
        category="format_string",
        severity=0.9,
        guard_pattern=None  # Should always use literal format string
    ),
]

ALL_DANGEROUS_PATTERNS = (
    UNBOUNDED_COPY_PATTERNS + 
    NULL_DEREF_PATTERNS + 
    MEMORY_PATTERNS + 
    ARRAY_PATTERNS + 
    INTEGER_PATTERNS +
    FORMAT_PATTERNS
)


# ============================================================
# REGEX PATTERNS FOR COUNTING
# ============================================================

POINTER_OP_PATTERN = re.compile(
    r'\*\s*\w+|'           # *ptr
    r'\w+\s*->\s*\w+|'     # ptr->member
    r'\w+\s*\[\s*[^\]]+\]', # arr[idx]
    re.MULTILINE
)

BOUNDS_CHECK_PATTERN = re.compile(
    r'\b(?:if|while|for)\s*\([^)]*(?:<|<=|>|>=)\s*(?:sizeof|strlen|size|len|count|num|max|limit|capacity)',
    re.MULTILINE | re.IGNORECASE
)

NULL_CHECK_PATTERN = re.compile(
    r'\b(?:if|while)\s*\([^)]*(?:==|!=)\s*NULL|'
    r'\b(?:if|while)\s*\(\s*!\s*\w+\s*\)|'
    r'\b(?:if|while)\s*\(\s*\w+\s*\)',  # if (ptr)
    re.MULTILINE
)

MALLOC_PATTERN = re.compile(r'\b(?:malloc|calloc|realloc)\s*\(', re.MULTILINE)
FREE_PATTERN = re.compile(r'\bfree\s*\(', re.MULTILINE)
SIZEOF_PATTERN = re.compile(r'\bsizeof\s*\(', re.MULTILINE)
STRLEN_PATTERN = re.compile(r'\bstrlen\s*\(', re.MULTILINE)

DANGEROUS_FUNC_PATTERN = re.compile(
    r'\b(?:strcpy|strcat|sprintf|gets|scanf|sscanf|fscanf|vsprintf)\s*\(',
    re.MULTILINE
)

SAFE_FUNC_PATTERN = re.compile(
    r'\b(?:strncpy|strncat|snprintf|fgets|memcpy_s|strcpy_s)\s*\(',
    re.MULTILINE
)


# ============================================================
# MAIN FEATURE EXTRACTION
# ============================================================

def extract_enhanced_features(code: str) -> Dict[str, float]:
    """
    Extract enhanced vulnerability features from C code.
    
    Returns features in 3 categories:
    1. RATIO-BASED: bounds_per_pointer_op, null_checks_per_malloc, defense_ratio
    2. BINARY PATTERNS: has_unbounded_strcpy, has_unchecked_malloc, etc.
    3. COMPOSITE SCORES: danger_score, missing_defense_score
    
    Total: ~35 features (all normalized 0-1 or ratio-based)
    """
    if not code or not code.strip():
        return _get_empty_features()
    
    features: Dict[str, float] = {}
    
    # ============================================================
    # 1. BASIC COUNTS (for computing ratios)
    # ============================================================
    loc = max(1, len([l for l in code.split('\n') if l.strip()]))
    
    pointer_ops = len(POINTER_OP_PATTERN.findall(code))
    bounds_checks = len(BOUNDS_CHECK_PATTERN.findall(code))
    null_checks = len(NULL_CHECK_PATTERN.findall(code))
    malloc_calls = len(MALLOC_PATTERN.findall(code))
    free_calls = len(FREE_PATTERN.findall(code))
    sizeof_uses = len(SIZEOF_PATTERN.findall(code))
    strlen_uses = len(STRLEN_PATTERN.findall(code))
    dangerous_calls = len(DANGEROUS_FUNC_PATTERN.findall(code))
    safe_calls = len(SAFE_FUNC_PATTERN.findall(code))
    
    # ============================================================
    # 2. RATIO-BASED FEATURES (Key improvement!)
    # ============================================================
    
    # Bounds checks per pointer operation (higher = safer)
    features['bounds_per_pointer_op'] = bounds_checks / max(pointer_ops, 1)
    
    # NULL checks per malloc (should be >= 1 if safe)
    features['null_checks_per_malloc'] = null_checks / max(malloc_calls, 1)
    
    # Free to malloc ratio (should be ~1 for no leaks)
    features['free_malloc_ratio'] = free_calls / max(malloc_calls, 1)
    
    # sizeof/strlen usage ratio (defensive coding indicator)
    features['size_check_per_dangerous'] = (sizeof_uses + strlen_uses) / max(dangerous_calls, 1)
    
    # Safe vs dangerous function ratio
    features['safe_func_ratio'] = safe_calls / max(safe_calls + dangerous_calls, 1)
    
    # Overall defense ratio
    total_risky = pointer_ops + malloc_calls + dangerous_calls
    total_defensive = bounds_checks + null_checks + sizeof_uses + safe_calls
    features['defense_ratio'] = total_defensive / max(total_risky, 1)
    
    # Missing defense indicator (inverse - higher = more vulnerable)
    features['missing_defense_ratio'] = 1.0 - min(1.0, features['defense_ratio'])
    
    # ============================================================
    # 3. DENSITY FEATURES (normalized by LOC)
    # ============================================================
    features['pointer_op_density'] = min(1.0, pointer_ops / loc)
    features['malloc_density'] = min(1.0, malloc_calls / loc * 10)  # Scale up
    features['dangerous_call_density'] = min(1.0, dangerous_calls / loc * 10)
    
    # ============================================================
    # 4. PATTERN-BASED BINARY FEATURES
    # ============================================================
    pattern_detections = _detect_dangerous_patterns(code)
    
    # Binary features (0 or 1)
    features['has_unbounded_strcpy'] = float(pattern_detections.get('unbounded_strcpy', False))
    features['has_unbounded_sprintf'] = float(pattern_detections.get('unbounded_sprintf', False))
    features['has_dangerous_gets'] = float(pattern_detections.get('dangerous_gets', False))
    features['has_unbounded_strcat'] = float(pattern_detections.get('unbounded_strcat', False))
    features['has_unchecked_malloc'] = float(pattern_detections.get('unchecked_malloc', False))
    features['has_unchecked_deref'] = float(pattern_detections.get('unchecked_deref', False))
    features['has_unchecked_free'] = float(pattern_detections.get('unchecked_free', False))
    features['has_potential_double_free'] = float(pattern_detections.get('potential_double_free', False))
    features['has_unbounded_array_access'] = float(pattern_detections.get('unbounded_array_access', False))
    features['has_unchecked_multiplication'] = float(pattern_detections.get('unchecked_multiplication', False))
    features['has_format_string_vuln'] = float(pattern_detections.get('format_string_vuln', False))
    
    # ============================================================
    # 5. CATEGORY COUNTS (how many patterns per category)
    # ============================================================
    category_counts = _count_by_category(pattern_detections)
    features['buffer_overflow_patterns'] = min(1.0, category_counts.get('buffer_overflow', 0) / 3)
    features['null_deref_patterns'] = min(1.0, category_counts.get('null_deref', 0) / 2)
    features['memory_patterns'] = min(1.0, category_counts.get('memory', 0) / 2)
    features['integer_overflow_patterns'] = min(1.0, category_counts.get('integer_overflow', 0))
    features['format_string_patterns'] = min(1.0, category_counts.get('format_string', 0))
    
    # ============================================================
    # 6. COMPOSITE DANGER SCORES
    # ============================================================
    
    # Weighted danger score based on pattern severities
    features['danger_score'] = _compute_danger_score(pattern_detections, code)
    
    # Missing defense score (high = likely vulnerable)
    features['vuln_likelihood_score'] = _compute_vuln_likelihood(features, pattern_detections)
    
    # ============================================================
    # 7. CONTEXT FEATURES
    # ============================================================
    features['has_error_handling'] = float(bool(re.search(
        r'\b(?:error|err|fail|errno|perror|strerror)\b', code, re.IGNORECASE
    )))
    features['has_logging'] = float(bool(re.search(
        r'\b(?:log|LOG|printf|fprintf\s*\(\s*stderr)\b', code
    )))
    features['has_assertions'] = float(bool(re.search(
        r'\b(?:assert|ASSERT|BUG_ON|WARN_ON)\s*\(', code
    )))
    
    # ============================================================
    # 8. RAW COUNTS (kept for backward compatibility, normalized)
    # ============================================================
    features['pointer_op_count_norm'] = min(1.0, pointer_ops / 50)
    features['malloc_count_norm'] = min(1.0, malloc_calls / 10)
    features['dangerous_call_count_norm'] = min(1.0, dangerous_calls / 10)
    features['null_check_count_norm'] = min(1.0, null_checks / 20)
    features['bounds_check_count_norm'] = min(1.0, bounds_checks / 10)
    
    return features


def _get_empty_features() -> Dict[str, float]:
    """Return empty feature dict with all zeros."""
    return {
        # Ratio features
        'bounds_per_pointer_op': 0.0,
        'null_checks_per_malloc': 0.0,
        'free_malloc_ratio': 0.0,
        'size_check_per_dangerous': 0.0,
        'safe_func_ratio': 0.0,
        'defense_ratio': 0.0,
        'missing_defense_ratio': 0.0,
        # Density features
        'pointer_op_density': 0.0,
        'malloc_density': 0.0,
        'dangerous_call_density': 0.0,
        # Binary pattern features
        'has_unbounded_strcpy': 0.0,
        'has_unbounded_sprintf': 0.0,
        'has_dangerous_gets': 0.0,
        'has_unbounded_strcat': 0.0,
        'has_unchecked_malloc': 0.0,
        'has_unchecked_deref': 0.0,
        'has_unchecked_free': 0.0,
        'has_potential_double_free': 0.0,
        'has_unbounded_array_access': 0.0,
        'has_unchecked_multiplication': 0.0,
        'has_format_string_vuln': 0.0,
        # Category counts
        'buffer_overflow_patterns': 0.0,
        'null_deref_patterns': 0.0,
        'memory_patterns': 0.0,
        'integer_overflow_patterns': 0.0,
        'format_string_patterns': 0.0,
        # Composite scores
        'danger_score': 0.0,
        'vuln_likelihood_score': 0.0,
        # Context features
        'has_error_handling': 0.0,
        'has_logging': 0.0,
        'has_assertions': 0.0,
        # Normalized counts
        'pointer_op_count_norm': 0.0,
        'malloc_count_norm': 0.0,
        'dangerous_call_count_norm': 0.0,
        'null_check_count_norm': 0.0,
        'bounds_check_count_norm': 0.0,
    }


def _detect_dangerous_patterns(code: str) -> Dict[str, bool]:
    """
    Detect dangerous patterns and check if they're guarded.
    Returns dict mapping pattern name -> True if dangerous (unguarded).
    """
    detections: Dict[str, bool] = {}
    lines = code.split('\n')
    
    for pattern in ALL_DANGEROUS_PATTERNS:
        matches = pattern.pattern.findall(code)
        if not matches:
            detections[pattern.name] = False
            continue
        
        # Check if guard pattern exists
        if pattern.guard_pattern:
            has_guard = bool(pattern.guard_pattern.search(code))
            # For malloc, need more sophisticated check
            if pattern.name == 'unchecked_malloc':
                detections[pattern.name] = not _check_malloc_guarded(code)
            elif pattern.name == 'potential_double_free':
                detections[pattern.name] = _check_double_free_risk(code)
            else:
                # Pattern is dangerous if no guard found
                detections[pattern.name] = not has_guard
        else:
            # No guard possible - always dangerous if pattern found
            detections[pattern.name] = True
    
    return detections


def _check_malloc_guarded(code: str) -> bool:
    """Check if malloc calls are properly guarded with NULL checks.
    
    Handles:
    - Standard: if (var == NULL), if (var != NULL)
    - Yoda conditions: if (NULL == var), if (NULL != var)
    - Boolean check: if (!var), if (var)
    - Lookahead: 10 lines (covers multi-line error handling)
    """
    # Find malloc assignments
    malloc_pattern = re.compile(r'(\w+)\s*=\s*(?:malloc|calloc|realloc)\s*\([^)]*\)', re.MULTILINE)
    
    lines = code.split('\n')
    
    for match in malloc_pattern.finditer(code):
        var_name = match.group(1)
        match_line = code[:match.start()].count('\n')
        
        # Look for NULL check in next N lines (configurable via constant)
        has_check = False
        
        for i in range(match_line + 1, min(match_line + NULL_CHECK_LOOKAHEAD_LINES + 1, len(lines))):
            line = lines[i] if i < len(lines) else ""
            
            # Standard: if (var == NULL) or if (var != NULL)
            if re.search(rf'\b{re.escape(var_name)}\s*(?:==|!=)\s*NULL\b', line):
                has_check = True
                break
            
            # Yoda conditions: if (NULL == var) or if (NULL != var)
            if re.search(rf'\bNULL\s*(?:==|!=)\s*{re.escape(var_name)}\b', line):
                has_check = True
                break
            
            # Boolean negation: if (!var)
            if re.search(rf'if\s*\(\s*!\s*{re.escape(var_name)}\s*\)', line):
                has_check = True
                break
            
            # Boolean truthy: if (var)
            if re.search(rf'if\s*\(\s*{re.escape(var_name)}\s*\)', line):
                has_check = True
                break
            
            # Ternary or inline: var ? ... : ... (implies check)
            if re.search(rf'\b{re.escape(var_name)}\s*\?', line):
                has_check = True
                break
            
            # Assert patterns: assert(var), BUG_ON(!var)
            if re.search(rf'\b(?:assert|Assert|ASSERT|BUG_ON|WARN_ON)\s*\([^)]*{re.escape(var_name)}', line):
                has_check = True
                break
        
        if not has_check:
            return False  # Found unguarded malloc
    
    return True  # All malloc calls are guarded


def _check_double_free_risk(code: str) -> bool:
    """Check for potential double-free (same var freed twice without reassignment)."""
    free_pattern = re.compile(r'\bfree\s*\(\s*(\w+)\s*\)', re.MULTILINE)
    freed_vars: Dict[str, int] = {}  # var -> line number of first free
    
    lines = code.split('\n')
    
    for match in free_pattern.finditer(code):
        var_name = match.group(1)
        current_line = code[:match.start()].count('\n')
        
        if var_name in freed_vars:
            first_free_line = freed_vars[var_name]
            # Check if var was reassigned between frees
            reassigned = False
            for i in range(first_free_line + 1, current_line):
                if i < len(lines):
                    line = lines[i]
                    if re.search(rf'\b{re.escape(var_name)}\s*=\s*(?!NULL)', line):
                        reassigned = True
                        break
                    # Also check if set to NULL (acceptable pattern)
                    if re.search(rf'\b{re.escape(var_name)}\s*=\s*NULL\b', line):
                        reassigned = True
                        break
            
            if not reassigned:
                return True  # Potential double free
        
        freed_vars[var_name] = current_line
    
    return False


def _count_by_category(detections: Dict[str, bool]) -> Dict[str, int]:
    """Count detected patterns by vulnerability category."""
    category_counts: Dict[str, int] = {}
    
    for pattern in ALL_DANGEROUS_PATTERNS:
        if detections.get(pattern.name, False):
            cat = pattern.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
    
    return category_counts


def _compute_danger_score(detections: Dict[str, bool], code: str) -> float:
    """
    Compute weighted danger score based on detected patterns.
    Score range: 0.0 - 1.0
    """
    if not any(detections.values()):
        return 0.0
    
    total_severity = 0.0
    max_possible = 0.0
    
    for pattern in ALL_DANGEROUS_PATTERNS:
        max_possible += pattern.severity
        if detections.get(pattern.name, False):
            total_severity += pattern.severity
    
    # Normalize
    if max_possible == 0:
        return 0.0
    
    return min(1.0, total_severity / max_possible * 2)  # Scale factor for sensitivity


def _compute_vuln_likelihood(features: Dict[str, float], detections: Dict[str, bool]) -> float:
    """
    Compute overall vulnerability likelihood score.
    Combines ratio features and pattern detections.
    """
    score = 0.0
    
    # Low defense ratio is bad
    score += features.get('missing_defense_ratio', 0) * 0.3
    
    # High danger density is bad
    score += features.get('dangerous_call_density', 0) * 0.2
    score += features.get('pointer_op_density', 0) * 0.1
    
    # Count of dangerous patterns
    pattern_count = sum(1 for v in detections.values() if v)
    score += min(1.0, pattern_count / 5) * 0.3
    
    # Specific high-severity patterns
    if detections.get('dangerous_gets', False):
        score += 0.1
    if detections.get('unbounded_strcpy', False):
        score += 0.1
    
    # Lack of defensive patterns
    if not features.get('has_error_handling', 0):
        score += 0.05
    if not features.get('has_assertions', 0):
        score += 0.05
    
    return min(1.0, score)


# ============================================================
# FEATURE NAMES FOR EXPORT
# ============================================================

def get_feature_names() -> List[str]:
    """Get list of all feature names in consistent order."""
    return list(_get_empty_features().keys())


def get_feature_dim() -> int:
    """Get total number of features."""
    return len(get_feature_names())


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    # Test with sample code
    test_code = '''
    void vulnerable_function(char *input) {
        char buffer[64];
        char *ptr = malloc(100);
        
        strcpy(buffer, input);  // unbounded copy
        sprintf(ptr, "%s", input);  // unbounded sprintf
        
        ptr->value = 10;  // no NULL check after malloc
        
        free(ptr);
        free(ptr);  // double free!
    }
    '''
    
    features = extract_enhanced_features(test_code)
    
    print("Enhanced Features:")
    print("=" * 50)
    for name, value in sorted(features.items()):
        if value > 0:
            print(f"  {name}: {value:.4f}")
    
    print(f"\nTotal features: {len(features)}")
    print(f"Danger score: {features['danger_score']:.4f}")
    print(f"Vuln likelihood: {features['vuln_likelihood_score']:.4f}")
