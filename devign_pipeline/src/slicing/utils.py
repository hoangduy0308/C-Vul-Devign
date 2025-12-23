"""Slicing utility functions.

Shared utilities for finding criterion lines for code slicing.
"""

import re
from typing import List

from ..tokenization.hybrid_tokenizer import DANGEROUS_APIS


def find_criterion_lines(code: str, dangerous_apis: set = None) -> List[int]:
    """Find criterion lines based on dangerous API calls.
    
    Searches for lines containing calls to security-sensitive APIs
    (memory allocation, string operations, etc.) that are likely
    vulnerability entry points.
    
    Args:
        code: Source code string to analyze
        dangerous_apis: Optional custom set of APIs to search for.
                       Defaults to DANGEROUS_APIS from hybrid_tokenizer.
    
    Returns:
        List of 1-indexed line numbers containing criterion points.
        Falls back to pointer/array patterns or middle of function
        if no dangerous APIs are found.
    """
    if dangerous_apis is None:
        dangerous_apis = DANGEROUS_APIS
    
    lines = code.split('\n')
    criterion_lines = []
    
    for i, line in enumerate(lines, 1):
        for api in dangerous_apis:
            if re.search(rf'\b{api}\s*\(', line):
                criterion_lines.append(i)
                break
    
    # Fallback: if no dangerous APIs found, use heuristic
    if not criterion_lines and lines:
        # Look for pointer ops, array access, etc.
        for i, line in enumerate(lines, 1):
            if re.search(r'\*\w+|->|\[\w*\]', line):  # pointer/array patterns
                criterion_lines.append(i)
        
        # Still nothing? Use middle
        if not criterion_lines:
            criterion_lines = [len(lines) // 2] if len(lines) > 1 else [1]
    
    return criterion_lines
