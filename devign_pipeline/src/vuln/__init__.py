"""Vulnerability detection utilities"""

from .dictionary import (
    VulnDictionary,
    VulnerabilityPattern,
    load_vuln_patterns,
    get_default_dictionary,
)
from .rules import (
    VulnRules,
    match_vulnerability,
    extract_vuln_features,
    extract_vuln_features_v2,
    find_dangerous_calls,
    analyze_pointer_usage,
    score_vulnerability_risk,
    get_vulnerability_summary,
)
from .vuln_lines import extract_vul_line_numbers
from .enhanced_features import (
    extract_enhanced_features,
    get_feature_names,
    get_feature_dim,
)

__all__ = [
    "VulnDictionary",
    "VulnerabilityPattern",
    "load_vuln_patterns",
    "get_default_dictionary",
    "VulnRules",
    "match_vulnerability",
    "extract_vuln_features",
    "extract_vuln_features_v2",
    "find_dangerous_calls",
    "analyze_pointer_usage",
    "score_vulnerability_risk",
    "get_vulnerability_summary",
    "extract_vul_line_numbers",
    # Enhanced features v3
    "extract_enhanced_features",
    "get_feature_names",
    "get_feature_dim",
]
