"""Slice-based vulnerability feature extraction.

"Local-first, Global-light" approach:
- 22 features computed on slice code (local context)
- 3 features computed on full function (global context)
"""

import logging
from typing import Dict, List, Optional, Set, TYPE_CHECKING
from concurrent.futures import ProcessPoolExecutor, as_completed

if TYPE_CHECKING:
    from ..graph.cfg import CFG
    from ..graph.dfg import DFG
    from ..parser.tree_sitter_parser import ParseResult

from .dictionary import VulnDictionary
from .rules import (
    _compute_basic_size_metrics,
    _compute_dangerous_call_features,
    _compute_pointer_deref_features,
    _compute_array_access_features,
    _compute_malloc_free_features,
    _compute_unchecked_return_features,
    _index_conditions_and_checks,
    _build_graph_context,
)

logger = logging.getLogger(__name__)

SLICE_FEATURE_NAMES: List[str] = [
    # Slice features (22)
    'loc_slice',
    'stmt_count_slice',
    'dangerous_call_count_slice',
    'dangerous_call_without_check_count_slice',
    'dangerous_call_without_check_ratio_slice',
    'pointer_deref_count_slice',
    'pointer_deref_without_null_check_count_slice',
    'pointer_deref_without_null_check_ratio_slice',
    'array_access_count_slice',
    'array_access_without_bounds_check_count_slice',
    'array_access_without_bounds_check_ratio_slice',
    'malloc_count_slice',
    'malloc_without_free_count_slice',
    'malloc_without_free_ratio_slice',
    'free_count_slice',
    'free_without_null_check_count_slice',
    'free_without_null_check_ratio_slice',
    'unchecked_return_value_count_slice',
    'unchecked_return_value_ratio_slice',
    'null_check_count_slice',
    'bounds_check_count_slice',
    'defense_ratio_slice',
    # Global features (3)
    'loc_full',
    'malloc_without_free_ratio_full',
    'defense_ratio_full',
]


def _compute_features_on_code(
    code: str,
    dictionary: VulnDictionary,
    parse_result: Optional['ParseResult'] = None,
    cfg: Optional['CFG'] = None,
    dfg: Optional['DFG'] = None,
) -> Dict[str, float]:
    """Compute all raw features on given code (reuses helpers from rules.py)."""
    features: Dict[str, float] = {}
    
    if not code or not code.strip():
        return {
            'loc': 0.0,
            'stmt_count': 0.0,
            'dangerous_call_count': 0.0,
            'dangerous_call_without_check_count': 0.0,
            'dangerous_call_without_check_ratio': 0.0,
            'pointer_deref_count': 0.0,
            'pointer_deref_without_null_check_count': 0.0,
            'pointer_deref_without_null_check_ratio': 0.0,
            'array_access_count': 0.0,
            'array_access_without_bounds_check_count': 0.0,
            'array_access_without_bounds_check_ratio': 0.0,
            'malloc_count': 0.0,
            'malloc_without_free_count': 0.0,
            'malloc_without_free_ratio': 0.0,
            'free_count': 0.0,
            'free_without_null_check_count': 0.0,
            'free_without_null_check_ratio': 0.0,
            'unchecked_return_value_count': 0.0,
            'unchecked_return_value_ratio': 0.0,
            'null_check_count': 0.0,
            'bounds_check_count': 0.0,
            'defense_ratio': 0.0,
        }
    
    parse_result, cfg, dfg = _build_graph_context(code, parse_result, cfg, dfg)
    
    size_metrics = _compute_basic_size_metrics(code)
    features.update(size_metrics)
    
    null_checked_vars, bounds_checked_vars, condition_lines = _index_conditions_and_checks(
        code, parse_result
    )
    
    dc_features = _compute_dangerous_call_features(
        code, dictionary, parse_result, cfg, dfg, null_checked_vars, bounds_checked_vars
    )
    features.update(dc_features)
    
    pd_features = _compute_pointer_deref_features(
        code, dfg, cfg, null_checked_vars
    )
    features.update(pd_features)
    
    aa_features = _compute_array_access_features(
        code, parse_result, cfg, bounds_checked_vars
    )
    features.update(aa_features)
    
    mf_features = _compute_malloc_free_features(
        code, parse_result, dfg, null_checked_vars
    )
    features.update(mf_features)
    
    ur_features = _compute_unchecked_return_features(
        code, parse_result, dictionary
    )
    features.update(ur_features)
    
    features['null_check_count'] = float(len(null_checked_vars))
    features['bounds_check_count'] = float(len(bounds_checked_vars))
    
    risk_ops = (
        features['dangerous_call_count'] + 
        features['pointer_deref_count'] + 
        features['array_access_count'] + 
        features['malloc_count']
    )
    defense_count = features['null_check_count'] + features['bounds_check_count']
    features['defense_ratio'] = defense_count / max(risk_ops, 1.0)
    
    return features


def extract_slice_features(
    slice_code: str,
    full_code: str,
    dictionary: VulnDictionary,
) -> Dict[str, float]:
    """
    Extract vulnerability features using "Local-first, Global-light" schema.
    
    Args:
        slice_code: The slice code (local context)
        full_code: The full function code (global context)
        dictionary: Vulnerability dictionary for dangerous function lookup
    
    Returns:
        Dict with 25 features:
        - 22 slice features (computed on slice_code)
        - 3 global features (computed on full_code)
    """
    result: Dict[str, float] = {name: 0.0 for name in SLICE_FEATURE_NAMES}
    
    slice_features = _compute_features_on_code(slice_code, dictionary)
    
    result['loc_slice'] = slice_features.get('loc', 0.0)
    result['stmt_count_slice'] = slice_features.get('stmt_count', 0.0)
    result['dangerous_call_count_slice'] = slice_features.get('dangerous_call_count', 0.0)
    result['dangerous_call_without_check_count_slice'] = slice_features.get('dangerous_call_without_check_count', 0.0)
    result['dangerous_call_without_check_ratio_slice'] = slice_features.get('dangerous_call_without_check_ratio', 0.0)
    result['pointer_deref_count_slice'] = slice_features.get('pointer_deref_count', 0.0)
    result['pointer_deref_without_null_check_count_slice'] = slice_features.get('pointer_deref_without_null_check_count', 0.0)
    result['pointer_deref_without_null_check_ratio_slice'] = slice_features.get('pointer_deref_without_null_check_ratio', 0.0)
    result['array_access_count_slice'] = slice_features.get('array_access_count', 0.0)
    result['array_access_without_bounds_check_count_slice'] = slice_features.get('array_access_without_bounds_check_count', 0.0)
    result['array_access_without_bounds_check_ratio_slice'] = slice_features.get('array_access_without_bounds_check_ratio', 0.0)
    result['malloc_count_slice'] = slice_features.get('malloc_count', 0.0)
    result['malloc_without_free_count_slice'] = slice_features.get('malloc_without_free_count', 0.0)
    result['malloc_without_free_ratio_slice'] = slice_features.get('malloc_without_free_ratio', 0.0)
    result['free_count_slice'] = slice_features.get('free_count', 0.0)
    result['free_without_null_check_count_slice'] = slice_features.get('free_without_null_check_count', 0.0)
    result['free_without_null_check_ratio_slice'] = slice_features.get('free_without_null_check_ratio', 0.0)
    result['unchecked_return_value_count_slice'] = slice_features.get('unchecked_return_value_count', 0.0)
    result['unchecked_return_value_ratio_slice'] = slice_features.get('unchecked_return_value_ratio', 0.0)
    result['null_check_count_slice'] = slice_features.get('null_check_count', 0.0)
    result['bounds_check_count_slice'] = slice_features.get('bounds_check_count', 0.0)
    result['defense_ratio_slice'] = slice_features.get('defense_ratio', 0.0)
    
    full_features = _compute_features_on_code(full_code, dictionary)
    
    result['loc_full'] = full_features.get('loc', 0.0)
    result['malloc_without_free_ratio_full'] = full_features.get('malloc_without_free_ratio', 0.0)
    result['defense_ratio_full'] = full_features.get('defense_ratio', 0.0)
    
    return result


def _extract_single(args: tuple) -> Dict[str, float]:
    """Helper for parallel processing."""
    slice_code, full_code, dictionary = args
    return extract_slice_features(slice_code, full_code, dictionary)


def extract_slice_features_batch(
    slice_codes: List[str],
    full_codes: List[str],
    dictionary: VulnDictionary,
    n_jobs: int = 1,
) -> List[Dict[str, float]]:
    """
    Batch extraction of slice features.
    
    Args:
        slice_codes: List of slice code strings
        full_codes: List of full function code strings (must match length of slice_codes)
        dictionary: Vulnerability dictionary
        n_jobs: Number of parallel jobs (1 = sequential)
    
    Returns:
        List of feature dictionaries, one per sample
    """
    if len(slice_codes) != len(full_codes):
        raise ValueError(
            f"Length mismatch: slice_codes={len(slice_codes)}, full_codes={len(full_codes)}"
        )
    
    if not slice_codes:
        return []
    
    if n_jobs == 1:
        results = []
        for i, (slice_code, full_code) in enumerate(zip(slice_codes, full_codes)):
            try:
                feat = extract_slice_features(slice_code, full_code, dictionary)
                results.append(feat)
            except Exception as e:
                logger.warning(f"Error extracting features for sample {i}: {e}")
                results.append({name: 0.0 for name in SLICE_FEATURE_NAMES})
        return results
    
    results = [None] * len(slice_codes)
    args_list = [(s, f, dictionary) for s, f in zip(slice_codes, full_codes)]
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_idx = {
            executor.submit(_extract_single, args): idx 
            for idx, args in enumerate(args_list)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning(f"Error extracting features for sample {idx}: {e}")
                results[idx] = {name: 0.0 for name in SLICE_FEATURE_NAMES}
    
    return results


if __name__ == "__main__":
    from .dictionary import VulnDictionary
    
    test_slice = """
    char *buf = malloc(100);
    strcpy(buf, user_input);
    if (buf != NULL) {
        free(buf);
    }
    """
    
    test_full = """
    void process_input(char *user_input) {
        char *buf = malloc(100);
        char *buf2 = malloc(200);
        strcpy(buf, user_input);
        if (buf != NULL) {
            free(buf);
        }
        // buf2 is leaked
    }
    """
    
    dictionary = VulnDictionary()
    
    print("=" * 60)
    print("Testing extract_slice_features()")
    print("=" * 60)
    
    features = extract_slice_features(test_slice, test_full, dictionary)
    
    print(f"\nTotal features: {len(features)}")
    print(f"Expected: {len(SLICE_FEATURE_NAMES)}")
    
    print("\n--- Slice Features (22) ---")
    for name in SLICE_FEATURE_NAMES[:22]:
        print(f"  {name}: {features[name]:.3f}")
    
    print("\n--- Global Features (3) ---")
    for name in SLICE_FEATURE_NAMES[22:]:
        print(f"  {name}: {features[name]:.3f}")
    
    print("\n" + "=" * 60)
    print("Testing extract_slice_features_batch()")
    print("=" * 60)
    
    batch_slices = [test_slice, "", test_slice]
    batch_fulls = [test_full, test_full, ""]
    
    batch_results = extract_slice_features_batch(
        batch_slices, batch_fulls, dictionary, n_jobs=1
    )
    
    print(f"\nBatch size: {len(batch_results)}")
    for i, feat in enumerate(batch_results):
        print(f"  Sample {i}: loc_slice={feat['loc_slice']:.0f}, "
              f"loc_full={feat['loc_full']:.0f}, "
              f"defense_ratio_slice={feat['defense_ratio_slice']:.3f}")
    
    print("\n✓ All tests passed!")
