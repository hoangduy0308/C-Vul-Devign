"""Multi-slice module combining backward and forward slicing.

Combines backward slice (data/control dependencies leading to criterion)
with forward slice (effects/lifecycle after criterion) for better
vulnerability pattern detection.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, 'F:/Work/C Vul Devign/devign_pipeline')

from src.slicing.slicer import CodeSlicer, SliceConfig, CodeSlice, SliceType
from src.graphs.cfg import CFG
from src.graphs.dfg import DFG


@dataclass
class MultiSliceConfig:
    """Configuration for multi-slice combining backward and forward."""
    backward_depth: int = 5
    backward_window: int = 15
    forward_depth: int = 3
    forward_window: int = 10
    max_combined_tokens: int = 512
    sep_token: str = "[SEP]"
    include_control_deps: bool = True
    include_data_deps: bool = True
    remove_comments: bool = True
    normalize_output: bool = True


@dataclass
class MultiSliceResult:
    """Result of multi-slice operation."""
    backward_slice: CodeSlice
    forward_slice: CodeSlice
    combined_code: str
    combined_lines: Set[int]
    
    def __post_init__(self):
        if isinstance(self.combined_lines, list):
            self.combined_lines = set(self.combined_lines)


class MultiCodeSlicer:
    """Multi-slice combining backward and forward slicing."""
    
    def __init__(self, config: MultiSliceConfig = None):
        self.config = config or MultiSliceConfig()
        
        self.backward_config = SliceConfig(
            slice_type=SliceType.BACKWARD,
            window_size=self.config.backward_window,
            max_depth=self.config.backward_depth,
            include_control_deps=self.config.include_control_deps,
            include_data_deps=self.config.include_data_deps,
            remove_comments=self.config.remove_comments,
            normalize_output=self.config.normalize_output
        )
        
        self.forward_config = SliceConfig(
            slice_type=SliceType.FORWARD,
            window_size=self.config.forward_window,
            max_depth=self.config.forward_depth,
            include_control_deps=self.config.include_control_deps,
            include_data_deps=self.config.include_data_deps,
            remove_comments=self.config.remove_comments,
            normalize_output=self.config.normalize_output
        )
        
        self.backward_slicer = CodeSlicer(self.backward_config)
        self.forward_slicer = CodeSlicer(self.forward_config)
    
    def multi_slice(self, code: str, criterion_lines: List[int],
                    cfg: Optional[CFG] = None,
                    dfg: Optional[DFG] = None) -> MultiSliceResult:
        """
        Perform multi-slice combining backward and forward slices.
        
        Args:
            code: Source code to slice
            criterion_lines: Lines to use as slicing criterion
            cfg: Optional precomputed CFG
            dfg: Optional precomputed DFG
        
        Returns:
            MultiSliceResult with backward, forward and combined slices
        """
        backward_slice = self.backward_slicer.slice(code, criterion_lines, cfg, dfg)
        forward_slice = self.forward_slicer.slice(code, criterion_lines, cfg, dfg)
        
        combined_code = self._combine_slices(backward_slice, forward_slice)
        combined_lines = backward_slice.included_lines | forward_slice.included_lines
        
        return MultiSliceResult(
            backward_slice=backward_slice,
            forward_slice=forward_slice,
            combined_code=combined_code,
            combined_lines=combined_lines
        )
    
    def _combine_slices(self, backward: CodeSlice, forward: CodeSlice) -> str:
        """
        Combine backward and forward slices with separator.
        Removes overlapping lines from forward to avoid duplication.
        """
        # Remove overlapping lines from forward
        forward_only_lines = forward.included_lines - backward.included_lines
        
        # Rebuild forward code from non-overlapping lines only
        if forward_only_lines and forward.original_code:
            forward_code = self._extract_lines_code(forward.original_code, forward_only_lines)
        else:
            forward_code = ""
        
        backward_code = backward.code.strip()
        forward_code = forward_code.strip()
        sep = f" {self.config.sep_token} "
        sep_tokens = 1
        
        # Both empty - return empty
        if not backward_code and not forward_code:
            return ""
        
        # Only forward - SEP at start
        if not backward_code:
            truncated = self._truncate_to_tokens(forward_code, self.config.max_combined_tokens - sep_tokens)
            return f"{sep}{truncated}"
        
        # Only backward - SEP at end
        if not forward_code:
            truncated = self._truncate_to_tokens(backward_code, self.config.max_combined_tokens - sep_tokens)
            return f"{truncated}{sep}"
        
        # Both non-empty
        backward_tokens = self._estimate_tokens(backward_code)
        forward_tokens = self._estimate_tokens(forward_code)
        total_tokens = backward_tokens + sep_tokens + forward_tokens
        
        if total_tokens <= self.config.max_combined_tokens:
            return f"{backward_code}{sep}{forward_code}"
        
        available = self.config.max_combined_tokens - sep_tokens
        
        if backward_tokens <= available // 2:
            backward_budget = backward_tokens
            forward_budget = available - backward_budget
        else:
            backward_budget = available * 2 // 3
            forward_budget = available - backward_budget
        
        truncated_backward = self._truncate_to_tokens(backward_code, backward_budget)
        truncated_forward = self._truncate_to_tokens(forward_code, forward_budget)
        
        return f"{truncated_backward}{sep}{truncated_forward}"
    
    def _extract_lines_code(self, code: str, line_numbers: Set[int]) -> str:
        """Extract specific lines from code with consistent dedent."""
        if not line_numbers:
            return ""
        
        lines = code.split('\n')
        extracted = []
        
        for line_num in sorted(line_numbers):
            if 1 <= line_num <= len(lines):
                line = lines[line_num - 1]
                if line.strip():  # Skip empty lines
                    extracted.append(line)
        
        if not extracted:
            return ""
        
        # Find minimum indentation and dedent consistently
        min_indent = float('inf')
        for line in extracted:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                min_indent = min(min_indent, indent)
        
        if min_indent == float('inf'):
            min_indent = 0
        
        # Dedent all lines by min_indent
        result = []
        for line in extracted:
            if len(line) >= min_indent:
                result.append(line[int(min_indent):])
            else:
                result.append(line.lstrip())
        
        return '\n'.join(result)
    
    def _estimate_tokens(self, code: str) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        if not code:
            return 0
        return max(1, len(code) // 4)
    
    def _truncate_to_tokens(self, code: str, max_tokens: int) -> str:
        """Truncate code to approximately max_tokens."""
        if not code:
            return ""
        
        current_tokens = self._estimate_tokens(code)
        if current_tokens <= max_tokens:
            return code
        
        max_chars = max_tokens * 4
        lines = code.split('\n')
        result = []
        char_count = 0
        
        for line in lines:
            if char_count + len(line) + 1 > max_chars:
                break
            result.append(line)
            char_count += len(line) + 1
        
        return '\n'.join(result)


def _multi_slice_single(args: Tuple[str, List[int], MultiSliceConfig]) -> MultiSliceResult:
    """Helper for multiprocessing - multi-slice a single code sample."""
    code, criterion_lines, config = args
    slicer = MultiCodeSlicer(config)
    return slicer.multi_slice(code, criterion_lines)


def multi_slice_batch(codes: List[str], criterion_lines_list: List[List[int]],
                      config: MultiSliceConfig = None, 
                      n_jobs: int = 4) -> List[MultiSliceResult]:
    """
    Batch multi-slicing with multiprocessing.
    
    Args:
        codes: List of source code strings
        criterion_lines_list: List of criterion lines for each code
        config: Multi-slice configuration
        n_jobs: Number of parallel workers
    
    Returns:
        List of MultiSliceResult objects
    """
    if config is None:
        config = MultiSliceConfig()
    
    if len(codes) != len(criterion_lines_list):
        raise ValueError("codes and criterion_lines_list must have same length")
    
    args_list = [(code, lines, config) for code, lines in zip(codes, criterion_lines_list)]
    
    if len(codes) <= 10 or n_jobs == 1:
        return [_multi_slice_single(args) for args in args_list]
    
    results: List[Optional[MultiSliceResult]] = [None] * len(codes)
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_idx = {
            executor.submit(_multi_slice_single, args): idx
            for idx, args in enumerate(args_list)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                slicer = MultiCodeSlicer(config)
                results[idx] = slicer.multi_slice(codes[idx], criterion_lines_list[idx])
    
    return results


if __name__ == "__main__":
    test_code = '''int process_buffer(char *buf, int size) {
    char *ptr = malloc(size);
    if (ptr == NULL) {
        return -1;
    }
    memcpy(ptr, buf, size);
    int result = validate(ptr);
    if (result < 0) {
        free(ptr);
        return result;
    }
    process_data(ptr);
    free(ptr);
    return 0;
}'''

    print("=== Multi-Slice Test ===\n")
    print("Original code:")
    print(test_code)
    print()
    
    config = MultiSliceConfig(
        backward_depth=5,
        backward_window=15,
        forward_depth=3,
        forward_window=10,
        max_combined_tokens=512,
        sep_token="[SEP]"
    )
    
    slicer = MultiCodeSlicer(config)
    
    criterion_lines = [6]
    result = slicer.multi_slice(test_code, criterion_lines)
    
    print(f"Criterion lines: {criterion_lines}")
    print(f"\n--- Backward Slice ---")
    print(f"Lines: {sorted(result.backward_slice.included_lines)}")
    print(f"Code:\n{result.backward_slice.code}")
    
    print(f"\n--- Forward Slice ---")
    print(f"Lines: {sorted(result.forward_slice.included_lines)}")
    print(f"Code:\n{result.forward_slice.code}")
    
    print(f"\n--- Combined ---")
    print(f"All lines: {sorted(result.combined_lines)}")
    print(f"Combined code:\n{result.combined_code}")
    
    print("\n=== Batch Test ===")
    codes = [test_code, test_code]
    criterion_list = [[6], [9]]
    
    batch_results = multi_slice_batch(codes, criterion_list, config, n_jobs=1)
    print(f"Batch processed {len(batch_results)} samples")
    for i, r in enumerate(batch_results):
        print(f"  Sample {i}: {len(r.combined_lines)} combined lines")
