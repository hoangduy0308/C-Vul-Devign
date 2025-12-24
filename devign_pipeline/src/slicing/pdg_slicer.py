"""PDG-based Code Slicer for vulnerability detection.

Uses Program Dependence Graph (PDG) for accurate slicing instead of
window-based approach. Follows actual data/control dependencies.

Key differences from window-based slicer:
1. Slices based on dependencies, not fixed window
2. Stops when no more dependencies exist
3. Produces shorter, more focused slices (~30-50 tokens)
4. Better for vulnerability pattern detection
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from enum import Enum

import sys
sys.path.insert(0, 'F:/Work/C Vul Devign/devign_pipeline')

from src.ast.parser import CFamilyParser, ParseResult
from src.graphs.cfg import CFG, build_cfg
from src.graphs.dfg import DFG, build_dfg
from src.graphs.pdg import PDG, PDGBuilder, build_pdg


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
    control_predicate_only: bool = True  # Only include condition lines, not full blocks
    
    # Output limits
    max_lines: int = 15              # Hard cap on output lines
    max_tokens: int = 100            # Hard cap on tokens (estimated)
    
    # Fallback settings
    fallback_window: int = 3         # Fallback window if PDG fails (reduced from 15)
    
    # Preprocessing
    remove_comments: bool = True
    normalize_output: bool = True


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
            control_lines = pdg.backward_slice(
                criterion_lines,
                max_depth=self.config.backward_depth,
                include_data=False,
                include_control=True
            )
            
            # If predicate-only mode, filter to only condition lines
            if self.config.control_predicate_only:
                control_lines = {
                    line for line in control_lines
                    if line in pdg.nodes and 
                    pdg.nodes[line].node_type == 'condition'
                }
        
        # Combine and apply limits
        all_lines = data_lines | control_lines | set(criterion_lines)
        limited_lines = self._apply_limits(code, all_lines, criterion_lines)
        
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
        limited_lines = self._apply_limits(code, all_lines, criterion_lines)
        
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
            backward_control = pdg.backward_slice(
                criterion_lines,
                max_depth=self.config.backward_depth,
                include_data=False,
                include_control=True
            )
            if self.config.control_predicate_only:
                backward_control = {
                    line for line in backward_control
                    if line in pdg.nodes and 
                    pdg.nodes[line].node_type == 'condition'
                }
        
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
        
        limited_lines = self._apply_limits(code, all_lines, criterion_lines)
        
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
        
        return self._build_result(
            code, included, criterion_lines,
            self.config.slice_type, set(), set(), True
        )
    
    def _apply_limits(self, code: str, lines: Set[int],
                      criterion_lines: List[int]) -> Set[int]:
        """Apply max_lines and max_tokens limits"""
        if len(lines) <= self.config.max_lines:
            return lines
        
        # Priority: criterion lines first, then by distance
        criterion_set = set(criterion_lines)
        
        # Sort by distance from nearest criterion
        def distance_from_criterion(line: int) -> int:
            return min(abs(line - c) for c in criterion_lines)
        
        sorted_lines = sorted(lines, key=distance_from_criterion)
        
        # Keep criterion lines + closest lines up to limit
        result: Set[int] = criterion_set & lines
        
        for line in sorted_lines:
            if len(result) >= self.config.max_lines:
                break
            result.add(line)
        
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
        
        # Extract statements
        statements = []
        for line_num in sorted(valid_lines):
            line = lines[line_num - 1]
            if line.strip():
                statements.append(line)
        
        # Build code
        if self.config.normalize_output:
            sliced_code = self._normalize_code(statements)
        else:
            sliced_code = '\n'.join(statements)
        
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
