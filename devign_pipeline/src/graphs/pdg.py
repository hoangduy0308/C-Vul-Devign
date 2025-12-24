"""Program Dependence Graph (PDG) = CFG + DFG combined.

PDG combines control dependencies (from CFG) and data dependencies (from DFG)
into a single graph for more accurate program slicing.

Based on SySeVR approach: https://github.com/SySeVR/SySeVR
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum
from collections import deque
import networkx as nx

from .cfg import CFG, BasicBlock, EdgeType, build_cfg
from .dfg import DFG, DFGEdge, VariableAccess, UseType, build_dfg
from ..ast.parser import ParseResult


class PDGEdgeType(Enum):
    """Edge types in PDG"""
    DATA_DEP = 'data'       # Data dependency (def-use)
    CONTROL_DEP = 'control' # Control dependency
    CALL = 'call'           # Function call
    PARAM = 'param'         # Parameter passing


@dataclass
class PDGNode:
    """Node in PDG representing a statement/line"""
    id: int
    line: int
    node_type: str          # 'statement', 'condition', 'call', 'declaration'
    text: str               # Statement text
    variables_def: List[str] = field(default_factory=list)  # Variables defined
    variables_use: List[str] = field(default_factory=list)  # Variables used


@dataclass
class PDGEdge:
    """Edge in PDG"""
    from_id: int
    to_id: int
    edge_type: PDGEdgeType
    label: str = ''         # Variable name for data deps


@dataclass
class PDG:
    """Program Dependence Graph"""
    nodes: Dict[int, PDGNode]  # line -> PDGNode
    edges: List[PDGEdge]
    entry_line: int
    exit_lines: List[int]
    
    def get_predecessors(self, line: int, edge_type: Optional[PDGEdgeType] = None) -> Set[int]:
        """Get predecessor lines (nodes that this node depends on)"""
        preds = set()
        for edge in self.edges:
            if edge.to_id == line:
                if edge_type is None or edge.edge_type == edge_type:
                    preds.add(edge.from_id)
        return preds
    
    def get_successors(self, line: int, edge_type: Optional[PDGEdgeType] = None) -> Set[int]:
        """Get successor lines (nodes that depend on this node)"""
        succs = set()
        for edge in self.edges:
            if edge.from_id == line:
                if edge_type is None or edge.edge_type == edge_type:
                    succs.add(edge.to_id)
        return succs
    
    def get_data_predecessors(self, line: int) -> Set[int]:
        """Get lines with data dependencies leading to this line"""
        return self.get_predecessors(line, PDGEdgeType.DATA_DEP)
    
    def get_control_predecessors(self, line: int) -> Set[int]:
        """Get lines with control dependencies (conditions guarding this line)"""
        return self.get_predecessors(line, PDGEdgeType.CONTROL_DEP)
    
    def backward_slice(self, criterion_lines: List[int], 
                       max_depth: int = 3,
                       include_data: bool = True,
                       include_control: bool = True) -> Set[int]:
        """
        Backward slicing from criterion lines.
        
        Unlike window-based slicing, this follows actual dependencies
        and stops when no more dependencies exist.
        
        Args:
            criterion_lines: Starting lines for slicing
            max_depth: Maximum dependency hops (prevents infinite loops)
            include_data: Include data dependencies
            include_control: Include control dependencies
            
        Returns:
            Set of line numbers in the slice
        """
        slice_lines: Set[int] = set(criterion_lines)
        visited: Set[int] = set()
        
        # BFS with depth tracking
        queue = deque([(line, 0) for line in criterion_lines])
        
        while queue:
            current_line, depth = queue.popleft()
            
            if current_line in visited:
                continue
            visited.add(current_line)
            
            if depth >= max_depth:
                continue
            
            # Get predecessors based on config
            preds: Set[int] = set()
            
            if include_data:
                preds.update(self.get_data_predecessors(current_line))
            
            if include_control:
                preds.update(self.get_control_predecessors(current_line))
            
            for pred_line in preds:
                if pred_line not in visited:
                    slice_lines.add(pred_line)
                    queue.append((pred_line, depth + 1))
        
        return slice_lines
    
    def forward_slice(self, criterion_lines: List[int],
                      max_depth: int = 2,
                      include_data: bool = True,
                      include_control: bool = False) -> Set[int]:
        """
        Forward slicing from criterion lines.
        
        Args:
            criterion_lines: Starting lines
            max_depth: Maximum dependency hops
            include_data: Include data dependencies
            include_control: Include control dependencies (usually False for forward)
            
        Returns:
            Set of line numbers affected by criterion
        """
        slice_lines: Set[int] = set(criterion_lines)
        visited: Set[int] = set()
        
        queue = deque([(line, 0) for line in criterion_lines])
        
        while queue:
            current_line, depth = queue.popleft()
            
            if current_line in visited:
                continue
            visited.add(current_line)
            
            if depth >= max_depth:
                continue
            
            succs: Set[int] = set()
            
            if include_data:
                succs.update(self.get_successors(current_line, PDGEdgeType.DATA_DEP))
            
            if include_control:
                succs.update(self.get_successors(current_line, PDGEdgeType.CONTROL_DEP))
            
            for succ_line in succs:
                if succ_line not in visited:
                    slice_lines.add(succ_line)
                    queue.append((succ_line, depth + 1))
        
        return slice_lines
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX for visualization"""
        G = nx.DiGraph()
        
        for line, node in self.nodes.items():
            G.add_node(line, 
                       node_type=node.node_type,
                       text=node.text[:50],
                       vars_def=node.variables_def,
                       vars_use=node.variables_use)
        
        for edge in self.edges:
            G.add_edge(edge.from_id, edge.to_id,
                       edge_type=edge.edge_type.value,
                       label=edge.label)
        
        return G
    
    def is_empty(self) -> bool:
        return not self.nodes


class PDGBuilder:
    """Build PDG from CFG and DFG"""
    
    def __init__(self):
        self.nodes: Dict[int, PDGNode] = {}
        self.edges: List[PDGEdge] = []
    
    def build(self, parse_result: ParseResult,
              cfg: Optional[CFG] = None,
              dfg: Optional[DFG] = None) -> Optional[PDG]:
        """
        Build PDG from parse result.
        
        Args:
            parse_result: Parsed AST
            cfg: Optional precomputed CFG
            dfg: Optional precomputed DFG
            
        Returns:
            PDG or None if building fails
        """
        if parse_result is None or not parse_result.nodes:
            return None
        
        self.nodes = {}
        self.edges = []
        
        # Build CFG if not provided
        if cfg is None or cfg.is_empty():
            cfg = build_cfg(parse_result)
        
        # Build DFG if not provided
        if dfg is None or dfg.is_empty():
            dfg = build_dfg(parse_result)
        
        if cfg is None and dfg is None:
            return None
        
        # Create nodes from source lines
        self._create_nodes_from_ast(parse_result)
        
        # Add control dependencies from CFG
        if cfg is not None and not cfg.is_empty():
            self._add_control_dependencies(cfg, parse_result)
        
        # Add data dependencies from DFG
        if dfg is not None and not dfg.is_empty():
            self._add_data_dependencies(dfg)
        
        # Determine entry/exit
        entry_line = min(self.nodes.keys()) if self.nodes else 1
        exit_lines = self._find_exit_lines(parse_result)
        
        return PDG(
            nodes=self.nodes,
            edges=self.edges,
            entry_line=entry_line,
            exit_lines=exit_lines
        )
    
    def _create_nodes_from_ast(self, parse_result: ParseResult):
        """Create PDG nodes from AST nodes"""
        line_to_text: Dict[int, str] = {}
        line_to_type: Dict[int, str] = {}
        line_to_vars_def: Dict[int, List[str]] = {}
        line_to_vars_use: Dict[int, List[str]] = {}
        
        for node in parse_result.nodes:
            line = node.start_line
            
            # Skip trivial nodes
            if node.node_type in ('translation_unit', 'compound_statement', 
                                  '(', ')', '{', '}', ';', ','):
                continue
            
            # Determine node type
            if node.node_type in ('if_statement', 'while_statement', 
                                  'for_statement', 'switch_statement'):
                node_type = 'condition'
            elif node.node_type == 'call_expression':
                node_type = 'call'
            elif node.node_type in ('declaration', 'init_declarator'):
                node_type = 'declaration'
            elif node.node_type == 'return_statement':
                node_type = 'return'
            else:
                node_type = 'statement'
            
            # Store best representation for each line
            if line not in line_to_text or len(node.text) > len(line_to_text[line]):
                line_to_text[line] = node.text.strip()
                line_to_type[line] = node_type
        
        # Create PDG nodes
        for line in sorted(line_to_text.keys()):
            self.nodes[line] = PDGNode(
                id=line,
                line=line,
                node_type=line_to_type.get(line, 'statement'),
                text=line_to_text[line],
                variables_def=line_to_vars_def.get(line, []),
                variables_use=line_to_vars_use.get(line, [])
            )
    
    def _add_control_dependencies(self, cfg: CFG, parse_result: ParseResult):
        """
        Add control dependency edges.
        
        A node Y is control-dependent on node X if:
        - X is a condition (if/while/for)
        - Y is in a branch controlled by X
        """
        # Find condition blocks and their controlled regions
        for block in cfg.blocks:
            if block.block_type in ('condition', 'loop_header'):
                cond_line = block.start_line
                
                # Find all lines controlled by this condition
                controlled_lines = self._get_controlled_lines(cfg, block)
                
                # Add control dependency edges
                for line in controlled_lines:
                    if line != cond_line and line in self.nodes:
                        self._add_edge(cond_line, line, PDGEdgeType.CONTROL_DEP)
    
    def _get_controlled_lines(self, cfg: CFG, cond_block: BasicBlock) -> Set[int]:
        """Get all lines controlled by a condition block"""
        controlled: Set[int] = set()
        visited: Set[int] = set()
        
        # BFS through successors
        successors = cfg.get_successors(cond_block.id)
        queue = deque(successors)
        
        while queue:
            block_id = queue.popleft()
            
            if block_id in visited:
                continue
            visited.add(block_id)
            
            block = cfg.get_block_by_id(block_id)
            if block is None:
                continue
            
            # Don't go past condition's end line (approximate scope)
            if block.start_line > cond_block.end_line:
                continue
            
            # Add lines from this block
            for line in range(block.start_line, block.end_line + 1):
                controlled.add(line)
            
            # Continue to successors within scope
            for succ_id in cfg.get_successors(block_id):
                if succ_id not in visited:
                    succ_block = cfg.get_block_by_id(succ_id)
                    if succ_block and succ_block.start_line <= cond_block.end_line:
                        queue.append(succ_id)
        
        return controlled
    
    def _add_data_dependencies(self, dfg: DFG):
        """
        Add data dependency edges from DFG.
        
        Edge from line A to line B if:
        - A defines a variable
        - B uses that variable
        - A's definition reaches B
        """
        # Group DFG nodes by line
        line_to_dfg_nodes: Dict[int, List[int]] = {}
        for idx, node in enumerate(dfg.nodes):
            if node.line not in line_to_dfg_nodes:
                line_to_dfg_nodes[node.line] = []
            line_to_dfg_nodes[node.line].append(idx)
        
        # Convert DFG edges to PDG edges (line-level)
        seen_edges: Set[Tuple[int, int, str]] = set()
        
        for edge in dfg.edges:
            if edge.from_idx >= len(dfg.nodes) or edge.to_idx >= len(dfg.nodes):
                continue
            
            from_node = dfg.nodes[edge.from_idx]
            to_node = dfg.nodes[edge.to_idx]
            
            from_line = from_node.line
            to_line = to_node.line
            var_name = from_node.var_name
            
            # Skip self-loops
            if from_line == to_line:
                continue
            
            # Avoid duplicates
            edge_key = (from_line, to_line, var_name)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            
            # Add data dependency edge
            if from_line in self.nodes and to_line in self.nodes:
                self._add_edge(from_line, to_line, PDGEdgeType.DATA_DEP, var_name)
                
                # Update node variable info
                if var_name not in self.nodes[from_line].variables_def:
                    self.nodes[from_line].variables_def.append(var_name)
                if var_name not in self.nodes[to_line].variables_use:
                    self.nodes[to_line].variables_use.append(var_name)
    
    def _add_edge(self, from_line: int, to_line: int, 
                  edge_type: PDGEdgeType, label: str = ''):
        """Add edge avoiding duplicates"""
        # Check for duplicate
        for edge in self.edges:
            if (edge.from_id == from_line and 
                edge.to_id == to_line and 
                edge.edge_type == edge_type):
                return
        
        self.edges.append(PDGEdge(
            from_id=from_line,
            to_id=to_line,
            edge_type=edge_type,
            label=label
        ))
    
    def _find_exit_lines(self, parse_result: ParseResult) -> List[int]:
        """Find return statement lines"""
        exit_lines = []
        for node in parse_result.nodes:
            if node.node_type == 'return_statement':
                exit_lines.append(node.start_line)
        return exit_lines if exit_lines else [max(self.nodes.keys())] if self.nodes else [1]


def build_pdg(parse_result: ParseResult,
              cfg: Optional[CFG] = None,
              dfg: Optional[DFG] = None) -> Optional[PDG]:
    """Convenience function to build PDG"""
    builder = PDGBuilder()
    return builder.build(parse_result, cfg, dfg)


def serialize_pdg(pdg: PDG) -> dict:
    """Serialize PDG to dict"""
    if pdg is None:
        return {}
    
    return {
        'nodes': {
            line: {
                'id': node.id,
                'line': node.line,
                'node_type': node.node_type,
                'text': node.text,
                'variables_def': node.variables_def,
                'variables_use': node.variables_use,
            }
            for line, node in pdg.nodes.items()
        },
        'edges': [
            {
                'from_id': e.from_id,
                'to_id': e.to_id,
                'edge_type': e.edge_type.value,
                'label': e.label,
            }
            for e in pdg.edges
        ],
        'entry_line': pdg.entry_line,
        'exit_lines': pdg.exit_lines,
    }


def deserialize_pdg(data: dict) -> Optional[PDG]:
    """Deserialize PDG from dict"""
    if not data or 'nodes' not in data:
        return None
    
    edge_type_map = {e.value: e for e in PDGEdgeType}
    
    nodes = {
        int(line): PDGNode(
            id=n['id'],
            line=n['line'],
            node_type=n['node_type'],
            text=n['text'],
            variables_def=n.get('variables_def', []),
            variables_use=n.get('variables_use', []),
        )
        for line, n in data['nodes'].items()
    }
    
    edges = [
        PDGEdge(
            from_id=e['from_id'],
            to_id=e['to_id'],
            edge_type=edge_type_map.get(e['edge_type'], PDGEdgeType.DATA_DEP),
            label=e.get('label', ''),
        )
        for e in data.get('edges', [])
    ]
    
    return PDG(
        nodes=nodes,
        edges=edges,
        entry_line=data.get('entry_line', 1),
        exit_lines=data.get('exit_lines', []),
    )


if __name__ == "__main__":
    # Test code
    import sys
    sys.path.insert(0, 'F:/Work/C Vul Devign/devign_pipeline')
    from src.ast.parser import CFamilyParser
    
    test_code = '''int process(char *buf, int size) {
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
    
    print("=== PDG Test ===\n")
    print("Source code:")
    print(test_code)
    print()
    
    parser = CFamilyParser()
    parse_result = parser.parse_with_fallback(test_code)
    
    if parse_result:
        pdg = build_pdg(parse_result)
        
        if pdg:
            print(f"PDG nodes: {len(pdg.nodes)}")
            print(f"PDG edges: {len(pdg.edges)}")
            print()
            
            print("Nodes:")
            for line, node in sorted(pdg.nodes.items()):
                print(f"  L{line}: [{node.node_type}] {node.text[:40]}...")
            print()
            
            print("Data dependencies:")
            for edge in pdg.edges:
                if edge.edge_type == PDGEdgeType.DATA_DEP:
                    print(f"  L{edge.from_id} -> L{edge.to_id} ({edge.label})")
            print()
            
            print("Control dependencies:")
            for edge in pdg.edges:
                if edge.edge_type == PDGEdgeType.CONTROL_DEP:
                    print(f"  L{edge.from_id} -> L{edge.to_id}")
            print()
            
            # Test backward slicing
            criterion = [7]  # Line with validate() call
            print(f"=== Backward slice from line {criterion} ===")
            slice_lines = pdg.backward_slice(criterion, max_depth=3)
            print(f"Slice lines: {sorted(slice_lines)}")
            print(f"Slice size: {len(slice_lines)} lines")
