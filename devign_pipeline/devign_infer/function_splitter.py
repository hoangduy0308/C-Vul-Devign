"""
Function Splitter using Tree-sitter.

Splits C/C++ source files into individual functions for more accurate
vulnerability detection (model was trained on individual functions).
"""

import tree_sitter_c as tsc
from tree_sitter import Language, Parser
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FunctionInfo:
    """Information about an extracted function."""
    name: str
    code: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    return_type: str
    parameters: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "code": self.code,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "return_type": self.return_type,
            "parameters": self.parameters,
            "loc": self.end_line - self.start_line + 1
        }


class FunctionSplitter:
    """Extract individual functions from C/C++ source code using Tree-sitter."""
    
    def __init__(self):
        """Initialize the Tree-sitter parser for C."""
        self.language = Language(tsc.language())
        self.parser = Parser(self.language)
    
    def split_file(self, code: str) -> List[FunctionInfo]:
        """
        Split source code into individual functions.
        
        Args:
            code: Source code string
            
        Returns:
            List of FunctionInfo objects
        """
        tree = self.parser.parse(bytes(code, "utf-8"))
        functions = []
        
        self._extract_functions(tree.root_node, code, functions)
        
        return functions
    
    def split_file_from_path(self, file_path: str) -> List[FunctionInfo]:
        """
        Split a source file into individual functions.
        
        Args:
            file_path: Path to source file
            
        Returns:
            List of FunctionInfo objects
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        
        return self.split_file(code)
    
    def _extract_functions(
        self, 
        node, 
        code: str, 
        functions: List[FunctionInfo]
    ) -> None:
        """Recursively extract function definitions."""
        
        # Function definition nodes in C
        if node.type == 'function_definition':
            func_info = self._parse_function_node(node, code)
            if func_info:
                functions.append(func_info)
        
        # Recurse into children
        for child in node.children:
            self._extract_functions(child, code, functions)
    
    def _parse_function_node(self, node, code: str) -> Optional[FunctionInfo]:
        """Parse a function_definition node into FunctionInfo."""
        try:
            # Extract function code
            start_byte = node.start_byte
            end_byte = node.end_byte
            func_code = code[start_byte:end_byte]
            
            # Get line numbers (1-indexed)
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            # Extract function name and return type
            func_name = ""
            return_type = ""
            parameters = []
            
            for child in node.children:
                if child.type == 'function_declarator':
                    # Get function name from declarator
                    func_name = self._get_function_name(child, code)
                    parameters = self._get_parameters(child, code)
                elif child.type in ['type_identifier', 'primitive_type', 'sized_type_specifier']:
                    return_type = code[child.start_byte:child.end_byte]
                elif child.type == 'pointer_declarator':
                    # Handle pointer return types
                    func_name = self._get_function_name(child, code)
                    parameters = self._get_parameters(child, code)
            
            if not func_name:
                # Try to get name from the declarator directly
                declarator = node.child_by_field_name('declarator')
                if declarator:
                    func_name = self._get_function_name(declarator, code)
            
            if not func_name:
                return None
            
            return FunctionInfo(
                name=func_name,
                code=func_code,
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                return_type=return_type,
                parameters=parameters
            )
            
        except Exception as e:
            return None
    
    def _get_function_name(self, node, code: str) -> str:
        """Extract function name from a declarator node."""
        if node.type == 'identifier':
            return code[node.start_byte:node.end_byte]
        
        if node.type == 'function_declarator':
            # The first child is usually the identifier
            for child in node.children:
                if child.type == 'identifier':
                    return code[child.start_byte:child.end_byte]
                elif child.type == 'parenthesized_declarator':
                    return self._get_function_name(child, code)
        
        if node.type == 'pointer_declarator':
            for child in node.children:
                result = self._get_function_name(child, code)
                if result:
                    return result
        
        if node.type == 'parenthesized_declarator':
            for child in node.children:
                result = self._get_function_name(child, code)
                if result:
                    return result
        
        # Recurse into children
        for child in node.children:
            result = self._get_function_name(child, code)
            if result:
                return result
        
        return ""
    
    def _get_parameters(self, node, code: str) -> List[str]:
        """Extract parameter list from declarator."""
        params = []
        
        if node.type == 'function_declarator':
            for child in node.children:
                if child.type == 'parameter_list':
                    for param in child.children:
                        if param.type == 'parameter_declaration':
                            param_text = code[param.start_byte:param.end_byte]
                            params.append(param_text.strip())
        
        return params
    
    def get_file_structure(self, code: str) -> Dict[str, Any]:
        """
        Get the structure of a source file.
        
        Returns:
            Dictionary with file structure information
        """
        functions = self.split_file(code)
        
        return {
            "total_functions": len(functions),
            "functions": [f.to_dict() for f in functions],
            "total_lines": len(code.split('\n'))
        }


# Singleton instance
_splitter = None

def get_splitter() -> FunctionSplitter:
    """Get or create the singleton FunctionSplitter instance."""
    global _splitter
    if _splitter is None:
        _splitter = FunctionSplitter()
    return _splitter


def split_into_functions(code: str) -> List[FunctionInfo]:
    """Convenience function to split code into functions."""
    return get_splitter().split_file(code)


def split_file_into_functions(file_path: str) -> List[FunctionInfo]:
    """Convenience function to split a file into functions."""
    return get_splitter().split_file_from_path(file_path)


# Test
if __name__ == "__main__":
    test_code = '''
#include <stdio.h>
#include <string.h>

void vulnerable_func(char *input) {
    char buffer[64];
    strcpy(buffer, input);
    printf("%s\\n", buffer);
}

int safe_func(int a, int b) {
    return a + b;
}

char* another_func(const char* str) {
    char* result = malloc(strlen(str) + 1);
    if (result) {
        strcpy(result, str);
    }
    return result;
}

int main(int argc, char *argv[]) {
    vulnerable_func(argv[1]);
    return 0;
}
'''
    
    splitter = FunctionSplitter()
    functions = splitter.split_file(test_code)
    
    print(f"Found {len(functions)} functions:\n")
    for func in functions:
        print(f"Function: {func.name}")
        print(f"  Lines: {func.start_line}-{func.end_line}")
        print(f"  Return type: {func.return_type}")
        print(f"  Parameters: {func.parameters}")
        print(f"  Code preview: {func.code[:50]}...")
        print()
