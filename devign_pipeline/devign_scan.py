#!/usr/bin/env python3
"""
Devign Vulnerability Scanner CLI

Scan C/C++ source code for security vulnerabilities using BiGRU deep learning model.

Usage:
    # Scan single file
    python devign_scan.py scan file.c
    
    # Scan directory
    python devign_scan.py scan src/ --recursive
    
    # Scan git diff (for CI/CD)
    python devign_scan.py scan-diff --base origin/main
    
    # Output SARIF for GitHub Code Scanning
    python devign_scan.py scan src/ --output results.sarif --format sarif
    
    # Set custom threshold
    python devign_scan.py scan src/ --threshold 0.7
"""

import argparse
import sys
import os
import json
import glob
import subprocess
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

# Try to import tree-sitter for accurate comment stripping
try:
    import tree_sitter_c as tsc
    from tree_sitter import Language, Parser
    _TREE_SITTER_AVAILABLE = True
except ImportError:
    _TREE_SITTER_AVAILABLE = False

from devign_infer import VulnerabilityDetector, SARIFReporter, InferenceConfig
from devign_infer.config import find_model_path, find_vocab_path, validate_paths

SCRIPT_DIR = Path(__file__).parent.resolve()

# Cached tree-sitter parser
_ts_parser = None

def _get_ts_parser():
    """Get or create tree-sitter C parser."""
    global _ts_parser
    if not _TREE_SITTER_AVAILABLE:
        return None
    if _ts_parser is None:
        try:
            _ts_parser = Parser(Language(tsc.language()))
        except Exception:
            return None
    return _ts_parser


@dataclass
class ScanResult:
    """Result of scanning a file."""
    file_path: str
    vulnerable: bool
    probability: float
    risk_level: str
    dangerous_apis: List[str]
    error: Optional[str] = None


class DevignScanner:
    """Main scanner class for vulnerability detection."""
    
    C_EXTENSIONS = {'.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx'}
    
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        config_path: Optional[str] = None,
        threshold: float = 0.5,
        device: str = "auto"
    ):
        self.detector = VulnerabilityDetector(
            model_path=model_path,
            vocab_path=vocab_path,
            config_path=config_path,
            device=device,
            threshold=threshold
        )
        self.threshold = threshold
    
    def is_c_file(self, path: str) -> bool:
        """Check if file is a C/C++ source file."""
        return Path(path).suffix.lower() in self.C_EXTENSIONS
    
    def find_c_files(self, path: str, recursive: bool = True) -> List[str]:
        """Find all C/C++ files in directory."""
        path = Path(path)
        
        if path.is_file():
            return [str(path)] if self.is_c_file(str(path)) else []
        
        files = []
        if recursive:
            for ext in self.C_EXTENSIONS:
                files.extend(glob.glob(str(path / "**" / f"*{ext}"), recursive=True))
        else:
            for ext in self.C_EXTENSIONS:
                files.extend(glob.glob(str(path / f"*{ext}")))
        
        return sorted(set(files))
    
    def scan_file(self, file_path: str) -> ScanResult:
        """Scan a single file for vulnerabilities."""
        try:
            result = self.detector.analyze_file(file_path)
            return ScanResult(
                file_path=file_path,
                vulnerable=result.vulnerable,
                probability=result.probability,
                risk_level=result.risk_level,
                dangerous_apis=result.details.get("dangerous_apis_found", [])
            )
        except Exception as e:
            return ScanResult(
                file_path=file_path,
                vulnerable=False,
                probability=0.0,
                risk_level="ERROR",
                dangerous_apis=[],
                error=str(e)
            )
    
    def scan_files(
        self,
        files: List[str],
        show_progress: bool = True
    ) -> List[ScanResult]:
        """Scan multiple files."""
        results = []
        
        for i, file_path in enumerate(files):
            if show_progress:
                print(f"\rScanning [{i+1}/{len(files)}] {Path(file_path).name}...", end="", flush=True)
            
            result = self.scan_file(file_path)
            results.append(result)
        
        if show_progress:
            print()
        
        return results
    
    def scan_directory(
        self,
        directory: str,
        recursive: bool = True,
        show_progress: bool = True
    ) -> List[ScanResult]:
        """Scan all C files in directory."""
        files = self.find_c_files(directory, recursive)
        
        if not files:
            print(f"No C/C++ files found in {directory}")
            return []
        
        if show_progress:
            print(f"Found {len(files)} C/C++ files")
        
        return self.scan_files(files, show_progress)
    
    def get_git_diff_files(self, base: str = "origin/main") -> List[str]:
        """Get list of changed C files from git diff."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=ACMR", base],
                capture_output=True,
                text=True,
                check=True
            )
            
            files = result.stdout.strip().split('\n')
            return [f for f in files if f and self.is_c_file(f)]
        except subprocess.CalledProcessError as e:
            print(f"Git error: {e.stderr}")
            return []
    
    def scan_diff(
        self,
        base: str = "origin/main",
        show_progress: bool = True
    ) -> List[ScanResult]:
        """Scan only files changed in git diff."""
        files = self.get_git_diff_files(base)
        
        if not files:
            print(f"No C/C++ files changed since {base}")
            return []
        
        if show_progress:
            print(f"Scanning {len(files)} changed C/C++ files")
        
        return self.scan_files(files, show_progress)


def format_results_text(results: List[ScanResult], verbose: bool = False) -> str:
    """Format results as human-readable text."""
    lines = []
    
    vuln_count = sum(1 for r in results if r.vulnerable)
    error_count = sum(1 for r in results if r.error)
    
    lines.append("=" * 60)
    lines.append("DEVIGN VULNERABILITY SCAN RESULTS")
    lines.append("=" * 60)
    lines.append(f"Files scanned: {len(results)}")
    lines.append(f"Vulnerabilities found: {vuln_count}")
    if error_count:
        lines.append(f"Errors: {error_count}")
    lines.append("-" * 60)
    
    for result in results:
        if result.error:
            lines.append(f"[ERROR] {result.file_path}")
            lines.append(f"  ERROR: {result.error}")
        elif result.vulnerable:
            risk_marker = {
                "CRITICAL": "[!!!]",
                "HIGH": "[!!]", 
                "MEDIUM": "[!]",
                "LOW": "[~]"
            }.get(result.risk_level, "[?]")
            
            lines.append(f"{risk_marker} {result.file_path}")
            lines.append(f"   Risk: {result.risk_level} ({result.probability:.1%})")
            if result.dangerous_apis:
                lines.append(f"   APIs: {', '.join(result.dangerous_apis[:5])}")
        elif verbose:
            lines.append(f"[OK] {result.file_path} (safe)")
    
    lines.append("=" * 60)
    
    if vuln_count > 0:
        lines.append(f"WARNING: Found {vuln_count} potential vulnerabilities!")
    else:
        lines.append("OK: No vulnerabilities detected")
    
    return "\n".join(lines)


def format_results_json(results: List[ScanResult]) -> str:
    """Format results as JSON."""
    data = {
        "summary": {
            "files_scanned": len(results),
            "vulnerabilities_found": sum(1 for r in results if r.vulnerable),
            "errors": sum(1 for r in results if r.error)
        },
        "results": [
            {
                "file": r.file_path,
                "vulnerable": r.vulnerable,
                "probability": r.probability,
                "risk_level": r.risk_level,
                "dangerous_apis": r.dangerous_apis,
                "error": r.error
            }
            for r in results
        ]
    }
    return json.dumps(data, indent=2)


def _strip_all_comments(content: str) -> str:
    """
    Remove all C/C++ comments from file content, handling multi-line block comments.
    
    Uses tree-sitter for accurate comment detection when available (handles edge cases
    like raw string literals R"(...)"), falls back to regex otherwise.
    
    Args:
        content: Full file content as string
        
    Returns:
        Content with all comments replaced by spaces (to preserve line numbers)
    """
    # Try tree-sitter first for accurate parsing
    result = _strip_comments_tree_sitter(content)
    if result is not None:
        return result
    
    # Fallback to regex
    return _strip_comments_regex(content)


def _strip_comments_tree_sitter(content: str) -> Optional[str]:
    """Use tree-sitter to accurately identify and remove comments."""
    parser = _get_ts_parser()
    if parser is None:
        return None
    
    try:
        tree = parser.parse(bytes(content, 'utf8'))
        
        # Collect all comment ranges
        comment_ranges = []
        
        def find_comments(node):
            if node.type == 'comment':
                comment_ranges.append((node.start_byte, node.end_byte, node.text))
            for child in node.children:
                find_comments(child)
        
        find_comments(tree.root_node)
        
        if not comment_ranges:
            return content  # No comments found
        
        # Build result by replacing comments with appropriate whitespace
        result = []
        last_end = 0
        content_bytes = content.encode('utf8')
        
        for start, end, comment_text in sorted(comment_ranges):
            # Add content before this comment
            result.append(content_bytes[last_end:start].decode('utf8', errors='ignore'))
            # Replace comment with newlines to preserve line numbers
            newline_count = comment_text.count(b'\n')
            result.append('\n' * newline_count)
            last_end = end
        
        # Add remaining content after last comment
        result.append(content_bytes[last_end:].decode('utf8', errors='ignore'))
        
        return ''.join(result)
        
    except Exception:
        return None  # Fallback to regex


def _strip_comments_regex(content: str) -> str:
    """Fallback regex-based comment stripping."""
    # Pattern to match strings, single-line comments (with line continuation), or block comments
    # Order matters: strings first to avoid matching // or /* inside strings
    pattern = r'''
        (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')  # String literals
        |(?P<line_comment>//(?:[^\n\\]|\\.|\\\n)*)       # Single-line comment (handles \ continuation)
        |(?P<block_comment>/\*[\s\S]*?\*/)               # Block comment (non-greedy)
    '''
    
    def replacer(match):
        if match.group('string'):
            return match.group('string')  # Keep strings
        elif match.group('line_comment'):
            # Preserve newlines from line continuations to maintain line numbers
            comment = match.group('line_comment')
            return '\n' * comment.count('\n')
        elif match.group('block_comment'):
            # Replace block comment with same number of newlines to preserve line numbers
            return '\n' * match.group('block_comment').count('\n')
        return ''
    
    return re.sub(pattern, replacer, content, flags=re.VERBOSE)


def find_dangerous_api_lines(
    file_path: str, 
    dangerous_apis: List[str],
    file_content: Optional[str] = None,
    stripped_content: Optional[str] = None
) -> Tuple[int, int, List[int]]:
    """Find line numbers where dangerous APIs are used.
    
    Uses regex with word boundaries to avoid false positives from:
    - Comments (// TODO: fix strcpy) - including multi-line block comments
    - Variable names (my_strcpy)
    - String literals ("use strcpy carefully")
    
    Args:
        file_path: Path to the file
        dangerous_apis: List of API names to search for
        file_content: Optional pre-loaded file content to avoid redundant I/O
        stripped_content: Optional pre-stripped content (comments already removed)
    
    Returns:
        Tuple of (min_line, max_line, all_lines):
        - min_line, max_line: bounding box for backward compatibility
        - all_lines: list of specific line numbers with dangerous calls
    """
    import re
    
    if not dangerous_apis:
        return 1, 1, []
    
    try:
        # Get file content
        if file_content is None:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
        
        # Strip all comments (including multi-line block comments)
        if stripped_content is None:
            stripped_content = _strip_all_comments(file_content)
        
        lines = stripped_content.splitlines()
        
        found_lines = []
        for i, line in enumerate(lines, 1):
            for api in dangerous_apis:
                # Use word boundary regex to match function calls: api_name(
                # This avoids matching my_strcpy or strcpy_wrapper
                pattern = rf'\b{re.escape(api)}\s*\('
                if re.search(pattern, line):
                    found_lines.append(i)
                    break
        
        if found_lines:
            return min(found_lines), max(found_lines), found_lines
    except Exception:
        pass
    
    return 1, 1, []


def generate_sarif(
    results: List[ScanResult],
    base_path: str = ".",
    threshold: float = 0.5,
    max_cache_bytes: int = 50 * 1024 * 1024,  # 50MB default
    max_file_size: int = 1024 * 1024,  # Skip caching files > 1MB
) -> Dict[str, Any]:
    """Generate SARIF report from scan results.
    
    Args:
        results: List of scan results
        base_path: Base path for relative file paths
        threshold: Vulnerability threshold
        max_cache_bytes: Maximum total bytes to cache (default 50MB)
        max_file_size: Skip caching files larger than this (default 1MB)
    """
    from collections import OrderedDict
    
    reporter = SARIFReporter(base_path=base_path)
    
    # LRU-style cache with byte-size limit to avoid memory issues with large files
    # Stores: path -> (content, stripped_content, size_bytes)
    file_cache: OrderedDict[str, tuple] = OrderedDict()
    cache_bytes = 0
    
    # Temporary cache for large files (single file at a time to avoid memory bloat)
    # This prevents re-reading the same large file for consecutive findings
    large_file_cache: Dict[str, tuple] = {}  # path -> (content, stripped)
    
    for result in results:
        if result.vulnerable and not result.error:
            # Get file content from cache or read from disk
            if result.file_path not in file_cache:
                try:
                    # Check large file cache first (for consecutive findings in same large file)
                    if result.file_path in large_file_cache:
                        content, stripped = large_file_cache[result.file_path]
                        start_line, end_line, all_lines = find_dangerous_api_lines(
                            result.file_path, 
                            result.dangerous_apis,
                            file_content=content,
                            stripped_content=stripped
                        )
                        reporter.add_finding(
                            file_path=result.file_path,
                            probability=result.probability,
                            risk_level=result.risk_level,
                            dangerous_apis=result.dangerous_apis,
                            start_line=start_line,
                            end_line=end_line,
                            threshold=threshold,
                            all_lines=all_lines
                        )
                        continue
                    
                    with open(result.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    content_size = len(content.encode('utf-8', errors='ignore'))
                    
                    # Skip LRU caching very large files, but use temporary cache
                    if content_size > max_file_size:
                        stripped = _strip_all_comments(content)
                        
                        # Cache for consecutive findings in same large file
                        # Clear previous large file to avoid memory bloat
                        large_file_cache.clear()
                        large_file_cache[result.file_path] = (content, stripped)
                        
                        start_line, end_line, all_lines = find_dangerous_api_lines(
                            result.file_path, 
                            result.dangerous_apis,
                            file_content=content,
                            stripped_content=stripped
                        )
                        reporter.add_finding(
                            file_path=result.file_path,
                            probability=result.probability,
                            risk_level=result.risk_level,
                            dangerous_apis=result.dangerous_apis,
                            start_line=start_line,
                            end_line=end_line,
                            threshold=threshold,
                            all_lines=all_lines
                        )
                        continue
                    
                    stripped = _strip_all_comments(content)
                    entry_size = content_size * 2  # Approximate: raw + stripped
                    
                    # Evict oldest entries until we have space
                    while cache_bytes + entry_size > max_cache_bytes and file_cache:
                        _, evicted = file_cache.popitem(last=False)
                        cache_bytes -= evicted[2]
                    
                    file_cache[result.file_path] = (content, stripped, entry_size)
                    cache_bytes += entry_size
                    
                except PermissionError as e:
                    # Critical error - log and skip this finding entirely
                    print(f"Warning: Permission denied reading {result.file_path}: {e}", file=sys.stderr)
                    continue  # Skip this finding, don't cache invalid data
                except OSError as e:
                    # File system errors - log and skip
                    print(f"Warning: Cannot read {result.file_path}: {e}", file=sys.stderr)
                    continue
                except Exception as e:
                    # Unexpected errors - log with details for debugging
                    print(f"Warning: Error processing {result.file_path}: {type(e).__name__}: {e}", file=sys.stderr)
                    continue  # Skip rather than pollute cache with empty data
            else:
                # Move to end (most recently used)
                file_cache.move_to_end(result.file_path)
            
            content, stripped, _ = file_cache[result.file_path]
            
            start_line, end_line, all_lines = find_dangerous_api_lines(
                result.file_path, 
                result.dangerous_apis,
                file_content=content,
                stripped_content=stripped
            )
            reporter.add_finding(
                file_path=result.file_path,
                probability=result.probability,
                risk_level=result.risk_level,
                dangerous_apis=result.dangerous_apis,
                start_line=start_line,
                end_line=end_line,
                threshold=threshold,
                all_lines=all_lines
            )
    
    return reporter.generate()


def cmd_scan(args):
    """Handle scan command."""
    model_path = find_model_path(args.model, SCRIPT_DIR)
    vocab_path = find_vocab_path(args.vocab, SCRIPT_DIR)
    
    errors = validate_paths(model_path, vocab_path)
    if errors:
        print("Error: Cannot find required files:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print("\nSearched locations (in priority order):", file=sys.stderr)
        print("  1. CLI arguments (--model, --vocab)", file=sys.stderr)
        print("  2. Environment variables (MODEL_PATH, VOCAB_PATH)", file=sys.stderr)
        print(f"  3. Relative to script: {SCRIPT_DIR}", file=sys.stderr)
        print("  4. Current working directory", file=sys.stderr)
        return 1
    
    scanner = DevignScanner(
        model_path=model_path,
        vocab_path=vocab_path,
        config_path=args.config,
        threshold=args.threshold,
        device=args.device
    )
    
    if os.path.isfile(args.path):
        results = [scanner.scan_file(args.path)]
    else:
        results = scanner.scan_directory(
            args.path,
            recursive=args.recursive,
            show_progress=not args.quiet
        )
    
    if not results:
        print("No files to scan")
        return 0
    
    if args.format == "sarif":
        sarif = generate_sarif(results, base_path=args.path, threshold=args.threshold)
        output = json.dumps(sarif, indent=2)
    elif args.format == "json":
        output = format_results_json(results)
    else:
        output = format_results_text(results, verbose=args.verbose)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        if not args.quiet:
            print(f"Results saved to {args.output}")
    else:
        print(output)
    
    vuln_count = sum(1 for r in results if r.vulnerable)
    
    if args.fail_on_findings and vuln_count > 0:
        return 1
    
    return 0


def cmd_scan_diff(args):
    """Handle scan-diff command (for CI/CD)."""
    model_path = find_model_path(args.model, SCRIPT_DIR)
    vocab_path = find_vocab_path(args.vocab, SCRIPT_DIR)
    
    errors = validate_paths(model_path, vocab_path)
    if errors:
        print("Error: Cannot find required files:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print("\nSearched locations (in priority order):", file=sys.stderr)
        print("  1. CLI arguments (--model, --vocab)", file=sys.stderr)
        print("  2. Environment variables (MODEL_PATH, VOCAB_PATH)", file=sys.stderr)
        print(f"  3. Relative to script: {SCRIPT_DIR}", file=sys.stderr)
        print("  4. Current working directory", file=sys.stderr)
        return 1
    
    scanner = DevignScanner(
        model_path=model_path,
        vocab_path=vocab_path,
        config_path=args.config,
        threshold=args.threshold,
        device=args.device
    )
    
    results = scanner.scan_diff(
        base=args.base,
        show_progress=not args.quiet
    )
    
    if not results:
        print("No changed C/C++ files to scan")
        return 0
    
    if args.format == "sarif":
        sarif = generate_sarif(results, base_path=".", threshold=args.threshold)
        output = json.dumps(sarif, indent=2)
    elif args.format == "json":
        output = format_results_json(results)
    else:
        output = format_results_text(results, verbose=args.verbose)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        if not args.quiet:
            print(f"Results saved to {args.output}")
    else:
        print(output)
    
    vuln_count = sum(1 for r in results if r.vulnerable)
    
    if args.fail_on_findings and vuln_count > 0:
        return 1
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Devign Vulnerability Scanner - AI-powered C/C++ security scanning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Scan a directory
    python devign_scan.py scan src/
    
    # Scan with SARIF output for GitHub
    python devign_scan.py scan src/ -o results.sarif -f sarif
    
    # Scan only changed files (for CI)
    python devign_scan.py scan-diff --base origin/main -o results.sarif -f sarif
    
    # Fail CI if vulnerabilities found
    python devign_scan.py scan src/ --fail-on-findings
        """
    )
    
    parser.add_argument('--model', '-m', type=str,
                       default=None,
                       help='Path to model checkpoint (auto-detected if not specified)')
    parser.add_argument('--vocab', '-v', type=str,
                       default=None,
                       help='Path to vocabulary file (auto-detected if not specified)')
    parser.add_argument('--config', '-c', type=str,
                       help='Path to model config file')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Vulnerability probability threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    scan_parser = subparsers.add_parser('scan', help='Scan files or directory')
    scan_parser.add_argument('path', type=str, help='File or directory to scan')
    scan_parser.add_argument('--recursive', '-r', action='store_true', default=True,
                            help='Scan directories recursively (default: true)')
    scan_parser.add_argument('--no-recursive', action='store_false', dest='recursive',
                            help='Do not scan recursively')
    scan_parser.add_argument('--output', '-o', type=str, help='Output file path')
    scan_parser.add_argument('--format', '-f', type=str, default='text',
                            choices=['text', 'json', 'sarif'],
                            help='Output format (default: text)')
    scan_parser.add_argument('--verbose', action='store_true',
                            help='Show all files including safe ones')
    scan_parser.add_argument('--quiet', '-q', action='store_true',
                            help='Suppress progress output')
    scan_parser.add_argument('--fail-on-findings', action='store_true',
                            help='Exit with code 1 if vulnerabilities found')
    scan_parser.set_defaults(func=cmd_scan)
    
    diff_parser = subparsers.add_parser('scan-diff', help='Scan git diff files (for CI/CD)')
    diff_parser.add_argument('--base', '-b', type=str, default='origin/main',
                            help='Git base ref to diff against (default: origin/main)')
    diff_parser.add_argument('--output', '-o', type=str, help='Output file path')
    diff_parser.add_argument('--format', '-f', type=str, default='text',
                            choices=['text', 'json', 'sarif'],
                            help='Output format (default: text)')
    diff_parser.add_argument('--verbose', action='store_true',
                            help='Show all files including safe ones')
    diff_parser.add_argument('--quiet', '-q', action='store_true',
                            help='Suppress progress output')
    diff_parser.add_argument('--fail-on-findings', action='store_true',
                            help='Exit with code 1 if vulnerabilities found')
    diff_parser.set_defaults(func=cmd_scan_diff)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        exit_code = args.func(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
