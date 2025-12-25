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
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from devign_infer import VulnerabilityDetector, SARIFReporter, InferenceConfig
from devign_infer.config import find_model_path, find_vocab_path, validate_paths

SCRIPT_DIR = Path(__file__).parent.resolve()


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


def generate_sarif(
    results: List[ScanResult],
    base_path: str = ".",
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Generate SARIF report from scan results."""
    reporter = SARIFReporter(base_path=base_path)
    
    for result in results:
        if result.vulnerable and not result.error:
            reporter.add_finding(
                file_path=result.file_path,
                probability=result.probability,
                risk_level=result.risk_level,
                dangerous_apis=result.dangerous_apis,
                threshold=threshold
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
