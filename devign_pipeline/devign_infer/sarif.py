"""
SARIF Reporter - Generate SARIF format reports for GitHub Code Scanning.

SARIF (Static Analysis Results Interchange Format) is the standard format
used by GitHub Code Scanning to display security alerts.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class Location:
    """Source code location."""
    file_path: str
    start_line: int = 1
    end_line: int = 1
    start_column: int = 1
    end_column: int = 1


@dataclass
class VulnerabilityFinding:
    """A single vulnerability finding."""
    rule_id: str
    message: str
    location: Location
    severity: str  # "error", "warning", "note"
    probability: float
    risk_level: str
    dangerous_apis: List[str]
    
    def fingerprint(self) -> str:
        """Generate unique fingerprint for deduplication."""
        content = f"{self.rule_id}:{self.location.file_path}:{self.location.start_line}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]


class SARIFReporter:
    """Generate SARIF format reports for GitHub Code Scanning."""
    
    TOOL_NAME = "devign-vuln-detector"
    TOOL_VERSION = "1.0.0"
    SARIF_VERSION = "2.1.0"
    SARIF_SCHEMA = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
    
    RULES = {
        "VULN001": {
            "id": "VULN001",
            "name": "PotentialVulnerability",
            "shortDescription": {"text": "Potential security vulnerability detected"},
            "fullDescription": {
                "text": "The BiGRU vulnerability detection model has identified code patterns that may indicate a security vulnerability. This could include buffer overflows, use-after-free, null pointer dereferences, or other common vulnerability types."
            },
            "helpUri": "https://cwe.mitre.org/",
            "help": {
                "text": "Review the flagged code for potential security issues. Common patterns include:\n- Buffer overflows from unsafe string functions\n- Memory corruption from improper memory management\n- Integer overflows in size calculations\n- Use of deprecated/dangerous APIs",
                "markdown": "## Potential Vulnerability\n\nThe AI model detected patterns associated with security vulnerabilities.\n\n### Common causes:\n- Buffer overflows from `strcpy`, `sprintf`, `gets`\n- Memory corruption from improper `malloc`/`free`\n- Integer overflows\n- Use of dangerous APIs\n\n### Recommended actions:\n1. Review the flagged code section\n2. Check for proper input validation\n3. Verify memory bounds\n4. Consider using safer alternatives"
            },
            "defaultConfiguration": {
                "level": "warning"
            },
            "properties": {
                "tags": ["security", "vulnerability", "ai-detected"],
                "precision": "medium",
                "problem.severity": "warning"
            }
        },
        "VULN002": {
            "id": "VULN002",
            "name": "HighRiskVulnerability",
            "shortDescription": {"text": "High-risk security vulnerability detected"},
            "fullDescription": {
                "text": "The model has detected code with high probability of containing a security vulnerability. Immediate review is recommended."
            },
            "helpUri": "https://cwe.mitre.org/",
            "help": {
                "text": "This code has been flagged as high-risk by the vulnerability detection model. Priority review is recommended.",
                "markdown": "## High-Risk Vulnerability\n\n⚠️ **This code has a high probability of containing a security vulnerability.**\n\n### Immediate actions:\n1. Do not deploy without review\n2. Check for dangerous function calls\n3. Verify all input validation\n4. Consider security audit"
            },
            "defaultConfiguration": {
                "level": "error"
            },
            "properties": {
                "tags": ["security", "vulnerability", "high-risk", "ai-detected"],
                "precision": "high",
                "problem.severity": "error"
            }
        },
        "VULN003": {
            "id": "VULN003",
            "name": "DangerousAPIUsage",
            "shortDescription": {"text": "Dangerous API function detected"},
            "fullDescription": {
                "text": "The code uses functions known to be dangerous or deprecated due to security concerns."
            },
            "helpUri": "https://wiki.sei.cmu.edu/confluence/display/c/SEI+CERT+C+Coding+Standard",
            "help": {
                "text": "Consider replacing dangerous functions with safer alternatives.",
                "markdown": "## Dangerous API Usage\n\n| Dangerous | Safer Alternative |\n|-----------|-------------------|\n| `strcpy` | `strncpy`, `strlcpy` |\n| `sprintf` | `snprintf` |\n| `gets` | `fgets` |\n| `scanf` | `fgets` + parsing |"
            },
            "defaultConfiguration": {
                "level": "warning"
            },
            "properties": {
                "tags": ["security", "dangerous-api"],
                "precision": "high"
            }
        }
    }
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path).resolve()
        self.findings: List[VulnerabilityFinding] = []
        self.files_analyzed: List[str] = []
    
    def add_finding(
        self,
        file_path: str,
        probability: float,
        risk_level: str,
        dangerous_apis: List[str],
        start_line: int = 1,
        end_line: Optional[int] = None,
        message: Optional[str] = None,
        threshold: float = 0.5
    ) -> None:
        """Add a vulnerability finding."""
        if probability < threshold:
            return
        
        if risk_level in ("CRITICAL", "HIGH"):
            rule_id = "VULN002"
            severity = "error"
        else:
            rule_id = "VULN001"
            severity = "warning"
        
        if message is None:
            message = f"Potential vulnerability detected (probability: {probability:.1%}, risk: {risk_level})"
            if dangerous_apis:
                message += f". Dangerous APIs found: {', '.join(dangerous_apis[:5])}"
        
        location = Location(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line or start_line
        )
        
        finding = VulnerabilityFinding(
            rule_id=rule_id,
            message=message,
            location=location,
            severity=severity,
            probability=probability,
            risk_level=risk_level,
            dangerous_apis=dangerous_apis
        )
        
        self.findings.append(finding)
        
        if file_path not in self.files_analyzed:
            self.files_analyzed.append(file_path)
    
    def _get_artifact_location(self, file_path: str) -> Dict[str, Any]:
        """Get artifact location relative to base path."""
        try:
            rel_path = Path(file_path).relative_to(self.base_path)
            uri = str(rel_path).replace("\\", "/")
        except ValueError:
            uri = str(file_path).replace("\\", "/")
        
        return {"uri": uri}
    
    def _build_result(self, finding: VulnerabilityFinding) -> Dict[str, Any]:
        """Build a SARIF result object from a finding."""
        return {
            "ruleId": finding.rule_id,
            "ruleIndex": list(self.RULES.keys()).index(finding.rule_id),
            "level": finding.severity,
            "message": {
                "text": finding.message
            },
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": self._get_artifact_location(finding.location.file_path),
                        "region": {
                            "startLine": finding.location.start_line,
                            "endLine": finding.location.end_line,
                            "startColumn": finding.location.start_column,
                            "endColumn": finding.location.end_column
                        }
                    }
                }
            ],
            "partialFingerprints": {
                "primaryLocationLineHash": finding.fingerprint()
            },
            "properties": {
                "probability": finding.probability,
                "riskLevel": finding.risk_level,
                "dangerousApis": finding.dangerous_apis
            }
        }
    
    def generate(self) -> Dict[str, Any]:
        """Generate SARIF report."""
        run = {
            "tool": {
                "driver": {
                    "name": self.TOOL_NAME,
                    "version": self.TOOL_VERSION,
                    "informationUri": "https://github.com/hoangduy0308/C-Vul-Devign",
                    "rules": list(self.RULES.values())
                }
            },
            "results": [self._build_result(f) for f in self.findings],
            "artifacts": [
                {"location": self._get_artifact_location(f)}
                for f in self.files_analyzed
            ],
            "invocations": [
                {
                    "executionSuccessful": True,
                    "endTimeUtc": datetime.now(timezone.utc).isoformat()
                }
            ]
        }
        
        sarif = {
            "$schema": self.SARIF_SCHEMA,
            "version": self.SARIF_VERSION,
            "runs": [run]
        }
        
        return sarif
    
    def save(self, output_path: str) -> None:
        """Save SARIF report to file."""
        sarif = self.generate()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sarif, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of findings."""
        risk_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for finding in self.findings:
            if finding.risk_level in risk_counts:
                risk_counts[finding.risk_level] += 1
        
        return {
            "total_findings": len(self.findings),
            "files_analyzed": len(self.files_analyzed),
            "by_risk_level": risk_counts,
            "high_risk_count": risk_counts["CRITICAL"] + risk_counts["HIGH"]
        }
