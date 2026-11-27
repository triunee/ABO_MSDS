"""
[김건하] Rule Engine 공용 타입 — Finding·Evidence·CheckContext.
출력 모양은 docs/frontend_api_spec.md와 맞춘다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

Severity = Literal["error", "warning", "info"]
ReviewStatus = Literal["OK", "NEEDS_REVIEW", "INCONSISTENT", "SCHEMA_ERROR"]

# app/rules/checks/*.py 파일명과 반드시 일치시킨다.
CHECK_NAMES = (
    "schema",
    "empty_value",
    "not_found",
    "evidence",
    "cas",
    "hcode",
    "phrase_to_hcode",
    "ghs_signal_hcode_consistency",
    "content_sum",
    "source_codes",
)

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class CheckContext:
    """모든 검사 함수에 공통으로 넘기는 값. 원문이 없으면 원문 대조 계열 검사는 건너뛴다."""

    source_text: Optional[str] = None


@dataclass
class Finding:
    check: str
    severity: Severity
    code: str
    field: Optional[str]
    message: str
    detail: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "code": self.code,
            "field": self.field,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class EvidenceItem:
    field: str
    value: str
    found: bool
    match_type: Literal["exact", "normalized", "not_found"]
    snippet: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "value": self.value,
            "found": self.found,
            "match_type": self.match_type,
            "snippet": self.snippet,
        }


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: _SEVERITY_ORDER.get(f.severity, 9))
