"""
[김건하] + 양세윤(API 통합) — 10개 검사 → review_status + 사유 코드 + evidence.
출력 모양은 docs/frontend_api_spec.md와 맞춘다. 그 문서를 고치지 않고 여기 모양만
바꾸지 말 것 — 바꾸면 문서도 같이 갱신한다.
"""
from __future__ import annotations

from typing import Optional

from .checks import cas as cas_check
from .checks import content_sum as content_sum_check
from .checks import empty_value as empty_value_check
from .checks import evidence as evidence_check
from .checks import ghs_signal_hcode_consistency as consistency_check
from .checks import hcode as hcode_check
from .checks import not_found as not_found_check
from .checks import phrase_to_hcode as phrase_to_hcode_check
from .checks import schema as schema_check
from .checks import source_codes as source_codes_check
from .types import CHECK_NAMES, CheckContext, Finding, ReviewStatus, sort_findings

_CHECKS = {
    "schema": schema_check.run,
    "empty_value": empty_value_check.run,
    "not_found": not_found_check.run,
    "evidence": evidence_check.run,
    "cas": cas_check.run,
    "hcode": hcode_check.run,
    "phrase_to_hcode": phrase_to_hcode_check.run,
    "ghs_signal_hcode_consistency": consistency_check.run,
    "content_sum": content_sum_check.run,
    "source_codes": source_codes_check.run,
}


def _decide_review_status(findings: list[Finding]) -> ReviewStatus:
    if any(f.check == "schema" and f.severity == "error" for f in findings):
        return "SCHEMA_ERROR"
    if any(
        f.check in ("ghs_signal_hcode_consistency", "phrase_to_hcode") and f.severity == "error"
        for f in findings
    ):
        return "INCONSISTENT"
    if any(f.severity in ("error", "warning") for f in findings):
        return "NEEDS_REVIEW"
    return "OK"


def run_rules(label: dict, source_text: Optional[str], doc_id: Optional[str] = None) -> dict:
    """10개 검사를 모두 돌려 docs/frontend_api_spec.md 모양의 dict를 반환한다."""
    ctx = CheckContext(source_text=source_text)

    findings: list[Finding] = []
    checks_run: list[str] = []
    for name in CHECK_NAMES:
        findings.extend(_CHECKS[name](label, ctx))
        checks_run.append(name)
    findings = sort_findings(findings)

    evidence_items = evidence_check.build_evidence(label, source_text) if source_text else []

    review_status = _decide_review_status(findings)
    schema_ok = not any(f.check == "schema" and f.severity == "error" for f in findings)

    return {
        "doc_id": doc_id,
        "review_status": review_status,
        "schema_valid": schema_ok,
        "source_text_available": bool(source_text),
        "checks_run": checks_run,
        "findings": [f.to_dict() for f in findings],
        "evidence": [e.to_dict() for e in evidence_items],
        "summary": {
            "n_errors": sum(1 for f in findings if f.severity == "error"),
            "n_warnings": sum(1 for f in findings if f.severity == "warning"),
            "n_info": sum(1 for f in findings if f.severity == "info"),
        },
    }
