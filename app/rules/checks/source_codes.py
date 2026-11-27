"""
[양세윤] 검사 규칙: source_codes — 원문에서 정규식으로 뽑은 CAS · H코드 집합과 모델 출력 집합을 대조한다.

plan 8절 "누락 재판정"(모델이 비웠는데 원문에 값이 있음 → 검토 필요 · NOT_FOUND)을 CAS · H코드에 대해 한다.
not_found 검사는 출력 값이 원문에 있는지(출력 → 원문)만 보므로, 출력에서 빠진 값(원문 → 출력)은 잡지 못한다.

- 원문 CAS: 숫자-숫자-숫자 모양 + 체크디지트가 맞는 것만. 전화번호 · 날짜 · 번호류는 체크디지트에서 대부분 걸러진다.
- 원문 H코드: H + 3자리(+ 영문 접미 0~2자, 예: H360FD). EUH는 뺀다. 조합 "H302+H312"는 낱개로 센다.
- 입력 텍스트는 1~3항(P문구 블록 제거)이라 CAS · H코드는 사실상 2 · 3항 것만 남는다.
- 두 방향:
  - CAS_IN_SOURCE_MISSING · HCODE_IN_SOURCE_MISSING: 원문에는 있는데 출력에 없음(warning)
  - HCODE_NOT_IN_SOURCE: 출력 H코드가 원문에 없음(warning, 무근거 생성 후보). 형식이 틀린 코드는 hcode 검사가 맡는다.
- 알려진 오경보: 한 칸에 CAS가 여러 개인 행(정답은 첫 CAS만), 대표 물질 + 세부조성 행(정답은 대표만). warning이라 검토 필요로만 표시된다.
- source_text가 없으면 아무것도 검사하지 않는다(EVIDENCE_SKIPPED_NO_SOURCE는 evidence 검사가 알린다).
"""
from __future__ import annotations

import re

from ..types import CheckContext, Finding
from .cas import _checksum_ok

_SRC_CAS_RE = re.compile(r"(?<![\d-])(\d{2,7})\s?-\s?(\d{2})\s?-\s?(\d)(?![\d-])")
_SRC_H_RE = re.compile(r"(?<![A-Za-z])H\s?(\d{3})([A-Za-z]{0,2})(?![0-9A-Za-z])")
_OUT_H_RE = re.compile(r"^H\d{3}[A-Za-z]{0,2}$")


def source_cas(text: str) -> set[str]:
    out = set()
    for a, b, c in _SRC_CAS_RE.findall(text):
        cas = f"{a}-{b}-{c}"
        if _checksum_ok(cas):
            out.add(cas)
    return out


def source_hcodes(text: str) -> set[str]:
    return {f"H{n}{s}" for n, s in _SRC_H_RE.findall(text)}


def _output_cas(label: dict) -> set[str]:
    return {
        ing["cas_number"]
        for ing in label.get("ingredients") or []
        if isinstance(ing, dict) and isinstance(ing.get("cas_number"), str)
    }


def _output_hcodes(label: dict) -> list[tuple[int, str]]:
    """(문구 위치, 낱개 H코드). 조합 코드는 '+'로 나눈다."""
    out = []
    for i, hs in enumerate(label.get("hazard_statements") or []):
        if isinstance(hs, dict) and isinstance(hs.get("code"), str):
            for part in hs["code"].replace(" ", "").split("+"):
                if part:
                    out.append((i, part))
    return out


def run(label: dict, ctx: CheckContext) -> list[Finding]:
    if not ctx.source_text:
        return []
    text = ctx.source_text
    findings: list[Finding] = []

    missing_cas = sorted(source_cas(text) - _output_cas(label))
    if missing_cas:
        findings.append(
            Finding(
                check="source_codes",
                severity="warning",
                code="CAS_IN_SOURCE_MISSING",
                field="$.ingredients",
                message=f"원문에 있는 CAS {', '.join(missing_cas)}이(가) 성분 목록에 없습니다. 누락된 성분이 있는지 확인하세요.",
                detail={"values": missing_cas},
            )
        )

    src_h = source_hcodes(text)
    out_h = _output_hcodes(label)
    missing_h = sorted(src_h - {c for _, c in out_h})
    if missing_h:
        findings.append(
            Finding(
                check="source_codes",
                severity="warning",
                code="HCODE_IN_SOURCE_MISSING",
                field="$.hazard_statements",
                message=f"원문에 있는 H코드 {', '.join(missing_h)}이(가) 유해·위험 문구에 없습니다.",
                detail={"values": missing_h},
            )
        )

    for i, code in out_h:
        if _OUT_H_RE.match(code) and code not in src_h:
            findings.append(
                Finding(
                    check="source_codes",
                    severity="warning",
                    code="HCODE_NOT_IN_SOURCE",
                    field=f"$.hazard_statements[{i}].code",
                    message=f"H코드 {code}이(가) 원문에 없습니다. 지식으로 채웠거나 잘못 읽었을 수 있습니다.",
                    detail={"value": code},
                )
            )
    return findings
