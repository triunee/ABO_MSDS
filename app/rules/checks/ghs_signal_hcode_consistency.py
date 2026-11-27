"""
[김건하] 검사 규칙: ghs_signal_hcode_consistency — 분류·신호어·H코드 3자 정합성 검사.

app/rules/tables/classification_signal_word.csv로 각 (분류명, 구분)의 기대 신호어·기대
H코드를 찾아 문서 실제값과 대조한다. docs/labeling_review_notes.md B절의 "원문 자체가
모순인 문서"가 여기서 INCONSISTENT로 잡혀야 한다.

문서 단위 판정(신호어 · 분류로 설명 안 되는 코드)은 모든 분류가 대응표에 있을 때만 한다. 표에 없는
분류가 하나라도 있으면 그 분류의 신호어 · H코드를 모르므로, 남은 분류만으로 판정하면 오경보가 난다
(예: 표에 없는 '위험' 분류를 빼고 '경고'를 기대, 그 분류의 H코드를 '설명 안 되는 코드'로 표시).
이때는 COMPLETENESS_SKIPPED(info)를 남기고, 신호어는 아는 분류만으로 본 기대보다 실제가 약할 때만 표시한다
(표에 없는 분류는 기대 신호어를 올릴 수만 있으므로 이 경우는 여전히 모순이다). 분류별 판정(HCODE_MISSING_FOR_CLASSIFICATION)은 그대로 한다.

알려진 한계: "H302+H312+H332" 같은 조합 코드는 구성 코드별로 풀어서 대조하지 않는다
(app/rules/tables/README.md 참고).
"""
from __future__ import annotations

from ..tables.loader import load_classification_table, load_h_code_phrases, normalize_hazard_class
from ..types import CheckContext, Finding

_SIGNAL_RANK = {None: 0, "경고": 1, "위험": 2}
_RANK_TO_SIGNAL = {0: None, 1: "경고", 2: "위험"}


def _split_combo(code: str) -> list[str]:
    """"H303+H313" 같은 조합 코드를 구성 코드로 풀어 대조를 관대하게 한다."""
    return code.split("+") if "+" in code else [code]


def _collect_actual_codes(label: dict) -> set[str]:
    """code가 있으면 그대로, code가 null이면 문구를 h_code_phrases표와 대조해 코드를 되짚는다.

    구서식 문서는 H코드 없이 문구만 나오는 경우가 흔하다(라벨링 규칙상 code=null이 정상).
    그런 문서까지 전부 '분류에 대응하는 H코드가 없다'고 오판하지 않도록 문구로도 찾는다.
    """
    phrase_table = load_h_code_phrases()
    phrase_to_code: dict[str, str] = {}
    for code, entry in phrase_table.items():
        for variant in entry["variants"]:
            phrase_to_code.setdefault(variant.strip(), code)

    codes: set[str] = set()
    for hs in label.get("hazard_statements") or []:
        if not isinstance(hs, dict):
            continue
        raw_code = hs.get("code")
        if isinstance(raw_code, str) and raw_code:
            for part in _split_combo(raw_code):
                codes.add(part)
            continue
        text = hs.get("text")
        if isinstance(text, str):
            matched = phrase_to_code.get(text.strip())
            if matched:
                codes.add(matched)
    return codes


def run(label: dict, ctx: CheckContext) -> list[Finding]:
    findings: list[Finding] = []
    table = load_classification_table()

    classifications = [c for c in (label.get("ghs_classification") or []) if isinstance(c, dict)]
    actual_signal = (label.get("signal_word") or {}).get("value")
    actual_codes = _collect_actual_codes(label)

    expected_signal_rank = 0
    all_expected_codes: set[str] = set()
    matched_any = False
    unmatched: list[str] = []

    for i, c in enumerate(classifications):
        raw_class = c.get("hazard_class") or ""
        category = c.get("category") or ""
        norm_class = normalize_hazard_class(raw_class)
        entry = table.get((norm_class, category))
        path = f"$.ghs_classification[{i}]"

        if entry is None:
            unmatched.append(f"{raw_class} {category}".strip())
            findings.append(
                Finding(
                    check="ghs_signal_hcode_consistency",
                    severity="info",
                    code="CLASSIFICATION_NOT_IN_TABLE",
                    field=path,
                    message=f"'{raw_class} {category}'가 대응표에 없어 신호어·H코드 대조를 건너뜁니다.",
                    detail={"hazard_class": raw_class, "category": category, "normalized": norm_class},
                )
            )
            continue

        matched_any = True
        expected_signal_rank = max(expected_signal_rank, _SIGNAL_RANK.get(entry["signal_word"], 0))
        expected_codes = entry["h_codes"]
        all_expected_codes.update(expected_codes)

        if expected_codes and not (actual_codes & set(expected_codes)):
            findings.append(
                Finding(
                    check="ghs_signal_hcode_consistency",
                    severity="error",
                    code="HCODE_MISSING_FOR_CLASSIFICATION",
                    field=path,
                    message=(
                        f"분류 '{raw_class} {category}'에 대응하는 H코드"
                        f"({'/'.join(expected_codes)}) 중 어느 것도 문서에 없습니다."
                    ),
                    detail={"hazard_class": raw_class, "category": category, "expected_codes": expected_codes},
                )
            )

    if matched_any and unmatched:
        if _SIGNAL_RANK.get(actual_signal, 0) < expected_signal_rank:
            expected_signal = _RANK_TO_SIGNAL[expected_signal_rank]
            findings.append(
                Finding(
                    check="ghs_signal_hcode_consistency",
                    severity="error",
                    code="SIGNAL_WORD_MISMATCH",
                    field="$.signal_word.value",
                    message=(
                        f"분류로 미루어 신호어는 적어도 '{expected_signal}'이어야 하는데 "
                        f"문서는 '{actual_signal or '없음/해당없음'}'입니다."
                    ),
                    detail={"expected": expected_signal, "actual": actual_signal, "at_least": True},
                )
            )
        findings.append(
            Finding(
                check="ghs_signal_hcode_consistency",
                severity="info",
                code="COMPLETENESS_SKIPPED",
                field="$.signal_word.value",
                message="대응표에 없는 분류가 있어 '분류로 설명 안 되는 H코드' 판정을 건너뛰고, 신호어는 기대보다 약할 때만 봅니다.",
                detail={"unmatched_classifications": unmatched},
            )
        )
    elif matched_any:
        expected_signal = _RANK_TO_SIGNAL[expected_signal_rank]
        if expected_signal != actual_signal:
            findings.append(
                Finding(
                    check="ghs_signal_hcode_consistency",
                    severity="error",
                    code="SIGNAL_WORD_MISMATCH",
                    field="$.signal_word.value",
                    message=(
                        f"분류로 미루어 신호어는 '{expected_signal or '없음/해당없음'}'이어야 하는데 "
                        f"문서는 '{actual_signal or '없음/해당없음'}'입니다."
                    ),
                    detail={"expected": expected_signal, "actual": actual_signal},
                )
            )

        unexplained = sorted(actual_codes - all_expected_codes)
        if unexplained:
            findings.append(
                Finding(
                    check="ghs_signal_hcode_consistency",
                    severity="error",
                    code="HCODE_MISSING_FOR_CLASSIFICATION",
                    field="$.hazard_statements",
                    message=f"문구 코드 {unexplained}는 목록에 있는 분류로 설명되지 않습니다.",
                    detail={
                        "unexplained_codes": unexplained,
                        "listed_classifications": [
                            {"hazard_class": c.get("hazard_class"), "category": c.get("category")}
                            for c in classifications
                        ],
                    },
                )
            )

    return findings
