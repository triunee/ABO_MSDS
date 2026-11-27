"""
[양세윤] 모델 출력 보정 — 원문에 적힌 값과 docs/plan.md 2절 출력 규칙만으로, 모델이 낸 JSON의 형식 실수를 고친다.

채점(eval/score.py --repair) · 교차검증(eval/cv.py report --repair) · 서빙(app/)이 모두 이 파일 하나만 쓴다.
지식으로 값을 채우거나 추정하지 않는다(CLAUDE.md 설계 원칙). 원문에 없는 값을 만드는 규칙은 넣지 않는다.

버전(REPAIRS): 결과를 만든 버전의 규칙은 바꾸지 않고, 고칠 때는 새 버전을 추가한다(프롬프트 버전과 같은 원칙).
  v1 (2026-11-27, 교차검증 r1 출력의 오류를 보고 만듦 — 확인용 문서에서는 아직 재지 않음)
    R1 H코드가 문구 앞에 붙어 나온 경우("H226 인화성 …", code null) → code로 옮기고 문구에서 뗀다
    R2 code가 null인데 원문 한 줄이 공백을 지우면 "H코드 + 그 문구"와 같고 그런 코드가 하나뿐이면 → 그 코드를 넣는다
    R3 code가 H코드 형식이 아니면("17" 등 깨져 인쇄된 코드) → null (plan 2절: 추정하지 않고 null)
    R4 분류명 앞의 묶음 머리("건강 유해성" · "물리적 위험성" · "환경 유해성")와 분류명 끝에 붙은 "구분"을 뗀다
    R5 "급성 독성-경피, 경구"처럼 한 줄에 경로가 여럿이면 경로별로 펼친다(plan 2절 "한 줄에 분류 두 개")

- 입력 dict는 바꾸지 않는다(깊은 복사). 파싱 실패 출력(None)은 고치지 않는다.
- 모델 원본은 그대로 남긴다: 출력 파일 {doc_id}.json과 DB raw_json에는 보정 전 원문을 둔다.
- 정답 라벨에 적용하면 바뀌는 것이 없어야 한다(test/test_repair.py가 data/labels로 검사).
"""
import copy
import re

DEFAULT_REPAIR = "v1"

_H_ONE = r"H\d{3}[A-Za-z]{0,2}"
_H_CODE = re.compile(rf"^{_H_ONE}(\s*\+\s*{_H_ONE})*$")
_H_LEAD = re.compile(rf"^\s*({_H_ONE}(?:\s*\+\s*{_H_ONE})*)\s*[:：.\-]?\s*(.+)$", re.S)
_GROUP_HEAD = re.compile(r"^\s*(건강\s*유해성|물리적\s*위험성|환경\s*유해성)\s*[:：\-]?\s*")
_TAIL_CAT = re.compile(r"\s*구분\s*$")
_ROUTES = ("경구", "경피", "흡입")
_ACUTE = re.compile(r"^\s*급성\s*독성(?:물질)?\s*[-:(（]?\s*(.+?)\s*[)）]?\s*$")
_WS = re.compile(r"\s+")


def _sq(s):
    return _WS.sub("", s or "")


def _code_from_source(text, source):
    """원문 줄 중 공백을 지운 뒤 'H코드 + 문구'가 문구와 같은 줄의 코드. 서로 다른 코드가 둘 이상이면 None."""
    t = _sq(text).rstrip(".")
    if not t or not source:
        return None
    found = set()
    for line in source.split("\n"):
        m = re.match(rf"^\W*({_H_ONE}(?:\+{_H_ONE})*)\W*(.*)$", _sq(line))
        if m and m.group(2).rstrip(".") == t:
            found.add(m.group(1))
    return found.pop() if len(found) == 1 else None


def _repair_statements(p, source, used):
    hs = p.get("hazard_statements")
    if not isinstance(hs, list):
        return
    for s in hs:
        if not isinstance(s, dict):
            continue
        code, text = s.get("code"), s.get("text")
        if isinstance(code, str) and not _H_CODE.match(code.strip()):
            s["code"] = None
            used.append("R3")
            code = None
        if code is not None or not isinstance(text, str):
            continue
        m = _H_LEAD.match(text)
        if m:
            s["code"] = _WS.sub("", m.group(1))
            s["text"] = m.group(2).strip()
            used.append("R1")
            continue
        c = _code_from_source(text, source)
        if c:
            s["code"] = c
            used.append("R2")


def _repair_classifications(p, used):
    gc = p.get("ghs_classification")
    if not isinstance(gc, list):
        return
    out = []
    for c in gc:
        if not isinstance(c, dict) or not isinstance(c.get("hazard_class"), str):
            out.append(c)
            continue
        name = c["hazard_class"]
        new = _TAIL_CAT.sub("", _GROUP_HEAD.sub("", name))
        if new != name:
            used.append("R4")
        m = _ACUTE.match(new)
        # 원문에 적힌 순서대로 펼친다
        routes = sorted((r for r in _ROUTES if r in m.group(1)), key=m.group(1).index) if m else []
        rest = re.sub("|".join(_ROUTES), "", m.group(1)) if m else ""
        if len(routes) >= 2 and not re.search(r"[가-힣A-Za-z]", re.sub(r"[,\s/및·]", "", rest)):
            out.extend({**c, "hazard_class": f"급성 독성({r})"} for r in routes)
            used.append("R5")
            continue
        out.append({**c, "hazard_class": new})
    p["ghs_classification"] = out


def _repair_v1(pred, source):
    p = copy.deepcopy(pred)
    used = []
    _repair_statements(p, source, used)
    _repair_classifications(p, used)
    return p, used


REPAIRS = {"v1": _repair_v1}


def repair_output(pred, source_text=None, version=DEFAULT_REPAIR):
    """→ (보정한 출력, 적용한 규칙 목록 — 같은 규칙이 여러 번 쓰이면 여러 번). dict가 아니면 그대로 돌려준다."""
    if version not in REPAIRS:
        raise ValueError(f"보정 버전이 없다: {version!r} (있는 버전: {sorted(REPAIRS)})")
    if not isinstance(pred, dict):
        return pred, []
    return REPAIRS[version](pred, source_text)
