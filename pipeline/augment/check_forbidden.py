"""
[강덕우] 성분·분류·H코드·함유량이 바뀐 변형 검출

원칙: 원본 텍스트에서 찾을 수 있던 정답 값은 변형 텍스트에서도 찾을 수 있어야 한다.
(원본에서도 못 찾는 값 — 셀 안 줄바꿈 등 — 은 비교 대상에서 뺀다. 변형 탓이 아니므로)
의도한 내용 변형(mutators.py)은 바뀐 라벨을 넘기고, 사라져야 할 값은 gone= 으로 넘긴다.

  problems = check(orig_text, new_text, new_label, gone=["1310-73-2"])
  빈 목록이면 통과.

라벨 값 대조만으로는 정규화 필드(신호어 · category · 상태값)와 함유량 부등호를 볼 수 없다
(라벨은 "위험" · "구분 2"인데 원문은 "DANGER" · "Category 2"일 수 있음). 그래서 이 값들의 원문 표기는
"보호 토큰"으로 세어, 렌더러 변형에서는 개수가 그대로인지, 내용 변형에서는 줄지 않았는지 본다.
"""
import re
from collections import Counter

H_CODE = re.compile(r"\bH\d{3}\b")

# 정답을 결정하는 원문 표기. 렌더러는 이 토큰을 만들거나 지우면 안 된다.
PROTECTED = {
    "신호어": re.compile(r"위[ \t]*험|경[ \t]*고|\bDANGER\b|\bWARNING\b", re.I),
    "구분": re.compile(r"(?:구[ \t]*분|Category)[ \t]*\d[A-C]?", re.I),
    "고압가스": re.compile(r"압축[ \t]*가스|냉동[ \t]*액화[ \t]*가스|액화[ \t]*가스|용해[ \t]*가스|Liquefied gas|Compressed gas", re.I),
    "상태": re.compile(r"자[ \t]*료[ \t]*없[ \t]*음|해[ \t]*당[ \t]*(?:사[ \t]*항[ \t]*)?없[ \t]*음|라벨[ \t]*부착[ \t]*규정[ \t]*없[ \t]*음"
                     r"|규정[ \t]*없[ \t]*음|분류[ \t]*되지[ \t]*않|Not applicable", re.I),
    "부등호": re.compile(r"[<>≤≥＜＞]=?|이상|이하|미만|초과"),
}


# 같은 정답으로 정규화되는 표기는 한 토큰으로 센다(plan 2절): Category N = 구분 N, 영문 가스 상태 = 한국어,
# 비해당 서술(해당 사항 없음 · 라벨 부착 규정 없음 · 규정 없음 · 분류되지 않음 · Not applicable) = 해당없음.
# 렌더러가 이 표기끼리 바꾸는 것은 허용하고(renderers.TARGETED_RENDERERS), 개수가 바뀌는 것은 그대로 막는다.
_CANON = [(re.compile(r"^CATEGORY"), "구분"), (re.compile(r"^LIQUEFIEDGAS$"), "액화가스"),
          (re.compile(r"^COMPRESSEDGAS$"), "압축가스"),
          (re.compile(r"^(?:해당사항없음|라벨부착규정없음|규정없음|분류되지않|NOTAPPLICABLE)$"), "해당없음")]


def protected_counts(text):
    """보호 토큰 종류별 개수. 공백 차이("위 험" / "위험")와 같은 정답이 되는 표기 차이는 같은 것으로 센다."""
    out = Counter()
    for name, rx in PROTECTED.items():
        for m in rx.findall(text or ""):
            key = re.sub(r"\s+", "", m).upper()
            for crx, rep in _CANON:
                key = crx.sub(rep, key)
            out[(name, key)] += 1
    return out


def _flat(s):
    return re.sub(r"\s+", "", s or "")


def label_values(label):
    """라벨에서 원문을 그대로 옮긴 값들 → [(필드 이름, 값)]"""
    out = []
    for key in ("product_name", "recommended_use", "use_restrictions"):
        v = (label.get(key) or {}).get("value")
        if v:
            out.append((key, v))
    for i, c in enumerate(label.get("ingredients", [])):
        for f in ("chemical_name", "cas_number", "ke_number"):
            if c.get(f):
                out.append((f"ingredients[{i}].{f}", c[f]))
        # 함유량은 숫자만 본다("80이상~90미만" ↔ 원문 "80 이상 ~ 90 % 미만")
        for n in re.findall(r"\d+(?:\.\d+)?", c.get("content") or ""):
            out.append((f"ingredients[{i}].content", n))
    for i, g in enumerate(label.get("ghs_classification", [])):
        out.append((f"ghs[{i}].hazard_class", g["hazard_class"]))
    for i, h in enumerate(label.get("hazard_statements", [])):
        if h.get("code"):
            for code in H_CODE.findall(h["code"]):
                out.append((f"hazard[{i}].code", code))
        out.append((f"hazard[{i}].text", h["text"]))
    return out


def check(orig_text, new_text, new_label, gone=(), content_mutation=False):
    """content_mutation=False(렌더러): 보호 토큰 개수가 원본과 같아야 한다.
    content_mutation=True(mutators): 보호 토큰이 줄면 안 된다(영업비밀 표기 "자료없음(영업비밀)"처럼 늘어나는 것은 허용)."""
    fo, fn = _flat(orig_text), _flat(new_text)
    problems = []
    po, pn = protected_counts(orig_text), protected_counts(new_text)
    for key in sorted(set(po) | set(pn)):
        a, b = po[key], pn[key]
        if b < a or (b > a and not content_mutation):
            problems.append(f"보호 토큰 {key[0]} {key[1]!r} 개수 변화 {a}→{b}")
    for where, v in label_values(new_label):
        fv = _flat(v)
        if fv and fv in fo and fv not in fn:
            problems.append(f"{where} 값이 변형 텍스트에서 사라짐: {v!r}")
    for v in gone:
        if _flat(v) in fn:
            problems.append(f"지워야 할 값이 남아 있음: {v!r}")
    # 라벨에 code가 하나도 없는데 텍스트에 H코드가 남아 있으면 모델이 배울 신호가 모순된다
    codes = {h.get("code") for h in new_label.get("hazard_statements", [])}
    if new_label.get("hazard_statements") and codes == {None} and H_CODE.search(new_text):
        problems.append("라벨은 H코드 없음인데 텍스트에 H코드가 남아 있음")
    return problems
