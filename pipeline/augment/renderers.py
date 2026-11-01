"""
[강덕우] 텍스트 겉모양만 바꾸는 렌더러 6종 — 정답 라벨은 그대로 둔다.

generate_msds.py 원본이 저장소에 없어 같은 역할로 새로 썼다. 렌더러는 한 번에 하나만 적용한다
(난수 조합 없음). 값이 들어 있는 글자는 건드리지 않는다: 줄 앞 표지·항 제목·콜론 주변 공백·
항 사이 머리글·CAS 칸 이름만 바꾼다. 결과는 check_forbidden.check()로 다시 검사한다.
스타일은 train 문서에서 본 표기와 plan 3.4 증강표에 있는 것만 쓴다(test 문서 표기를 흉내 내지 않는다).

각 렌더러: (text, rng: random.Random) -> text
"""
import re

BULLETS = ["◦", "○", "-", "•", "·", "▶", "※", ""]
_BULLET_LINE = re.compile(r"^(\s*)([◦○•·▶※\-]|[ㅇo](?=\s))\s*")

# 1~3항 제목. 앞 번호 표기만 바꾸고 제목 글자는 그대로 둔다
_SECTION = re.compile(r"^\s*(?:제\s*)?([123])\s*(?:항)?\s*[.)]?\s*(?=(?:화학\s*제품|유해|구성\s*성분))")
SECTION_STYLES = ["{n}. ", "{n}) ", "제{n}항 ", "【{n}】 ", "{n}.", "SECTION {n}. "]

# 하위 항목 번호: "가. " "나) " "1) " "(1) "
_SUB_HANGUL = re.compile(r"^(\s*)([가나다라마바사아자차카타파하])\s*[.)]\s*")
_SUB_DIGIT = re.compile(r"^(\s*)\(?(\d{1,2})\)\s*")
HANGUL = "가나다라마바사아자차카타파하"
# "{d}. "는 넣지 않는다 — "3. 공급사 정보"가 3항 제목처럼 보인다
SUB_STYLES = ["{h}. ", "{h}) ", "({h}) ", "{d}) ", "({d}) "]

# "제품명 : 값" 의 콜론. 앞에 한글 라벨이 있을 때만 바꾼다(시각 "10:30" 같은 숫자 사이는 안 건드림)
_COLON = re.compile(r"(?<=[가-힣)])[ \t]*[:：][ \t]*")   # 줄바꿈은 먹지 않는다
COLON_STYLES = [" : ", ": ", ":", " ： ", " - "]

PAGE_HEADERS = [
    "물질안전보건자료(MSDS)", "Page {p} of {t}", "페이지 {p} / {t}", "- {p} -",
    "MSDS번호 : AA{num}", "개정일자 : 2024.0{m}.1{p}", "{p}/{t}",
]


def bullets(text, rng):
    """줄 앞 불릿 기호를 하나로 통일해서 바꾼다."""
    b = rng.choice(BULLETS)
    rep = (b + " ") if b else ""
    return "\n".join(_BULLET_LINE.sub(lambda m: m.group(1) + rep, ln) for ln in text.split("\n"))


def section_headers(text, rng):
    """'1. 화학제품과 …' → '제1항 화학제품과 …' 등."""
    style = rng.choice(SECTION_STYLES)
    return "\n".join(_SECTION.sub(lambda m: style.format(n=m.group(1)), ln) for ln in text.split("\n"))


def sub_numbering(text, rng):
    """하위 번호 체계를 바꾼다. 한글 순번(가나다)과 숫자 순번을 서로 바꿀 수 있다."""
    style = rng.choice(SUB_STYLES)

    def fmt(ind, idx):
        return ind + style.format(h=HANGUL[idx % len(HANGUL)], d=idx + 1)

    out = []
    for ln in text.split("\n"):
        m = _SUB_HANGUL.match(ln)
        if m and ln[m.end():m.end() + 1] not in ("", " "):
            ln = fmt(m.group(1), HANGUL.index(m.group(2))) + ln[m.end():]
        else:
            m = _SUB_DIGIT.match(ln)
            # 숫자 뒤에 곧바로 한글 라벨이 오는 줄만(함유량 "1) 30~40" 같은 오탐 방지)
            if m and re.match(r"[가-힣]", ln[m.end():m.end() + 1] or " ") and int(m.group(2)) <= 14:
                ln = fmt(m.group(1), int(m.group(2)) - 1) + ln[m.end():]
        out.append(ln)
    return "\n".join(out)


def colons(text, rng):
    """라벨 뒤 콜론 표기를 바꾼다. 괄호 안 콜론은 값의 일부라 두다("급성 독성(흡입: 가스)")."""
    style = rng.choice(COLON_STYLES)

    def inside_paren(s):
        depth = 0
        for ch in s:
            if ch == "(":
                depth += 1
            elif ch == ")" and depth:   # "3) 급성 독성" 의 번호 괄호처럼 짝 없는 ")"는 무시
                depth -= 1
        return depth > 0

    def one_line(ln):
        return _COLON.sub(lambda m: m.group(0) if inside_paren(ln[:m.start()]) else style, ln)

    return "\n".join(one_line(ln) for ln in text.split("\n"))


def page_headers(text, rng):
    """항 제목 줄 앞에 PDF 머리글·쪽번호 같은 잡음 줄을 끼운다(실제 쪽 넘김 흉내)."""
    t = rng.randint(3, 12)
    lines, out, p = text.split("\n"), [], 1
    for ln in lines:
        if _SECTION.match(ln) and out:
            p += 1
            h = rng.choice(PAGE_HEADERS).format(p=p, t=t, num=rng.randint(10**9, 10**10 - 1), m=rng.randint(1, 9))
            out.append(h)
        out.append(ln)
    return "\n".join(out)


def cas_labels(text, rng):
    """CAS 칸 이름 표기를 바꾼다(plan 3.4 "라벨 표기"). 뒤에 오는 CAS 값과 "또는 식별번호" 같은 부기는 그대로."""
    style = rng.choice(CAS_LABEL_STYLES)
    return _CAS_LABEL.sub(style, text)


# "CAS번호" "CAS 번호" "CAS No." "CAS NO.:" "CAS Registry No." — 콜론은 뒤에 남긴다
_CAS_LABEL = re.compile(r"\bCAS[ \t]*(?:Registry[ \t]*)?(?:No\.?|NO\.?|번호|Number)", re.I)
CAS_LABEL_STYLES = ["CAS No.", "CAS 번호", "CAS NO.", "CAS번호", "CAS Registry No."]


RENDERERS = {
    "bullet": bullets,
    "section": section_headers,
    "subnum": sub_numbering,
    "colon": colons,
    "pagehdr": page_headers,
    "caslabel": cas_labels,
}


# ---------------------------------------------------------------- 오류 겨냥 렌더러 (r1_aug, build_jsonl.py --targeted)
# 교차검증 기준선 실패 분석(cv/prereg_r1_aug.md)을 보고 고른 4종. 같은 45건을 보고 골랐다는 선택 편향은 보고서에 밝힌다.
# RENDERERS에 넣지 않는다 — 넣으면 r1 학습셋(렌더러 6종 · 같은 난수 순서)이 바뀐다. train에만 쓴다.
# 바꿀 곳이 없는 문서에만 None을 돌려준다(해당 없음). 고른 표기가 원본과 같으면 텍스트를 그대로 돌려줘
# build_jsonl이 다시 뽑게 한다. 값 글자는 그대로 두고, 같은 정답으로 읽히는 표기만 바꾼다
# (check_forbidden.protected_counts가 Category N = 구분 N, Liquefied gas = 액화가스, 비해당 서술 = 해당없음으로 센다).
_CAS = re.compile(r"(?<![\d-])(\d{2,7})-(\d{2})-(\d)(?![\d-])")
_RANGE = re.compile(r"(\d)[ \t]*([~∼～])[ \t]*(\d)")
_SIGNAL = re.compile(r"(?<![가-힣])(위험|경고)(?![가-힣])")
CAS_SEPS = [" - ", " -", "- "]

# 2항 안의 비해당 서술. 바꾸는 쪽은 해당없음 · 라벨 부착 규정 없음만(행 단위 "분류되지 않음"은 두고)
_NONAPP = re.compile(r"해당[ \t]*사항[ \t]*없음|해당[ \t]*없음|라벨[ \t]*부착[ \t]*규정[ \t]*없음")
NONAPP_STYLES = ["해당없음", "해당 사항 없음", "라벨 부착 규정 없음", "규정 없음", "분류되지 않음"]

_CATEGORY = re.compile(r"(?:구[ \t]*분|Category)[ \t]*(\d[A-C]?)(?![\dA-Za-z])", re.I)
CATEGORY_STYLES = ["구분 {n}", "구분{n}", "Category {n}"]
_GAS = re.compile(r"(냉동[ \t]*액화|액화|압축|용해)[ \t]*가스")
GAS_STYLES = {"냉동액화": ["냉동액화가스", "냉동 액화가스"], "액화": ["액화가스", "액화 가스", "Liquefied gas"],
              "압축": ["압축가스", "압축 가스", "Compressed gas"], "용해": ["용해가스", "용해 가스"]}

_H_LINE = re.compile(r"^\s*(?:[◦○•·▶※\-]\s*)?H\d{3}")
JOIN_SEPS = [" ▷ ", " ▷", ". "]


def _section2(lines):
    """2항 제목 다음 줄 ~ 3항 제목 앞 줄의 범위 (start, end). 못 찾으면 None."""
    s2 = next((i for i, ln in enumerate(lines) if (m := _SECTION.match(ln)) and m.group(1) == "2"), None)
    if s2 is None:
        return None
    s3 = next((i for i in range(s2 + 1, len(lines)) if (m := _SECTION.match(lines[i])) and m.group(1) == "3"), len(lines))
    return s2 + 1, s3


def value_spacing(text, rng):
    """값 안 공백: CAS 하이픈 양옆(67 - 64 - 1), 함유량 범위 구분자 양옆(10 ~ 20), 신호어 줄의 위 험 · 경 고.
    plan 3.4 "값 공백 · 구두점". 공백만 넣는다. 어느 구분자를 골라도 바뀌므로, 안 바뀌면 바꿀 곳이 없는 것."""
    sep = rng.choice(CAS_SEPS)
    out = _CAS.sub(lambda m: sep.join(m.groups()), text)
    out = _RANGE.sub(r"\1 \2 \3", out)
    lines = out.split("\n")
    for i, ln in enumerate(lines):
        if "신호어" not in ln:
            continue
        head, _, tail = ln.partition("신호어")
        if _SIGNAL.search(tail):
            lines[i] = head + "신호어" + _SIGNAL.sub(lambda m: " ".join(m.group(1)), tail)
        elif i + 1 < len(lines):
            lines[i + 1] = _SIGNAL.sub(lambda m: " ".join(m.group(1)), lines[i + 1])
    out = "\n".join(lines)
    return out if out != text else None


def nonapplicable(text, rng):
    """2항의 비해당 서술(해당없음 · 라벨 부착 규정 없음)을 다른 비해당 표현 하나로 바꾼다. 정답 상태는 그대로 해당없음
    (plan 2절 상태 값 표, core/prompt.py v1 규칙 2)."""
    lines = text.split("\n")
    span = _section2(lines)
    if span is None or not any(_NONAPP.search(lines[i]) for i in range(*span)):
        return None
    style = rng.choice(NONAPP_STYLES)
    for i in range(*span):
        lines[i] = _NONAPP.sub(style, lines[i])
    return "\n".join(lines)


def category_notation(text, rng):
    """분류 구분 표기(구분 2A · 구분2A · Category 2A)와 고압가스 상태명 표기(액화가스 · 액화 가스 · Liquefied gas)를 바꾼다.
    세부구분 글자(1A · 2A)와 숫자는 그대로. plan 2절 "분류 표기" · "고압가스"의 정규화 대상 표기만 쓴다."""
    if not (_CATEGORY.search(text) or _GAS.search(text)):
        return None
    style = rng.choice(CATEGORY_STYLES)
    out = _CATEGORY.sub(lambda m: style.format(n=m.group(1).upper()), text)
    return _GAS.sub(lambda m: rng.choice(GAS_STYLES[re.sub(r"\s+", "", m.group(1))]), out)


def join_lines(text, rng):
    """2항에서 분류 줄(구분 N · 가스 상태가 있는 줄)과 H코드 줄이 2줄 이상 이어지면 ▷ · 마침표로 한 줄에 잇는다(구서식).
    이어 붙이는 줄 앞 불릿만 뗀다. 정답은 줄마다 1건 그대로(plan 2절)."""
    lines = text.split("\n")
    span = _section2(lines)
    if span is None:
        return None
    sep = rng.choice(JOIN_SEPS)

    def kind(ln):
        if _H_LINE.match(ln):
            return "h"
        if _CATEGORY.search(ln) or _GAS.search(ln):
            return "ghs"
        return None

    out, i, joined = lines[:span[0]], span[0], False
    while i < span[1]:
        k, j = kind(lines[i]), i + 1
        while k and j < span[1] and kind(lines[j]) == k:
            j += 1
        if j - i >= 2:
            parts = [lines[i].rstrip()] + [_BULLET_LINE.sub("", ln).strip() for ln in lines[i + 1:j]]
            line = parts[0]
            for p in parts[1:]:
                line += (" " if sep == ". " and line.endswith(".") else sep) + p
            out.append(line)
            joined = True
        else:
            out += lines[i:j]
        i = j
    out += lines[span[1]:]
    return "\n".join(out) if joined else None


TARGETED_RENDERERS = {
    "valspace": value_spacing,
    "nonapp": nonapplicable,
    "catnote": category_notation,
    "join": join_lines,
}
