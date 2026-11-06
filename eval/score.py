"""
[양세윤] 파싱률·스키마·필드별·CAS F1·pair F1·H코드 F1

outputs/{condition}/{split}/{doc_id}.json(infer.py가 저장한 모델 출력 원문)을 정답과 비교해
report/scores.csv에 (condition, split, subset, metric, value)로 기록한다.
subset은 val·val_en은 언어(ko/en), test는 test 담당이 넘긴 서식 목록(--subset-csv)으로 나누고 합산하지 않는다.
시간·토큰은 같은 폴더의 _log.jsonl에서 읽는다. 문서별 상세는 _score_detail.jsonl(*.json이 아닌 이름, CLAUDE.md 규약).
--prompt v2: outputs/prompt_v2/{condition}/{split}/을 채점해 report/scores_prompt_v2.csv에 쓴다(v1 기록을 덮어쓰지 않음).
출력 폴더의 _run.jsonl에 적힌 프롬프트 버전이 --prompt와 다르면 거부한다.
--scores-csv: 점수를 다른 파일에 쓴다(기본값 · 동작 불변). 본 비교군이 아닌 결과(외부 API 비교군 · 배포 경량화)가
report/scores.csv에 섞이지 않게(docs/deploy_quant_plan.md 6절). test에는 쓰지 않는다.

  python3 eval/score.py --condition base_zs --split val
  python3 eval/score.py --condition base_fs --split val --prompt v2
  python3 eval/score.py --split val --self-check                  # 정답을 예측으로 넣어 1.0이 나오는지 점검
  python3 eval/score.py --condition qlora_final --split test --allow-test --out-root <저장소 밖> \
                         --label-dir <저장소 밖> --subset-csv <저장소 밖> --text-dir <저장소 밖>
  # 코드 점검(fixture, 옛 라벨): --splits test/fixtures/splits.csv --label-dir test/fixtures/labels \
  #                              --text-dir test/fixtures/text --split val --self-check

채점 규칙 (docs/plan.md 5절)
- 파싱률·스키마 준수율: 전체 문서 기준. 출력 파일이 없거나 NOT_FOUND로 건너뛴 문서도 실패로 센다.
- 파싱에 실패한 문서는 "빈 예측"으로 보고 모든 필드를 채점한다(정답 항목은 전부 FN).
- 제품명: (상태, 값) 완전 일치. 값은 비교할 때만 NFKC · 괄호 부기 · 쉼표류 · 공백 제거 · 소문자로 맞춘다
  (core/normalize.py norm_product_name. 정답에는 원문 괄호를 그대로 둔다).
- 신호어: 위험/경고 정규화 후 (상태, 값) 완전 일치.
- GHS 분류: hazard_class_alias.csv로 정규 분류명으로 바꾼 뒤 (정규 분류명, 구분 N) 집합 비교 → micro P/R/F1.
  별칭표 밖의 분류명은 공백만 무시하고 비교하며, 목록을 따로 남긴다(별칭표 보강용). val 등은 화면에 출력하고,
  test는 화면에 내지 않고 저장소 밖 _unknown_hazard_classes.txt에만 쓴다(test 담당이 보고 별칭표 행 추가, plan 5절).
  "(GHS 외 ...)"로 정규화되는 항목은 기본 제외(--include-non-ghs로 포함, 착수 회의 판정 대기).
- 분류·문구 목록 상태(기재/자료없음/해당없음): 문서별 일치율. 해당없음 문서는 여기서 채점된다.
- H코드: code가 있는 문구만, "H302+H332"는 개별 코드로 쪼개 집합 비교 → micro P/R/F1.
- H코드 없는 문구: code가 null인 문구끼리 문구(NFKC·공백 제거·끝 마침표 제거) 집합 비교 → 별도 P/R/F1.
- CAS: 문서별 CAS 집합(중복 1건) → micro P/R/F1. cas_number가 null인 성분은 제외.
- pair: CAS로 짝짓고, 같은 CAS 안에서 함유량 min/max가 같은 것끼리 1:1 매칭(중복 CAS는 각각 셈).
  content_acc = CAS로 짝지어진 성분 중 함유량까지 맞은 비율(pair 오류가 CAS 탓인지 함유량 탓인지 구분용).
- CAS 없는 성분(영업비밀 포함): (is_substitute_data, 함유량)으로 1:1 매칭 → nocas P/R/F1.
- 다른 칸 오입력: ke_number에 KE가 아닌 값(EC 등), content에 CAS 형식 값 → 건수.
- 무근거 생성: 원문 텍스트에 없는 H코드·CAS, 비어야 할 분류 목록을 채운 문서 → 건수.
- 문서 완전 정답: 기존 기준(doc_exact_rate)과 확장 기준(doc_exact_ext_rate, + nocas · 오입력 없음)을 따로 낸다.
- ke_number·chemical_name·supplier 등 참고 필드는 채점하지 않는다.
- test는 --allow-test · 실험 고정(eval/experiment.json) · 저장소 밖 정답 · 서식 목록 · 텍스트 · 출력 폴더 · 봉인 대조
  (eval/seal.py) · 모든 문서의 출력 존재를 확인한 뒤에만 정답을 읽는다. 문서별 상세(정답 값 포함)는 출력 폴더(저장소 밖)에만 쓴다.
- 다른 test 세트(cv/prereg_newtest.md): --experiment · --manifest로 그 세트의 고정 파일 · 봉인을 쓴다. 이때 점수는
  report/scores.csv에 쓰지 않고 저장소 밖 <out-root>/_scores.csv(또는 저장소 밖 --scores-csv)에 쓴다(팀 기록과 섞지 않음).
- test 최초 채점: 비교군마다 처음 채점한 결과를 _score_first.jsonl(메타 + 지표)과 _score_detail_first.jsonl로
  따로 남기고 읽기 전용으로 둔다. 다시 채점해도(별칭표 보강 등) 이 두 파일은 바꾸지 않는다.
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.normalize import (  # noqa: E402
    canon_hazard_class,
    class_key,
    content_equal,
    is_non_ghs,
    norm_cas,
    norm_category,
    norm_product_name,
    norm_signal_word,
    norm_statement_text,
    split_hcodes,
    squash,
)
from core.prompt import DEFAULT_PROMPT, PROMPTS, prompt_sha256, prompt_subdir, recorded_prompt_version  # noqa: E402
from core.schema import check_schema, extract_json  # noqa: E402
from eval.seal import require_sealed  # noqa: E402

SPLITS_CSV = ROOT / "data" / "splits.csv"
SCORES_CSV = ROOT / "report" / "scores.csv"  # v1. 다른 프롬프트 버전은 scores_csv_for()


def scores_csv_for(version):
    """프롬프트 버전별 점수 파일. v1은 report/scores.csv(기존), 그 밖은 report/scores_prompt_{버전}.csv."""
    sub = prompt_subdir(version)
    return SCORES_CSV if not sub else SCORES_CSV.with_name(f"scores_{sub}.csv")


OUTPUT_ROOT = ROOT / "outputs"
EXPERIMENT_JSON = ROOT / "eval" / "experiment.json"  # 단계 7(모델·실험 고정)에서 PM이 커밋
# train: Base 난이도 진단 전용 — scores.csv에 쓰지 않는다. test: 정답은 저장소 밖(--label-dir 필수)
GOLD_DIRS = {"val": ROOT / "data" / "labels", "train": ROOT / "data" / "labels", "val_en": ROOT / "eval" / "val_en"}
SPLITS = ("val", "train", "val_en", "test")
FEWSHOT_JSON = ROOT / "eval" / "fewshot.json"
TEXT_DIR = ROOT / "data" / "text"
LOG_FILE = "_log.jsonl"  # eval/infer.py가 쓰는 문서별 시간·토큰 기록
DETAIL_FILE = "_score_detail.jsonl"  # *.json이 아닌 이름(검사기가 폴더의 *.json을 모두 검사함)
UNKNOWN_FILE = "_unknown_hazard_classes.txt"  # test 전용: 별칭표 밖 분류명(화면에 내지 않음)
FIRST_FILE = "_score_first.jsonl"  # test 전용: 최초 채점 메타 + 지표(덮어쓰지 않음)
FIRST_DETAIL_FILE = "_score_detail_first.jsonl"
TEST_CONDITIONS = ("base_zs", "base_fs", "qlora_final")  # eval/infer.py와 같음(plan 5절)
# 채점 결과를 좌우하는 파일 — 최초 채점 기록에 해시를 남긴다(단계 7 고정값과 대조용)
SCORER_FILES = (ROOT / "eval" / "score.py", ROOT / "core" / "normalize.py", ROOT / "core" / "schema.py")
ALIAS_CSV = ROOT / "eval" / "hazard_class_alias.csv"
H_CODE = re.compile(r"H\d{3}")
KE_FULL = re.compile(r"KE-\d{3,6}")
CAS_LIKE = re.compile(r"\d{2,7}-\d{2}-\d")


# ---------------------------------------------------------------- 모델 출력은 스키마를 어길 수 있으니 방어적으로 꺼낸다
def _dict(x):
    return x if isinstance(x, dict) else {}


def _items(x):
    return [i for i in x if isinstance(i, dict)] if isinstance(x, list) else []


def _str(x):
    if x is None or isinstance(x, str):
        return x
    return str(x)


def value_field(obj, key):
    f = _dict(obj.get(key))
    return _str(f.get("source_status")), _str(f.get("value"))


def list_status(obj, key):
    return _str(_dict(obj.get("list_status")).get(key))


# ---------------------------------------------------------------- 항목 → 비교 키
def ghs_keys(obj, include_non_ghs, unknown, who):
    keys = set()
    for it in _items(obj.get("ghs_classification")):
        raw = _str(it.get("hazard_class"))
        if not raw:
            continue
        canon, known = canon_hazard_class(raw)
        if not known:
            unknown[raw].add(who)
        if is_non_ghs(canon) and not include_non_ghs:
            continue
        keys.add((class_key(canon), norm_category(_str(it.get("category")))))
    return keys


def hcode_set(obj):
    codes = set()
    for it in _items(obj.get("hazard_statements")):
        codes.update(split_hcodes(_str(it.get("code"))))
    return codes


def htext_set(obj):
    return {
        norm_statement_text(_str(it.get("text")))
        for it in _items(obj.get("hazard_statements"))
        if not _str(it.get("code")) and _str(it.get("text"))
    }


def ingredients(obj):
    """[(정규화 CAS | None, 함유량 원문), ...]"""
    return [(norm_cas(_str(it.get("cas_number"))), it.get("content")) for it in _items(obj.get("ingredients"))]


def nocas_items(obj):
    """CAS가 null인 성분(영업비밀 포함) → [(is_substitute_data, 함유량 원문)]. plan 5절: 영업비밀은 is_substitute_data와 content로 판정"""
    return [(bool(it.get("is_substitute_data")), it.get("content"))
            for it in _items(obj.get("ingredients")) if not norm_cas(_str(it.get("cas_number")))]


def nocas_match(gold_items, pred_items):
    """(is_substitute_data 같음, 함유량 min/max 같음)으로 1:1 매칭 → tp, fp, fn"""
    left = list(gold_items)
    tp = 0
    for sub, content in pred_items:
        for i, (gs, gc) in enumerate(left):
            if gs == sub and content_equal(gc, content):
                left.pop(i)
                tp += 1
                break
    return tp, len(pred_items) - tp, len(gold_items) - tp


def misplaced(obj):
    """다른 칸에 넣은 값: ke_number에 KE가 아닌 값(EC 번호 등), content에 CAS 형식 값"""
    ke = content = 0
    for it in _items(obj.get("ingredients")):
        k = squash(_str(it.get("ke_number")) or "")
        ke += bool(k) and not KE_FULL.fullmatch(k.upper())
        content += bool(CAS_LIKE.search(_str(it.get("content")) or ""))
    return ke, content


def pair_match(gold_ings, pred_ings):
    """CAS로 짝짓고 같은 CAS 안에서 함유량이 같은 것끼리 1:1 매칭.
    반환: tp, fp, fn, CAS로 짝지어진 수, 상세(함유량 불일치 목록)"""
    g_by, p_by = defaultdict(list), defaultdict(list)
    for cas, content in gold_ings:
        if cas:
            g_by[cas].append(content)
    for cas, content in pred_ings:
        if cas:
            p_by[cas].append(content)
    tp = cas_paired = 0
    mismatch = []
    for cas in sorted(set(g_by) & set(p_by)):
        g_left = list(g_by[cas])
        matched = 0
        for pv in p_by[cas]:
            for i, gv in enumerate(g_left):
                if content_equal(gv, pv):
                    g_left.pop(i)
                    matched += 1
                    break
        n_pair = min(len(g_by[cas]), len(p_by[cas]))
        cas_paired += n_pair
        tp += matched
        if matched < n_pair or len(g_by[cas]) != len(p_by[cas]):
            mismatch.append({"cas": cas, "gold_content": g_by[cas], "pred_content": p_by[cas]})
    n_gold = sum(map(len, g_by.values()))
    n_pred = sum(map(len, p_by.values()))
    return tp, n_pred - tp, n_gold - tp, cas_paired, mismatch


# ---------------------------------------------------------------- 집계
class PRF:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, tp, fp, fn):
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def add_sets(self, gold, pred):
        tp, fp, fn = gold & pred, pred - gold, gold - pred
        self.add(len(tp), len(fp), len(fn))
        s = lambda x: sorted(x, key=str)  # noqa: E731  (None이 섞인 튜플 정렬용)
        return {"tp": s(tp), "fp": s(fp), "fn": s(fn)}

    def result(self):
        """(P, R, F1). 정답·예측이 모두 0건이면 채점 대상 없음 → None."""
        if self.tp + self.fp + self.fn == 0:
            return None, None, None
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        return p, r, f1


class Acc:
    PRF_NAMES = ("ghs", "hcode", "htext", "cas", "pair", "nocas")
    HIT_NAMES = ("product_name", "signal_word", "ghs_status", "hazard_status")

    def __init__(self):
        self.n = self.parsed = self.schema_ok = self.missing = self.hit_max = 0
        self.hits = defaultdict(int)
        self.prf = defaultdict(PRF)
        self.cas_paired = 0
        self.exact = self.exact_ext = 0
        self.fab = defaultdict(int)  # 지어낸 값: 원문 텍스트에 없는 H코드·CAS, 비어야 할 분류 목록을 채운 문서
        self.gen_time, self.in_tok, self.out_tok = [], [], []

    def rows(self):
        n = self.n or 1
        out = [
            ("n_docs", self.n),
            ("n_missing_output", self.missing),
            ("parse_rate", self.parsed / n),
            ("schema_rate", self.schema_ok / n),
        ]
        out += [(f"{k}_acc", self.hits[k] / n) for k in self.HIT_NAMES]
        for name in self.PRF_NAMES:
            p, r, f1 = self.prf[name].result()
            out += [(f"{name}_p", p), (f"{name}_r", r), (f"{name}_f1", f1)]
        out.append(("content_acc", self.prf["pair"].tp / self.cas_paired if self.cas_paired else None))
        # F1은 "지어낸 값"을 평균에 묻는다(H코드 13개를 지어내도 F1 0.9). 안전 문서에서 가장 위험한 오류라 따로 센다
        out += [
            ("doc_exact_rate", self.exact / n),
            ("doc_exact_ext_rate", self.exact_ext / n),  # 확장 기준: 기존 + nocas 정답 + 다른 칸 오입력 없음
            ("fab_hcode_n", self.fab["hcode_n"]),
            ("fab_hcode_docs", self.fab["hcode_docs"]),
            ("fab_ghs_docs", self.fab["ghs_docs"]),
            ("fab_cas_n", self.fab["cas_n"]),
            ("fab_any_docs", self.fab["any_docs"]),
            ("misplace_ke_n", self.fab["misplace_ke"]),
            ("misplace_content_n", self.fab["misplace_content"]),
        ]
        out += [
            ("gen_time_mean_s", mean(self.gen_time) if self.gen_time else None),
            ("input_tokens_mean", mean(self.in_tok) if self.in_tok else None),
            ("output_tokens_mean", mean(self.out_tok) if self.out_tok else None),
            ("n_hit_max_new_tokens", self.hit_max),
        ]
        return out


# ---------------------------------------------------------------- 문서 1건 채점
def fabrications(gold, pred, source):
    """원문에 없는 값을 지어냈는지. source(1~3항 텍스트)가 없으면 None.
    → {"hcode": [원문에 없는 H코드], "ghs": 비어야 할 분류 목록을 채웠는지, "cas": [원문에 없는 CAS]}"""
    if source is None:
        return None
    src_codes = set(H_CODE.findall(source))
    flat = re.sub(r"\s+", "", source)
    fab_codes = sorted(c for c in hcode_set(pred) if c not in src_codes)
    fab_ghs = list_status(gold, "ghs_classification") != "기재" and bool(_items(pred.get("ghs_classification")))
    # PDF 표에서 CAS가 "134759-18-" / "5"로 줄이 갈리는 경우가 있어, 마지막 체크디짓 앞까지만 있어도 원문에 있는 것으로 본다
    fab_cas = sorted({c for c, _ in ingredients(pred) if c and c not in flat and c.rsplit("-", 1)[0] + "-" not in flat})
    return {"hcode": fab_codes, "ghs": fab_ghs, "cas": fab_cas}


def doc_exact(detail, pred):
    """문서 완전 정답 → (기존 기준, 확장 기준).

    기존: 스키마 통과 + 제품명·신호어·목록 상태 + GHS·H코드·H문구·CAS·pair 모두 맞음 (정의를 바꾸지 않는다)
    확장: 기존 + CAS 없는 성분(nocas) 정답 + 다른 칸 오입력(misplaced) 없음 — plan 5절 영업비밀 판정까지 포함
    두 기준은 따로 보고한다(report/decisions.md "평가 기준 사전 고정").
    """
    exact = (bool(pred) and check_schema(pred)[0]
             and all(detail[k] == "ok" for k in ("ghs", "hcode", "htext", "cas", "pair"))
             and detail["product_name"]["ok"] and detail["signal_word"]["ok"]
             and "ghs_status" not in detail and "hazard_status" not in detail)
    ext = exact and detail.get("nocas", "ok") == "ok" and not detail.get("misplaced")
    return exact, ext


def score_doc(gold, pred, acc, include_non_ghs, unknown, source=None):
    detail = {}

    gs, gv = value_field(gold, "product_name")
    ps, pv = value_field(pred, "product_name")
    ok = gs == ps and norm_product_name(gv) == norm_product_name(pv)
    acc.hits["product_name"] += ok
    detail["product_name"] = {"ok": ok, "gold": [gs, gv], "pred": [ps, pv]}

    gs, gv = value_field(gold, "signal_word")
    ps, pv = value_field(pred, "signal_word")
    ok = gs == ps and norm_signal_word(gv) == norm_signal_word(pv)
    acc.hits["signal_word"] += ok
    detail["signal_word"] = {"ok": ok, "gold": [gs, gv], "pred": [ps, pv]}

    for key, name in (("ghs_classification", "ghs_status"), ("hazard_statements", "hazard_status")):
        g, p = list_status(gold, key), list_status(pred, key)
        acc.hits[name] += g == p
        if g != p:
            detail[name] = {"gold": g, "pred": p}

    detail["ghs"] = acc.prf["ghs"].add_sets(
        ghs_keys(gold, include_non_ghs, unknown, "gold"), ghs_keys(pred, include_non_ghs, unknown, "pred")
    )
    detail["hcode"] = acc.prf["hcode"].add_sets(hcode_set(gold), hcode_set(pred))
    detail["htext"] = acc.prf["htext"].add_sets(htext_set(gold), htext_set(pred))

    g_ings, p_ings = ingredients(gold), ingredients(pred)
    detail["cas"] = acc.prf["cas"].add_sets({c for c, _ in g_ings if c}, {c for c, _ in p_ings if c})
    tp, fp, fn, paired, mismatch = pair_match(g_ings, p_ings)
    acc.prf["pair"].add(tp, fp, fn)
    acc.cas_paired += paired
    detail["pair"] = {"tp": tp, "fp": fp, "fn": fn, "content_mismatch": mismatch}

    # 변수 이름을 따로 둔다 — 아래 pair "ok" 판정이 위의 pair tp/fp/fn을 쓴다
    n_tp, n_fp, n_fn = nocas_match(nocas_items(gold), nocas_items(pred))
    acc.prf["nocas"].add(n_tp, n_fp, n_fn)
    detail["nocas"] = "ok" if not n_fp and not n_fn else {"tp": n_tp, "fp": n_fp, "fn": n_fn,
                                                          "gold": nocas_items(gold), "pred": nocas_items(pred)}
    mk, mc = misplaced(pred)
    acc.fab["misplace_ke"] += mk
    acc.fab["misplace_content"] += mc
    if mk or mc:
        detail["misplaced"] = {"ke_number": mk, "content": mc}

    # 상세 파일을 읽기 쉽게: 틀린 게 없는 항목은 "ok"로 줄인다
    for k in ("ghs", "hcode", "htext", "cas"):
        if not detail[k]["fp"] and not detail[k]["fn"]:
            detail[k] = "ok"
    if not mismatch and not fp and not fn:
        detail["pair"] = "ok"

    fab = fabrications(gold, pred, source)
    if fab is not None:
        acc.fab["hcode_n"] += len(fab["hcode"])
        acc.fab["hcode_docs"] += bool(fab["hcode"])
        acc.fab["ghs_docs"] += fab["ghs"]
        acc.fab["cas_n"] += len(fab["cas"])
        acc.fab["any_docs"] += bool(fab["hcode"] or fab["ghs"] or fab["cas"])
        if fab["hcode"] or fab["ghs"] or fab["cas"]:
            detail["fabricated"] = fab
    return detail


# ---------------------------------------------------------------- 입출력
def read_splits(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["doc_id"]: r for r in csv.DictReader(f)}


def read_log(pred_dir):
    """_log.jsonl → {doc_id: 마지막 기록}. --overwrite로 다시 돈 문서는 마지막 줄이 유효하다."""
    path = Path(pred_dir) / LOG_FILE
    if not path.exists():
        return {}
    recs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["doc_id"]] = r
    return recs


def load_prediction(pred_dir, doc_id, log):
    """→ (예측 dict | None, 로그 기록 | None, 실패 사유). {doc_id}.json에는 모델 출력 원문이 들어 있다."""
    rec = log.get(doc_id)
    if rec and rec.get("skipped"):
        return None, rec, f"건너뜀({rec['skipped']})"
    path = Path(pred_dir) / f"{doc_id}.json"
    if not path.exists():
        return None, rec, "출력 파일 없음"
    obj, err = extract_json(path.read_text(encoding="utf-8"))
    return obj, rec, err


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def upsert_scores(condition, split, rows, path=SCORES_CSV):
    """같은 (condition, split)의 이전 기록을 지우고 새로 쓴다."""
    kept = []
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            kept = [r for r in csv.DictReader(f) if not (r["condition"] == condition and r["split"] == split)]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "split", "subset", "metric", "value"], lineterminator="\n")
        w.writeheader()
        w.writerows(kept)
        for subset, metric, value in rows:
            w.writerow({"condition": condition, "split": split, "subset": subset, "metric": metric, "value": fmt(value)})


def read_subset_csv(path):
    """test 담당이 넘긴 (doc_id, subset) 목록 → {doc_id: subset}. subset은 서식(현행 · 수입품 국문판 · 구서식 등)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["doc_id"]: r["subset"] for r in csv.DictReader(f)}


def score_docs(doc_ids, subset_of, gold_dir, pred_dir, text_dir, self_check=False, include_non_ghs=False,
               quiet=False):
    """문서 목록 전체를 채점 → (subset별 Acc, 별칭표 밖 분류명, 문서별 상세).
    목록의 모든 문서가 분모에 들어간다. 출력 파일이 없거나 파싱에 실패한 문서는 빈 예측으로 채점한다.
    quiet(test): 문서 ID · 정답 내용을 화면에 내지 않는다(상세 파일에만 남김)."""
    accs = defaultdict(Acc)
    unknown = defaultdict(set)
    details = {}
    log = {} if self_check else read_log(pred_dir)
    n_bad_gold = 0
    for doc_id in doc_ids:
        gold_path = Path(gold_dir) / f"{doc_id}.json"
        if not gold_path.exists():
            sys.exit(f"정답 파일 없음: {gold_path}")
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        g_ok, g_err = check_schema(gold)
        if not g_ok:
            n_bad_gold += 1
            if not quiet:
                print(f"[경고] 정답 {doc_id}가 스키마를 어김: {g_err[:3]}")

        if self_check:
            pred, rec, err = gold, None, None
        else:
            pred, rec, err = load_prediction(pred_dir, doc_id, log)

        acc = accs[subset_of[doc_id]]
        acc.n += 1
        d = {"doc_id": doc_id, "subset": subset_of[doc_id]}
        if not g_ok:
            d["gold_schema_errors"] = g_err[:10]
        if not self_check and (err == "출력 파일 없음" or (rec and rec.get("skipped"))):
            acc.missing += 1
        if rec and not rec.get("skipped") and rec.get("gen_time_sec") is not None:
            acc.gen_time.append(rec["gen_time_sec"])
            acc.in_tok.append(rec["input_tokens"])
            acc.out_tok.append(rec["output_tokens"])
            acc.hit_max += bool(rec.get("hit_max_new_tokens"))

        if pred is None:
            d["parse_error"] = err
            pred = {}
        else:
            acc.parsed += 1
            s_ok, s_err = check_schema(pred)
            acc.schema_ok += s_ok
            if not s_ok:
                d["schema_errors"] = s_err[:10]
        tp_ = Path(text_dir) / f"{doc_id}.txt"
        source = tp_.read_text(encoding="utf-8") if tp_.exists() else None
        d.update(score_doc(gold, pred, acc, include_non_ghs, unknown, source))
        d["exact"], d["exact_ext"] = doc_exact(d, pred)
        acc.exact += d["exact"]
        acc.exact_ext += d["exact_ext"]
        details[doc_id] = d
    if quiet and n_bad_gold:
        print(f"[경고] 스키마를 어긴 정답 {n_bad_gold}건 — 문서별 상세의 gold_schema_errors 참고(test 담당 확인)")
    return accs, unknown, details


def outside_repo(path):
    try:
        Path(path).resolve().relative_to(ROOT)
        return False
    except ValueError:
        return True


def check_prompt_version(pred_dir, version):
    """출력 폴더를 만든 프롬프트(_run.jsonl 첫 줄)가 --prompt와 같아야 한다. 기록이 없으면(출력 없음) 넘어간다."""
    run_path = Path(pred_dir) / "_run.jsonl"
    if not run_path.exists():
        return
    lines = run_path.read_text(encoding="utf-8").splitlines()
    run = json.loads(lines[0]) if lines else {}
    got = recorded_prompt_version(run)
    if got != version:
        sys.exit(f"[거부] {pred_dir}의 출력은 프롬프트 {got}로 만든 것이다(--prompt {version}). "
                 "버전이 다른 점수를 섞지 않는다")
    if run.get("prompt_text_sha256") and run["prompt_text_sha256"] != prompt_sha256(version):
        sys.exit(f"[거부] {pred_dir}를 만든 뒤 프롬프트 {version}의 문구가 바뀌었다. 새 버전으로 다시 돌릴 것")


def experiment_path(args):
    return Path(getattr(args, "experiment", None) or EXPERIMENT_JSON)


def gate_test(args, pred_dir):
    """test 채점 전 확인. 통과하기 전에는 정답 폴더를 열지 않는다. → 채점할 doc_id 목록"""
    if not args.allow_test:
        sys.exit("[거부] test 채점은 --allow-test가 있어야 한다(비교군 생성이 끝난 뒤)")
    if not experiment_path(args).exists():
        sys.exit(f"[거부] {experiment_path(args).name}(실험 고정)이 없다. 실험을 고정한 뒤에만 test를 채점한다")
    if args.self_check:
        sys.exit("[거부] test에는 self-check를 돌리지 않는다(봉인된 정답을 여는 일이라서). 채점기 점검은 val로")
    if args.no_write:
        sys.exit("[거부] test에는 --no-write를 쓰지 않는다(최초 채점 기록을 남겨야 함)")
    if args.condition not in TEST_CONDITIONS:
        sys.exit(f"[거부] test 비교군은 {TEST_CONDITIONS}뿐이다")
    for name in ("label_dir", "subset_csv", "text_dir", "out_root"):
        v = getattr(args, name)
        if not v or not outside_repo(v):
            sys.exit(f"[거부] test는 저장소 밖 --{name.replace('_', '-')}가 필요하다"
                     "(정답 · 서식 목록 · 텍스트는 test 담당이 보관, 출력 · 채점 상세도 저장소 밖에 둠)")
    for kind, path in (("label", args.label_dir), ("text", args.text_dir), ("subset", args.subset_csv)):
        require_sealed(kind, path, args.manifest)  # 봉인한 정답 · 텍스트 · 서식 목록으로만 채점(파일 해시만 보고, 정답은 아직 안 연다)
    subset_of = read_subset_csv(args.subset_csv)
    log = read_log(pred_dir)
    missing = [d for d in subset_of if not (pred_dir / f"{d}.json").exists() and not log.get(d, {}).get("skipped")]
    if missing:
        sys.exit(f"[거부] 추론이 끝나지 않았다: 출력 없음 {len(missing)}건. eval/infer.py를 먼저 끝낼 것")
    return subset_of


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", help="outputs/ 아래 조건 폴더 이름 (base_zs, base_fs, qlora_r1, ...)")
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--include-non-ghs", action="store_true", help="Simple Asphyxiant 같은 GHS 외 항목도 채점")
    ap.add_argument("--self-check", action="store_true", help="정답을 예측으로 넣어 채점기 자체를 점검(파일 안 씀)")
    ap.add_argument("--no-write", action="store_true", help="scores.csv·상세 파일을 쓰지 않고 화면에만 출력")
    ap.add_argument("--allow-test", action="store_true", help="test 채점 허용(실험 고정 · 전 비교군 생성 뒤)")
    ap.add_argument("--splits", default=SPLITS_CSV, help="분할표(코드 점검 때 test/fixtures/splits.csv)")
    ap.add_argument("--label-dir", help="정답 폴더(기본: val·train은 data/labels, val_en은 eval/val_en, test는 저장소 밖 필수)")
    ap.add_argument("--text-dir", help="무근거 생성 대조용 1~3항 텍스트 폴더(기본 data/text, test는 저장소 밖 필수)")
    ap.add_argument("--subset-csv", help="test 전용: (doc_id, subset) 목록(test 담당이 넘김)")
    ap.add_argument("--out-root", help="출력 루트(기본 outputs/, test는 저장소 밖 필수 — infer.py와 같은 폴더)")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default=DEFAULT_PROMPT,
                    help=f"프롬프트 버전(기본 {DEFAULT_PROMPT}). v1이 아니면 <out-root>/prompt_<버전>/을 채점하고 "
                         "report/scores_prompt_<버전>.csv에 쓴다")
    ap.add_argument("--scores-csv", help="점수 기록 파일(기본: --prompt에 따라 report/scores.csv 또는 "
                                         "report/scores_prompt_<버전>.csv). 본 비교군이 아닌 결과를 따로 둘 때")
    ap.add_argument("--experiment", help="test 실험 고정 파일(기본 eval/experiment.json = 팀 test)")
    ap.add_argument("--manifest", help="test 봉인 기록(기본 report/test_manifest.csv = 팀 test)")
    args = ap.parse_args(argv)
    other_set = args.split == "test" and bool(args.experiment)  # 팀 test가 아닌 test 세트
    if args.scores_csv and args.split == "test" and not (other_set and outside_repo(args.scores_csv)):
        ap.error("--scores-csv는 팀 test에 쓰지 않는다(test 점수는 기본 경로 · 최초 채점 기록으로 남긴다). "
                 "다른 test 세트(--experiment)는 저장소 밖 경로만 받는다")
    if (args.experiment or args.manifest) and args.split != "test":
        ap.error("--experiment · --manifest는 test 전용이다")
    if other_set and not args.manifest:
        ap.error("다른 test 세트(--experiment)는 그 세트의 봉인 --manifest도 함께 준다")

    if not args.self_check and not args.condition:
        ap.error("--condition이 필요함(--self-check일 때만 생략 가능)")
    if args.split != "test":  # test는 기본값 없이 저장소 밖 폴더를 받아야 한다(gate_test)
        args.text_dir = args.text_dir or TEXT_DIR
        args.out_root = args.out_root or OUTPUT_ROOT
    is_test = args.split == "test"
    pred_dir = (Path(args.out_root or OUTPUT_ROOT) / prompt_subdir(args.prompt) / (args.condition or "_self_check")
                / args.split)
    scores_csv = Path(args.scores_csv) if args.scores_csv else scores_csv_for(args.prompt)
    if other_set and not args.scores_csv:  # gate_test가 저장소 밖 --out-root를 확인한다
        scores_csv = Path(args.out_root or OUTPUT_ROOT) / "_scores.csv"
    if not args.self_check:
        check_prompt_version(pred_dir, args.prompt)

    if is_test:
        subset_of = gate_test(args, pred_dir)
        doc_ids = sorted(subset_of)
        gold_dir = Path(args.label_dir)
    else:
        splits = read_splits(args.splits)
        doc_ids = sorted(d for d, r in splits.items() if r["split"] == args.split)
        if args.split == "train":
            fs = set(json.loads(FEWSHOT_JSON.read_text(encoding="utf-8")).get("fewshot_doc_ids", []))
            doc_ids = [d for d in doc_ids if d not in fs]
        subset_of = {d: splits[d]["lang"] for d in doc_ids}
        gold_dir = Path(args.label_dir or GOLD_DIRS[args.split])
    if not doc_ids:
        sys.exit(f"{args.split} 문서가 0건")

    accs, unknown, details = score_docs(doc_ids, subset_of, gold_dir, pred_dir, args.text_dir,
                                        args.self_check, args.include_non_ghs, quiet=is_test)
    subsets = sorted(accs)
    rows = [(subset, m, v) for subset in subsets for m, v in accs[subset].rows()]

    # 화면 출력: 지표 × subset 표
    table = {s: dict(accs[s].rows()) for s in subsets}
    print(f"\n[{args.condition or 'self-check'} / {args.split}]  " + "  ".join(f"{s}={accs[s].n}건" for s in subsets))
    print(f"{'metric':<24}" + "".join(f"{s:>10}" for s in subsets))
    for metric in table[subsets[0]]:
        print(f"{metric:<24}" + "".join(f"{fmt(table[s][metric]):>10}" for s in subsets))

    unknown_lines = [f"  - {raw!r}  ({'/'.join(sorted(unknown[raw]))})" for raw in sorted(unknown)]
    if unknown and not is_test:
        print("\n[별칭표에 없는 분류명] eval/hazard_class_alias.csv 보강 후보:")
        print("\n".join(unknown_lines))

    if args.self_check:
        bad = sorted({m for s in subsets for m, v in table[s].items()
                      if m.endswith(("_acc", "_rate", "_p", "_r", "_f1")) and v is not None and v < 1.0}
                     | {m for s in subsets for m, v in table[s].items() if m.startswith(("fab_", "misplace_")) and v})
        print("\nself-check:", "통과 (채점 대상 지표 모두 1.0)" if not bad else f"실패 — 1.0이 아닌 지표 {bad}")
        sys.exit(0 if not bad else 1)

    if not args.no_write:
        if args.split != "train":  # 진단 점수는 보고용 scores.csv와 섞지 않는다
            scores_csv.parent.mkdir(parents=True, exist_ok=True)
            upsert_scores(args.condition, args.split, rows, scores_csv)
        detail_path = pred_dir / DETAIL_FILE
        with open(detail_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"include_non_ghs": args.include_non_ghs,
                                "unknown_hazard_classes": sorted(unknown)}, ensure_ascii=False) + "\n")
            for d in details.values():
                f.write(json.dumps(d, ensure_ascii=False, default=list) + "\n")
        where = "scores.csv에는 안 씀(진단 전용)" if args.split == "train" else f"기록: {scores_csv}"
        print(f"\n{where}  /  문서별 상세: {detail_path}")
        if is_test:
            write_test_extras(args, pred_dir, rows, unknown_lines, detail_path)


def sha256_files(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def write_test_extras(args, pred_dir, rows, unknown_lines, detail_path):
    """test 전용 기록(모두 저장소 밖 pred_dir): 별칭표 밖 분류명 목록, 최초 채점 보존."""
    unknown_path = pred_dir / UNKNOWN_FILE
    if unknown_lines:
        unknown_path.write_text("[별칭표에 없는 분류명] eval/hazard_class_alias.csv 보강 후보(test 담당 확인):\n"
                                + "\n".join(unknown_lines) + "\n", encoding="utf-8")
        print(f"별칭표 밖 분류명 있음 → {unknown_path} (화면에 출력하지 않음, test 담당이 확인)")
    else:
        unknown_path.unlink(missing_ok=True)  # 별칭표 보강 뒤 재채점하면 이전 목록이 남지 않게

    first_path = pred_dir / FIRST_FILE
    if first_path.exists():
        print(f"재채점: 최초 채점 기록은 그대로 둠 → {first_path}")
        return
    meta = {
        "condition": args.condition,
        "split": args.split,
        "scored_at": datetime.now().isoformat(timespec="seconds"),
        "include_non_ghs": args.include_non_ghs,
        "experiment_sha256": sha256_files([experiment_path(args)]),
        "manifest_sha256": sha256_files([args.manifest]) if args.manifest else None,
        "scorer_sha256": sha256_files(SCORER_FILES),
        "alias_sha256": sha256_files([ALIAS_CSV]),
    }
    with open(first_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for subset, metric, value in rows:
            f.write(json.dumps({"subset": subset, "metric": metric, "value": value}, ensure_ascii=False) + "\n")
    first_detail = pred_dir / FIRST_DETAIL_FILE
    shutil.copyfile(detail_path, first_detail)
    for p in (first_path, first_detail):
        p.chmod(0o444)  # 실수로 덮어쓰지 않게 읽기 전용
    print(f"최초 채점 보존 → {first_path} · {first_detail}")


if __name__ == "__main__":
    main()
