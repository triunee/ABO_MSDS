"""
제조사 그룹 k-fold 교차검증 하네스 — 라벨이 있는 train + val 45건을 fold마다 학습 / 평가로 나눠, 모든 문서를
"그 제조사를 본 적 없는 모델"로 한 번씩 채점한다. test는 쓰지 않는다(봉인 자료는 저장소에 없고, 이 도구도 열지 않음).

  python3 eval/cv.py folds                          # cv/folds.csv · fold별 분할표 · few-shot 예시(있으면 같은지만 확인)
  python3 eval/cv.py run --cond r1 [--folds 0,1]    # build → train → infer. 끝난 단계는 건너뛰고 이어서 돈다
  python3 eval/cv.py run --cond base_fs --dry-run   # 돌릴 명령만 출력
  python3 eval/cv.py status                         # 조건 × fold 진행 상황
  python3 eval/cv.py report                         # 45건을 모아 채점 → cv/scores.csv · cv/paired.csv · cv/report.md
  python3 eval/cv.py stopcheck --recipe r1_aug       # 중간 멈춤 규칙(seed 42 짝). 종료 코드 0 계속 · 3 멈춤 · 2 판단 불가
  # 동작 점검(학습 2 step, 추론 2건, 저장소 밖): --root <폴더> run --cond r1 --folds 0 --smoke

설계
- data/splits.csv는 바꾸지 않는다. fold마다 <root>/fold{k}/splits.csv(평가 fold = val, 나머지 = train)를 만들어
  기존 스크립트에 넘긴다(build_jsonl · infer는 --splits, train_qlora는 설정의 splits:).
- fold는 split_group 단위로 나눈다. 큰 그룹부터 문서 수가 가장 적은 fold에 넣고, 동률이면 그 그룹의 서식이 가장 적은 fold에
  넣는다. 크기가 같은 그룹끼리의 순서만 seed로 섞는다.
- base_fs 예시: eval/fewshot.json의 2건이 그 fold의 train에 있으면 그대로 쓴다. 평가 fold에 있으면 같은 서식의 train 문서 중
  doc_id 순 첫 문서로 바꾼다(결과를 보기 전에 정한 규칙). 같은 제조사 문서가 예시로 들어가 평가가 새는 것을 막는다.
- 조건은 cv/experiments.yaml에 적는다. qlora 조건의 train은 pipeline/configs/r1.yaml 위에 덮어쓸 값, build는
  pipeline/build_jsonl.py 옵션이다. 개선 레시피는 기준(r1)에서 한 가지만 바꾼다(plan 6절 원칙).
- 단계마다 subprocess로 돌려 GPU 메모리를 단계마다 비운다. 로그는 <root>/logs/, 한 번에 하나만 돈다(<root>/.lock).
- 채점은 eval/score.py의 score_docs, 짝 비교는 eval/paired.py의 doc_outcomes · compare를 그대로 쓴다(채점 규칙을 새로
  만들지 않음). 점수는 report/scores.csv가 아니라 <root>/scores.csv에만 쓴다.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from core.prompt import DEFAULT_PROMPT, prompt_subdir  # noqa: E402
from eval.paired import FIELDS, compare, doc_outcomes  # noqa: E402
from eval.score import DETAIL_FILE, check_prompt_version, score_docs  # noqa: E402

SPLITS_CSV = ROOT / "data" / "splits.csv"
LABEL_DIR = ROOT / "data" / "labels"
TEXT_DIR = ROOT / "data" / "text"
FEWSHOT_JSON = ROOT / "eval" / "fewshot.json"
CV_ROOT = ROOT / "cv"
EXPERIMENTS = CV_ROOT / "experiments.yaml"
PY = sys.executable
# r1 학습셋을 만든 build_jsonl 옵션(runs/0928_r1/config.json의 augmentation.build_args). 조건의 build가 이 위에 덮어쓴다
BASE_BUILD = {"seed": 42, "secret": 1, "renderers": 6}
INFER_CONDITION = {"base_zs": "base_zs", "base_fs": "base_fs", "qlora": "qlora_r1"}  # infer.py 조건 이름(폴더는 조건마다 따로)
SMOKE_TRAIN_STEPS, SMOKE_INFER_LIMIT = 2, 2
TRAIN_SEED = 42  # pipeline/train_qlora.py DEFAULTS의 seed. 조건의 train.seed가 없으면 이 값
PREREG_GUARD = ("parse_rate", "schema_rate", "cas_f1", "pair_f1")  # 레시피 판정의 악화 불허 지표(cv/prereg_r1_aug.md)
REPORT_METRICS = ("parse_rate", "schema_rate", "product_name_acc", "signal_word_acc", "ghs_f1", "hcode_f1", "cas_f1",
                  "pair_f1", "nocas_f1", "doc_exact_rate", "doc_exact_ext_rate", "fab_any_docs", "misplace_ke_n",
                  "input_tokens_mean", "gen_time_mean_s")


# ---------------------------------------------------------------- 공용
def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def labeled_docs():
    """라벨이 있는 train · val 문서 → {doc_id: splits.csv 행}. test는 splits.csv에 없다."""
    rows = {r["doc_id"]: r for r in read_csv(SPLITS_CSV) if r["split"] in ("train", "val")}
    missing = [d for d in rows if not (LABEL_DIR / f"{d}.json").exists()]
    if missing:
        sys.exit(f"정답 파일 없음: {missing}")
    return rows


def load_experiments(path):
    exp = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for name, c in exp["conditions"].items():
        if c.get("type") not in INFER_CONDITION:
            sys.exit(f"{path}: {name}의 type은 {sorted(INFER_CONDITION)} 중 하나여야 함")
        if c.get("replicate_of") and c["replicate_of"] not in exp["conditions"]:
            sys.exit(f"{path}: {name}의 replicate_of {c['replicate_of']}이 conditions에 없음")
    if exp.get("baseline") not in exp["conditions"]:
        sys.exit(f"{path}: baseline {exp.get('baseline')}이 conditions에 없음")
    return exp


# ---------------------------------------------------------------- fold 나누기
def assign_folds(docs, k, seed):
    """split_group 단위 → {doc_id: fold}. 큰 그룹부터 가장 작은 fold에, 동률이면 그 서식이 적은 fold에."""
    groups = defaultdict(list)
    for d, r in docs.items():
        groups[r["split_group"]].append(d)
    rng = random.Random(seed)
    names = sorted(groups)
    rng.shuffle(names)
    names.sort(key=lambda g: -len(groups[g]))  # 안정 정렬: 크기가 같은 그룹끼리만 seed 순서
    size, forms = [0] * k, [Counter() for _ in range(k)]
    fold_of = {}
    for g in names:
        g_forms = Counter(docs[d]["form"] for d in groups[g])
        best = min(range(k), key=lambda i: (size[i], sum(forms[i][f] for f in g_forms), i))
        for d in groups[g]:
            fold_of[d] = best
        size[best] += len(groups[g])
        forms[best].update(g_forms)
    return fold_of


def pick_fewshot(docs, fold_of, k):
    """fold k의 base_fs 예시 2건. 고정 예시가 평가 fold에 있으면 같은 서식의 train 문서 중 doc_id 순 첫 문서로 바꾼다."""
    fixed = json.loads(FEWSHOT_JSON.read_text(encoding="utf-8"))["fewshot_doc_ids"]
    chosen = []
    for d in fixed:
        if fold_of[d] != k:
            chosen.append(d)
            continue
        alt = sorted(x for x, r in docs.items()
                     if fold_of[x] != k and r["form"] == docs[d]["form"] and x not in fixed and x not in chosen)
        if not alt:
            sys.exit(f"fold {k}: {d}를 대신할 같은 서식 train 문서가 없음")
        chosen.append(alt[0])
    return chosen


def cmd_folds(args):
    exp = load_experiments(args.experiments)
    k, seed = exp["folds"]["k"], exp["folds"]["seed"]
    docs = labeled_docs()
    fold_of = assign_folds(docs, k, seed)
    root = Path(args.root)
    folds_csv = root / "folds.csv"
    rows = [{"doc_id": d, "fold": fold_of[d], "split_group": docs[d]["split_group"], "form": docs[d]["form"]}
            for d in sorted(docs)]
    if folds_csv.exists():
        old = {r["doc_id"]: int(r["fold"]) for r in read_csv(folds_csv)}
        if old != fold_of:
            sys.exit(f"[거부] {folds_csv}가 지금 규칙 · 데이터로 만든 fold와 다르다. 이미 돈 결과와 섞이지 않게 멈춘다")
    write_csv(folds_csv, rows, ["doc_id", "fold", "split_group", "form"])
    for i in range(k):
        fold_dir = root / f"fold{i}"
        out = []
        for d, r in sorted(docs.items()):
            out.append({**r, "split": "val" if fold_of[d] == i else "train"})
        write_csv(fold_dir / "splits.csv", out, list(read_csv(SPLITS_CSV)[0].keys()))
        fs = pick_fewshot(docs, fold_of, i)
        (fold_dir / "fewshot.json").write_text(json.dumps({"fewshot_doc_ids": fs}, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    print(f"{k} fold · seed {seed} → {rel(folds_csv)}")
    print(f"{'fold':<6}{'docs':>5}{'groups':>8}  서식")
    for i in range(k):
        ds = [d for d in docs if fold_of[d] == i]
        forms = Counter(docs[d]["form"] for d in ds)
        groups = sorted({docs[d]["split_group"] for d in ds})
        fs = json.loads((root / f"fold{i}" / "fewshot.json").read_text(encoding="utf-8"))["fewshot_doc_ids"]
        print(f"{i:<6}{len(ds):>5}{len(groups):>8}  {dict(forms)}  few-shot {fs}")
    return fold_of


def read_folds(root):
    path = Path(root) / "folds.csv"
    if not path.exists():
        sys.exit(f"{rel(path)} 없음 — 먼저 python3 eval/cv.py folds")
    return {r["doc_id"]: int(r["fold"]) for r in read_csv(path)}


# ---------------------------------------------------------------- 경로 · 완료 판정
class Paths:
    def __init__(self, root, cond, k):
        self.root, self.cond, self.k = Path(root), cond, k
        self.fold = self.root / f"fold{k}"
        self.splits = self.fold / "splits.csv"
        self.fewshot = self.fold / "fewshot.json"
        self.data = self.root / "data" / cond / f"fold{k}"
        self.config = self.root / "configs" / f"{cond}_f{k}.yaml"
        self.run = self.root / "runs" / cond / f"fold{k}"
        self.adapter = self.run / "adapter"
        self.out_root = self.root / "outputs" / cond / f"fold{k}"

    def pred_dir(self, ctype, prompt=DEFAULT_PROMPT):
        return self.out_root / prompt_subdir(prompt) / INFER_CONDITION[ctype] / "val"


def build_args(cond_cfg):
    return {**BASE_BUILD, **(cond_cfg.get("build") or {})}


def build_done(p, cond_cfg):
    rep = p.data / "build_report.json"
    if not rep.exists() or not (p.data / "train.jsonl").exists():
        return False
    r = json.loads(rep.read_text(encoding="utf-8"))
    want = build_args(cond_cfg)
    if any(r["args"].get(key) != v for key, v in want.items()) or r["inputs"]["splits_sha256"] != sha256_file(p.splits):
        sys.exit(f"[거부] {rel(p.data)}가 지금 설정(build 옵션 · fold 분할)과 다르게 만들어졌다. 폴더를 지우고 다시 돌릴 것")
    return True


def train_done(p):
    cfg = p.run / "config.json"
    return p.adapter.exists() and cfg.exists() and "result" in json.loads(cfg.read_text(encoding="utf-8"))


def held_out(fold_of, k):
    return sorted(d for d, f in fold_of.items() if f == k)


def infer_count(p, ctype, fold_of):
    """(끝난 문서 수, 평가 문서 수). 끝남 = 출력 파일이 있거나 _log에 건너뜀(NOT_FOUND)으로 남음."""
    docs = held_out(fold_of, p.k)
    pred = p.pred_dir(ctype)
    skipped = set()
    log = pred / "_log.jsonl"
    if log.exists():
        skipped = {json.loads(x)["doc_id"] for x in log.read_text(encoding="utf-8").splitlines()
                   if x.strip() and json.loads(x).get("skipped")}
    return sum((pred / f"{d}.json").exists() or d in skipped for d in docs), len(docs)


# ---------------------------------------------------------------- 실행
def run_step(cmd, log_path, dry_run):
    print("  $ " + " ".join(str(c) for c in cmd))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {datetime.now().isoformat(timespec='seconds')}  {' '.join(str(c) for c in cmd)}\n")
        log.flush()
        proc = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            sys.stdout.write("    " + line)
        rc = proc.wait()
    print(f"  → 종료 코드 {rc}, {time.perf_counter() - t0:.0f}s, 로그 {rel(log_path)}")
    if rc:
        sys.exit(f"[중단] 단계 실패(종료 코드 {rc}). 로그: {rel(log_path)} — 원인을 고친 뒤 같은 명령으로 이어서 돌린다")


def write_train_config(p, cond_cfg):
    cfg = dict(cond_cfg.get("train") or {})
    cfg.update(train_jsonl=rel(p.data / "train.jsonl"), val_jsonl=rel(p.data / "val.jsonl"),
               build_report=rel(p.data / "build_report.json"), splits=rel(p.splits))
    p.config.parent.mkdir(parents=True, exist_ok=True)
    p.config.write_text("# eval/cv.py가 만든 fold 설정 — pipeline/configs/r1.yaml 위에 덮어쓴다\n"
                        + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_condition(root, name, cond_cfg, folds, fold_of, args):
    ctype = cond_cfg["type"]
    for k in folds:
        p = Paths(root, name, k)
        logs = Path(root) / "logs"
        print(f"\n[{name} / fold {k}] 평가 {len(held_out(fold_of, k))}건")
        adapter = None
        if ctype == "qlora":
            if build_done(p, cond_cfg):
                print("  build: 끝남")
            else:
                cmd = [PY, "pipeline/build_jsonl.py", "--splits", p.splits, "--out-dir", p.data]
                for key, v in build_args(cond_cfg).items():
                    if v is True:
                        cmd.append(f"--{key}")
                    elif v not in (False, None):
                        cmd += [f"--{key}", v]
                run_step(cmd, logs / f"{name}_f{k}_build.log", args.dry_run)
            if train_done(p):
                print("  train: 끝남")
            else:
                write_train_config(p, cond_cfg)
                cmd = [PY, "pipeline/train_qlora.py", "--config", p.config, "--name", f"cv_{name}_f{k}",
                       "--run-dir", p.run]
                if args.smoke:
                    cmd += ["--smoke", SMOKE_TRAIN_STEPS]
                run_step(cmd, logs / f"{name}_f{k}_train.log", args.dry_run)
            adapter = p.adapter
        done, total = infer_count(p, ctype, fold_of)
        limit = SMOKE_INFER_LIMIT if args.smoke else None
        if done >= (limit or total):
            print(f"  infer: 끝남 ({done}/{total})")
            continue
        cmd = [PY, "eval/infer.py", "--condition", INFER_CONDITION[ctype], "--split", "val", "--splits", p.splits,
               "--out-root", p.out_root]
        if adapter:
            cmd += ["--adapter", adapter]
        if ctype == "base_fs":
            cmd += ["--fewshot-json", p.fewshot]
        if limit:
            cmd += ["--limit", limit]
        run_step(cmd, logs / f"{name}_f{k}_infer.log", args.dry_run)


def cmd_run(args):
    exp = load_experiments(args.experiments)
    root = Path(args.root)
    fold_of = read_folds(root)
    k = exp["folds"]["k"]
    folds = [int(x) for x in args.folds.split(",")] if args.folds else list(range(k))
    names = args.cond.split(",")
    unknown = [n for n in names if n not in exp["conditions"]]
    if unknown:
        sys.exit(f"experiments.yaml에 없는 조건: {unknown}")
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f"[거부] 다른 eval/cv.py run이 {rel(root)}에서 돌고 있다(GPU 작업은 한 번에 하나)")
        for n in names:
            run_condition(root, n, exp["conditions"][n], folds, fold_of, args)
    print("\n완료. 진행 상황: python3 eval/cv.py status")


def cmd_status(args):
    exp = load_experiments(args.experiments)
    root = Path(args.root)
    fold_of = read_folds(root)
    k = exp["folds"]["k"]
    print(f"{'조건':<14}" + "".join(f"{'fold ' + str(i):>16}" for i in range(k)))
    for name, c in exp["conditions"].items():
        cells = []
        for i in range(k):
            p = Paths(root, name, i)
            done, total = infer_count(p, c["type"], fold_of)
            if c["type"] == "qlora":
                b = "B" if (p.data / "build_report.json").exists() else "·"
                t = "T" if train_done(p) else ("t" if p.run.exists() else "·")
                cells.append(f"{b}{t} {done}/{total}")
            else:
                cells.append(f"{done}/{total}")
        print(f"{name:<14}" + "".join(f"{c_:>16}" for c_ in cells))
    print("\nB = 학습셋 생성, T = 학습 끝남(t = 진행 중이거나 중단됨), n/m = 추론 끝난 문서 / 평가 문서")


# ---------------------------------------------------------------- 모으기 · 채점 · 짝 비교
def pool(root, name, ctype, fold_of, partial):
    """fold별 출력 → <root>/pooled/{name}/ 한 폴더(score_docs가 한 폴더를 읽으므로). → (pred_dir, doc_ids)"""
    dst = Path(root) / "pooled" / name / INFER_CONDITION[ctype] / "val"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    logs, runs, docs = [], [], []
    for k in sorted(set(fold_of.values())):
        p = Paths(root, name, k)
        src = p.pred_dir(ctype)
        done, total = infer_count(p, ctype, fold_of)
        if done < total and not partial:
            return None, f"fold {k} 추론 {done}/{total}"
        if not src.exists():
            continue
        log = src / "_log.jsonl"
        recs = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()] if log.exists() else []
        logged = {r["doc_id"] for r in recs}
        for d in held_out(fold_of, k):
            if (src / f"{d}.json").exists():
                shutil.copyfile(src / f"{d}.json", dst / f"{d}.json")
            if (src / f"{d}.json").exists() or d in logged:
                docs.append(d)
        logs += recs
        run = src / "_run.jsonl"
        if run.exists():
            runs.append({**json.loads(run.read_text(encoding="utf-8").splitlines()[0]), "fold": k})
    if not docs:
        return None, "출력 없음"
    (dst / "_log.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in logs), encoding="utf-8")
    # score.py의 프롬프트 버전 검사가 읽는 첫 줄 = fold 0 기록, 나머지 줄은 fold별 기록(어댑터 · few-shot · 입력 해시)
    (dst / "_run.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in runs), encoding="utf-8")
    versions = {r.get("prompt_version") for r in runs}
    if len(versions) > 1:
        sys.exit(f"[거부] {name}의 fold마다 프롬프트 버전이 다르다: {versions}")
    return dst, sorted(docs)


def score_condition(pred_dir, doc_ids, docs):
    check_prompt_version(pred_dir, DEFAULT_PROMPT)
    out = {}
    for label, subset_of in (("all", {d: "all" for d in doc_ids}), ("form", {d: docs[d]["form"] for d in doc_ids})):
        accs, _, details = score_docs(doc_ids, subset_of, LABEL_DIR, pred_dir, TEXT_DIR, quiet=True)
        out[label] = {s: dict(a.rows()) for s, a in accs.items()}
        if label == "all":
            with open(pred_dir / DETAIL_FILE, "w", encoding="utf-8") as f:
                f.write(json.dumps({"pooled": True}) + "\n")
                for d in details.values():
                    f.write(json.dumps(d, ensure_ascii=False, default=list) + "\n")
            out["details"] = details
    return out


def comparisons(exp):
    """짝 비교 목록: 설정의 comparisons, 없으면 qlora 조건마다 base_fs 대비 + baseline 밖 qlora 조건은 baseline 대비."""
    if exp.get("comparisons"):
        return [tuple(x) for x in exp["comparisons"]]
    conds, base = exp["conditions"], exp["baseline"]
    pairs = [(n, "base_fs") for n, c in conds.items() if c["type"] == "qlora" and "base_fs" in conds]
    pairs += [(n, "base_zs") for n, c in conds.items() if n == base and "base_zs" in conds]
    pairs += [(n, base) for n, c in conds.items() if c["type"] == "qlora" and n != base]
    return pairs


def condition_count(exp, names):
    """보고서의 조건 수 문장. seed 반복(replicate_of)은 고를 후보가 아니라 잡음 측정용이라 따로 센다."""
    reps = [n for n in names if exp["conditions"][n].get("replicate_of")]
    text = f"{len(names) - len(reps)}개"
    if reps:
        text += " (그 밖에 seed 반복 " + " · ".join(f"{n} = {exp['conditions'][n]['replicate_of']}의 반복" for n in reps) + \
                ". 고를 후보가 아니라 학습 잡음 측정용)"
    return text


def recipe_seeds(exp, recipe):
    """레시피와 그 seed 반복 → {학습 seed: 조건 이름}"""
    names = [recipe] + [n for n, c in exp["conditions"].items() if c.get("replicate_of") == recipe]
    return {(exp["conditions"][n].get("train") or {}).get("seed", TRAIN_SEED): n for n in names}


def seed_mean_verdict(outs, metrics, groups, ids, rs, bs, metric="doc_exact_ext"):
    """사전 등록 판정(cv/prereg_r1_aug.md): 레시피 rs와 기준 bs({seed: 조건})를 같은 seed 묶음으로 비교한다.
    outs[조건] = {doc: doc_outcomes}, metrics[조건] = 전체 지표. 셋 다 만족해야 "개선 확인":
      1. 문서마다 seed 평균 맞음 비율의 차이(레시피 − 기준), 그룹 부트스트랩 95% 구간 하한 > 0
      2. 같은 seed끼리의 평균 차이가 모든 seed에서 > 0
      3. PREREG_GUARD 지표의 seed 평균이 기준보다 낮지 않음
    → dict(판정 · 근거)."""
    seeds = sorted(rs)
    avg = lambda names: {d: {f: sum(outs[n][d][f] for n in names) / len(names) for f in outs[names[0]][d]} for d in ids}
    main = next(r for r in compare(avg([rs[x] for x in seeds]), avg([bs[x] for x in seeds]), groups, ids)
                if r["metric"] == metric)
    per_seed = {}
    for x in seeds:
        diffs = [outs[rs[x]][d][metric] - outs[bs[x]][d][metric] for d in ids]
        per_seed[x] = (sum(diffs) / len(diffs), sum(v > 0 for v in diffs), sum(v < 0 for v in diffs))
    guard = {}
    for m in PREREG_GUARD:
        a = sum(metrics[rs[x]][m] for x in seeds) / len(seeds)
        b = sum(metrics[bs[x]][m] for x in seeds) / len(seeds)
        guard[m] = (a, b, a >= b)
    failed = ([] if main["ci_low"] is not None and main["ci_low"] > 0 else ["seed 평균 구간 하한이 0 이하"])
    failed += [f"seed {x}에서 기준보다 낫지 않음" for x, (mdiff, _, _) in per_seed.items() if mdiff <= 0]
    failed += [f"{m} 악화" for m, (_, _, ok) in guard.items() if not ok]
    flips = {name: sum(outs[c[seeds[0]]][d][metric] != outs[c[x]][d][metric] for d in ids for x in seeds[1:])
             for name, c in (("recipe", rs), ("base", bs))}
    return {"main": main, "per_seed": per_seed, "guard": guard, "flips": flips, "failed": failed,
            "verdict": "개선 확인" if not failed else "개선을 확인하지 못했다"}


def sign_test_p(win, loss):
    """양측 정확 부호 검정(무승부 제외) p값. 승 · 패가 모두 0이면 1."""
    n = win + loss
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(win, loss) + 1)) / 2 ** n)


def consistency(out_a, out_b, ids, fold_of, groups, metric="doc_exact_ext"):
    """주지표 일관성: fold별 평균 차이, 문서 · 그룹 단위 승 · 패와 부호 검정. 그룹 단위는 그룹 안 차이의 합의 부호."""
    diffs = {d: out_a[d][metric] - out_b[d][metric] for d in ids}
    by_fold, by_group = defaultdict(list), defaultdict(list)
    for d, x in diffs.items():
        by_fold[fold_of[d]].append(x)
        by_group[groups[d]].append(x)
    g = [sum(v) for v in by_group.values()]
    w, lo = sum(x > 0 for x in diffs.values()), sum(x < 0 for x in diffs.values())
    gw, gl = sum(x > 0 for x in g), sum(x < 0 for x in g)
    return {"folds": {k: (sum(v) / len(v), len(v)) for k, v in sorted(by_fold.items())},
            "doc": (w, lo, len(diffs) - w - lo, sign_test_p(w, lo)),
            "group": (gw, gl, len(g) - gw - gl, sign_test_p(gw, gl))}


def fmt(v, metric=""):
    if v is None:
        return "—"
    if isinstance(v, float) and not metric.endswith(("_n", "_docs")):
        return f"{v:.1f}" if metric.endswith(("_mean", "_s")) else f"{v:.3f}"
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def manifest(root, exp, scored):
    runs = {}
    for name, c in exp["conditions"].items():
        if c["type"] != "qlora" or name not in scored:
            continue
        for k in range(exp["folds"]["k"]):
            cfg = Paths(root, name, k).run / "config.json"
            if cfg.exists():
                m = json.loads(cfg.read_text(encoding="utf-8"))
                runs[f"{name}/fold{k}"] = {"git": m.get("git"), "smoke": m.get("smoke"),
                                           "train_jsonl_sha256": m["data"]["train_jsonl_sha256"],
                                           "n_train": m["data"]["n_train"], "result": m.get("result")}
    code = {f: sha256_file(ROOT / f) for f in ("eval/cv.py", "eval/score.py", "eval/paired.py", "eval/infer.py",
                                                "pipeline/train_qlora.py", "pipeline/build_jsonl.py", "core/prompt.py",
                                                "core/normalize.py", "src/schema.py", "eval/hazard_class_alias.csv")}
    return {"created_at": datetime.now().isoformat(timespec="seconds"),
            "folds_csv_sha256": sha256_file(Path(root) / "folds.csv"),
            "experiments_sha256": sha256_file(EXPERIMENTS) if EXPERIMENTS.exists() else None,
            "conditions": exp["conditions"], "code_sha256": code, "runs": runs}


def cmd_report(args):
    exp = load_experiments(args.experiments)
    root = Path(args.root)
    fold_of = read_folds(root)
    docs = labeled_docs()
    groups = {d: docs[d]["split_group"] for d in docs}
    scored, skipped = {}, {}
    for name, c in exp["conditions"].items():
        pred_dir, info = pool(root, name, c["type"], fold_of, args.partial)
        if pred_dir is None:
            skipped[name] = info
            continue
        scored[name] = {**score_condition(pred_dir, info, docs), "docs": info}
    if not scored:
        sys.exit(f"채점할 조건이 없다: {skipped}")

    score_rows = [{"condition": n, "subset": s, "metric": m, "value": v}
                  for n, r in scored.items() for part in ("all", "form") for s, ms in r[part].items()
                  for m, v in ms.items()]
    write_csv(root / "scores.csv", score_rows, ["condition", "subset", "metric", "value"])

    paired_rows, consist = [], {}
    for a, b in comparisons(exp):
        if a not in scored or b not in scored:
            continue
        common = sorted(set(scored[a]["docs"]) & set(scored[b]["docs"]))
        out_a = {d: doc_outcomes(scored[a]["details"][d]) for d in common}
        out_b = {d: doc_outcomes(scored[b]["details"][d]) for d in common}
        consist[(a, b)] = consistency(out_a, out_b, common, fold_of, groups)
        subsets = [("all", common)] + [(f, [d for d in common if docs[d]["form"] == f])
                                       for f in sorted({docs[d]["form"] for d in common})]
        for subset, ids in subsets:
            if not ids:
                continue
            for r in compare(out_a, out_b, groups, ids):
                paired_rows.append({"a": a, "b": b, "subset": subset, **r})
    write_csv(root / "paired.csv", paired_rows,
              ["a", "b", "subset", "metric", "n_docs", "n_groups", "win", "loss", "tie", "mean_diff", "ci_low", "ci_high"])
    (root / "manifest.json").write_text(json.dumps(manifest(root, exp, scored), ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    prereg = {}
    for name, c in exp["conditions"].items():
        base = exp["baseline"]
        if c["type"] != "qlora" or name == base or c.get("replicate_of"):
            continue
        rs, bs = recipe_seeds(exp, name), recipe_seeds(exp, base)
        if len(rs) < 2 or set(rs) != set(bs) or not all(n in scored for n in [*rs.values(), *bs.values()]):
            continue
        conds = [*rs.values(), *bs.values()]
        common = sorted(set.intersection(*(set(scored[n]["docs"]) for n in conds)))
        outs = {n: {d: doc_outcomes(scored[n]["details"][d]) for d in common} for n in conds}
        prereg[(name, base)] = (rs, bs, seed_mean_verdict(outs, {n: scored[n]["all"]["all"] for n in conds},
                                                         groups, common, rs, bs))
    write_report(root, exp, scored, skipped, paired_rows, docs, args.partial, consist, prereg)


def cmd_stopcheck(args):
    """중간 멈춤 규칙(cv/prereg_r1_aug.md 3절): 한 seed에서 레시피 − 기준의 doc_exact_ext 평균 차이를 본다.
    > 0이면 다음 seed를 돌리고(종료 코드 0), 0 이하면 '같은 seed끼리 모두 이김' 조건이 이미 깨졌으니 멈춘다(3).
    두 조건 중 하나라도 5 fold가 끝나지 않았으면 판단하지 않는다(2)."""
    exp = load_experiments(args.experiments)
    root = Path(args.root)
    fold_of, docs = read_folds(root), labeled_docs()
    a, b = recipe_seeds(exp, args.recipe).get(args.seed), recipe_seeds(exp, exp["baseline"]).get(args.seed)
    if not a or not b:
        sys.exit(f"seed {args.seed}의 조건이 experiments.yaml에 없다(레시피 {a}, 기준 {b})")
    outs = {}
    for name in (a, b):
        pred_dir, info = pool(root, name, exp["conditions"][name]["type"], fold_of, False)
        if pred_dir is None:
            print(f"[판단 불가] {name}: {info}")
            sys.exit(2)
        details = score_condition(pred_dir, info, docs)["details"]
        outs[name] = {d: doc_outcomes(details[d])["doc_exact_ext"] for d in info}
    ids = sorted(set(outs[a]) & set(outs[b]))
    diffs = [outs[a][d] - outs[b][d] for d in ids]
    mean = sum(diffs) / len(diffs)
    go = mean > 0
    print(f"seed {args.seed}: {a} − {b} doc_exact_ext {mean:+.3f} (문서 {len(ids)}, 승 {sum(x > 0 for x in diffs)} · "
          f"패 {sum(x < 0 for x in diffs)}) → {'계속(다음 seed를 돌림)' if go else '멈춤(개선을 확인하지 못했다)'}")
    sys.exit(0 if go else 3)


def write_report(root, exp, scored, skipped, paired_rows, docs, partial, consist=None, prereg=None):
    names = list(scored)
    n_docs = {n: len(scored[n]["docs"]) for n in names}
    n_groups = len({docs[d]["split_group"] for n in names for d in scored[n]["docs"]})
    lines = ["# 그룹 교차검증 결과", ""]
    if partial:
        lines += ["> **부분 결과(--partial)** — 끝난 문서만 채점했다. 보고 수치로 쓰지 않는다.", ""]
    lines += [f"- {exp['folds']['k']}-fold, 제조사 그룹(split_group) 단위, seed {exp['folds']['seed']}. "
              f"문서 {max(n_docs.values())}건 · 그룹 {n_groups}개(train + val 라벨, test 미사용).",
              "- 문서마다 그 문서의 제조사를 학습에서 빼고 만든 모델로 한 번씩 채점했다. 채점 규칙은 eval/score.py와 같다.",
              "- 공식 r1 · test 수치(report/)와 다른 평가 방식이다. 섞어 보지 않는다.",
              f"- 생성: {datetime.now():%Y-%m-%d %H:%M}, `python3 eval/cv.py report`. 재현 정보는 manifest.json.", ""]
    if skipped:
        lines += ["채점하지 않은 조건: " + ", ".join(f"{n}({why})" for n, why in skipped.items()), ""]
    lines += ["## 전체 지표", "", "| 지표 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| n_docs | " + " | ".join(str(n_docs[n]) for n in names) + " |"]
    for m in REPORT_METRICS:
        lines.append(f"| {m} | " + " | ".join(fmt(scored[n]["all"]["all"].get(m), m) for n in names) + " |")
    forms = sorted({f for n in names for f in scored[n]["form"]})
    lines += ["", "## 서식별 문서 완전 정답(확장)", "", "| 서식 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for f in forms:
        cells = []
        for n in names:
            r = scored[n]["form"].get(f)
            cells.append(f"{fmt(r['doc_exact_ext_rate'], 'x')} ({r['n_docs']})" if r else "—")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    lines += ["", "## 짝 비교 (a − b, 문서별 맞음 1 / 틀림 0, 제조사 그룹 부트스트랩 95% 구간)", ""]
    shown = ("doc_exact_ext", "schema_rate", "ghs_classification", "hazard_statements", "cas", "pair", "nocas")
    for a, b in dict.fromkeys((r["a"], r["b"]) for r in paired_rows):
        lines += [f"### {a} vs {b}", "", "| 지표 | 문서 · 그룹 | 승 · 패 · 무 | 차이 [95% 구간] |", "|---|---|---|---|"]
        for r in paired_rows:
            if r["a"] == a and r["b"] == b and r["subset"] == "all" and r["metric"] in shown:
                ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]" if r["ci_low"] is not None else "(구간 없음)"
                lines.append(f"| {r['metric']} | {r['n_docs']} · {r['n_groups']} | {r['win']} · {r['loss']} · {r['tie']} "
                             f"| {r['mean_diff']:+.3f} {ci} |")
        lines.append("")
    if consist:
        ks = sorted({k for c in consist.values() for k in c["folds"]})
        lines += ["## 주지표 일관성 (doc_exact_ext, a − b)", "",
                  "| 비교 | " + " | ".join(f"fold {k}" for k in ks) + " | 문서 승 · 패 · 무 (부호 검정 p) | 그룹 승 · 패 · 무 (p) |",
                  "|---|" + "---|" * (len(ks) + 2)]
        for (a, b), c in consist.items():
            cells = [f"{c['folds'][k][0]:+.2f} ({c['folds'][k][1]})" if k in c["folds"] else "—" for k in ks]
            (w, lo, t, p), (gw, gl, gt, gp) = c["doc"], c["group"]
            lines.append(f"| {a} vs {b} | " + " | ".join(cells) + f" | {w} · {lo} · {t} (p={p:.3f}) | {gw} · {gl} · {gt} (p={gp:.3f}) |")
        lines += ["", "- fold 칸: 그 fold 문서들의 평균 차이(문서 수). 문서 단위 부호 검정은 같은 그룹 문서끼리의 상관을 무시해 "
                  "낙관적이다. 그룹 단위가 더 보수적이다.", ""]
    for (name, base), (rs, bs, v) in (prereg or {}).items():
        m = v["main"]
        seeds = sorted(rs)
        lines += [f"## 사전 등록 판정: {name} vs {base} (seed {' · '.join(map(str, seeds))} 평균, doc_exact_ext)", "",
                  f"- 기준: cv/prereg_r1_aug.md. 레시피 {' · '.join(rs[x] for x in seeds)} / 기준 {' · '.join(bs[x] for x in seeds)}.",
                  f"- **판정: {v['verdict']}**" + (f" — {', '.join(v['failed'])}" if v["failed"] else ""), "",
                  "| 항목 | 값 | 조건 | 통과 |", "|---|---|---|---|",
                  f"| seed 평균 차이 [95% 구간] | {m['mean_diff']:+.3f} [{m['ci_low']:+.3f}, {m['ci_high']:+.3f}] "
                  f"(문서 {m['n_docs']} · 그룹 {m['n_groups']}) | 하한 > 0 | {'예' if m['ci_low'] > 0 else '아니오'} |"]
        for x, (md, w, lo) in v["per_seed"].items():
            lines.append(f"| seed {x}: {rs[x]} − {bs[x]} | {md:+.3f} (승 {w} · 패 {lo}) | > 0 | {'예' if md > 0 else '아니오'} |")
        for k, (a, b, ok) in v["guard"].items():
            lines.append(f"| {k} (seed 평균) | {a:.3f} vs {b:.3f} | 기준 이상 | {'예' if ok else '아니오'} |")
        lines += ["", f"- 참고(판정에 쓰지 않음): seed끼리 뒤집힌 문서 — {name} {v['flips']['recipe']}건, {base} {v['flips']['base']}건.", ""]
    lines += ["## 읽을 때 주의", "",
              "- 구서식 문서는 모두 한 제조사 그룹이라 한 fold에 몰린다. 그 fold의 모델은 구서식을 학습에서 보지 못한다.",
              "- 같은 45건으로 여러 레시피를 비교해 고르면 고른 레시피의 점수는 조금 낙관적이다. 비교한 조건 수를 함께 적는다.",
              f"- 이 보고서의 조건 수: {condition_count(exp, names)}.", ""]
    (Path(root) / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"→ {rel(Path(root) / 'report.md')} · scores.csv · paired.csv · manifest.json")
    print("\n".join(lines[lines.index("## 전체 지표"):]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(CV_ROOT), help="결과 폴더(기본 cv/)")
    ap.add_argument("--experiments", default=str(EXPERIMENTS), help="조건 설정(기본 cv/experiments.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("folds", help="fold 나누기")
    r = sub.add_parser("run", help="build → train → infer")
    r.add_argument("--cond", required=True, help="조건 이름(쉼표로 여러 개)")
    r.add_argument("--folds", help="돌릴 fold(예: 0,1). 기본 전부")
    r.add_argument("--dry-run", action="store_true", help="명령만 출력")
    r.add_argument("--smoke", action="store_true",
                   help=f"동작 점검: 학습 {SMOKE_TRAIN_STEPS} step, 추론 {SMOKE_INFER_LIMIT}건. --root를 저장소 밖으로 줄 것")
    sub.add_parser("status", help="진행 상황")
    rp = sub.add_parser("report", help="모아서 채점 · 짝 비교 · 보고서")
    rp.add_argument("--partial", action="store_true", help="끝난 문서만 채점(동작 점검용, 보고 수치 아님)")
    sc = sub.add_parser("stopcheck", help="중간 멈춤 규칙(cv/prereg_r1_aug.md 3절)")
    sc.add_argument("--recipe", required=True, help="레시피 조건 이름(예: r1_aug)")
    sc.add_argument("--seed", type=int, default=TRAIN_SEED, help=f"먼저 돌린 seed(기본 {TRAIN_SEED})")
    args = ap.parse_args(argv)
    if getattr(args, "smoke", False) and Path(args.root).resolve() == CV_ROOT.resolve():
        ap.error("--smoke는 cv/에 쓰지 않는다. --root <저장소 밖 폴더>를 줄 것")
    {"folds": cmd_folds, "run": cmd_run, "status": cmd_status, "report": cmd_report,
     "stopcheck": cmd_stopcheck}[args.cmd](args)


if __name__ == "__main__":
    main()
