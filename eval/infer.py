"""
[양세윤] 조건(base_zs/base_fs/qlora_*) × split(val/test) 추론
--split test는 명시 플래그 없이는 실행 거부

세 조건 모두 같은 런타임(Transformers, 4bit NF4)·같은 입력·thinking 끔·greedy로 돌린다.
파싱·스키마 검사·채점은 하지 않는다 → eval/score.py

출력 (CLAUDE.md 규약)
  outputs/{condition}/{split}/{doc_id}.json   모델이 낸 문자열 그대로. 파싱에 실패해도 그대로 저장해 실패로 센다
  outputs/{condition}/{split}/_log.jsonl      문서별 시간·입출력 토큰·max_new_tokens 도달 여부·건너뛴 사유
  outputs/{condition}/{split}/_run.jsonl      첫 실행 설정(베이스 · 리비전 · 어댑터 · few-shot · 생성 길이 · 프롬프트 버전 ·
                                              프롬프트 문구 해시 · 입력 해시 = 문서 목록 + 1~3항 텍스트 + _cut_log 상태)
  --prompt v2(core/prompt.py PROMPTS)의 출력은 outputs/prompt_v2/{condition}/{split}/에 따로 쌓인다(--out-root를
  줘도 그 아래 prompt_v2/). v1 폴더 · report/scores.csv를 덮어쓰지 않는다. 폴더의 _run.jsonl과 버전이 다르면 거부
  - qlora_*: 어댑터를 학습한 프롬프트 버전(어댑터 옆 run 폴더의 config.json)과 --prompt가 다르면 거부한다.
    버전을 확인할 수 없는 어댑터(config.json 없음 등)도 거부한다
  - _cut_log.csv에서 NOT_FOUND인 문서는 모델에 넣지 않고 _log.jsonl에 skipped로 남긴다(채점에선 실패로 셈)
  - 이미 출력 파일이 있으면 건너뛴다(중단 후 이어 돌리기). 다시 돌리려면 --overwrite(test는 불가)
  - 이어 돌릴 때 설정이 _run.jsonl과 다르면 거부한다(한 폴더에 두 모델 · 두 입력의 출력이 섞이지 않게)
  - 베이스는 pipeline/configs/r1.yaml의 model · model_revision(학습과 같은 스냅샷)으로 불러온다. 리비전이 없거나
    (--base만 주고 --revision을 안 준 경우), 불러온 스냅샷을 확인할 수 없거나, 요청과 다르면 멈춘다.
    확인한 실제 스냅샷은 _log.jsonl의 문서별 base_commit에 남는다

  python3 eval/infer.py --condition base_zs --split val
  python3 eval/infer.py --condition base_fs --split val
  python3 eval/infer.py --condition qlora_r1 --split val --adapter runs/MMDD_r1/adapter
  python3 eval/infer.py --condition base_fs --split val --prompt v2     # → outputs/prompt_v2/base_fs/val/
  # 코드 점검(fixture, 옛 라벨): --splits test/fixtures/splits.csv --text-dir test/fixtures/text --out-root /tmp/x

test 규칙 (docs/plan.md 3.3 · 9절 단계 7~8)
  test 문서·텍스트는 저장소 밖(test 담당)에 있다. --allow-test + 실험 고정(eval/experiment.json) +
  저장소 밖 --text-dir · --out-root가 모두 있어야 돈다. 비교군마다 1회, --overwrite · --limit 불가.
  - --text-dir의 텍스트는 봉인(eval/seal.py, report/test_manifest.csv)과 같아야 한다. 봉인 전에는 돌지 않는다.
  - 비교군은 base_zs · base_fs · qlora_final만(plan 5절). qlora_r1 · r2 · r3는 val에서만 비교한다.
  - --max-new-tokens를 주면 experiment.json의 값과 같아야 하고, qlora_final의 --adapter는 폴더 해시가
    experiment.json의 adapter_sha256과 같아야 한다(고정한 뒤 다른 값·다른 어댑터로 도는 것을 막음).
  - 출력(모델 출력 원문 · _log.jsonl)은 --out-root 아래 {condition}/test/에 생긴다. 저장소 안에 두지 않는다.

eval/experiment.json (단계 7에서 PM이 커밋)
  {"max_new_tokens": 1268,                       # train·val 정답 최장 토큰 × 1.3 (plan 6절)
   "adapter": "runs/MMDD_r1/adapter",           # qlora_final로 고른 어댑터(참고용 경로)
   "adapter_sha256": "...",                     # python3 -c "from eval.infer import sha256_dir; print(sha256_dir('<폴더>'))"
   "prompt_version": "v1"}                      # 비교군 3개가 모두 이 프롬프트로 돈다(qlora_final을 학습한 버전)
  채점기 · 별칭표 해시 등 다른 키를 더 넣어도 된다. 위 세 키(max_new_tokens · adapter_sha256 · prompt_version)는 필수.

다른 test 세트(cv/prereg_newtest.md): --experiment cv/newtest/experiment.json --manifest cv/newtest/test_manifest.csv
  - 고정 파일에 adapter_sha256 대신 "adapters": {이름: 해시}를 둘 수 있다. --adapter 해시가 그 목록에 있어야 돈다.
    어댑터마다 --out-root를 따로 준다(<NEWTEST_ROOT>/out/r1_f0 등, 조건 이름은 qlora_final 그대로).
  - "allow_resume": true면, 끊긴 뒤 같은 설정(_run.jsonl 대조)으로 이어 돌릴 때 없는 문서만 생성한다.
    이미 생성한 문서는 다시 만들지 않는다(--overwrite는 여전히 불가). 기본(팀 test)은 이어 돌리기도 거부.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.prompt import (  # noqa: E402
    CHAT_TEMPLATE_KWARGS,
    DEFAULT_PROMPT,
    LEGACY_V1_FILE_SHA256,
    PROMPTS,
    build_messages,
    prompt_sha256,
    prompt_subdir,
    recorded_prompt_version,
)
from eval.seal import require_sealed  # noqa: E402

CONDITIONS = ("base_zs", "base_fs", "qlora_r1", "qlora_r2", "qlora_r3", "qlora_final")
TEST_CONDITIONS = ("base_zs", "base_fs", "qlora_final")  # test는 이 셋만 1회씩(plan 5절 · 9절 단계 8)
SPLITS = ("val", "train", "val_en", "test")  # train: Base 난이도 진단 전용(few-shot 예시 제외, 보고 금지)

SPLITS_CSV = ROOT / "data" / "splits.csv"
TEXT_DIR = ROOT / "data" / "text"
LABEL_DIR = ROOT / "data" / "labels"
FEWSHOT_JSON = ROOT / "eval" / "fewshot.json"
EXPERIMENT_JSON = ROOT / "eval" / "experiment.json"  # 단계 7(모델·실험 고정)에서 PM이 커밋
DEFAULT_CONFIG = ROOT / "pipeline" / "configs" / "r1.yaml"
OUTPUT_ROOT = ROOT / "outputs"
DEFAULT_MAX_NEW_TOKENS = 2048
# 실험 고정 전 임시값(2026-09-28, docs/plan.md 6절 "생성 길이 부족" 참고).
# - 진짜 고정값은 train·val 정답 최장 토큰 × 1.3이다. 새 라벨이 나오면 pipeline/length_stats.py로 재서
#   eval/experiment.json에 채우고 실험을 고정한다(단계 7). 그 전까지 이 상수는 val·train 진단에만 쓰인다
#   (test는 이 상수를 쓰지 않는다 — main()에서 experiment.json 값을 쓴다).
# - 1024는 부족했다: fixture 문서(KR-HENKEL-001, 입력 2,141토큰)가 1024에서 잘려 JSON이 깨졌다. 옛 라벨
#   기준 정답 최장이 약 975토큰이라 × 1.3 ≈ 1,268이었는데도 1024로는 못 미친 사례라, 여유를 더 둔다.
# - 4096(=r1.yaml의 max_length)을 그대로 쓰지 않는다. max_length는 학습 때 입력+정답 합계 길이의 상한이고,
#   max_new_tokens는 추론 때 새로 생성하는 토큰 수만의 상한이라 용도가 다르다. EOS 없이 도는 실패 사례가
#   나오면 이 값까지 다 채우고서야 멈추므로, 진단 단계(실패가 잦을 수 있는 구간)에서 4096은 문서당 생성
#   시간을 크게 늘릴 위험이 있다. 2048은 1024보다 넉넉하면서 그 위험을 줄인 절충값이다.

LOG_FILE = "_log.jsonl"
RUN_FILE = "_run.jsonl"


# ---------------------------------------------------------------- 입력 준비
def read_splits(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["doc_id"]: r for r in csv.DictReader(f)}


def read_cut_status(text_dir):
    cut_log = Path(text_dir) / "_cut_log.csv"
    if not cut_log.exists():
        return {}
    with open(cut_log, encoding="utf-8-sig", newline="") as f:
        return {r["doc_id"]: r["status"] for r in csv.DictReader(f) if r.get("doc_id")}


def fewshot_ids_in_file(path=FEWSHOT_JSON):
    return json.loads(Path(path).read_text(encoding="utf-8")).get("fewshot_doc_ids", [])


def load_fewshot(splits, text_dir, label_dir, path=FEWSHOT_JSON):
    """few-shot 예시 파일(기본 eval/fewshot.json)의 train 문서 2건 → (ids, [(텍스트, 정답 dict), ...])"""
    ids = fewshot_ids_in_file(path)
    if len(ids) != 2:
        sys.exit(f"eval/fewshot.json의 fewshot_doc_ids는 2건이어야 함(현재 {ids}). train에서 골라 PM 승인 후 고정")
    for doc_id in ids:
        if splits.get(doc_id, {}).get("split") != "train":
            sys.exit(f"few-shot 예시는 train 문서만 쓸 수 있음: {doc_id}")
        for p in (Path(text_dir) / f"{doc_id}.txt", Path(label_dir) / f"{doc_id}.json"):
            if not p.exists():
                sys.exit(f"few-shot 예시 파일 없음: {p}")
    pairs = [
        ((Path(text_dir) / f"{d}.txt").read_text(encoding="utf-8"),
         json.loads((Path(label_dir) / f"{d}.json").read_text(encoding="utf-8")))
        for d in ids
    ]
    return ids, pairs


def base_model_spec(base, revision):
    """→ (베이스 모델 이름, 리비전). --base가 없으면 학습과 같은 r1.yaml의 model · model_revision."""
    if base:
        return base, revision
    import yaml

    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    return cfg["model"], revision or cfg.get("model_revision")


def inputs_digest(doc_ids, text_dir, cut):
    """이 폴더의 출력을 만든 입력 전체의 해시: 문서 목록 + 문서별 _cut_log 상태 + 1~3항 텍스트.
    --limit과 상관없이 split 전체 문서로 계산한다(나눠 돌려도 같은 값)."""
    h = hashlib.sha256()
    for d in sorted(doc_ids):
        p = Path(text_dir) / f"{d}.txt"
        h.update(f"{d}\0{cut.get(d, '')}\0".encode())
        h.update(p.read_bytes() if p.exists() else b"<no text>")
        h.update(b"\0")
    return h.hexdigest()


def sha256_dir(path):
    """어댑터 폴더의 파일 이름·내용 해시(같은 조건 이름에 다른 어댑터가 섞이는 것을 막는 용도)"""
    h = hashlib.sha256()
    for p in sorted(Path(path).rglob("*")):
        if p.is_file():
            h.update(p.relative_to(path).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def adapter_prompt_version(adapter):
    """어댑터를 학습한 프롬프트 버전 ← 같은 run 폴더의 config.json(pipeline/train_qlora.py가 씀). 모르면 None.
    - 09-29 이후 학습: augmentation.prompt의 버전 + 문구 해시(같은 버전 이름에 다른 문구면 None)
    - 그 전 학습(r1): 버전 기록 없이 증강 보고서의 core/prompt.py 파일 해시만 있음 → 그 해시가 v1이면 v1"""
    cfg_path = Path(adapter).resolve().parent / "config.json"
    if not cfg_path.exists():
        return None
    aug = json.loads(cfg_path.read_text(encoding="utf-8")).get("augmentation") or {}
    rec = aug.get("prompt") or {}
    if rec.get("version") in PROMPTS:
        return rec["version"] if rec.get("text_sha256") == prompt_sha256(rec["version"]) else None
    return "v1" if (aug.get("code_sha256") or {}).get("core/prompt.py") == LEGACY_V1_FILE_SHA256 else None


def run_record(args, base_name, revision, fewshot_ids, fewshot_pairs, n_docs, inputs_sha256):
    """출력 폴더 하나를 만든 설정(시간 제외). 이어 돌릴 때 이 값이 같아야 한다."""
    return {
        "condition": args.condition,
        "split": args.split,
        "base_model": base_name,
        "base_revision": revision,
        "adapter": args.adapter,
        "adapter_sha256": sha256_dir(args.adapter) if args.adapter else None,
        "fewshot_doc_ids": fewshot_ids,
        "fewshot_sha256": hashlib.sha256(json.dumps(fewshot_pairs, ensure_ascii=False).encode()).hexdigest()
        if fewshot_pairs else None,
        "max_new_tokens": args.max_new_tokens,
        "prompt_version": args.prompt,
        "prompt_text_sha256": prompt_sha256(args.prompt),
        "n_docs": n_docs,
        "inputs_sha256": inputs_sha256,
    }


def check_resume(out_dir, run):
    """폴더에 첫 실행 설정이 있으면 지금 설정과 같아야 한다. 없으면 새로 남긴다."""
    run_path = out_dir / RUN_FILE
    if run_path.exists():
        first = json.loads(run_path.read_text(encoding="utf-8").splitlines()[0])
        first.pop("created_at", None)
        if "prompt_version" not in first:  # 09-29 이전 기록: 프롬프트 파일 해시만 있음(v1이면 그 해시로 판정)
            v = recorded_prompt_version(first)
            first.update(prompt_version=v, prompt_text_sha256=prompt_sha256(v) if v else None)
        if first["prompt_version"] != run["prompt_version"]:
            sys.exit(f"[거부] {out_dir}는 프롬프트 {first['prompt_version']}의 결과 폴더다(지금 {run['prompt_version']}). "
                     "다른 버전의 출력을 섞거나 덮어쓰지 않는다")
        if first.get("prompt_text_sha256") != run["prompt_text_sha256"]:
            sys.exit(f"[거부] {out_dir}를 만든 뒤 프롬프트 {run['prompt_version']}의 문구가 바뀌었다. "
                     "기존 버전 문구는 고치지 말고 core/prompt.py에 새 버전(예: v2_1)을 추가해 --prompt로 돌릴 것")
        diff = sorted(k for k in run if first.get(k) != run[k])
        if diff:
            sys.exit(f"[거부] {out_dir}의 첫 실행과 설정이 다름: {diff}\n"
                     f"  → 같은 설정으로 이어 돌리거나, 다른 조건 이름/폴더를 쓸 것")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps({**run, "created_at": now()}, ensure_ascii=False) + "\n", encoding="utf-8")


def now():
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- 모델
def verify_commit(model, revision):
    """실제로 불러온 베이스 스냅샷(HF 커밋 해시) → 요청한 리비전과 같을 때만 그 해시를 돌려준다.
    확인할 수 없거나(None) 다르면 멈춘다."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    commit = getattr(getattr(base, "config", None), "_commit_hash", None)
    if not commit:
        sys.exit("[거부] 불러온 베이스 스냅샷(커밋 해시)을 확인할 수 없다")
    if commit != revision:
        sys.exit(f"[거부] 불러온 베이스 스냅샷 {commit}이 요청한 리비전 {revision}과 다르다")
    return commit


def load_model(base_name, adapter, revision=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(base_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        base_name, revision=revision, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def generate(tok, model, messages, max_new_tokens):
    """→ (출력 문자열, 생성 시간 초, 입력 토큰 수, 출력 토큰 수)"""
    import torch

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    n_in = int(enc["input_ids"].shape[1])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            temperature=None,  # 모델 generation_config의 샘플링 값 무시
            top_p=None,
            top_k=None,
            pad_token_id=pad_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = out[0, n_in:]
    return tok.decode(new_tokens, skip_special_tokens=True), elapsed, n_in, int(new_tokens.shape[0])


# ---------------------------------------------------------------- 실행
def outside_repo(path):
    try:
        Path(path).resolve().relative_to(ROOT)
        return False
    except ValueError:
        return True


def load_experiment(path=None):
    """실험 고정 파일(기본 eval/experiment.json, 단계 7 고정값). 필수 키가 없으면 거부.
    어댑터는 adapter_sha256(1개) 또는 adapters({이름: 해시}, 다른 test 세트용) 중 하나가 있어야 한다."""
    path = Path(path or EXPERIMENT_JSON)
    if not path.exists():
        sys.exit(f"[거부] {path.name}(실험 고정)이 없다. 실험을 고정해 커밋한 뒤에만 test를 돌린다")
    exp = json.loads(path.read_text(encoding="utf-8"))
    lacking = [k for k in ("max_new_tokens", "prompt_version") if not exp.get(k)]
    if not exp.get("adapter_sha256") and not exp.get("adapters"):
        lacking.append("adapter_sha256 또는 adapters")
    if lacking:
        sys.exit(f"[거부] {path.name}에 {lacking}가 없다")
    return exp


def allowed_adapters(exp):
    """→ {해시: 이름}. 고정 파일에 적힌 어댑터만 test에서 돈다."""
    out = {h: n for n, h in (exp.get("adapters") or {}).items()}
    if exp.get("adapter_sha256"):
        out[exp["adapter_sha256"]] = exp.get("adapter") or "adapter_sha256"
    return out


def check_args(args):
    if args.condition.startswith("qlora") and not args.adapter:
        sys.exit(f"{args.condition}에는 --adapter(어댑터 폴더)가 필요함")
    if args.condition.startswith("base") and args.adapter:
        sys.exit(f"{args.condition}은 베이스 모델 조건이라 --adapter를 줄 수 없음")
    if args.adapter and not Path(args.adapter).exists():
        sys.exit(f"어댑터 폴더 없음: {args.adapter}")
    if args.adapter:
        trained = adapter_prompt_version(args.adapter)
        if trained is None:
            sys.exit(f"[거부] {args.adapter}를 학습한 프롬프트 버전을 확인할 수 없다"
                     f"({Path(args.adapter).resolve().parent / 'config.json'}의 augmentation 기록)")
        if trained != args.prompt:
            sys.exit(f"[거부] {args.adapter}는 프롬프트 {trained}로 학습했다(--prompt {args.prompt}). "
                     "학습과 다른 프롬프트로 추론하지 않는다")
    if args.split == "train" and not args.condition.startswith("base"):
        sys.exit("[거부] train은 Base 난이도 진단 전용. 학습한 모델을 학습 문서로 채점하는 건 의미가 없음")
    if args.split == "test":
        if not args.allow_test:
            sys.exit("[거부] test 추론은 --allow-test가 있어야 한다(모델·실험 고정 뒤, 비교군마다 1회)")
        if args.condition not in TEST_CONDITIONS:
            sys.exit(f"[거부] test 비교군은 {TEST_CONDITIONS}뿐이다. qlora_r1 · r2 · r3는 val에서 비교해 qlora_final로 고른다")
        exp = load_experiment(args.experiment)
        exp_name = Path(args.experiment or EXPERIMENT_JSON).name
        if not args.text_dir or not outside_repo(args.text_dir):
            sys.exit("[거부] test 텍스트는 저장소 밖에 있어야 한다: --text-dir <test 담당이 넘긴 폴더>")
        if not outside_repo(args.out_root):
            sys.exit("[거부] test 출력은 저장소 밖에 둔다: --out-root <저장소 밖 폴더>")
        require_sealed("text", args.text_dir, args.manifest)  # 넘겨받은 텍스트가 봉인(--manifest, 기본 팀 test)과 같아야 함
        if args.overwrite or args.limit:
            sys.exit("[거부] test에는 --overwrite · --limit을 쓰지 않는다(비교군마다 전체 문서 1회)")
        if Path(args.fewshot_json).resolve() != FEWSHOT_JSON.resolve():
            sys.exit("[거부] test의 base_fs는 고정한 eval/fewshot.json으로만 돈다(--fewshot-json 불가)")
        if args.max_new_tokens is not None and args.max_new_tokens != exp["max_new_tokens"]:
            sys.exit(f"[거부] --max-new-tokens {args.max_new_tokens} ≠ {exp_name}의 {exp['max_new_tokens']}. "
                     "test는 고정값으로만 돈다(생략하면 고정값을 씀)")
        if args.prompt != exp["prompt_version"]:
            sys.exit(f"[거부] --prompt {args.prompt} ≠ {exp_name}의 prompt_version {exp['prompt_version']}. "
                     "test 비교군은 모두 고정한 프롬프트로 돈다")
        if args.adapter and sha256_dir(args.adapter) not in allowed_adapters(exp):
            sys.exit(f"[거부] --adapter {args.adapter}의 해시가 {exp_name}의 adapter_sha256 · adapters 목록에 없다"
                     f"(고정한 어댑터: {sorted(allowed_adapters(exp).values())})")


def select_docs(args, splits, text_dir):
    """split 전체 문서(--limit 적용 전)."""
    if args.split == "test":  # 실패 문서는 .txt가 없어도 _cut_log에 남는다
        cut = read_cut_status(text_dir)
        return sorted({p.stem for p in Path(text_dir).glob("*.txt")} | set(cut))
    doc_ids = sorted(d for d, r in splits.items() if r["split"] == args.split)
    if args.split == "train":  # few-shot 예시는 base_fs가 정답을 보고 푸는 셈이라 두 조건 모두에서 뺀다
        doc_ids = [d for d in doc_ids if d not in set(fewshot_ids_in_file())]
    return doc_ids


def append_log(out_dir, rec):
    with open(out_dir / LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--adapter", help="qlora_* 조건의 LoRA 어댑터 폴더")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default=DEFAULT_PROMPT,
                    help=f"프롬프트 버전(기본 {DEFAULT_PROMPT}). v1이 아니면 출력은 <out-root>/prompt_<버전>/ 아래. "
                         "qlora_*는 어댑터를 학습한 버전과 같아야 한다")
    ap.add_argument("--base", help="베이스 모델 이름(기본: pipeline/configs/r1.yaml의 model)")
    ap.add_argument("--revision", help="베이스 리비전(기본: --base가 없으면 r1.yaml의 model_revision)")
    ap.add_argument("--max-new-tokens", type=int, help=f"기본: test는 eval/experiment.json 고정값, 그 밖은 {DEFAULT_MAX_NEW_TOKENS}")
    ap.add_argument("--allow-test", action="store_true", help="test 추론 허용(실험 고정 뒤에만)")
    ap.add_argument("--overwrite", action="store_true", help="이미 있는 출력도 다시 생성(test 불가)")
    ap.add_argument("--limit", type=int, help="앞에서 N건만(동작 점검용)")
    ap.add_argument("--splits", default=SPLITS_CSV, help="분할표(코드 점검 때 test/fixtures/splits.csv)")
    ap.add_argument("--text-dir", help="1~3항 텍스트 폴더(기본 data/text, test는 저장소 밖 폴더 필수)")
    ap.add_argument("--label-dir", default=LABEL_DIR, help="few-shot 예시 정답 폴더(base_fs만 씀)")
    ap.add_argument("--fewshot-json", default=FEWSHOT_JSON,
                    help="base_fs 예시 목록 파일(기본 eval/fewshot.json). 교차검증(eval/cv.py)이 fold마다 따로 넘긴다")
    ap.add_argument("--out-root", default=OUTPUT_ROOT, help="출력 루트(기본 outputs/, test는 저장소 밖 폴더 필수)")
    ap.add_argument("--experiment", help="test 실험 고정 파일(기본 eval/experiment.json = 팀 test)")
    ap.add_argument("--manifest", help="test 봉인 기록(기본 report/test_manifest.csv = 팀 test)")
    args = ap.parse_args(argv)
    check_args(args)
    if args.max_new_tokens is None:
        args.max_new_tokens = (load_experiment(args.experiment)["max_new_tokens"] if args.split == "test"
                               else DEFAULT_MAX_NEW_TOKENS)

    text_dir = Path(args.text_dir or TEXT_DIR)
    splits = {} if args.split == "test" else read_splits(args.splits)
    all_ids = select_docs(args, splits, text_dir)
    doc_ids = all_ids[: args.limit] if args.limit else all_ids
    if not doc_ids:
        sys.exit(f"{args.split} 문서가 0건({args.splits})")
    cut = read_cut_status(text_dir)
    fewshot_ids, fewshot = (load_fewshot(read_splits(args.splits), TEXT_DIR if args.split == "test" else text_dir,
                                         args.label_dir, args.fewshot_json)
                            if args.condition == "base_fs" else ([], []))
    base_name, revision = base_model_spec(args.base, args.revision)
    if not revision:
        sys.exit("[거부] 베이스 리비전이 없다. --base를 줄 때는 --revision도 준다(기본은 r1.yaml의 model_revision)")
    out_dir = Path(args.out_root) / prompt_subdir(args.prompt) / args.condition / args.split

    # 모델을 올리기 전에 입력부터 전부 확인(모델 로딩에 1분 넘게 걸림)
    missing = [d for d in doc_ids if cut.get(d) != "NOT_FOUND" and not (text_dir / f"{d}.txt").exists()]
    if missing:
        sys.exit(f"전처리 텍스트 없음 {len(missing)}건: {missing}\n→ pipeline/extract_text.py를 먼저 돌릴 것")
    check_resume(out_dir, run_record(args, base_name, revision, fewshot_ids, fewshot,
                                     len(all_ids), inputs_digest(all_ids, text_dir, cut)))
    if args.split == "test" and any(out_dir.glob("*.json")) and not load_experiment(args.experiment).get("allow_resume"):
        sys.exit(f"[거부] {out_dir}에 이미 출력이 있다. test는 비교군마다 1회만 생성한다")
    todo = [d for d in doc_ids if args.overwrite or not (out_dir / f"{d}.json").exists()]
    if not todo:
        print(f"할 일 없음: {out_dir}에 {len(doc_ids)}건 모두 있음(다시 돌리려면 --overwrite)")
        return

    print(f"[{args.condition} / {args.split}] {len(todo)}/{len(doc_ids)}건, base={base_name}@{revision}, adapter={args.adapter}")
    tok, model = load_model(base_name, args.adapter, revision)
    commit = verify_commit(model, revision)
    print(f"베이스 스냅샷: {commit}")
    generate(tok, model, [{"role": "user", "content": "warm-up"}], 8)  # CUDA 초기화 시간이 첫 문서 속도에 섞이지 않게

    for n, doc_id in enumerate(todo, 1):
        rec = {"doc_id": doc_id, "skipped": None, "gen_time_sec": None, "input_tokens": None,
               "output_tokens": None, "hit_max_new_tokens": False, "base_commit": commit, "created_at": now()}
        if cut.get(doc_id) == "NOT_FOUND":
            rec["skipped"] = "NOT_FOUND"
            (out_dir / f"{doc_id}.json").unlink(missing_ok=True)  # --overwrite 때 이전 출력이 남지 않게
            print(f"  ({n}/{len(todo)}) {doc_id}  건너뜀: _cut_log NOT_FOUND")
        else:
            text = (text_dir / f"{doc_id}.txt").read_text(encoding="utf-8")
            msgs = build_messages(text, fewshot, prompt=args.prompt)
            raw, sec, n_in, n_out = generate(tok, model, msgs, args.max_new_tokens)
            rec.update(gen_time_sec=round(sec, 3), input_tokens=n_in, output_tokens=n_out,
                       hit_max_new_tokens=n_out >= args.max_new_tokens)
            (out_dir / f"{doc_id}.json").write_text(raw, encoding="utf-8")
            flag = "  ← max_new_tokens 도달(잘렸을 수 있음)" if rec["hit_max_new_tokens"] else ""
            print(f"  ({n}/{len(todo)}) {doc_id}  {sec:.1f}s  in={n_in} out={n_out}{flag}")
        append_log(out_dir, rec)

    nxt = " --allow-test --out-root <같은 폴더> --label-dir · --subset-csv · --text-dir <저장소 밖>" if args.split == "test" else ""
    pv = f" --prompt {args.prompt}" if args.prompt != DEFAULT_PROMPT else ""
    print(f"완료 → {out_dir}   다음: python3 eval/score.py --condition {args.condition} --split {args.split}{pv}{nxt}")


if __name__ == "__main__":
    main()
