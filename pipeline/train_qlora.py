"""
[강덕우] + 양세윤 — QLoRA 학습, runs/{날짜}_{r1|r2|r3}/에 config.json·loss 기록

  python3 pipeline/train_qlora.py --config pipeline/configs/r1.yaml --memcheck   # 가장 긴 예시로 1 step, 최대 메모리만 확인
  python3 pipeline/train_qlora.py --config pipeline/configs/r1.yaml --smoke 10   # 스모크: 10 step, runs/ 밖(/tmp)에 기록
  python3 pipeline/train_qlora.py --config pipeline/configs/r1.yaml              # 학습(PM 승인 후)
  python3 pipeline/train_qlora.py --config pipeline/configs/r2.yaml              # r1.yaml 위에 r2.yaml 값만 덮어씀(r3.yaml도 같음)

입력: data/train.jsonl, data/val.jsonl (pipeline/build_jsonl.py 결과, 한 줄 = {"doc_id", "split", "variant", "messages"})
출력: runs/{MMDD}_{이름}/
  adapter/       LoRA 어댑터 (eval/infer.py --adapter 로 넘긴다, 가중치는 gitignore)
  config.json    실제로 쓴 설정 + 데이터 해시 + 증강 보고서 요약 + 환경 + 결과(step · 시간 · 최대 메모리)
  loss.csv       step별 train loss, epoch별 val loss(val이 있을 때)

추론과 한 글자도 다르지 않게:
  프롬프트 = tok.apply_chat_template(system+user, add_generation_prompt=True, enable_thinking=False)  ← eval/infer.py와 같은 호출
  정답     = format_target(label) + "<|im_end|>"
  loss는 정답 토큰에만 건다(프롬프트는 -100).

모델을 올리기 전에 멈추는 경우(precheck): train · val JSONL이 없거나 빔, 데이터 용도 위반, 증강 보고서 없음,
보고서와 지금의 train · val JSONL · 라벨 · 분할표가 다름, JSONL의 시스템 프롬프트가 설정의 prompt 버전과 다름
(학습은 v1, 추론은 v2처럼 어긋나는 것을 막음).

프롬프트 버전: 설정의 prompt(기본 v1, core/prompt.py PROMPTS). v1이 아니면 데이터 기본 경로가 data/prompt_<버전>/
(pipeline/build_jsonl.py --prompt <버전>의 출력)이다. r3(프롬프트 v2_1)처럼 설정에 prompt: v2_1 한 줄만 적는다(데이터 경로를 같이 적어도 된다). --memcheck는 여유 5% 미만이면 실패로 끝난다(종료 코드 1).

스모크(--smoke)는 동작 · 메모리 · 속도 확인용이다. runs/에 쓰지 않고 r1 · r2 · r3 이름을 쓰지 않으며, 결과를 조건 선택에 쓰지 않는다.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.prompt import CHAT_TEMPLATE_KWARGS, DEFAULT_PROMPT, PROMPTS, prompt_sha256, prompt_subdir  # noqa: E402
from pipeline.guard import load_splits, refuse_sealed, require_splits  # noqa: E402

BASE_CONFIG = ROOT / "pipeline" / "configs" / "r1.yaml"
TRAIN_JSONL = ROOT / "data" / "train.jsonl"   # 설정의 train_jsonl로 바꿀 수 있다(r2: 증강 배수 축소)
VAL_JSONL = ROOT / "data" / "val.jsonl"
SPLITS_CSV = ROOT / "data" / "splits.csv"
BUILD_REPORT = ROOT / "data" / "build_report.json"   # 설정의 build_report로 바꿀 수 있다(train_jsonl을 바꿀 때 같이)
SMOKE_ROOT = Path("/tmp/msds_smoke")
RESERVED_NAMES = {"r1", "r2", "r3", "final"}        # 스모크 · 부분 실행에 쓰면 안 되는 이름
DEFAULTS = {"warmup_ratio": 0.03, "weight_decay": 0.0, "max_grad_norm": 1.0, "seed": 42, "gradient_checkpointing": True,
            "prompt": DEFAULT_PROMPT}


# ---------------------------------------------------------------- 설정 · 기록
def load_config(path):
    """r1.yaml을 바탕으로 두고 주어진 설정을 덮어쓴다(r2는 바뀐 값만 적는다)."""
    cfg = dict(DEFAULTS)
    cfg.update(yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")) or {})
    if Path(path).resolve() != BASE_CONFIG.resolve():
        cfg.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return cfg


def splits_path(cfg):
    """분할표. 설정의 splits로 바꿀 수 있다(교차검증 eval/cv.py가 fold마다 넘김). 기본 data/splits.csv."""
    return ROOT / cfg["splits"] if cfg.get("splits") else SPLITS_CSV


def data_paths(cfg):
    """→ (train JSONL, val JSONL, 증강 보고서). 설정에 경로가 없으면 프롬프트 버전의 기본 위치(v1은 data/ 그대로)."""
    sub = prompt_subdir(cfg["prompt"])
    return tuple(ROOT / cfg[k] if cfg.get(k) else p.parent / sub / p.name
                 for k, p in (("train_jsonl", TRAIN_JSONL), ("val_jsonl", VAL_JSONL), ("build_report", BUILD_REPORT)))


def rel_root(path):
    """기록용 경로: 저장소 안이면 저장소 기준 상대 경로, 밖(교차검증 스모크 등)이면 절대 경로."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def env_info():
    import accelerate
    import bitsandbytes
    import peft
    import torch
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__, "accelerate": accelerate.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_mem_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else None}


def _labels_sha(label_dir, jsonl_path):
    """build_jsonl.py와 같은 방식: JSONL에 든 문서의 라벨을 doc_id 순으로 이어 붙여 해시. 라벨이 없으면 None."""
    docs = sorted({json.loads(l)["doc_id"] for l in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if l.strip()})
    try:
        return hashlib.sha256(b"".join((label_dir / f"{d}.json").read_bytes() for d in docs)).hexdigest()
    except FileNotFoundError:
        return None


def build_report_summary(train_path, val_path=None, report_path=None, splits_csv=None):
    """증강 보고서(build_jsonl.py)에서 r1 설정에 남길 항목(① 배수 ② 유형별 ③ val 섞임 ④ 초과 길이 · 거부 · 해시).
    보고서가 없으면 None. splits_csv 기본은 SPLITS_CSV."""
    splits_csv = splits_csv or SPLITS_CSV
    report_path = Path(report_path or BUILD_REPORT)
    if not report_path.exists():
        return None
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    t = rep["splits"]["train"]
    jsonl_sha = (t.get("jsonl") or {}).get("sha256")
    # 보고서를 만든 뒤 라벨 · 분할표가 바뀌었는지
    label_dir = Path(rep["args"]["label_dir"])
    label_dir = label_dir if label_dir.is_absolute() else ROOT / label_dir
    out = {
        "matches_train_jsonl": jsonl_sha == sha256(train_path),   # False면 보고서와 학습 데이터가 다름
        "labels_unchanged_since_build": _labels_sha(label_dir, train_path) == rep["inputs"].get("train_labels_sha256"),
        "splits_unchanged_since_build": sha256(splits_csv) == rep["inputs"].get("splits_sha256"),
    }
    if val_path is not None and Path(val_path).exists():
        v = rep["splits"].get("val") or {}
        out["matches_val_jsonl"] = (v.get("jsonl") or {}).get("sha256") == sha256(val_path)
        out["val_labels_unchanged_since_build"] = (_labels_sha(label_dir, val_path)
                                                   == rep["inputs"].get("val_labels_sha256"))
    return {
        **out,
        "build_args": {k: rep["args"].get(k) for k in ("seed", "secret", "renderers", "ecnum", "max_length")},
        "docs": t["docs"], "examples": t["examples"], "multiplier_incl_orig": t["multiplier_incl_orig"],
        "variants": t["variants"], "by_form": t["by_form"], "docs_with": t["docs_with"], "examples_with": t["examples_with"],
        "checks": rep.get("checks"),
        "tokens": {k: v for k, v in (t.get("tokens") or {}).items() if k != "over_limit"},
        "over_limit": (t.get("tokens") or {}).get("over_limit"),
        "n_rejected": len(t.get("rejected", [])),
        "code_sha256": rep.get("code_sha256"),
    }


def check_data(train_path, val_path, splits_csv=None):
    """학습 · 조건 선택 데이터 용도 검사(CLAUDE.md: splits.csv가 유일한 기준). splits_csv 기본은 SPLITS_CSV."""
    splits_csv = splits_csv or SPLITS_CSV
    refuse_sealed(train_path, val_path, splits_csv)
    splits = load_splits(splits_csv)
    for path, allowed in ((train_path, {"train"}), (val_path, {"val"})):
        if Path(path).exists() and Path(path).stat().st_size:
            ids = {json.loads(l)["doc_id"] for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()}
            require_splits(ids, allowed, splits, who=Path(path).name)


PRECHECK_KEYS = ("matches_train_jsonl", "labels_unchanged_since_build", "splits_unchanged_since_build",
                 "matches_val_jsonl", "val_labels_unchanged_since_build")


def prompt_mismatches(path, version):
    """JSONL에서 시스템 프롬프트가 PROMPTS[version]과 다른 예시 → ["doc_id/variant", ...]"""
    expected = PROMPTS[version]
    bad = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        msgs = r.get("messages") or [{}]
        if msgs[0].get("role") != "system" or msgs[0].get("content") != expected:
            bad.append(f"{r.get('doc_id')}/{r.get('variant')}")
    return bad


def precheck(train_path, val_path, report_path=None, prompt=DEFAULT_PROMPT, splits_csv=None):
    """GPU에 모델을 올리기 전에 멈출 것들 → 증강 보고서 요약(config.json에 남긴다).
    - train · val JSONL이 없거나 비었다
    - 데이터 용도 위반(check_data)
    - 증강 보고서(data/build_report.json)가 없다
    - 보고서와 지금의 train · val JSONL · 라벨 · 분할표가 다르다(보고서를 만든 뒤 무언가 바뀜)
    - JSONL의 시스템 프롬프트가 prompt 버전의 문구와 다르다"""
    for p in (train_path, val_path):
        if not Path(p).exists() or not Path(p).stat().st_size:
            sys.exit(f"[중단] {p}가 없거나 비었다 — pipeline/build_jsonl.py로 만들 것")
    check_data(train_path, val_path, splits_csv)
    aug = build_report_summary(train_path, val_path, report_path, splits_csv)
    if aug is None:
        sys.exit(f"[중단] 증강 보고서가 없다({report_path or BUILD_REPORT}) — pipeline/build_jsonl.py로 다시 만들 것")
    bad = [k for k in PRECHECK_KEYS if not aug.get(k)]
    if bad:
        sys.exit(f"[중단] 증강 보고서와 지금의 데이터가 다르다: {bad} — pipeline/build_jsonl.py로 다시 만들 것")
    for p in (train_path, val_path):
        wrong = prompt_mismatches(p, prompt)
        if wrong:
            sys.exit(f"[중단] {p}의 시스템 프롬프트가 설정의 prompt {prompt} 문구와 다르다({len(wrong)}건, 예: {wrong[:3]}) "
                     f"— pipeline/build_jsonl.py --prompt {prompt}로 다시 만들 것")
    aug["prompt"] = {"version": prompt, "text_sha256": prompt_sha256(prompt)}
    return aug


def mem_verdict(reserved_gib, total_gib):
    """memcheck 판정. 여유 5% 미만이면 실패."""
    return reserved_gib < total_gib * 0.95


# ---------------------------------------------------------------- 데이터
def encode(tok, messages, max_length):
    """→ (input_ids, labels) 또는 max_length를 넘으면 None. 정답 끝을 자르면 닫히지 않은 JSON을 배우므로 자르지 않고 뺀다."""
    assert messages[-1]["role"] == "assistant"
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)
    target = messages[-1]["content"] + "<|im_end|>"
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    t_ids = tok(target, add_special_tokens=False).input_ids
    ids = p_ids + t_ids
    if len(ids) > max_length:
        return None
    return ids, [-100] * len(p_ids) + t_ids


def load_split(tok, path, max_length):
    """→ (예시 목록, max_length 초과로 뺀 목록). 뺀 예시는 조용히 버리지 않고 config.json에 남긴다."""
    rows, dropped = [], []
    if not Path(path).exists():
        return rows, dropped
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        enc = encode(tok, r["messages"], max_length)
        if enc is None:
            dropped.append(f"{r['doc_id']}/{r['variant']}")
        else:
            rows.append(enc)
    if dropped:
        print(f"[경고] {Path(path).name}: max_length({max_length}) 초과로 {len(dropped)}건 제외 {dropped[:5]}")
    return rows, dropped


def batches(rows, size, pad_id):
    import torch

    for i in range(0, len(rows), size):
        chunk = rows[i:i + size]
        n = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), n), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), n), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for j, (a, b) in enumerate(chunk):
            ids[j, :len(a)] = torch.tensor(a)
            lab[j, :len(b)] = torch.tensor(b)
            att[j, :len(a)] = 1
        yield ids, lab, att


# ---------------------------------------------------------------- 모델
def load_model(cfg):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rev = cfg.get("model_revision")
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=rev)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)  # eval/infer.py와 같은 양자화
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=rev, quantization_config=bnb,
                                                 device_map={"": 0}, dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg["gradient_checkpointing"])
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.train()  # from_pretrained 직후는 eval 모드 — gradient checkpointing은 train 모드에서만 켜진다
    return tok, model


def model_commit(model):
    """실제로 불러온 베이스 모델 스냅샷(HF 커밋 해시). 실험 고정 때 env_check.md의 리비전과 대조한다."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return getattr(base.config, "_commit_hash", None)


def make_optimizer(model, cfg):
    import bitsandbytes as bnb

    params = [p for p in model.parameters() if p.requires_grad]
    return bnb.optim.PagedAdamW8bit(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])


def lr_at(step, total, cfg):
    """선형 warmup → cosine 감쇠"""
    warm = max(1, int(total * cfg["warmup_ratio"]))
    if step < warm:
        return cfg["learning_rate"] * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return cfg["learning_rate"] * 0.5 * (1 + math.cos(math.pi * prog))


def forward_loss(model, ids, lab, att):
    """정답 구간의 logits만 계산해 loss를 낸다(값은 labels=로 넘길 때와 같음).

    3.8K 토큰 × 어휘 15만의 logits 전체는 fp32로 2GB가 넘고 기울기까지 두 배라 8GB GPU에서 OOM이 난다.
    loss는 정답 토큰에만 걸리므로, 첫 정답 토큰 바로 앞 위치부터만 logits를 뽑는다(logits_to_keep).
    """
    import torch.nn.functional as F

    ids, lab, att = ids.to(0), lab.to(0), att.to(0)
    first = int((lab != -100).int().argmax(dim=1).min())  # 배치에서 가장 앞선 정답 시작 위치
    keep = ids.shape[1] - first + 1
    logits = model(input_ids=ids, attention_mask=att, logits_to_keep=keep).logits[:, :-1, :]
    target = lab[:, first:]
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100)


def peak_mem():
    import torch

    return {"allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}


# ---------------------------------------------------------------- 실행
def memcheck(cfg):
    """가장 긴 train 예시 1건으로 forward+backward+optimizer step 1회 → 최대 GPU 메모리."""
    import torch

    train_path, val_path, report_path = data_paths(cfg)
    precheck(train_path, val_path, report_path, cfg["prompt"], splits_path(cfg))
    tok, model = load_model(cfg)
    rows, _ = load_split(tok, train_path, cfg["max_length"])
    val_rows, _ = load_split(tok, val_path, cfg["max_length"])
    longest = max(rows, key=lambda r: len(r[0]))
    print(f"가장 긴 예시: {len(longest[0])} 토큰 (정답 {sum(x != -100 for x in longest[1])} 토큰), "
          f"max_length {cfg['max_length']}, gradient_checkpointing {cfg['gradient_checkpointing']}")
    opt = make_optimizer(model, cfg)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ids, lab, att = next(batches([longest] * cfg["per_device_batch_size"], cfg["per_device_batch_size"], tok.pad_token_id))
    loss = forward_loss(model, ids, lab, att)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    m = peak_mem()
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"loss {loss.item():.4f} · 1 step {time.perf_counter() - t0:.1f}s")
    if val_rows:  # val loss 계산(역전파 없음)도 같은 GPU 상태(옵티마이저 상태가 올라간 뒤)에서 가장 긴 예시로 확인.
        v_longest = max(val_rows, key=lambda r: len(r[0]))  # val 예시는 train보다 길 수 있다(v2: train 3,551 · val ~4,096)
        print(f"가장 긴 val 예시: {len(v_longest[0])} 토큰(역전파 없음)")
        evaluate(model, [v_longest], tok.pad_token_id)
        torch.cuda.synchronize()
        m = peak_mem()
    ok = mem_verdict(m["reserved_gib"], total)
    print(f"최대 메모리 allocated {m['allocated_gib']:.2f} GiB / reserved {m['reserved_gib']:.2f} GiB / GPU {total:.2f} GiB "
          f"→ {'통과' if ok else '위험: 여유 5% 미만'}")
    if not ok:
        sys.exit(1)


def evaluate(model, rows, pad_id):
    import torch

    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for ids, lab, att in batches(rows, 1, pad_id):
            tot += forward_loss(model, ids, lab, att).item()
            n += 1
    model.train()
    return tot / n


def train(cfg, name, run_dir, max_steps=None, smoke=False):
    import torch

    train_path, val_path, report_path = data_paths(cfg)
    aug = precheck(train_path, val_path, report_path, cfg["prompt"], splits_path(cfg))
    if (run_dir / "adapter").exists():
        sys.exit(f"이미 있음: {run_dir} — 지우거나 이름을 바꿀 것(--name)")
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    tok, model = load_model(cfg)
    train_rows, train_dropped = load_split(tok, train_path, cfg["max_length"])
    val_rows, val_dropped = load_split(tok, val_path, cfg["max_length"])
    bs, accum = cfg["per_device_batch_size"], cfg["grad_accum"]
    steps_per_epoch = len(train_rows) // (bs * accum)  # 꼬리 micro-batch는 버리므로 내림(스케줄 끝이 실제 마지막 step과 맞게)
    total = steps_per_epoch * cfg["num_epochs"]
    if max_steps:
        total = min(total, max_steps)

    meta = {
        "name": name, "smoke": smoke, "created_at": datetime.now().isoformat(timespec="seconds"), "git": git_state(),
        "config": cfg, "chat_template_kwargs": CHAT_TEMPLATE_KWARGS, "env": env_info(), "model_commit": model_commit(model),
        "data": {"train_jsonl": rel_root(train_path), "train_jsonl_sha256": sha256(train_path),
                 "val_jsonl": rel_root(val_path),
                 "val_jsonl_sha256": sha256(val_path) if val_path.exists() else None,
                 "splits": rel_root(splits_path(cfg)), "splits_sha256": sha256(splits_path(cfg)),
                 "n_train": len(train_rows), "n_val": len(val_rows),
                 "dropped_over_max_length": {"train": train_dropped, "val": val_dropped}},
        "augmentation": aug,
        "schedule": {"batch": bs, "grad_accum": accum, "effective_batch": bs * accum, "steps_per_epoch": steps_per_epoch,
                     "optimizer_steps": total, "tail_microbatches_dropped_per_epoch": len(train_rows) % (bs * accum)},
    }
    write_meta(run_dir, meta)
    print(f"[{name}] train {len(train_rows)} · val {len(val_rows)} · batch {bs}×accum {accum} · "
          f"{cfg['num_epochs']} epoch = optimizer step {total} → {run_dir}")

    opt = make_optimizer(model, cfg)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    log = open(run_dir / "loss.csv", "w", encoding="utf-8", newline="")
    w = csv.writer(log)
    w.writerow(["epoch", "step", "lr", "train_loss", "val_loss", "elapsed_s"])
    step, t0 = 0, time.perf_counter()
    for epoch in range(1, cfg["num_epochs"] + 1):
        random.shuffle(train_rows)
        acc_loss, micro = 0.0, 0
        for ids, lab, att in batches(train_rows, bs, tok.pad_token_id):
            loss = forward_loss(model, ids, lab, att) / accum
            loss.backward()
            acc_loss += loss.item()
            micro += 1
            if micro % accum:
                continue
            lr = lr_at(step, total, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["max_grad_norm"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            w.writerow([epoch, step, f"{lr:.2e}", f"{acc_loss:.4f}", "", f"{time.perf_counter() - t0:.0f}"])
            log.flush()
            print(f"  epoch {epoch} step {step}/{total}  loss {acc_loss:.4f}  lr {lr:.2e}  {time.perf_counter() - t0:.0f}s")
            acc_loss = 0.0
            if step >= total:
                break
        # 남은 micro-batch(accum으로 나누어떨어지지 않는 꼬리)는 버린다 — epoch마다 최대 accum-1건
        opt.zero_grad(set_to_none=True)
        if val_rows:
            vl = evaluate(model, val_rows, tok.pad_token_id)
            w.writerow([epoch, step, "", "", f"{vl:.4f}", f"{time.perf_counter() - t0:.0f}"])
            log.flush()
            print(f"  epoch {epoch} 끝 — val loss {vl:.4f}")
        if step >= total:
            break
    log.close()
    elapsed = time.perf_counter() - t0
    model.save_pretrained(run_dir / "adapter")
    meta["result"] = {"optimizer_steps_done": step, "train_seconds": round(elapsed),
                      "seconds_per_step": round(elapsed / max(step, 1), 1), "peak_gpu_memory": peak_mem(),
                      "finished_at": datetime.now().isoformat(timespec="seconds")}
    write_meta(run_dir, meta)
    print(f"완료 → {run_dir}/adapter · {step} step · {elapsed:.0f}s ({elapsed / max(step, 1):.1f}s/step) · "
          f"최대 메모리 {meta['result']['peak_gpu_memory']}")
    if not smoke:
        rel = rel_root(run_dir / "adapter")
        print(f"다음: python3 eval/infer.py --condition qlora_{name} --split val --adapter {rel}")


def write_meta(run_dir, meta):
    (run_dir / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", help="실행 이름(기본: 설정 파일 이름, 예 r1)")
    ap.add_argument("--memcheck", action="store_true", help="가장 긴 예시로 1 step만 돌려 최대 메모리 확인")
    ap.add_argument("--smoke", type=int, metavar="STEPS", help=f"스모크: STEPS step만, {SMOKE_ROOT}/ 아래에 기록(runs/ 아님)")
    ap.add_argument("--run-dir", help="기록 폴더(기본 runs/{MMDD}_{이름}). 교차검증(eval/cv.py)이 fold마다 넘긴다")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.memcheck:
        memcheck(cfg)
        return
    if args.smoke:
        name = args.name or "smoke"
        if name in RESERVED_NAMES:
            sys.exit(f"스모크에는 {sorted(RESERVED_NAMES)} 이름을 쓰지 않는다")
        run_dir = Path(args.run_dir).resolve() if args.run_dir else SMOKE_ROOT / f"{datetime.now():%m%d_%H%M}_{name}"
        train(cfg, name, run_dir, max_steps=args.smoke, smoke=True)
        return
    name = args.name or Path(args.config).stem
    train(cfg, name, Path(args.run_dir).resolve() if args.run_dir else ROOT / "runs" / f"{datetime.now():%m%d}_{name}")


if __name__ == "__main__":
    main()
