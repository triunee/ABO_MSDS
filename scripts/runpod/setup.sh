#!/usr/bin/env bash
# RunPod pod에서 한 번: venv · 모델 스냅샷 준비 → 버전 · GPU 확인 → 하드웨어 기록 → 테스트 → 동작 점검(smoke)
#
#   cd /workspace && git clone -b feat/group-kfold-cv https://github.com/triunee/ABO_MSDS.git && cd ABO_MSDS
#   bash scripts/runpod/setup.sh <사전 등록 커밋 해시>
#
# 버전은 노트북 r1 학습과 같게 고정한다(requirements.txt, torch 2.14.0+cu130). 다르면 멈춘다.
# 학습 설정(batch 1 · accum 8 · gradient checkpointing · max_length 4096)은 GPU 메모리가 커도 바꾸지 않는다.
set -euo pipefail
PREREG=${1:?"사용법: bash scripts/runpod/setup.sh <사전 등록 커밋 해시>"}
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/../.."
check_repo "$PREREG"

log "1/6 GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
gpu_idle_or_die

log "2/6 venv ($VENV)"
if [ ! -x "$VENV/bin/python" ]; then
  if command -v python3.12 >/dev/null; then
    python3.12 -m venv "$VENV"
  else   # 이미지에 3.12가 없으면 uv로 받는다
    python3 -m pip install -q uv
    python3 -m uv venv --seed --python 3.12 "$VENV"
  fi
fi
source "$VENV/bin/activate"
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
pip install -q --upgrade pip
pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
pip install -q -r requirements.txt

log "3/6 버전 확인(노트북 r1 학습 환경과 같아야 함)"
python - <<'EOF'
import importlib.metadata as md, sys, torch
want = {"torch": "2.14.0", "transformers": "5.17.0", "peft": "0.21.0", "bitsandbytes": "0.50.2", "accelerate": "1.15.0"}
bad = {k: md.version(k) for k, v in want.items() if not md.version(k).startswith(v)}
if bad:
    sys.exit(f"[거부] 버전이 다르다: {bad}")
if not torch.cuda.is_available():
    sys.exit("[거부] torch가 GPU를 못 본다. 드라이버가 CUDA 13.0을 지원하는 pod인지 확인(RunPod 'CUDA Version' 필터)")
if not torch.cuda.is_bf16_supported():
    sys.exit("[거부] bf16을 지원하지 않는 GPU")
print("ok", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
EOF

log "4/6 모델 스냅샷(pipeline/configs/r1.yaml의 model_revision)"
python - <<'EOF'
import yaml
from huggingface_hub import snapshot_download
cfg = yaml.safe_load(open("pipeline/configs/r1.yaml", encoding="utf-8"))
print(snapshot_download(cfg["model"], revision=cfg["model_revision"]))
EOF

log "5/6 하드웨어 기록 → $CV_ROOT/hardware.txt"
mkdir -p "$CV_ROOT"
{
  echo "# 판정용 실행 환경(scripts/runpod/setup.sh가 기록)"
  echo "recorded_at: $(date -Iseconds)"
  echo "repo_head: $(git rev-parse HEAD)"
  echo "prereg_commit: $PREREG"
  echo "runpod_pod_id: ${RUNPOD_POD_ID:-?}"
  echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
  echo "cpu: $(nproc) cores, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
  python -c 'import sys, torch, transformers, peft, bitsandbytes, accelerate; print(f"python: {sys.version.split()[0]}\ntorch: {torch.__version__} (cuda {torch.version.cuda})\ntransformers: {transformers.__version__}\npeft: {peft.__version__}\nbitsandbytes: {bitsandbytes.__version__}\naccelerate: {accelerate.__version__}")'
} > "$CV_ROOT/hardware.txt"
cat "$CV_ROOT/hardware.txt"

log "6/6 테스트 + 동작 점검(학습 2 step · 추론 2건, 저장소 밖 $WS/smoke)"
python test/test_augment.py
python test/test_cv.py
rm -rf "$WS/smoke"
python eval/cv.py --root "$WS/smoke" folds
python eval/cv.py --root "$WS/smoke" run --cond r1_aug --folds 0 --smoke
log "준비 끝. 다음: nohup bash scripts/runpod/run_all.sh $PREREG > $WS/run_all.out 2>&1 &"
