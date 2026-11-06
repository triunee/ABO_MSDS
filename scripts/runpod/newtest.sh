#!/usr/bin/env bash
# 새 test 평가(cv/prereg_newtest.md 8절 4번) — run_all.sh 대신 이것을 돌린다. 학습은 없고 추론 600회 + 채점 + 판정만.
#
# 준비(pod에서 한 번):
#   cd /workspace && git clone -b feat/group-kfold-cv https://github.com/triunee/ABO_MSDS.git && cd ABO_MSDS
#   bash scripts/runpod/setup.sh bb81da1             # venv · 버전 · 모델 스냅샷 · 테스트(r1_aug 사전 등록 커밋으로 검사)
#   tar xzf /workspace/newtest_adapters.tar.gz -C /workspace/ABO_MSDS   # 어댑터 10개 → cv/runs/{r1,r1_aug}/fold*/adapter
# 실행:
#   nohup bash scripts/runpod/newtest.sh <고정 커밋> > /workspace/newtest.out 2>&1 &
#   tail -f /workspace/newtest.out                   # 단계 · 건수 · 판정만 나온다(문서 이름 없음)
#
# <고정 커밋> = manifest · experiment.json을 커밋한 해시. 그 뒤로 사전 등록 · 고정 파일 · 봉인 · 코드가 바뀌었으면 멈춘다.
# 문서별 추론 로그(문서 이름 있음)는 $NT_ROOT/out/_run.log(저장소 밖)에만 쓴다. 개발 세션에 붙여 넣지 않는다.
# 중간에 끊기면 같은 명령으로 다시 돌린다. 끝난 단위 · 문서는 건너뛴다(experiment.json allow_resume). GPU 종류가 바뀌면 멈춘다.
set -euo pipefail
FIXED=${1:?"사용법: bash scripts/runpod/newtest.sh <고정 커밋>"}
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/../.."
NT_ROOT=${NT_ROOT:-$WS/newtest}
EXP=cv/newtest/experiment.json
MAN=cv/newtest/test_manifest.csv
HW=cv/newtest/hardware.txt

log "1/7 저장소 확인(고정 커밋 $FIXED)"
git merge-base --is-ancestor "$FIXED" HEAD || { echo "[거부] HEAD가 고정 커밋 $FIXED 뒤가 아니다"; exit 1; }
git diff --quiet "$FIXED" HEAD -- cv/prereg_newtest.md "$EXP" "$MAN" eval/ core/ src/ prompts/ 2>/dev/null \
  || { echo "[거부] 고정 커밋 뒤에 사전 등록 · 고정 파일 · 봉인 · 코드가 바뀌었다"; git diff --stat "$FIXED" HEAD; exit 1; }
git diff --quiet HEAD || { echo "[거부] 추적 파일에 커밋하지 않은 수정이 있다"; git status --short; exit 1; }
[ -f "$MAN" ] || { echo "[거부] $MAN 이 없다(봉인 manifest를 먼저 커밋)"; exit 1; }
[ -x "$VENV/bin/python" ] || { echo "[거부] venv가 없다. scripts/runpod/setup.sh bb81da1 을 먼저"; exit 1; }
source "$VENV/bin/activate"

log "2/7 GPU"
gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)
echo "$gpu"
case "$gpu" in *4090*) ;; *) echo "[거부] 사전 등록 5절은 RTX 4090. 바꾸려면 첫 추론 전에 사전 등록을 고쳐 다시 커밋"; exit 1 ;; esac
gpu_idle_or_die

log "3/7 어댑터 10개 해시"
python eval/newtest.py verify-adapters

log "4/7 봉인 대조(정답 · 텍스트 · 서식 · 그룹, 이름은 내지 않음)"
python eval/seal.py --verify --manifest "$MAN" --label-dir "$NT_ROOT/labels" --text-dir "$NT_ROOT/text" \
  --subset-csv "$NT_ROOT/subset.csv" --groups-csv "$NT_ROOT/groups.csv"

log "5/7 하드웨어 기록 → $HW"
if [ -f "$HW" ]; then   # 이어 돌릴 때: 처음 기록한 GPU 종류와 같아야 한다(비교군을 다른 GPU에서 섞지 않음)
  first=$(grep '^gpu:' "$HW" | cut -d' ' -f2- | cut -d, -f1)
  [ "$first" = "$(echo "$gpu" | cut -d, -f1)" ] || { echo "[거부] 처음 GPU($first)와 지금 GPU가 다르다"; exit 1; }
  echo "이미 있음(이어 돌림): $first"
else
  {
    echo "# 새 test 추론 환경(scripts/runpod/newtest.sh가 첫 추론 전에 기록)"
    echo "recorded_at: $(date -Iseconds)"
    echo "repo_head: $(git rev-parse HEAD)"
    echo "fixed_commit: $FIXED"
    echo "runpod_pod_id: ${RUNPOD_POD_ID:-?}"
    echo "gpu: $gpu"
    echo "cpu: $(nproc) cores, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
    python -c 'import sys, torch, transformers, peft, bitsandbytes, accelerate; print(f"python: {sys.version.split()[0]}\ntorch: {torch.__version__} (cuda {torch.version.cuda})\ntransformers: {transformers.__version__}\npeft: {peft.__version__}\nbitsandbytes: {bitsandbytes.__version__}\naccelerate: {accelerate.__version__}")'
  } > "$HW"
  cat "$HW"
fi

log "6/7 추론 · 채점(단위 12개 × 문서, 로그 $NT_ROOT/out/_run.log)"
mkdir -p "$NT_ROOT/out"
python eval/newtest.py run --root "$NT_ROOT" >> "$NT_ROOT/out/_run.log" 2>&1 \
  || { python eval/newtest.py status --root "$NT_ROOT"; echo "[멈춤] 실패. 원인은 $NT_ROOT/out/_run.log 끝부분(문서 이름 있음 — 개발 세션에 붙이지 말 것)"; exit 1; }
python eval/newtest.py status --root "$NT_ROOT"

log "7/7 판정 → cv/newtest/report.md · scores.csv · paired.csv"
python eval/newtest.py report --root "$NT_ROOT"
stamp=$(date +%m%d_%H%M)
tar czf "$WS/newtest_results_$stamp.tar.gz" cv/newtest/report.md cv/newtest/scores.csv cv/newtest/paired.csv "$HW"
log "끝. 받기: (pod) runpodctl send $WS/newtest_results_$stamp.tar.gz → (Mac) runpodctl receive <코드> → 저장소 루트에서 tar xzf"
