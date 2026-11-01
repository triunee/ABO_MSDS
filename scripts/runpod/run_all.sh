#!/usr/bin/env bash
# 판정에 쓰는 전 과정을 RunPod GPU 한 종류에서 돌린다(cv/prereg_r1_aug.md 1 · 7절).
#   fold 만들기(cv/folds.csv와 같은지 확인) → 조건 × fold 학습 · 추론 → 보고서 → Rule Engine → 변형 시험
#
#   nohup bash scripts/runpod/run_all.sh <사전 등록 커밋> > /workspace/run_all.out 2>&1 &
#   nohup bash scripts/runpod/run_all.sh <사전 등록 커밋> r1_aug r1_aug_s43 > ... &   # 조건 일부만
#
# 끊겨도 같은 명령으로 다시 돌리면 이어서 간다(cv.py · cv_perturb.py가 끝난 단계 · 출력을 건너뜀). 끊긴 fold의 학습은
# 처음부터 다시 돈다. 진행 상황: python eval/cv.py --root cv/runpod status
set -euo pipefail
PREREG=${1:?"사용법: bash scripts/runpod/run_all.sh <사전 등록 커밋 해시> [조건 ...]"}
shift
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/../.."
source "$VENV/bin/activate"
check_repo "$PREREG"
[ -f "$CV_ROOT/hardware.txt" ] || { echo "[거부] $CV_ROOT/hardware.txt 없음 — setup.sh 먼저"; exit 1; }
gpu_idle_or_die
read -r -a CONDS <<< "${*:-$CONDS_DEFAULT}"
GPU_LOG="$CV_ROOT/gpu_log.csv"   # fold마다 온도 · 메모리 한 줄(참고 기록)

log "fold — 노트북과 같은 분할이어야 한다"
python eval/cv.py --root "$CV_ROOT" folds
cmp "$CV_ROOT/folds.csv" cv/folds.csv
for k in 0 1 2 3 4; do
  cmp "$CV_ROOT/fold$k/splits.csv" "cv/fold$k/splits.csv"
  cmp "$CV_ROOT/fold$k/fewshot.json" "cv/fold$k/fewshot.json"
done

[ -f "$GPU_LOG" ] || echo "time,cond,fold,gpu_temp_c,mem_used_mib" > "$GPU_LOG"
for c in "${CONDS[@]}"; do
  for k in 0 1 2 3 4; do
    log "$c / fold $k"
    python eval/cv.py --root "$CV_ROOT" run --cond "$c" --folds "$k"
    echo "$(date -Iseconds),$c,$k,$(nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ')" >> "$GPU_LOG"
  done
done
python eval/cv.py --root "$CV_ROOT" status

log "보고서 · Rule Engine (끝난 조건만, 나머지는 '채점하지 않은 조건'에 적힘)"
python eval/cv.py --root "$CV_ROOT" report
python eval/cv_rules.py --root "$CV_ROOT"

log "변형 시험(layout · value · glyph)"
python eval/cv_perturb.py --root "$CV_ROOT" make
PERTURB_CONDS=$(printf '%s\n' "${CONDS[@]}" | grep -vx base_zs | paste -sd, -)
python eval/cv_perturb.py --root "$CV_ROOT" run --cond "$PERTURB_CONDS"
python eval/cv_perturb.py --root "$CV_ROOT" report

log "끝. 결과 묶기: bash scripts/runpod/collect.sh"
