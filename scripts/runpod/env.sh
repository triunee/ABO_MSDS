# scripts/runpod/*.sh가 함께 쓰는 설정. source로 읽는다. 값은 환경 변수로 바꿀 수 있다.
WS=${WS:-/workspace}                        # RunPod 볼륨(pod를 멈춰도 남음). 모든 것을 여기 둔다
REPO_DIR=${REPO_DIR:-$WS/ABO_MSDS}
VENV=${VENV:-$WS/venv}
CV_ROOT=${CV_ROOT:-cv/runpod}               # 판정용 결과 폴더(저장소 기준). 노트북 결과 cv/와 섞지 않는다
CONDS_DEFAULT="r1 r1_aug r1_s43 r1_aug_s43 base_fs base_zs"   # seed 42 짝을 먼저 — 중간에 멈춰도 비교 한 쌍은 남게
export HF_HOME=${HF_HOME:-$WS/hf}
export PYTHONUNBUFFERED=1                   # 로그가 끝날 때 한꺼번에 찍히지 않게

log() { echo "[$(date '+%F %T')] $*"; }

gpu_idle_or_die() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
  if [ -n "$apps" ]; then
    echo "[거부] GPU를 쓰는 다른 프로세스가 있다:"; echo "$apps"; exit 1
  fi
}

# 사전 등록 커밋 뒤의 코드인지, 추적 파일에 손댄 곳이 없는지 확인한다(결과를 본 뒤 기준 · 코드를 바꾸지 않았다는 근거)
check_repo() {
  local prereg=$1
  git merge-base --is-ancestor "$prereg" HEAD || { echo "[거부] HEAD가 사전 등록 커밋 $prereg 뒤가 아니다"; exit 1; }
  git diff --quiet "$prereg" HEAD -- cv/prereg_r1_aug.md || { echo "[거부] 사전 등록 뒤에 cv/prereg_r1_aug.md가 바뀌었다"; exit 1; }
  git diff --quiet HEAD || { echo "[거부] 추적 파일에 커밋하지 않은 수정이 있다"; git status --short; exit 1; }
}
