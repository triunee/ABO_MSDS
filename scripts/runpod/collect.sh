#!/usr/bin/env bash
# 결과를 두 묶음으로 만든다. 받은 뒤 저장소 루트에서 풀면 제자리에 들어간다.
#   results_*.tar.gz  — 커밋할 파일만(cv/runpod/ 중 .gitignore에 걸리지 않는 것: 보고서 · 점수 · 출력 · config.json · loss.csv · hardware.txt)
#   adapters_*.tar.gz — 어댑터(커밋하지 않음, 보관 · 나중 추론용)
#
#   bash scripts/runpod/collect.sh
#   받기 예: (pod) runpodctl send /workspace/results_*.tar.gz  →  (내 PC) runpodctl receive <코드>
#   풀기: tar xzf results_*.tar.gz -C <저장소 루트>  →  git add cv/runpod
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/../.."
stamp=$(date +%m%d_%H%M)

git ls-files -z --others --exclude-standard "$CV_ROOT" | tar czf "$WS/results_$stamp.tar.gz" --null -T -
echo "→ $WS/results_$stamp.tar.gz ($(tar tzf "$WS/results_$stamp.tar.gz" | wc -l)개 파일, $(du -h "$WS/results_$stamp.tar.gz" | cut -f1))"

if compgen -G "$CV_ROOT/runs/*/fold*/adapter" >/dev/null; then
  tar czf "$WS/adapters_$stamp.tar.gz" "$CV_ROOT"/runs/*/fold*/adapter
  echo "→ $WS/adapters_$stamp.tar.gz ($(du -h "$WS/adapters_$stamp.tar.gz" | cut -f1))"
fi
