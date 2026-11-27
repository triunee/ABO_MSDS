# ABO_MSDS

MSDS 1~3항 → 핵심 5필드 고정 JSON 추출(Qwen3-4B QLoRA) + Rule Engine 검토 상태 + SQLite 적재 + 검토 화면. 4인 팀 과업(2026-09-28 재시작).
상세: @docs/plan.md · 라벨 표기 세부: data/README.md · 결정 기록: report/decisions.md

## 재시작 전제
- **`feat/web` 브랜치는 PM이 혼자 먼저 해 본 파일럿이다.** 그 라벨 · 모델 출력 · 점수 · 문서를 재시작의 근거로 쓰지 않는다. `feat/web`를 main에 병합하지 않는다. 파일럿 코드를 가져올 때는 파일 단위 PR로 올리고 리뷰를 받는다.
- **파일럿에서 본 문서 87건(제조사 44곳)은 모두 노출 문서다.** 목록은 파일럿 `data/sources.csv`에 있다. train · val 후보다. **예외(2026-09-28 결정):** 새 문서를 모을 수 없어 노출 문서를 test로 쓴다 — 국문 18건에 영문 13건을 더한 31건(2026-09-30 봉인). 이 31건은 개발에 쓰지 않고, 결론에 파일럿 노출 이력을 밝힌다(report/decisions.md).
- **정답은 처음부터 다시 만든다.** 09-23 기존 라벨(`archive/labels_0923/`, `archive/val_en_0923/`)은 초안이나 참고로 쓰지 않는다. 옛 test 정답(`eval/test/`)은 지금 test 문서와 겹쳐 저장소에서 지웠다(git 이력에는 남음 — 열지 않는다).

## 절대 규칙
- **test 봉인:**
  - test 문서 · 텍스트 · 정답은 저장소 밖에만 보관한다(현재 PM 로컬). 정답은 파일럿 정답을 열지 않고 원문에서 새로 만든다.
  - 개발 세션(사람 · AI 모두)은 test 자료를 열지 않고, 첨부받지도 않는다. 개발 쪽이 받는 정보는 건수 · 서식별 건수 · 봉인 해시뿐이다.
  - test 추론은 실험 고정 뒤 비교군마다 1회만 한다.
  - test를 학습 · few-shot · 프롬프트 예시 · 화면 샘플 · 조건 선택에 쓰지 않는다.
  - test 자료가 개발 세션에 들어오면 작업을 멈추고 PM에게 알리고, report/decisions.md에 남긴다.
- **공개 저장소:**
  - 평가가 끝나기 전에는 test 문서의 파일명 · 제조사 · 정답에서 나온 값(성분 수 · 신호어 등)을 어떤 파일에도 적지 않는다. data/sources.csv도 마찬가지다.
  - 커밋한 값은 지워도 공개 이력과 PR 기록에 남는다.
- **데이터 용도는 data/splits.csv가 유일한 기준:**
  - 학습(증강 · JSONL) · few-shot은 split=train만 쓴다. 조건 선택(r1 · r2 · r3 비교, 모델 고정)은 split=val만 쓴다.
  - splits.csv는 PM 승인 없이 바꾸지 않는다.
- **정답은 사람이 확정한다.**
  - LLM 초안을 쓰면 원문과 모두 대조한 뒤 확정한다. 요청받지 않은 정답 값 수정은 하지 않는다.
  - 수정 순서: JSON 수정 → `python3 scripts/validate_labels.py data/labels` → docs/labeling_review_notes.md에 이유 한 줄 → 커밋 `fix(label): KR-XXX-001 …`
- **스키마 동결:**
  - 출력 JSON 키 이름, 상태값(기재 · 자료없음 · 해당없음), DB 테이블 · 열 이름을 바꾸지 않는다.
  - 출력 스키마는 src/schema.py가 단일 기준이다. 다른 곳에 별도 정의를 만들지 않는다.
- **커밋 금지:**
  - 원본 PDF(`data/raw/`, `*.pdf`) · zip
  - runs/의 가중치 · 체크포인트(`*.safetensors` · `*.bin` · `*.gguf` · `checkpoint-*/`). config.json · loss 로그만 커밋한다.
  - `.env`, DB 파일(`*.db`), 기획서 원본 · 수요조사 자료
  - 평가 전의 test 자료(목록 · 정답 · 텍스트). 봉인 해시 목록만 커밋한다.
- **Git:**
  - main에 직접 커밋 · push하지 않는다. 작업 브랜치에서 PR로 병합한다.
  - `git push --force` · `git branch -M main`은 쓰지 않는다. main 변경은 merge로 받는다.
  - `git add .` 대신 경로를 지정한다.
- **승인 필요:** 학습 실행, 조건 선택, 분할 확정, 봉인, test 평가는 PM 승인 후에 한다.
- **작업 폴더:** 한 폴더에서는 한 세션(사람 · AI)만 쓴다.

## 설계 원칙 (코드 작성 시)
- 모델은 원문에 적힌 값만 읽는다. 프롬프트 · 후처리에서 H코드 · 분류 · CAS를 지식으로 채우거나 추정하지 않는다.
- 모델 출력은 스키마 JSON뿐이다. evidence · review_status는 Rule Engine(app/rules/)이 붙인다.
- P문구는 출력에서 뺀다. 입력에서도 2항 P문구 블록을 잘라낸다.
- 전처리 · 프롬프트는 학습 · 평가 · 서빙이 core/ 하나를 공유한다. core/ 수정은 담당자 리뷰 필수다.
- 추출 실패는 조용히 버리지 않는다. 사유를 기록하고, 실패 건을 평가 분모에서 빼지 않는다.
- 기능을 늘리지 않는다. 범위는 docs/plan.md 1 · 8절이다.

## 환경
- WSL2 Ubuntu 24.04, 노트북 GPU 8GB(RTX 4060급). venv 활성화 후 저장소 루트에서 `python3`로 실행한다.
- 검사: `python3 scripts/validate_labels.py data/labels` / `python3 scripts/check_splits.py`

## 규약
- doc_id는 `KR|EN-제조사약칭-번호`, 정답 · 출력 파일명은 `{doc_id}.json`이다.
- 모델 출력: `outputs/{base_zs,base_fs,qlora_r1,qlora_r2,qlora_r3,qlora_final}/val/{doc_id}.json`. test는 저장소 밖 `--out-root` 아래 `{base_zs,base_fs,qlora_final}/test/`(infer · score가 강제, `.gitignore`에도 `outputs/*/test/`)
  - 이 파일에는 모델이 낸 JSON만 둔다. 파싱 실패도 모델 출력 그대로 저장해 실패로 집계한다.
  - 시간 · 토큰 · 원문 로그는 `.jsonl`로, 채점 상세는 `*.json`이 아닌 이름으로 둔다. 검사기가 폴더의 `*.json`을 모두 검사한다.
- 프롬프트 버전(`core/prompt.py`의 `PROMPTS`: v1 · v2 · v2_1, 기본 v1): 결과를 만든 버전의 문구는 바꾸지 않고, 고칠 때는 새 버전을 추가한다. v1 밖의 버전은 v1 자리를 덮어쓰지 않도록 따로 둔다 — 출력 `outputs/prompt_{버전}/{조건}/val/`, 점수 `report/scores_prompt_{버전}.csv`, 학습셋 `data/prompt_{버전}/`. 스크립트는 `--prompt {버전}`(학습은 설정 `prompt:`)으로 쓰고, 폴더 · JSONL의 버전이나 문구 해시가 다르면 거부한다.
- 출력 보정(`core/repair.py`의 `REPAIRS`: v1): 원문 값과 plan 2절 출력 규칙만으로 모델 JSON의 형식 실수를 고친다. 채점 `--repair v1`(점수 행 `{조건}+repair_v1`, 상세 `_score_detail_repair_v1.jsonl`), 교차검증 `eval/cv.py report --repair v1`(→ `cv/repair_v1/`), 서빙은 항상 적용. 출력 파일 · DB `raw_json`에는 보정 전 모델 원본을 둔다. 규칙을 고칠 때는 v1을 바꾸지 않고 새 버전을 추가한다.
- 전처리: `data/text/{doc_id}.txt`. 실패 기록은 `data/text/_cut_log.csv` + `data/text/review_required/`
- 점수: `report/scores.csv` (condition, split, subset, metric, value). 평가셋별로 행을 나누고 합산하지 않는다.
- 짝 비교: `eval/paired.py` → `report/paired.csv`(승 · 패 · 무, 제조사 그룹 부트스트랩 95% 구간). test 봉인: `eval/seal.py` → `report/test_manifest.csv`(내용 해시만, 파일 이름 없음)
- 학습: `pipeline/configs/r1.yaml`이 기준이고, r2 · r3는 각각 r1에서 조건 1개만 바꾼다(r2 = epoch 1, r3 = 프롬프트 v2_1, 3회째 학습은 PM 결정 2026-09-29). 기록은 `runs/{날짜}_{r1|r2|r3}/`
- 결정 기록: report/decisions.md
- 커밋 prefix: `feat:` / `fix:` / `fix(label):` / `docs:` / `chore:`
- Git 태그: `label-rules-frozen`(라벨 규칙 동결) → `split-frozen`(train · val 분할 확정) → `test-sealed`(test 정답 확정 · 봉인 해시 커밋 직후)
