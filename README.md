# ABO_MSDS

> **출처:** 이 저장소는 4인 팀 프로젝트 [EOCHAEEUN/ABO_MSDS](https://github.com/EOCHAEEUN/ABO_MSDS)의 fork입니다. 아래 본문과 팀 과업(라벨 · 학습 · test 평가 · API · 검토 화면)은 팀원 4명이 함께 만들었습니다. 그 뒤에 이 fork에서 더한 제조사 그룹 교차검증(`cv/`), 새 test 평가 · 확인용 문서 봉인, Rule Engine 개선, 출력 보정(`core/repair.py`)은 양세윤 개인 확장이며, 경위는 `report/decisions.md`와 `cv/`의 사전 등록 · 보고서에 있습니다.

실제 MSDS 1~3항의 핵심 5개 항목을 고정 JSON으로 추출하는 Qwen3-4B QLoRA 모델, 원문 근거와 검토 상태를 붙이는 Rule Engine, SQLite 적재, 검토 화면을 만드는 4인 팀 과업입니다. 작업공간에는 별도 RAG 인덱스로 1~16항 근거를 검색하는 `MSDS 질의` 화면도 제공합니다. 과업 정의는 기획서 v8(MSDS-PL-2609-08)을 따릅니다. 기획서 원본은 팀 드라이브에만 둡니다.

작업 기준은 `docs/plan.md`(재시작판)이고, 작업 규칙은 `CLAUDE.md`입니다. 두 파일은 `docs/restart` PR로 main에 들어옵니다.

## 현재 상태 (2026-09-28 재시작)

- **데이터 · 정답 · 평가 절차를 처음부터 다시 합니다.**
  - PM이 혼자 먼저 진행한 파일럿(`feat/web` 브랜치, 09-23~28)에서 test가 개발 과정에 노출됐습니다.
  - 파일럿의 라벨 · 모델 출력 · 점수는 재시작의 근거로 쓰지 않습니다.
  - `feat/web`는 main에 병합하지 않습니다. 필요한 코드는 파일 단위 PR로 가져옵니다.
- **09-23 기존 라벨은 쓰지 않습니다.** `archive/labels_0923/` · `archive/val_en_0923/`로 옮겼고, 옛 test 정답(`eval/test/`)은 지웠습니다. 재시작 정답 45건(train 35 · val 10)은 `data/labels/`에 있습니다.
- **test는 새 문서로 만듭니다.** 개발에 참여하지 않는 test 담당이 모으고 라벨링해 저장소 밖에 보관합니다. 파일럿에서 본 문서 87건은 train · val 후보로만 씁니다.
- **코드는 대부분 뼈대입니다.** `core/` · `pipeline/` · `eval/` · `app/`의 파일은 담당 표기만 있습니다. 동작하는 것은 `src/schema.py`(출력 스키마), `scripts/`(라벨 · 분할 검사), `web/`(검토 화면, 예시 모드)입니다.
- **다음 할 일:** 착수 회의(`docs/plan.md` 13절) → 라벨 규칙 파일럿 · 동결 → train · val 라벨링 · 분할 확정.

## 진행 단계

날짜는 PM이 기간을 정한 뒤 `docs/plan.md` 9절에 채웁니다.

| # | 단계 | 통과 조건 |
|---|---|---|
| 0 | 착수 회의 | 13절 확정, test 담당 지정, `docs/restart` 병합 |
| 1 | test 수집 시작 (test 담당) | 수집 조건 대조표. 개발 쪽에는 건수만 전달 |
| 2 | 라벨 규칙 파일럿 · 동결 | tag `label-rules-frozen` |
| 3 | train · val 라벨링 · 분할 확정 | `validate_labels.py` · `check_splits.py` 통과, tag `split-frozen` |
| 4 | test 라벨링 · 봉인 (test 담당 + 2차 검수자) | 봉인 해시 커밋, tag `test-sealed` |
| 5 | 전처리 · 프롬프트 · 채점기 | 추출 실패 기록, 채점기 회귀 테스트 통과 |
| 6 | Base 진단 · 학습 r1 → r2 | val 채점 · 짝 비교 |
| 7 | 모델 · 실험 고정 | `qlora_final` · `max_new_tokens` · 채점기 해시 · 결론 문구 커밋 |
| 8 | test 평가 1회 | 비교군 3개(base_zs · base_fs · qlora_final) 각 1회 |
| 9 | 보고 | `report/final_table.md` · `report/failures.md` |

제품 트랙(API · DB · Rule Engine · 화면)은 5단계부터 함께 진행합니다.

## 원칙

- **test 봉인:** test 문서 · 텍스트 · 정답은 저장소 밖에 둡니다. 개발 세션(사람 · AI 모두)은 열지 않습니다. 평가가 끝나기 전에는 test 파일명 · 제조사 · 정답에서 나온 값을 어떤 파일에도 적지 않습니다.
- **분할:** `data/splits.csv`가 유일한 기준입니다. 학습 · few-shot은 train만, 조건 선택은 val만 씁니다.
- **공유 코드:** 전처리 · 프롬프트는 학습 · 평가 · 서빙이 `core/` 하나를 같이 씁니다. `core/` 수정은 담당자 리뷰가 필요합니다.
- **스키마 동결:** 출력 JSON 키, 상태값(기재 · 자료없음 · 해당없음), DB 테이블 · 열 이름을 바꾸지 않습니다. 출력 스키마는 `src/schema.py` 하나입니다.
- **범위:** 기능을 늘리지 않습니다. 범위는 `docs/plan.md` 1 · 8절입니다.
- **Git:** main에 직접 push하지 않고 작업 브랜치에서 PR로 병합합니다. `git add .` 대신 경로를 지정합니다. PDF · zip · 가중치 · `.env` · `*.db`는 커밋하지 않습니다.

## 처음 받을 때

```bash
git clone https://github.com/EOCHAEEUN/ABO_MSDS.git && cd ABO_MSDS
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
git switch -c <prefix>/<작업>

# 검사
python3 scripts/validate_labels.py data/labels
python3 scripts/check_splits.py

# 검토 화면 (예시 모드)
npm --prefix web ci && npm --prefix web run dev

# MSDS 질의 기능은 docs/rag_question.md의 Ollama·인덱스 준비 후 사용
```

- 한 폴더에서는 한 세션(사람 · AI)만 씁니다. 각자 따로 clone합니다.
- 원본 PDF(`data/raw/`)와 어댑터 가중치는 git에 없습니다. 팀 드라이브로 받습니다.

## 파일트리와 담당

역할은 `docs/plan.md` 11절 제안 기준입니다. 착수 회의에서 확정합니다.
**데** 강덕우 · **규** 김건하 · **평** 양세윤 · **프** 어채은(PM)

```
ABO_MSDS/
├── README.md · CLAUDE.md        # [프] 과업 개요 · 작업 규칙
├── requirements.txt             # [평] 버전 고정
├── core/                        # [평 · 데] 학습 · 평가 · 서빙 공용 — 전처리 · 스키마 검사 · 프롬프트 · 정규화
├── data/                        # [데 · 규]
│   ├── README.md                # 라벨 표기 규칙, doc_id 규칙
│   ├── raw/                     # 원본 PDF (git 제외)
│   ├── text/                    # 전처리 결과 {doc_id}.txt, _cut_log.csv
│   ├── sources.csv              # 출처 (test 행에는 정답 파생값을 적지 않음)
│   ├── splits.csv               # 분할의 유일한 기준, 확정 후 tag split-frozen
│   └── labels/                  # train · val 정답 {doc_id}.json
├── pipeline/                    # [데] 텍스트 추출 · 증강 · JSONL · 길이 측정 · 학습, configs/r1.yaml · r2.yaml
├── eval/                        # [평] 추론 · 채점 · 봉인, fewshot.json, hazard_class_alias.csv
├── outputs/                     # [평] 모델 출력 {조건}/{val|test}/{doc_id}.json
├── runs/                        # [데] {날짜}_{r1|r2}/ — config.json · loss 로그만 커밋
├── app/                         # [평] FastAPI · 모델 로드 · DB, rules/ [규] Rule Engine
├── web/                         # [프] 검토 화면 (Vite + React, README.md에 API 연결 계약)
├── src/schema.py                # 출력 스키마 단일 기준
├── scripts/                     # validate_labels.py · check_splits.py
├── docs/                        # [프] plan.md, labeling_review_notes.md, frontend_api_spec.md
└── report/                      # decisions · scores · length_stats · test_manifest · final_table · failures
```

## Git 태그

- `label-rules-frozen`: 라벨 규칙 동결
- `split-frozen`: train · val 분할 확정
- `test-sealed`: test 정답 확정 · 봉인 해시 커밋 직후
