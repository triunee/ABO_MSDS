# Rule Engine 오류 탐지 (교차검증 출력)

- 대상: 그룹 5-fold 교차검증 출력(cv/pooled/) 45건 × 조건별. Rule Engine은 app/rules/(검사 10개) 그대로.
- 틀림 = 문서 완전 정답(확장) 0. 표시 = review_status가 OK가 아님(NEEDS_REVIEW · INCONSISTENT · SCHEMA_ERROR).
- gold 행은 정답 라벨을 그대로 넣은 결과다. 원문 모순 · 검사 잡음으로 생기는 표시의 바닥값이다.
- 생성: `python3 eval/cv_rules.py`

## 문서 단위

| 조건 | 틀린 문서 | 표시 문서 | 탐지율(틀린 것 중 표시) | 정밀도(표시 중 틀림) | OK 통과율 | OK 문서 정확도 |
|---|---|---|---|---|---|---|
| gold | 0/45 | 6 | — (0/0) | 0.0% | 86.7% | 100.0% |
| base_zs | 35/45 | 34 | 85.7% (30/35) | 88.2% | 24.4% | 54.5% |
| base_fs | 27/45 | 23 | 63.0% (17/27) | 73.9% | 48.9% | 54.5% |
| r1 | 17/45 | 11 | 29.4% (5/17) | 45.5% | 75.6% | 64.7% |
| r1_s43 | 20/45 | 17 | 50.0% (10/20) | 58.8% | 62.2% | 64.3% |
| r1_aug | 17/45 | 13 | 47.1% (8/17) | 61.5% | 71.1% | 71.9% |

## 필드 단위 (틀린 필드 중 그 필드에 표시가 붙은 수 / 맞은 필드에 붙은 표시 수)

| 조건 | product_name | ingredients | ghs_classification | signal_word | hazard_statements |
|---|---|---|---|---|---|
| gold | 0/0 · 오경보 0 | 0/0 · 오경보 0 | 0/0 · 오경보 3 | 0/0 · 오경보 2 | 0/0 · 오경보 2 |
| base_zs | 0/1 · 오경보 1 | 9/14 · 오경보 3 | 3/22 · 오경보 3 | 1/3 · 오경보 0 | 3/6 · 오경보 5 |
| base_fs | 0/7 · 오경보 0 | 6/10 · 오경보 1 | 4/17 · 오경보 7 | 0/5 · 오경보 2 | 3/7 · 오경보 7 |
| r1 | 2/3 · 오경보 0 | 4/6 · 오경보 0 | 2/10 · 오경보 3 | 1/5 · 오경보 2 | 1/7 · 오경보 3 |
| r1_s43 | 2/3 · 오경보 0 | 3/7 · 오경보 5 | 2/11 · 오경보 5 | 1/5 · 오경보 2 | 1/8 · 오경보 3 |
| r1_aug | 2/3 · 오경보 0 | 6/8 · 오경보 1 | 2/10 · 오경보 3 | 1/3 · 오경보 1 | 1/4 · 오경보 3 |

## OK로 통과한 틀린 문서

- base_zs: KR-HENKEL-001 KR-ITW-001 KR-ITW-002 KR-NOROO-002 KR-NOROO-007
- base_fs: KR-3DSYS-001 KR-ITW-002 KR-KUMHO-002 KR-KUMHO-004 KR-KUMHO-005 KR-NOROO-004 KR-NOROO-007 KR-OCI-002 KR-OCI-004 KR-THERMO-001
- r1: KR-3DSYS-001 KR-HENKEL-001 KR-ITW-002 KR-KCC-001 KR-KUMHO-004 KR-KUMHO-005 KR-NOROO-003 KR-NOROO-004 KR-NOROO-006 KR-NOROO-007 KR-OCI-005 KR-SOTL-001
- r1_s43: KR-HANIL-004 KR-HENKEL-001 KR-ITW-002 KR-KCC-001 KR-KUMHO-004 KR-KUMHO-005 KR-NOROO-003 KR-NOROO-004 KR-NOROO-006 KR-NOROO-007
- r1_aug: KR-3DSYS-001 KR-DUKSAN-001 KR-HENKEL-001 KR-ITW-002 KR-KCC-001 KR-NOROO-003 KR-NOROO-004 KR-NOROO-006 KR-NOROO-007
