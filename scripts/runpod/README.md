# RunPod 실행 — r1_aug 사전 등록 판정용

판정에 쓰는 교차검증(r1 · r1_s43 · r1_aug · r1_aug_s43 + base_fs · base_zs)과 변형 시험을 RunPod GPU 한 종류에서 처음부터 다시 돌린다. 왜 다시 돌리는지와 무엇을 같게 두는지는 `cv/prereg_r1_aug.md` 1.1절에 있다. 결과는 `cv/runpod/`에 쌓이고, 노트북 결과 `cv/`와 섞지 않는다.

## 0. 시작 전

- 사전 등록(`cv/prereg_r1_aug.md`)과 이 스크립트가 커밋 · push돼 있어야 한다. 그 커밋 해시를 아래 `<PREREG>`에 쓴다.
- 스크립트는 HEAD가 그 커밋 뒤인지, 사전 등록 파일과 추적 파일이 바뀌지 않았는지 확인한 뒤에만 돈다.
- test 자료는 쓰지 않는다. pod에는 공개 저장소의 train · val 텍스트 · 라벨만 들어간다.

## 1. pod 만들기

| 항목 | 값 |
|---|---|
| GPU | **RTX 4090 24GB × 1** (사전 등록에 적은 종류. 바꾸려면 첫 학습 전에 사전 등록부터 고친다) |
| CUDA | 드라이버가 **CUDA 13.0 이상**을 지원하는 머신(생성 화면의 CUDA Version 필터). torch 2.14.0+cu130을 쓴다 |
| 이미지 | Ubuntu 24.04 계열 PyTorch 템플릿. Python 3.12가 없으면 setup.sh가 uv로 받는다 |
| 볼륨 | `/workspace` 50GB 이상(모델 약 8GB + venv 약 8GB + 어댑터 · 출력). pod를 멈춰도 남는다 |

## 2. 준비 (약 20~30분)

```bash
cd /workspace && git clone -b feat/group-kfold-cv https://github.com/triunee/ABO_MSDS.git && cd ABO_MSDS
bash scripts/runpod/setup.sh <PREREG>
```

setup.sh가 하는 일:
1. GPU를 다른 프로세스가 쓰고 있지 않은지 확인한다.
2. `/workspace/venv`를 만들고 버전을 고정해 설치한 뒤, 노트북 학습 환경과 버전이 같은지 확인한다.
3. 모델 스냅샷을 받는다(r1.yaml의 `model_revision`).
4. `cv/runpod/hardware.txt`를 남긴다.
5. 테스트를 돌리고, 저장소 밖 `/workspace/smoke`에서 동작 점검을 한다(학습 2 step · 추론 2건).

## 3. 실행

```bash
nohup bash scripts/runpod/run_all.sh <PREREG> > /workspace/run_all.out 2>&1 &
```

- 순서: fold 분할 확인(`cv/folds.csv`와 바이트 비교) → r1 · r1_aug · r1_s43 · r1_aug_s43 · base_fs · base_zs를 fold마다 학습 · 추론 → `cv.py report` → `cv_rules.py` → 변형 시험(layout · value · glyph) make · run · report
- **중간 멈춤 규칙(사전 등록 3절):** seed 43 조건에 들어가기 전에 `cv.py stopcheck --recipe r1_aug`로 seed 42 짝을 본다. r1_aug − r1이 0 이하이면 r1_s43 · r1_aug_s43을 건너뛰고(학습 약 40% 절약) 나머지만 돈다. 결과는 `cv/runpod/stop_rule.txt`에 남는다.
- 진행 상황:
  ```bash
  source /workspace/venv/bin/activate && python eval/cv.py --root cv/runpod status
  ```
  `tail -f /workspace/run_all.out`, `nvidia-smi`로도 볼 수 있다.
- 끊기면 같은 명령을 다시 돌린다. 끝난 단계는 건너뛰고, 중간에 끊긴 fold의 학습만 처음부터 다시 한다.
- 일부 조건만 돌리려면 커밋 해시 뒤에 조건 이름을 붙인다. 예: `run_all.sh <PREREG> r1 r1_aug`

**시간 (추정, 첫 fold에서 실측해 고칠 것):**
- 학습: 1,644 step(r1 · r1_s43 712 + r1_aug 두 seed 932)이다. 4090에서 step당 12~16초로 보면 약 5.5~7.5시간이다.
- 추론: CV 30회 + 변형 시험 75회(회당 9건)로 약 3~4시간이다.
- 합계 약 8~12시간이다. 노트북은 step당 56~61초였다.

## 4. 결과 받기

```bash
bash scripts/runpod/collect.sh
```

- `/workspace/results_*.tar.gz`: 커밋할 파일만 들어 있다. 보고서 · 점수 · 출력 · `config.json` · `loss.csv` · `hardware.txt` · `gpu_log.csv`이다.
- `/workspace/adapters_*.tar.gz`: 어댑터 20개(약 0.9GB)다. 커밋하지 않는다.
- 받는 방법은 `runpodctl send <파일>`(받는 쪽은 `runpodctl receive <코드>`) 또는 pod의 SSH를 이용한 scp다.
- 저장소 루트에서 `tar xzf results_*.tar.gz` 한 뒤 `git add cv/runpod`로 커밋한다.
- 받은 뒤에는 pod를 멈추거나 지운다. 볼륨은 멈춰 있어도 요금이 붙는다.

## 판정 보는 곳

- `cv/runpod/report.md`: "사전 등록 판정: r1_aug vs r1" 절
- `cv/runpod/perturb/report.md`: glyph · value 세트의 "r1_aug 하락 − r1 하락 (두 seed 평균)" 행
