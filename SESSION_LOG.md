# 세션 로그 — SK Hynix Defect Test Prediction

> 최종 업데이트: 2026-07-03 (세션 3 종료 시점)

---

## 1. 프로젝트 개요

- **과제**: FDC Trace 데이터 → Defect Test(C65) 회귀 예측 (난이도 下)
- **평가지표**: RMSE (낮을수록 좋음)
- **목표 RMSE**: ~40 (현재 최선: ~62 → 약 20pt 추가 개선 필요)
- **데이터**: `문제1(하)/` (train 123,614행 × 65컬럼, valid/test 각 ~20,500행 × 64컬럼)
- **WF 수**: train 11,939 / valid 1,990 / test 1,990
- **프로젝트 규칙**: `CLAUDE.md` 참조

---

## 2. 데이터 구조 핵심 (EDA에서 확인)

| 항목 | 내용 |
|------|------|
| 행 구조 | 1행 = 1 WF × 1 Step. WF당 약 10행 |
| C65 (타깃) | WF 내 상수 → WF 단위로 집계 후 예측 |
| 결측 | 부분 결측 0개 (전부 결측 컬럼만 제거) |
| C65 분포 | 평균 973, 왜도 0.05, 이상치 0개 |
| Covariate Shift | 없음 (Adversarial AUC ≈ 0.50) |
| 핵심 피처 | C17(r=-0.797), C10(0.478), C39(0.478), C12(0.338) |
| PM(C33) | 직접 상관 미미 (r=-0.013) |
| FDC 센서 | 제거 후 36개 수치형 컬럼 남음 |
| 범주형 | C6: 2종(99.2% C6_0), C7: 5종 Step |

---

## 3. 컬럼 역할 정리 (CLAUDE.md에도 있음)

| 구분 | 컬럼 | 처리 |
|------|------|------|
| WF ID (그룹키) | C64, C34, C35, C38 | C64만 그룹키, 나머지 drop |
| Recipe ID | C6(2종), C23, C30 | C6만 사용 |
| Step 번호 | C7(5종), C36 | C7 사용 |
| Lot ID | C20, C21, C22 | drop (일반화 불가) |
| Chamber | C24 | 상수 → drop |
| 장비 | C14 | 상수 → drop |
| 시간 | C10, C39, C40, C41 | C10/C39 → datetime 변환, C41 경과시간 |
| PM time | C33 | drift 지표 활용 |
| 제외 지정 | C26, C28, C29, C37 | drop |
| 전부 결측 | C2, C13, C43, C47, C53, C55 | drop |
| X (FDC 센서) | 나머지 수치형 36개 | WF 단위 집계 피처 |
| Y | C65 | 타깃 (WF 내 상수) |

---

## 4. 모델 실험 결과 종합

### 4.1 RMSE 비교표

| 버전 | 내용 | CV OOF | Valid | Test | 피처 수 | 결과 |
|------|------|--------|-------|------|---------|------|
| 베이스라인 (평균) | 전부 평균 예측 | — | 258.97 | — | — | — |
| **baseline** (=v1 정리) | 기본 통계 집계 + 수동 파라미터 | 62.88 | 62.53 | 61.15 | 315 | 기준선 |
| **v2** | Step(C7)별 센서 집계 | 63.12 | 62.72 | 61.25 | 816 | 악화 |
| **v3** | Optuna 100 trials 튜닝 | 62.19 | 62.31 | 60.51 | 299 | **최선** |
| **v4** | 피처 근본 재설계 (5개 그룹) | 63.09 | 63.42 | 61.19 | 901 | 악화 |
| **v5** | Row-level 예측 + C23 재탐색 반영 | 62.63 | **61.38** | 60.52 | row단위 | Valid 최선, Test 정체 |
| **v6** | 시퀀스 1D-CNN (딥러닝) | 68.92 | 68.26 | 66.04 | 3D텐서 | 악화(딥러닝 부적합) |
| **목표** | — | — | ~40 | ~40 | — | — |

### 4.2 각 버전 상세

**baseline (= 기존 v1 정리본, `modeling_baseline.ipynb`)**
- WF 단위 통계 집계: mean/std/min/max/median/range/delta/slope (8종 × 36센서)
- C6/C7 one-hot 비율, n_rows, C41_total, C33_first/max
- LightGBM: lr=0.05, num_leaves=63, subsample=0.8, colsample=0.8
- 현재 프로젝트 루트의 기준 노트북

**v2 (`modeling_v2/`)**
- baseline + Step(C7)별 센서 mean/std, step 전환 피처, 센서 교차(C17×C12 등), C33 binning
- 실패 원인: WF당 step별 행이 1~2개 → step별 통계가 global 통계와 거의 동일

**v3 (`modeling_v3/`)**
- baseline 피처(299개) + Optuna TPE 100 trials 자동 튜닝
- 최적 파라미터: `optuna_best_params_v3.json`에 저장
- lr=0.00576, num_leaves=189, max_depth=10, min_child_samples=14, subsample=0.967, colsample=0.655, reg_alpha=4.256, reg_lambda=0.003, min_split_gain=0.758
- 개선폭 ~0.5pt → 하이퍼파라미터는 거의 한계

**v4 (`modeling_v4/`)**
- 5개 신규 피처 그룹: 분포 형태(skew/kurt/IQR/CV), 센서 교차(120쌍×3), FFT top-3, 안정성(Cpk/drift), Target Encoding
- 실패 원인: 노이즈 피처 폭증(360+개 교차 피처), FFT에 쓸 시계열이 너무 짧음(~10행), v3 params가 901개 피처에 부적합
- 알려진 버그: `.values`로 학습시켜 컬럼명 소실 → 피처 그룹 분석 전부 0%

**v6 (`modeling_v6/`)** — 시퀀스 딥러닝
- WF당 (16 step × 41채널: 36센서 표준화 + C7 one-hot) 3D 텐서, C23 임베딩, 1D-CNN + masked mean pooling, GroupKFold(C64).
- 결과: Test 66.04 → tabular(60.5) 대비 +5.5pt 악화. **딥러닝 방향 종료.**
- 원인: 표본 부족(11,939 WF), 짧고 약한 시퀀스(step 9~16, C40 간격 균일). GBDT가 소표본 tabular에서 우위.
- **오차 분석(v5 Test)**: R²=0.947, corr 0.973으로 이미 강함. 오차는 극단값 수축 + **742~976 구간 편향 +58.8**(특정 regime 미포착 신호)에 집중. C23_14/C23_12에서 오차 최대.

**v5 (`modeling_v5/`)**
- **데이터 재탐색 발견**: (1) C23(28종 Recipe)이 이전 버전에서 잘못 누락됨 — C6은 99.2% 상수인데 C23은 실제 변동, 일반화 안전(미관측 0). (2) step×센서 상호작용 강함(C17 상관 step별 0.19~0.80). (3) C36=C7 완전중복, C30 상수, C40/C41 무신호 → drop.
- **설계**: row(step) 단위 C65 직접 예측 → WF 평균. WF 전역 context broadcast + C7 native categorical + C23 out-of-fold TE. DataFrame 학습(v4 `.values` 버그 방지). GroupKFold(C64).
- **결과**: Valid 61.38(역대 최선, v3 62.31 대비 -0.93)이나 Test 60.52로 v3(60.51)과 동일 → **천장 미돌파**.
- 원인: row에 WF 집계를 함께 넣어 트리가 여전히 WF 평균 센서(C17_wf_mean, r=-0.797)에 의존. C23 기여폭 자체가 작음(eta² ~4.8%).

### 4.3 핵심 진단

**"WF 단위 통계 집계 → tabular ML" 프레임워크 자체가 RMSE ~60의 천장을 가짐. Row-level(v5)도 이 천장을 넘지 못함.**

| 시도 | 시사점 |
|------|--------|
| 피처 추가 (v2, v4) | 무차별적 피처는 노이즈만 증가 |
| 하이퍼파라미터 튜닝 (v3) | 개선 여지 ~0.5pt 수준 |
| 데이터 재탐색 + Row-level (v5) | Valid 소폭 개선, Test 정체 — tabular 계열 표현력 소진 |
| 결론 | tabular 프레임(집계·row 모두) ~60 수렴. ~40은 시퀀스 모델만 가능성 |

---

## 5. 현재 프로젝트 파일 구조

```
defect_test_prediction_MK/
├── CLAUDE.md                     # 프로젝트 규칙/컬럼 명세
├── README.md                     # 프로젝트 개요
├── REPORT.md                     # v1 결과 보고서
├── SESSION_LOG.md                # ← 이 파일
├── requirements.txt              # Python 의존성
├── train.py                      # CLI 학습 파이프라인 (python train.py --tune)
├── modeling_baseline.ipynb       # 기준 모델 노트북 (v1 정리본)
├── valid_Y_submit.csv            # baseline 제출 파일
├── test_Y_submit.csv             # baseline 제출 파일
├── 제9회_Data 분석 경진대회_문제1.pptx
│
├── EDA/
│   ├── 01_EDA.ipynb
│   ├── 01_EDA_README.md
│   └── 01_EDA_REPORT.md
│
├── modeling_v2/
│   ├── modeling_v2.ipynb
│   ├── modeling_v2_README.md
│   └── modeling_v2_REPORT.md
│
├── modeling_v3/
│   ├── modeling_v3.ipynb
│   ├── modeling_v3_README.md
│   ├── modeling_v3_REPORT.md
│   └── optuna_best_params_v3.json
│
├── modeling_v4/
│   ├── modeling_v4.ipynb
│   ├── modeling_v4_README.md
│   └── modeling_v4_REPORT.md
│
├── modeling_v5/
│   ├── modeling_v5.ipynb
│   ├── modeling_v5_README.md
│   ├── modeling_v5_REPORT.md
│   └── outputs/  (valid/test_Y_submit.csv, results.json)
│
├── modeling_v6/
│   ├── modeling_v6.ipynb   (1D-CNN + Cell8 LSTM)
│   ├── modeling_v6_README.md
│   ├── modeling_v6_REPORT.md
│   └── outputs/  (valid/test_Y_submit.csv, results.json)
│
├── 문제1(하)/                    # 원본 데이터
│   ├── train_data.csv
│   ├── valid_X.csv / valid_Y_problem.csv
│   └── test_X.csv / test_Y_problem.csv
│
├── 문제1_하_answer/              # 정답 (평가용)
│   ├── valid_Y_answer.csv
│   └── test_Y_answer.csv
│
└── venv/                         # Python 가상환경
```

---

## 6. train.py 사용법

프로젝트 루트에 `train.py`가 있음. baseline과 동일한 파이프라인을 CLI로 실행 가능:

```bash
python train.py                    # 기본 파라미터로 학습
python train.py --tune             # Optuna 100 trials 튜닝 후 학습
python train.py --tune --n_trials 200  # trial 수 지정
```

결과는 `outputs/` 폴더에 저장됨: `results.json`, `valid_Y_submit.csv`, `test_Y_submit.csv`

---

## 7. 환경 설정 메모

| 항목 | 내용 |
|------|------|
| venv 활성화 | `.\venv\Scripts\Activate.ps1` |
| pip 사용 | `python -m pip install` (pip.exe 직접 실행은 회사 보안정책에 의해 차단) |
| Jupyter 커널 등록 | `python -m ipykernel install --user --name venv --display-name "venv"` |
| Jupyter 커널 에러 시 | `taskkill /F /IM jupyter* /T` 후 VS Code 재시작 |
| 의존성 설치 | `python -m pip install -r requirements.txt --break-system-packages` |

---

## 8. 다음 세션에서 해야 할 일 (RMSE ~60 → ~40)

**세션3 완료**: 데이터 재탐색 + Row-level(v5) + 시퀀스 딥러닝(v6) 실행.
- tabular 프레임(집계·row) ~60 수렴 확정.
- **딥러닝(v6) 종료** — 표본 부족·짧은 시퀀스로 GBDT에 패배(Test 66).
- 오차 분석 결과: 남은 오차는 **742~976 구간 편향 +58.8** 등 특정 regime 미포착에 집중.

### 우선순위별 다음 시도 (갱신)

| 순위 | 방향 | 핵심 아이디어 | 기대 | 상태 |
|------|------|-------------|------|------|
| ~~-~~ | ~~데이터 재탐색 / Row-level / 시퀀스~~ | — | — | ✅ 완료(세션3) |
| **1** | **regime 진단 → 세그먼트 피처/모델 (v7)** | 742~976 구간·C23_14/12 과대예측 WF들의 공통점(C23/C7 조합/센서 임계) 발굴 → 피처화 또는 구간별 모델 | 미지수, 최대 잠재력 | 다음 |
| 2 | **멀티모델 앙상블** | v3 LGB + v5 row-level + XGB + CatBoost 가중평균 | 1~3pt, ~59 안정 | 대기 |
| 3 | **OOF 사후 보정** | 극단 수축을 isotonic/선형 보정 (OOF only) | 0~2pt | 대기 |
| ~~4~~ | ~~딥러닝(LSTM 등)~~ | 표본 한계 확인 | 종료 | ✅ v6 |

### 구체적 실행 계획 (v7: regime 진단)

1. v5 OOF 잔차를 타깃으로, 어떤 피처(C23, C7 구성비, 주요 센서 구간)가 큰 잔차를 예측하는지 분석(잔차에 트리 학습 → 중요도).
2. 특히 정답 742~976 구간에서 과대예측되는 WF들을 격리해 공통 특성 도출.
3. 발견 시 → (a) regime 지시 피처 추가 후 재학습, 또는 (b) regime별 분리 모델.
4. 현실성: R² 0.947 → 목표 40은 R² 0.977 필요. regime 신호가 유일한 열쇠일 가능성. 없으면 앙상블로 ~59 확정 후 마무리 검토.

---

## 9. 작업 규칙 리마인더

1. **ipynb는 직접 수정하지 말 것** — 수정할 부분만 텍스트로 안내, 사용자가 직접 수정
2. **버전 관리**: 피처/모델/EDA 변경 시 새 파일 생성 (버그 수정은 기존 파일에)
3. **"컴파일 완료 됐어!" 시 자동 작업**: README + REPORT 생성 → 다음 단계 제안
4. **Lot ID/WF ID 피처 사용 금지**: C20/C21/C22, C64/C34/C35/C38
5. **GroupKFold(C64)** 필수: row 누수 방지
