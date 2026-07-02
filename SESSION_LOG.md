# 세션 로그 — SK Hynix Defect Test Prediction

> 작성일: 2026-07-02

---

## 1. 이번 세션에서 한 일

### 1-1. EDA 노트북 (01_EDA.ipynb) 디버깅 및 완료

노트북을 셀 단위로 실행하면서 발생한 오류 3건을 수정했습니다.

| 셀 | 오류 | 원인 | 수정 내용 |
|-----|------|------|----------|
| 셀 6 (C65 분포 분석) | `TypeError: unexpected keyword argument 'labels'` | 최신 matplotlib에서 `boxplot()`의 파라미터명 변경 | `labels=` → `tick_labels=` |
| 셀 9 (Adversarial Validation) | `TypeError: dtype 'str' does not support operation 'mean'` | 최신 pandas에서 `groupby().mean()`이 문자열 컬럼을 자동 제외하지 않음 | `.mean()` → `.mean(numeric_only=True)` |
| 셀 11 (KDE 오버레이) | `LinAlgError: singular data covariance matrix` | 상수/분산 0인 컬럼에서 KDE 추정 실패 | `s.nunique() > 1` 체크 + `try/except` 추가 |

### 1-2. EDA 문서 생성

| 파일 | 내용 |
|------|------|
| `01_EDA_README.md` | 비전공자용 노트북 안내서 — 각 셀 설명, 용어 정리, 실행 방법 |
| `01_EDA_REPORT.md` | 분석 결과 보고서 — 10개 항목별 결과, 해석, 조치 방향, 종합 요약 |

### 1-3. 모델링 파이프라인 (modeling_pipeline.ipynb) 디버깅

| 셀 | 오류 | 원인 | 수정 내용 |
|-----|------|------|----------|
| 셀 1 (imports) | `ModuleNotFoundError: No module named 'pandas'` | Jupyter 커널이 venv가 아닌 시스템 Python으로 설정됨 | 커널을 `venv/Scripts/python.exe`로 변경 |
| 셀 5 (피처 추출) | `ValueError: 'C64' is both an index level and a column label` | `get_dummies()` 후 인덱스와 컬럼명 중복 | `c6_dummies[WF_ID]=...` 삭제, `groupby(level=0)` 사용 |
| 셀 12 (Feature Importance) | `ValueError: Length of values (315) does not match length of index (314)` | Train/Valid/Test 간 더미 컬럼 수 불일치 | `index=common_cols` → `index=model.feature_name_` |

---

## 2. EDA 핵심 발견사항 요약

| # | 항목 | 결과 | 시사점 |
|---|------|------|--------|
| 1 | 상수 컬럼 | 20개 (전부 결측 8개 포함) | 모두 제거 → 68개 → 48개 컬럼 |
| 2 | 결측값 | 부분 결측 **0개** | 결측 처리 불필요 |
| 3 | WF 구조 | 웨이퍼당 약 10행, C65는 WF 내 상수 | 웨이퍼 단위 집계 후 예측 |
| 4 | C65 분포 | 평균 973, 왜도 0.05, 이상치 0개 | 타깃 변환 불필요 |
| 5 | 시간 분포 | Train/Valid/Test 시간 범위 완전 겹침 | 랜덤 분할, temporal drift 무시 가능 |
| 6 | Covariate Shift | Adversarial AUC ≈ 0.50 (모든 쌍) | 분포 동일, 도메인 적응 불필요 |
| 7 | 핵심 피처 | C17(-0.797), C10(0.478), C39(0.478), C12(0.338) | C17이 가장 강력한 예측 변수 |
| 8 | PM(C33) | 상관계수 -0.013 | 직접 영향 미미, 구간화 파생 변수 고려 |
| 9 | 이상치 | C65 IQR 이상치 0개 | 이상치 처리 불필요 |
| 10 | 범주형 | C6: 99.2% C6_0 / C7: 5개 step, 분포 동일 | 단순 변수로 포함 |

---

## 3. 모델링 파이프라인 현재 상태

`modeling_pipeline.ipynb`는 9개 섹션으로 구성되어 있으며, 디버깅 후 실행 가능한 상태입니다.

```
[완료] 셀 1: imports & 설정
[완료] 셀 2: 데이터 로드
[완료] 셀 3: 전처리 — 불필요 컬럼 제거
[완료] 셀 4: Feature Engineering 함수 정의 (WF 단위 집계)
[완료] 셀 5: 피처 추출 실행 (train/valid/test)
[완료] 셀 6: 피처/타깃 분리 & 정렬
[완료] 셀 7: GroupKFold CV + LightGBM 학습
[완료] 셀 8: Valid/Test RMSE 평가
[완료] 셀 9: Feature Importance 시각화
[완료] 셀 10: 제출 파일 생성
[완료] 셀 11: 일반화 점검 — 예측 vs 실제 분포
```

### 현재 모델 구성

- **모델**: LightGBM (GBDT)
- **CV**: GroupKFold 5-fold (C64 기준, row 누수 방지)
- **피처**: WF 단위 집계 (mean/std/min/max/median/range/delta/slope) + C6/C7 one-hot + PM(C33) + 경과시간(C41)
- **하이퍼파라미터**: lr=0.05, num_leaves=63, subsample=0.8, colsample=0.8, early_stopping=100

---

## 4. 앞으로 해야 할 일

### 4-1. 즉시 해야 할 것 (현재 파이프라인 완성)

- [ ] `modeling_pipeline.ipynb` 전체 셀 실행 완료 확인
- [ ] Valid RMSE / Test RMSE 결과 확인 및 기록
- [ ] CV↔Valid 격차로 과적합 여부 판단
- [ ] Feature Importance Top 30 확인 → 불필요 피처 정리 검토
- [ ] 제출 파일(`valid_Y_submit.csv`, `test_Y_submit.csv`) 생성 확인

### 4-2. 성능 개선 (RMSE 낮추기)

| 우선순위 | 방법 | 설명 |
|---------|------|------|
| 1 | **하이퍼파라미터 튜닝** | Optuna로 LightGBM 파라미터 자동 탐색 (learning_rate, num_leaves, min_child_samples 등) |
| 2 | **피처 선택** | Importance 하위 피처 제거, Permutation Importance 기반 정제 |
| 3 | **추가 피처 엔지니어링** | Step(C7)별 센서 통계, Step 간 변화율, C33 구간화 파생 변수 |
| 4 | **모델 앙상블** | XGBoost, CatBoost 추가 → 가중 평균 앙상블 |
| 5 | **타깃 변환 실험** | 왜도가 낮긴 하지만 Box-Cox / 로그 변환 시 성능 변화 확인 |
| 6 | **이상치 처리** | 센서 피처의 극단값 클리핑 실험 |

### 4-3. 보고서 및 정리

- [ ] 최종 모델 성능 기록 (RMSE 비교표)
- [ ] `modeling_pipeline.ipynb` README 작성
- [ ] 최종 결과 보고서 업데이트 (`01_EDA_REPORT.md`에 모델링 결과 추가 또는 별도 파일)

---

## 5. 프로젝트 파일 구조 (현재)

```
SKHynix_defect_test_prediction/
├── 문제1(하)/                    # 원본 데이터
│   ├── train_data.csv
│   ├── valid_X.csv
│   ├── valid_Y_problem.csv
│   ├── test_X.csv
│   └── test_Y_problem.csv
├── 문제1_하_answer/              # 정답 데이터 (평가용)
│   ├── valid_Y_answer.csv
│   └── test_Y_answer.csv
├── venv/                         # Python 가상환경
├── 01_EDA.ipynb                  # 탐색적 데이터 분석 (완료)
├── 01_EDA_README.md              # EDA 안내서 (비전공자용)
├── 01_EDA_REPORT.md              # EDA 결과 보고서
├── modeling_pipeline.ipynb       # 모델링 파이프라인 (디버깅 완료)
├── CLAUDE.md                     # 프로젝트 규칙/컬럼 명세
├── requirements.txt              # Python 의존성
├── 제9회_Data 분석 경진대회_문제1.pptx  # 대회 문제 설명
└── SESSION_LOG.md                # ← 이 파일
```
