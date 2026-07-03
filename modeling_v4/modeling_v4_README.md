# modeling_v4.ipynb — 노트북 안내서

> 피처 엔지니어링 근본 재설계 실험 (결과: 실패)
>
> 이전 버전: v1(수동), v2(Step별), v3(Optuna)

---

## 실행 전 준비

1. **커널**: `venv`
2. **실행 시간**: 약 15~20분 (피처 추출 7분 + 학습 10분)

---

## 셀 구성

| 섹션 | 셀 번호 | 내용 |
|------|---------|------|
| 설정~전처리 | 0~5 | imports, 데이터 로드, 컬럼 정리 |
| 피처 엔지니어링 | 6~7 | `extract_wf_features_v4()` — 6개 그룹 피처 생성 |
| 피처 추출 | 8~9 | train/valid/test 적용 → 899개 피처 |
| Target Encoding | 10~11 | C6/C7 WF 대표값 추출 |
| 데이터 분리 | 12~13 | fold-aware TE 준비, valid/test에 global TE 적용 |
| 학습 | 14~15 | v3 Optuna 파라미터로 5-Fold CV |
| 평가 | 16~17 | v1~v4 RMSE 비교 |
| 중요도 | 18~19 | Top 30 + 신규 피처 기여도 |
| 제출 | 20~21 | CSV 생성 |
| 점검 | 22~23 | 히스토그램 + 산점도 |
| 그룹 분석 | 24~25 | 피처 그룹별 Importance 비중 |

---

## 추가된 피처 그룹

| 그룹 | 피처 수 (추정) | 내용 |
|------|---------------|------|
| B. 분포 형태 | ~144 | skewness, kurtosis, IQR, CV (36센서 × 4) |
| C. 센서 교차 | ~360 | 16센서 C(16,2)=120쌍 × 3(차이/비율/곱) |
| D. FFT | ~48 | 16센서 × top-3 magnitude |
| E. 안정성 | ~48 | 16센서 × 3(range_ratio/cpk/drift) |
| F. Target Encoding | 2 | te_c6, te_c7 |

---

## 알려진 버그

셀 25 (피처 그룹별 Importance 분석)에서 모든 그룹이 0%로 나옵니다.
원인: `model.fit(X_tr.values, ...)` 에서 `.values`로 넘기면서 컬럼명이 소실되어 `model.feature_name_`이 auto-generated 이름이 됨.
