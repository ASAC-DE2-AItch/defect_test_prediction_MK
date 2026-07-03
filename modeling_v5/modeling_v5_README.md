# modeling_v5.ipynb — 노트북 안내서

> Row-level 예측(step 단위 → WF 평균) + 데이터 재탐색 반영(C23 추가) v5 파이프라인
> **결과: Valid RMSE 61.38 (역대 최선) / Test 60.52.** 결과·오차분석은 `modeling_v5_REPORT.md` 참조
> 이전 버전: `modeling_baseline.ipynb` (v1), `modeling_v2/` (v2), `modeling_v3/` (v3), `modeling_v4/` (v4)
>
> 이 문서는 **노트북 사용법**입니다. 실험 결과·해석은 같은 폴더의 REPORT를 보세요.

---

## 실행 전 준비

1. **커널 설정**: `venv` 선택
2. **폴더 위치**: 이 노트북은 `modeling_v5/` 안에 있어야 함 (데이터 경로가 `../문제1(하)`, `../문제1_하_answer` 상대경로)
3. **실행 시간**: 약 5~10분 (row-level 123,614행 × 5-Fold LightGBM)

---

## 핵심 아이디어

기존 v1~v4는 모두 **WF 단위로 센서를 집계한 뒤 예측**했고, 이 프레임의 RMSE 천장이 ~62로 확인됨. v5는 프레임을 전환:

- **row(=step) 단위로 C65를 직접 예측 → WF 내 예측값 평균**
- C65가 WF 내 상수이므로 각 row가 같은 타깃을 공유 → GroupKFold(C64)로 누수 방지 필수
- 목적: WF 평균으로 뭉개지던 step×센서 상호작용을 트리가 직접 학습

---

## 셀 구성

| 셀 | 내용 |
|----|------|
| 1 | imports & 설정 (제외 컬럼, 범주형, TE 대상 정의) |
| 2 | 데이터 로드 (train / valid / test / 정답) |
| 3 | row-level 피처 빌더 — 원본 센서 + WF 전역 context(집계) broadcast + row_pos |
| 4 | C23 out-of-fold 타깃인코딩 헬퍼 (스무딩 m=20) |
| 5 | GroupKFold 5-Fold row-level LightGBM 학습 |
| 6 | row 예측 → WF 평균 집계 → WF-level RMSE 평가 |
| 7 | 피처 중요도(gain) 출력 + 제출 CSV / results.json 저장 |

---

## 세션2 데이터 재탐색 반영 사항

| 발견 | 처리 |
|------|------|
| **C23** (28종 Recipe, 이전 버전 누락) | out-of-fold 타깃인코딩으로 추가 |
| **C36 == C7** 완전중복 | drop |
| **C30** 진짜 상수(1종) | drop |
| **C40** step 간격 ~3.0초 일정 / **C41** row-level 상관 ≈ 0 | drop |
| **step×센서 상호작용** (C17 상관 step별 0.19~0.80) | C7을 LightGBM 범주형으로 명시 |
| v4 `.values` 버그 | DataFrame으로 학습(컬럼명 유지)하여 방지 |

---

## v3(최선) 대비 변경사항

| 항목 | v3 | v5 |
|------|----|----|
| 예측 단위 | WF (집계 후 1행/WF) | row (step 단위, ~10행/WF) |
| C23 | 미사용 | 타깃인코딩 추가 |
| 범주형 처리 | one-hot 비율 집계 | C6/C7 LightGBM native categorical |
| 학습 입력 | numpy `.values` | pandas DataFrame (컬럼명 유지) |
| 하이퍼파라미터 | Optuna 최적값 | 수동 (lr=0.03, num_leaves=127) — 미튜닝 |

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `outputs/valid_Y_submit.csv` | Valid 예측 결과 |
| `outputs/test_Y_submit.csv` | Test 예측 결과 |
| `outputs/results.json` | OOF / Valid / Test RMSE |
