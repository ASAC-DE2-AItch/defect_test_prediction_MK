# modeling_v11.ipynb — 노트북 안내서

> **modeling_v10 에서 시간/레짐 피처 7개만 제거**한 버전 (ablation 실험).
> 피처 = combined 원본센서 36 + WF집계 144 + 구조 2 + 범주형 2 + `C23_te` + `C20_te` = **186개** (v10 193개 − 7)
> **결과: valid RMSE 34.70 / test 35.10** — v10(34.29/34.33) 대비 소폭 악화(+0.4 / +0.8).
> 이전 흐름: `modeling_v5` → `combined_model_full_handover` → `modeling_v10`(+Lot TE) → **v11**(v10 − 시간7)
>
> 이 문서는 **노트북 사용법**입니다. 실험 해석은 컴파일 후 생성될 REPORT를 보세요.

---

## 제거한 시간/레짐 7개

`is_post_loud_pm`, `days_since_last_pm`, `hour`, `dslp_x_hour`, `hour_x_c33`, `post_pm_days`, `is_special_recipe`

## v10 대비 결과

| 버전 | 피처 수 | OOF | valid | test |
|------|--------|-----|-------|------|
| modeling_v10 (combined + Lot TE) | 193 | 34.87 | **34.29** | **34.33** |
| **modeling_v11 (v10 − 시간7)** | 186 | 35.63 | **34.70** | **35.10** |
| 차이 | −7 | +0.76 | +0.41 | +0.77 |

**해석:** Lot 타깃인코딩(`C20_te`)이 점수를 지배하는 상황에서도, 시간/레짐 7개는 **여전히 작지만 실질적인 기여**(valid −0.4, test −0.8)를 한다. 즉 **빼면 소폭 손해** → 대회 점수 관점에선 v10(시간피처 유지)이 우세. 이 노트북은 그 ablation 근거를 남기는 용도.

## ⚠️ 누수 주의 (v10과 동일)

`C20_te`(Lot 평균 C65)는 valid/test 가 train 과 Lot 을 99.9~100% 공유해서 점수를 낮춘다.
신규 Lot 기준(GroupKFold C20)으론 무너지며 **실무 일반화가 안 된다.** 프로젝트 원칙(`CLAUDE.md`:
"Lot ID 피처 금지")과 배치되는 **대회 점수 전용** 계열이다.

## 실행 전 준비

1. **커널**: `venv`
2. **폴더 위치**: `modeling_v11/` 안. 상대경로 참조:
   - 피처 코드: `../combined_model_full_handover/src`
   - PM 설정: `../combined_model_full_handover/data/processed/{pm_bins,pm_log}.json`
   - 데이터: `../문제1(하)/{train_data,valid_X,test_X}.csv`, `../문제1_하_answer/{valid,test}_Y_answer.csv`
3. **실행 시간**: 약 15~20분 (row-level 123,614행 × 5-Fold LightGBM)

## 셀 구성

| 셀 | 내용 |
|----|------|
| 1 | 설정 — 경로, import (`TIME_FEATS` 포함) |
| 2 | combined row-level 피처 빌드 + Lot(C20) 부착 |
| 3 | **★ 시간/레짐 7개 제거**(`feats_notime`) + C23/Lot 타깃인코딩 정의 |
| 4 | GroupKFold(C64) 5-Fold LightGBM 학습 |
| 5 | WF 집계 + valid/test RMSE 평가 |
| 6 | 제출 CSV + `results.json` 저장 |

## 출력 파일

| 파일 | 내용 |
|------|------|
| `outputs/valid_Y_submit.csv` | Valid 예측 (wafer_id, predicted_C65) |
| `outputs/test_Y_submit.csv` | Test 예측 |
| `outputs/results.json` | OOF / Valid / Test RMSE |

## 누적 실험 로그

| 날짜 | 변경 | OOF | valid | test |
|------|------|-----|-------|------|
| 2026-07-11 | v11 생성 = v10 − 시간/레짐 7개 | 35.63 | 34.70 | 35.10 |
| (참고) | v10 = combined(193) | 34.87 | 34.29 | 34.33 |

## 스냅샷 인덱스 (REPORT)

| # | 파일 | 내용 |
|---|------|------|
| 01 | `REPORT/modeling_v11_REPORT_01_LotTE.md` | 시간/레짐 7개 제거 ablation 스냅샷: v10 대비 소폭 악화(빼면 손해) 결론 |
