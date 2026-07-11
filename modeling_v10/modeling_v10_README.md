# modeling_v10.ipynb — 노트북 안내서

> **combined(192) + Lot 타깃인코딩(C20_te)** — 대회 점수 극대화 버전 (누수 감수)
> **결과: valid RMSE 34.29 / test 34.33** (combined 48.68/48.65 → −14pt). 상세·해석은 REPORT 참조.
> 이전 흐름: `modeling_v5`(row-level, valid 61.38) → `combined_model_full_handover`(v5 + 시간피처, valid 48.68) → **v10**(+ Lot TE)
>
> 이 문서는 **노트북 사용법**입니다. 실험 결과·해석은 컴파일 후 생성될 REPORT를 보세요.

---

## ⚠️ 먼저 읽을 것 — 이건 '대회 점수 전용' 버전

- valid/test 가 train 과 **Lot 을 99.9~100% 공유**하는 구조를 활용해 점수를 낮춘다.
- **신규 Lot 기준(GroupKFold C20)으론 70~100대**이고 **실무 일반화가 안 된다.**
- 프로젝트 원칙(`CLAUDE.md`: "Lot ID(C20/C21/C22)는 피처로 사용하지 않음")과 배치된다.
- 정직한 성능이 필요하면 `combined_model_full_handover`(48.68)나 챔피언(ExtraTrees 35.6) 라인을 쓸 것.
- 근거·수치 상세는 팀 `modeling_v5/modeling_v5_leakage_check_REPORT.md` 및 세션 로그 참조.

## v5 / v9 대비 무엇이 다른가

| 항목 | modeling_v5 | v9 (Lot 집계) | **v10 (이 노트북)** |
|------|------|------|------|
| 뼈대 | row(step)→WF평균 | wafer | row(step)→WF평균 (= combined) |
| Lot 활용 방식 | 안 씀(그룹키만) | `LOT_mean/std_*`(센서집계) | **`C20_te`(Lot 평균 C65 타깃인코딩)** |
| Lot 활용 결과 | valid 61.38 | valid 100+ (붕괴) | **valid 34.29 (성공)** |
| 왜 다른가 | — | 집계값 train/valid 불안정→깨짐 | train에서 Lot→점수 지도 만들어 valid 조회(안정) |

> 핵심: "Lot 정보를 붙인다"도 **방식에 따라 정반대**. 센서집계(v9)는 valid 붕괴, **타깃인코딩(v10)은 성공.**

## 실행 전 준비

1. **커널**: `venv`
2. **폴더 위치**: 이 노트북은 `modeling_v10/` 안에 있어야 함. 상대경로로 참조:
   - 피처 코드: `../combined_model_full_handover/src` (config·preprocessing·feature_engineering·combined_model)
   - PM 설정: `../combined_model_full_handover/data/processed/pm_bins.json`, `pm_log.json`
   - 데이터: `../문제1(하)/{train_data,valid_X,test_X}.csv`, `../문제1_하_answer/{valid,test}_Y_answer.csv`
3. **실행 시간**: 약 15~20분 (row-level 123,614행 × 5-Fold LightGBM, n_estimators=4000)

## 셀 구성

| 셀 | 내용 |
|----|------|
| 1 | 설정 — 경로, import, `rmse` |
| 2 | combined row-level 피처 빌드(build_rows) + Lot(C20) 부착 (train/valid/test) |
| 3 | C23 + Lot(C20) 타깃인코딩 헬퍼 + 모델 입력 피처(`USE`) 정의 |
| 4 | GroupKFold(C64) 5-Fold LightGBM 학습 (fold별 TE 재계산, row예측→WF평균) |
| 5 | WF 집계 + valid/test RMSE 평가 |
| 6 | 제출 CSV(`outputs/`) + `results.json` 저장 |

## 핵심 아이디어

combined 의 row-level 피처(원본센서 + WF집계 + 시간/레짐 + C23_te) 위에 **Lot 단위 타깃인코딩
`C20_te`** 를 얹는다. 각 웨이퍼에 "그 Lot 의 평균 불량 점수"를 주입하는 것. 대회 valid/test 가
train 과 Lot 을 공유하므로, train 에서 만든 Lot→점수 매핑이 valid/test 에 거의 그대로 전이된다.
누수 차단을 위해 C20_te 는 **fold 내부에서만** 계산(GroupKFold C64) 하고, valid/test 는 fold별
인코더의 5-fold 앙상블 평균으로 예측한다.

## 출력 파일

| 파일 | 내용 |
|------|------|
| `outputs/valid_Y_submit.csv` | Valid 예측 (wafer_id, predicted_C65) |
| `outputs/test_Y_submit.csv` | Test 예측 |
| `outputs/results.json` | OOF / Valid / Test RMSE |

## 누적 실험 로그

| 날짜 | 변경 | OOF | valid | test |
|------|------|-----|-------|------|
| 2026-07-11 | v10 생성 = combined(192) + C20_te | 34.87 | 34.29 | 34.33 |
| (참고) | combined(192) 기준 | 49.86 | 48.68 | 48.65 |

## 스냅샷 인덱스 (REPORT)

| # | 파일 | 내용 |
|---|------|------|
| 01 | `REPORT/modeling_v10_REPORT_01_LotTE.md` | combined + Lot TE 채택 스냅샷: 성능·오차분석(고불량 tail 과소예측)·누수해석 |
