# lean-85 wafer별 예측/잔차 — 다른 파트 전달용

> 산출 2026-07-19 · **프리뷰(xgboost 3.2.0)** — 로컬(3.3.0) 재현으로 확정 · 스크립트 `produce_lean85_wafer_pred.py`

## 파일: `lean-85_wafer_pred.csv` (11,016행 = seg2 전체)

| 컬럼 | 의미 |
|---|---|
| `C64` | Wafer ID |
| `y_true` | 실측 C65 (정답) |
| `y_pred_holdout` | **무재학습** 예측 — seg1 TRAIN(3,922)로 1회 학습 → seg2 예측 |
| `y_pred_retrain` | **재학습** 예측 — seg2 주간(7일) 확장창 롤링(B2′) |
| `is_onset` | 1 = 요란 PM 직후 7일(레짐-온셋, 예측 신뢰 낮음), 0 = 정착 |

**잔차는 수신 측에서 `|y_true − y_pred_holdout|` 또는 `|y_true − y_pred_retrain|`로 계산.**

## 요청서 대비 확정 사항 (3가지)
1. **모델 = XGBoost** (요청서의 "LightGBM 예측"은 오기 — lean-85는 처음부터 XGB. 동결 85피처·246r·seed42).
2. **y_pred 2컬럼** — "홀드아웃이랑 재학습 두 개 다" 요청 반영. 요청 스펙(단일 y_pred)이 필요하면 원하는 열 하나만 취하면 됨.
3. **대상 = seg2 전체(post-PM)** — 온셋 첫 주 포함, `is_onset` 라벨로 필터 가능.

## seg 방식 (민지님 방식, 데이터로 재현 확인)
- 병합 = train + valid⋈answer + test⋈answer = **15,919 WF**, wafer 시각순 정렬(C10 epoch 기준).
- 경계 = **C33 리셋(대PM, 2018-12-23) = `C64_9664`(seq 4904)**.
- **seg1** 4,903 (C65 중앙 636) = 학습 지형 → 시간순 앞 80% **TRAIN 3,922** / 뒤 20% **VAL 981**(채점자).
- **seg2** 11,016 (C65 중앙 1082) = **홀드아웃**(평가 전용, 불가침).
- ⚠️ `test_X`의 C40은 엑셀 손상('MM:SS.f') → **C10(unix epoch)+9h(KST)**로 복원해 정렬.

## 예측 방식
- **holdout(무재학습)**: TRAIN만 학습 → seg2 고정 예측. seg2를 전혀 안 본 순수 홀드아웃(불가침 엄격).
- **retrain(재학습)**: B2′ 롤링 — TRAIN 초기 → seg2를 주간(7일) 확장창으로 재학습(지난 seg2 편입 = 운영 시뮬). fold 7개.
- 랜덤 CV 미사용(요청: "랜덤 CV는 과대라 지양").

## 수치 (프리뷰 · xgboost 3.2.0)
| 예측 | seg2 전체 RMSE | 온셋 제외(정착) RMSE |
|---|---|---|
| holdout(무재학습) | 507.2 | 451.3 |
| **retrain(재학습)** | 304.5 | **99.75** |

- **재학습 정착 99.75 ≈ handoff 시간축 벤치마크 99.84**(B2′ 재학습) — 정합.
- holdout이 큰 이유: seg1(저레짐 636)만 학습해 seg2(고레짐 1082) **레벨 자체를 못 넘음** = "재학습 없으면 배포 불가"의 극단. 온셋(급등 직후 첫 주)이 retrain 전체 RMSE를 304로 끌어올림 → 정착만 보면 99.75.

## 재현
```
# 프로젝트 venv (xgboost 3.3.0), 이 폴더에서
python produce_lean85_wafer_pred.py
```
`handoff_lean85/lean85_pipeline.py`(동결 스펙·피처 빌더)를 자동 참조. 로컬 실행 시 `lean-85_wafer_pred.csv` 덮어씀 = **확정본**. XGB는 결정론적이라 프리뷰와 근사(버전 미세차만).
