# baseline_vanilla REPORT 01 — 바닐라 Baseline 확정 (스냅샷)

> **스냅샷 (고정)** · 작성 2026-07-19 · 노트북 `baseline_vanilla.ipynb` 로컬 실행(lightgbm 4.6.0, sklearn) · 이후 수정 금지
> 목적: 도메인 처리 없는 **기본 ML + 전체 피처**의 성능을 고정해 프로젝트 모델(after)의 "before" 출발선으로 삼는다.

---

## 0. 한 줄 요약

**아무 처리 없는 바닐라의 최선 = ID제외 LightGBM, same-era test RMSE 85.0** (valid 87.3). 프로젝트 모델의 동일-채점 same-era 성능(v8 코어10 test ~35.6, v10 test 34.3)과 대비하면 **약 50pt(≈60%) 개선** — 피처 엔지니어링·모델링의 가치를 정량화하는 before 기준선이다. (최종 운영 모델 lean-85는 *시간축* 재학습 99.84/R² 0.8545로 축이 다름 — §3 참조.)

## 1. 결과 (로컬 실측 · 4구성)

`{ID포함, ID제외} × {LinearRegression, LGBMRegressor(default)}`. valid/test = 정답(`valid_Y_answer`/`test_Y_answer`) 대비 RMSE.

| 범위 | 모델 | 피처 | train_insample | train_cv5 | valid | test | honest R²(test) |
|---|---|---|---|---|---|---|---|
| ID포함 | linear | 63 | 81.5 | 81.7 | 1545.6 | 1542.2 | 붕괴 |
| ID포함 | lgbm | 63 | 47.8 | 58.2 | 101.0 | 92.8 | 0.874 |
| ID제외 | linear | 57 | 86.7 | 86.9 | 434.4 | 433.0 | 붕괴 |
| **ID제외** | **lgbm** | 57 | 49.0 | 60.0 | **87.3** | **85.0** | **0.895** |

*honest R² = 1 − (RMSE/σ)², σ(C65)=261.7.*

## 2. before → after (동일 채점축 = same-era 정답 채점)

| 단계 | 모델 | valid | test | 비고 |
|---|---|---|---|---|
| **before** | 바닐라 ID제외 LGBM | 87.3 | **85.0** | 무처리·전체피처·디폴트 |
| after (실무 라인) | v8 코어10 ExtraTrees | 35.66 | 35.64 | 시간/레짐/PM 백본 10피처 (기존 REPORT) |
| after (대회 최고) | v10 Lot 타깃인코딩 | 34.29 | 34.33 | same-era 전용(신규 Lot 붕괴) |

**개선폭**: test 85.0 → 34~36 = **약 50pt**. same-era 축에서 프로젝트 작업이 만든 실질 이득의 크기.

> ⚠️ **축 주의**: 위 표는 전부 same-era(대회) 정답 채점으로 **동일 축**이라 직접 비교 가능. 반면 **최종 운영 모델 lean-85의 현업 수치(시간축 재학습 99.84 / R² 0.8545)는 다른 축**(미래 Lot 일반화)이므로 이 표와 직접 비교 금지.

> 📝 **정정(2026-07-19, REPORT_02 참조)**: 초판은 "same-era로 lean-85를 대비하려면 lot-CV 66.96을 쓸 것"이라 적었으나, 위 valid/test는 **홀드아웃 직접 채점**이고 lean-85 66.96은 **GroupKFold(C20) train-CV**라 축이 다르다. 정합 비교(GKF C20 축)는 baseline에 GKF 열을 추가해 산출했다 → **REPORT_02** 로 분리. 본 표(홀드아웃)의 v8/v10 대비는 그대로 유효.

## 3. 관찰

1. **바닐라도 LGBM은 test 85까지는 간다.** 트리 모델이 원값 집계만으로 잡는 하한. 여기서 프로젝트가 50pt를 더 좁혔다 = 개선의 대부분은 **모델이 아니라 피처·시간 설계**에서 나옴(바닐라도 같은 LGBM 계열).
2. **Linear는 붕괴(valid 434~1546).** 원값 범주(Lot 코드·C40 시각 코드)를 정수 label로 넣으면 선형에서 미관측 코드에 발산. → "무처리+전체피처+선형"이 성립 불가임을 시연. 실질 before 기준선은 트리(LGBM)로 잡는 게 타당.
3. **단순 ID label-encoding은 누수를 만들지 않는다(가설 정정).** ID포함 LGBM(test 92.8)이 ID제외(85.0)보다 오히려 나쁨. valid Lot의 99.9%가 train에 존재해도, label 코드는 순서 의미가 없어 트리가 깔끔히 외우지 못하고 노이즈만 추가. 프로젝트가 실증한 Lot 누수는 **타깃 인코딩**(v9/v10 C20_te로 test 34까지 내려갔다가 신규 Lot에서 122로 붕괴)이라는 **더 강한 메커니즘**이었음을 역으로 확인. 단 랜덤 CV(58.2)만 ID포함이 미세 우위 → valid/test로 미전이 = 순진한 CV 낙관의 축소판.

## 4. 방법 (재현)

- **집계**: row→웨이퍼(C64), 수치=평균·범주=첫값. 그 외 도메인 가공 없음.
- **피처**: 남는 전 컬럼. 범주형은 train 기준 label 인코딩(미관측=NaN), 끝에 수치 강제. ID제외는 `[C20·C21·C22·C34·C35·C38]` 6종 제거(63→57).
- **모델**: `LinearRegression()`(NaN=train 중앙값 대치, 스케일 없음) · `LGBMRegressor(random_state=42)`(NaN 네이티브). 튜닝 없음.
- **평가**: train_insample(참고) · train_cv5(랜덤 5-fold) · valid/test(정답 직접 채점).
- **데이터 처리**: test_X 중복 C64(C64.1)·빈 C65 제거. train 11,939 / valid·test 각 1,990 웨이퍼, 정답 100% 커버.

## 5. 한계·주의

1. `train_insample`(LGBM 47.8)은 과적합 지표일 뿐 성능 아님 — 인용 금지.
2. C40(시각)은 label 코드화돼 valid/test에선 대부분 미관측(NaN)→기여 미미.
3. Linear "붕괴" 수치는 방법 부적합의 산물 — before 기준선으로는 **ID제외 LGBM(85.0)** 을 쓸 것.
4. 이 문서는 **same-era 축 전용**. 시간축(미래 일반화) 대비는 lean-85 handoff 문서를 별도 참조.
5. lightgbm 버전차로 소수 2자리는 흔들릴 수 있음(LGBM은 4.6.0 기준, 결정론적). Linear는 sklearn 버전에 민감.

*(스냅샷 고정 — 수치·분석 변경 시 새 REPORT)*
