# baseline_vanilla — 바닐라 Baseline (before→after 스토리용)

> 제작 2026-07-19 · baseline 전 지표 로컬 확정(lightgbm 4.6.0) · 대비 대상 = **최종 운영 모델 lean-85**
> 목적: 도메인 처리 없이 **기본 ML + 전체 피처**로 "before" 출발선을 고정 → lean-85(after)와 대비.

## 실행
프로젝트 venv에서 이 폴더의 `baseline_vanilla.ipynb`를 셀 순서대로 실행. 끝나면 `baseline_vanilla_results.csv` 생성.
(경로는 자동 탐색 — 루트나 하위 어디서 열어도 `문제1(하)`/`문제1_하_answer`를 찾음.)

## 무엇이 '바닐라'인가
- row→웨이퍼(C64): 수치=평균, 범주=첫값. 그 외 가공(slope·PM·레짐·상호작용) 없음.
- 전체 피처: 명세 제외/집계 규칙 무시, 남는 컬럼 전부. 범주형은 label 인코딩.
- 모델: `LinearRegression()` · `LGBMRegressor(random_state=42)` — 디폴트, 튜닝 없음.

## baseline 결과 (4구성 · 로컬 확정 lightgbm 4.6.0)

`{ID포함, ID제외} × {linear, lgbm}`. **세 개의 다른 축**:

| 범위 | 모델 | 피처 | train_insample | cv5_random | cv5_gkf_c20 | valid | test |
|---|---|---|---|---|---|---|---|
| ID포함 | linear | 63 | 81.5 | 81.7 | 82.3 | 1545.6 | 1542.2 |
| ID포함 | lgbm | 63 | 47.8 | 58.2 | 73.6 | 101.0 | 92.8 |
| ID제외 | linear | 57 | 86.7 | 86.9 | 87.4 | 434.4 | 433.0 |
| **ID제외** | **lgbm** | 57 | 49.0 | 60.0 | **69.9** | **87.3** | **85.0** |

**바닐라 before 출발선 = ID제외 LGBM** (Linear는 §관찰3처럼 붕괴, 트리로 잡음).

## before → after : 바닐라 → lean-85 (85피처 XGB, 최종 운영 모델)

| 축 | 바닐라 (before) | **lean-85 (after)** | 개선 | 무엇을 재나 |
|---|---|---|---|---|
| **홀드아웃** (valid/test same-era 직접 채점) | test **85.0** | test **59.3**ᵖ (valid 60.8ᵖ) | **25.7pt** | 미지 웨이퍼(같은 시기) 실채점 |
| **GKF(C20)** train-CV | **69.9** | **66.96** | 2.9pt | 같은 시기 **새 Lot** 일반화 |
| **시간축** (미래·현업 인용) | — (측정 안 함) | 재학습 **99.84** / 무재학습 254.9 | — | 미래 Lot 운영 |

ᵖ lean-85 홀드아웃 = 프리뷰(xgb 3.2.0). 로컬(3.3.0) 확정 권장 — `handoff_lean85/score_same_era_lean85.py`.

- **홀드아웃 축**: 정제 피처(lean-85)가 lot-mate까지 활용해 바닐라를 25.7pt 앞섬.
- **GKF 축**: 새 Lot만 보면 이득이 2.9pt로 줄어듦 — same-era에서 센서/피처 이득은 원래 작다.
- **시간축**: baseline이 못 재는 축. lean-85의 **현업 인용 수치는 여기(99.84)** 이며, 재학습이 필수(무재학습 254.9)임을 보여줌. 위 홀드아웃·GKF(59~70)는 same-era 후향치라 현업 인용 금지.

## 관찰
1. **랜덤 KFold는 못 믿는다(누수 시연).** ID제외 LGBM `cv5_random 60.0 → cv5_gkf_c20 69.9` = **GKF로 막으니 9.85pt 상승**. ID포함은 58.2→73.6(15.5pt, Lot 코드로 더 외움). 같은 Lot 웨이퍼가 fold를 가로질러 lot-mate를 외운 것.
2. **같은 same-era라도 축따라 그림이 다르다.** 홀드아웃은 바닐라 85 vs lean-85 59(25.7pt), GKF는 69.9 vs 66.96(2.9pt). **어느 축을 인용하느냐로 스토리가 갈리니 반드시 축을 명시.**
3. **Linear는 붕괴(valid 434~1546).** 원값 범주(Lot·C40 코드)를 정수 label로 넣으면 선형에서 미관측 코드에 발산 → before 기준선은 트리(LGBM)로 잡을 것.
4. **단순 ID label-encoding은 누수 안 만듦.** ID포함 LGBM(test 92.8)이 ID제외(85.0)보다 나쁨 — 순서 의미 없는 정수 코드는 깔끔한 누수 대신 노이즈만 더함.

## 축 주의 (핵심)
- **lean-85 현업 인용 = 시간축 99.84 / R² 0.8545**(미래 일반화). 홀드아웃(59.3)·GKF(66.96)는 same-era 후향치 — 현업 근거로 인용 금지.
- `train_insample`(LGBM 47.8)은 과적합 지표일 뿐 성능 아님.

## 실험 로그
| 날짜 | 작업 | 결과 |
|---|---|---|
| 2026-07-19 | 노트북 제작 → 로컬 첫 실행 시 셀5 문자열 오류 | build_matrix dtype 판정 버그 |
| 2026-07-19 | 버전-무관 인코딩 패치 → 로컬 재실행 완료 | 4구성 확정, ID제외 LGBM test 85.0 |
| 2026-07-19 | GKF(C20) OOF 열 추가 → 로컬 확정 | 랜덤60→GKF69.9(누수 9.85pt), lean-85 GKF 66.96과 2.9pt |
| 2026-07-19 | lean-85 홀드아웃 산출(프리뷰) | valid 60.8 / test 59.3 → 바닐라 85 대비 25.7pt |

산출물: `baseline_vanilla.ipynb` · `baseline_vanilla_results.csv` · `REPORT/baseline_vanilla_REPORT_01_baseline.md` · `REPORT/baseline_vanilla_REPORT_02_GKF정합.md`
