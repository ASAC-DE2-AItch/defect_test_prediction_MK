# modeling_v13 REPORT 06 — M4 듀얼(LGBM·ET) + 보조(SegLGBM·XGBoost) 정직축 맞대결

> **스냅샷 (고정)** · 작성 2026-07-14 · 노트북 `modeling_v13_final_compare_dual.ipynb` + `modeling_v13_compare_seg_xgb.ipynb` (로컬 venv 실행)
> 결과 출처: `final_compare_dual_results.{json,csv}` · `compare_seg_xgb_results.{json,csv}` (로컬). 이후 마일스톤이 나와도 **수정하지 않는다.**
> **CV: v2.2 동결 `stable_group_kfold`(환경독립).** REPORT_04(v2.0 LGBM-only)는 보존 — 본 06이 듀얼/멀티모델 정본.

---

## 0. 한 줄 요약

2셋(Cons-GA 99 / Bal-GA 108) × **4모델**(LGBM·ET·SegLGBM·XGBoost) 정직축(GKF C20) 맞대결. 사전등록 **§4.2 규칙3 → 확정 셋 = Conservative-GA(99)**, **B1_LGBM = 71.366 동결**(R² 0.9256). 최대 발견: **XGBoost가 정직축에서 LGBM을 양쪽 −0.6 이기고(KFold는 열세 = 반-암기)**, ET는 정직축 열세(v8 M7 답), SegLGBM은 무변화. **단 XGBoost 우위는 잠정(R7 — 단일 GKF, 다중시드 미확인).**

---

## 1. 결과 (로컬 · v2.2 stable 폴드맵 · 2셋 × 4모델)

| 모델 | Cons-GA(99) GKF ① | Bal-GA(108) GKF ① | KFold ②(Cons/Bal) | Δ vs B1_LGBM (Cons/Bal) | 성격 |
|---|---|---|---|---|---|
| **LGBM (기준선)** | **71.366** | **71.272** | 50.677 / 51.412 | — (기준) | 공식 B1 |
| ET (순수 median) | 72.415 | 72.245 | 48.481 / 48.857 | +1.049 / +0.973 | KFold 최우·GKF 최악 = **lot-mate 암기** |
| SegLGBM(레짐 세그) | 71.289 | 71.527 | 50.707 / 51.193 | −0.077 / +0.255 | ≈ 무변화 |
| **XGBoost** | **70.773** | **70.651** | 52.259 / 52.568 | **−0.593 / −0.621** | KFold 열세·GKF 최우 = **신규-Lot 견고** |

- ① 현업축(GroupKFold C20) = 의사결정 축. ② KFold = 기록만(R1).
- 정직축 R²: LGBM 0.9256 · ET 0.9234 · SegLGBM 0.9258 · **XGBoost 0.9269**(Cons 기준) — 4모델 전부 ≥ 0.9 [가드레일 통과].
- B0 = diet∪core10 Conservative151 = 71.212 (M2). B1_LGBM 71.366 > B0 → 공식 기준선 B = min = 71.212.

---

## 2. 정합 확인 ✅

- **LGBM 재현 게이트 [통과]**: Cons 71.366 / Bal 71.272 = 동결값 **정확 재현(Δ0.000)** → 로더·stable 폴드맵·M8_PARAMS·타깃 전부 무결.
- **미러 ↔ 로컬**: LGBM·ET·SegLGBM **12값 전부 Δ0.000 일치**(stable 폴드맵 환경독립 확증). **XGBoost만** 로컬이 미러보다 **+0.15~0.26 드리프트**(로컬 xgboost 버전 ≠ 컨테이너 3.2.0 — R6/P0-3). **결론(XGB > LGBM ≥0.3·KFold 열세)은 불변.**
- `json ↔ csv` 수치 일치. `floor_ok` 4모델 × 2셋 **전부 True**(R10 — 필수 5센서 유지).
- ET median 손실 플래그(양쪽 GKF > B0+2=73.212) = **False** → ET 조기종료 미발동, 규칙상 M5 진출 유지.

---

## 3. 판정 — 사전등록 §4.2 / R4 (결과 보기 전 고정)

**G4 [셋 확정 · 통과]**
```
1) 모델별 GKF 셋간 비교 (|Δ|≥0.3 유의):
   LGBM |71.366−71.272|=0.094<0.3 → tie   ·   ET |72.415−72.245|=0.170<0.3 → tie
   (보조 4모델도 전부 셋간 |Δ|<0.3: SegLGBM 0.238 · XGBoost 0.122)
2) 두 모델 합의? 양쪽 tie → 아니오
3) 양쪽 tie → 두 모델 GKF 평균 낮은 셋: Cons (71.366+72.415)/2=71.891 · Bal 71.759
   평균차 0.132<0.3 → 피처 적은 쪽 = Conservative-GA(99)
   (보조 근거: §1.3 안정코어 — Cons-GA RFECV 교차생존 98%)
→ 확정 셋 = Conservative-GA (99)
```
- **B1_LGBM = 71.366 동결** (공식 기준선) · **B1_ET = 72.415 기록** (도전자, 기준선 아님 — 순환 방지 §0.3).
- **가드레일 [통과]**: 정직축 R² = 1 − (71.366/261.7)² = **0.9256 ≥ 0.9**.
- **[모델 트랙]** M4에서 모델 미확정 — LGBM·ET 규칙상 M5 진출. SegLGBM·XGBoost 는 도전자 기록(사용자 요청 추가).

---

## 4. 정직한 발견 (전략 함의)

1. **XGBoost 정직축 우위 [주목·잠정]** — 양쪽 −0.593/−0.621(≥0.3 유의) + KFold 열세(52.3/52.6 > LGBM 50.7/51.4). KFold 열세·GKF 우위 = ET와 정반대 = **lot-mate 과적합이 덜해 신규 Lot에 견고**. 프로젝트 통틀어 정직축 −0.5 목표를 넘긴 첫 신호. **그러나 채택 아님(R7)**: ① 단일 GKF 5-fold(다중시드 lot-CV 미확인) ② XGB 미튜닝 — 이 우위엔 **정규화 효과**가 섞였을 수 있음(M8_PARAMS num_leaves=175 공격적 → LGBM이 lot-mate 과적합). → **M5에서 LGBM·XGBoost를 GKF 목적으로 동시 튜닝**해 공정 재대결 필요.
2. **ET = 정직축 최약체 (v8 M7 미결의 답)** — KFold 최우(48.5/48.9)인데 GKF 최악. KFold 우위는 lot-mate 암기이며 R1대로 **채택 근거 불인정**. ET는 F5′ 앙상블 재료로만 동결 보존.
3. **SegLGBM 무이득** — `is_high_regime`가 이미 피처라 레짐 세그로 얻는 정직축 이득 없음(−0.08/+0.26). 챔피언 후보 아님.
4. **miss_frac 재확인** — Cons-GA step5 결측 65% vs Bal-GA 0%인데 **ET는 양쪽 다 열세** → ET 열세는 median 대치 손실이 아님. F3′ missing-indicator 기대이득 낮음.
5. **정직-CV축 정당성 자기검증** — KFold로 골랐다면 최악의 정직축 모델(ET)을 뽑고 최선(XGBoost)을 버렸을 것. 축 선택이 결과를 갈랐다.

---

## 5. 게이트 앵커 (여전히 사용자 승인 대기)

B1_LGBM 71.366 > B0 71.212 → PLAN 두 조항 상충:
- §0.1 정의: 목표 = **B − 0.5 = 70.712** (B = min(B0,B1))
- §0.3 문구: 목표 = **B1_LGBM − 0.5 = 70.866**

권고 = **70.712**(정의 일치·더 엄격). **이제 학술적이지 않다**: XGBoost Cons **70.773** 은 70.866 아래(§0.3 통과)·70.712 위(§0.1 미달) — **앵커 결정이 XGB(확정 시)의 G5 통과 여부를 직접 가른다.** (단 G5는 M5 튜닝 챔피언 대상 — 원 XGB는 참고.)

---

## 6. 산출물 & 다음

- 산출: `final_compare_dual_results.{json,csv}` · `compare_seg_xgb_results.{json,csv}` (로컬).
- 확정 셋: fixed 15(core10+champion) + Cons-GA prunable 84 = **99**. floor {C17:4, C11:5, C31:3, C15:3, C16:2}.
- **다음**: (1) **XGBoost 다중시드 lot-CV**(seed 1/2/3) — R7 잠정→확정. 3시드 평균도 LGBM −0.5 이상이면 §6.2 챔피언 후보 편입(승인 안건). (2) **M4.5 진단**(확정 셋에 Null Importance 80셔플). (3) **M5 재튜닝**을 **LGBM·XGBoost(±ET) GKF 동시 튜닝**으로 확장 검토(§6.2 개정 안건). (4) 게이트 앵커 확정. (5) P0-3 버전 핀에 **xgboost 추가**.


---

## 부록 A. 확정·후보 피처셋 물리 명세 (사후 추가 — 2026-07-16 · 동결 예외 승인)

> ⚠️ **동결 예외 로그**: 이 부록은 원본 M4 스냅샷(§0~6, 2026-07-14) **이후** 사용자 승인(2026-07-16) 하에 추가됐다. CLAUDE.md/R8 REPORT 동결 원칙의 **명시적 1회 예외**이며, **원 스냅샷의 결론(셋 확정·B1 71.366 동결·XGB 잠정우위)은 일절 수정하지 않았다.** 피처↔물리 매핑 출처 = 업로드 **데이터 사전 v3**(train_data 컬럼 최종 *유추본* — 등급 `?`=단서 강함/`??`=구조만 확정, 물리명 미확정). 셋 구성 출처 = `select_result_{Conservative,Balanced}_GA.json` + v8 `CORE10` + champion. 재구성 검증: Cons **99**·Bal **108** 정확 일치.

### A.0 세 셋의 지위 (혼동 방지)

| 셋 | n | 역할 | 로컬 GKF (공식) |
|---|---|---|---|
| B0 = diet∪core10 (Conservative) | 151 | **기준선(자)** | 71.212 |
| **Conservative-GA** | **99** | **확정 작업셋 → M5** | **B1 71.366 (동결)** |
| Balanced-GA | 108 | 준우승 (F3′ 예비 동결) | — (Colab 70.262) |

> 주의: `select_result`의 GKF(70.355/70.262)는 **Colab 산 상대비교 전용(R6)**. 로컬 공식 기준선은 **B1_LGBM 71.366**. GA '개선'은 환경 오프셋(REPORT_04 §3).

### A.1 공유 백본 — fixed 15 (두 셋 100% 동일)

GA 이전에 **항상 포함**되는 고정 15피처. floor(필수 5센서≥1)와 시간/레짐 백본을 자동 보장.

| 블록 | 피처 | 물리 의미 |
|---|---|---|
| 시간·레짐 (7) | `is_high_regime`, `high_regime_days`, `days_since_last_pm`, `hour`, `dslp_x_hour`, `hour_x_c33`, `is_special_recipe` | PM 레짐·경과일·시각 교호 (엔지니어링 신호 = 정직축 지배 백본) |
| PM 카운터 (1) | `C33` | PM 경과 카운터(리셋=PM 이벤트) |
| 멀티플렉스 (2) | `C59_mean_step4`, `C60_mean_step4` | 동일물리량 2채널 교대판독 ⚠️**물리량 미확정**(사전 우선순위③) |
| champion 5센서 (5) | `C17_max_step4`, `C11_min_step4`, `C31_mean_step4`, `C15_max_step1`, `C16_max_step1` | 온도·Vdc·RF출력·가스A·가스B = **floor 물리 앵커** |

### A.2 확정 셋 — Conservative-GA (99)

**Conservative-GA (n=99)** — GKF 70.355 / KFold 50.614 *(Colab 산, R6 상대비교 전용)* · floor {C17:4, C11:5, C31:3, C15:3, C16:2} · floor_ok ✅

| 카테고리 | 피처수 |
|---|---|
| 시간·레짐·PM(비센서 집계) | 8 |
| RF·플라즈마 | 49 |
| 가스유량 | 19 |
| 온도 | 9 |
| 장비상태·카운터 | 12 |
| 공정·시간(C46) | 2 |
| **합계** | **99** |

| 센서 | 수 | 물리 의미 (사전 v3, 등급 ?/??=유추 신뢰도) | 피처(통계_스텝) |
|---|---|---|---|
| C25 | 10 | 영점/베이스라인 드리프트(노후도 보조, 달력시간 corr −0.30) | last_step1, last_step4, last_step6, last_step7, max_step6, mean_step4, mean_step5, min_step4, std_step1, std_step4 |
| C18 | 8 | RF 매칭 과도 편차 | last_step1, last_step4, max_step4, mean_step1, min_step4, min_step7, std_step6, std_step7 |
| C62 | 7 | RF 전극 전압(Vpp 계열) | last_step4, max_step4, max_step6, max_step7, mean_step1, mean_step4, std_step6 |
| C58 | 6 | He Backside 압력 후보(공정압력 기각) | last_step4, mean_step1, mean_step4, min_step7, std_step1, std_step7 |
| C59 | 6 | 동일물리량 2채널 교대판독(멀티플렉스) ⚠️물리량 미확정 | mean_step4, mean_step5, std_step1, std_step4, std_step6, std_step7 |
| C11 | 5 | DC Self-Bias 전압 Vdc ★필수(플라즈마) | last_step7, max_step6, mean_step5, min_step1, min_step4 |
| C52 | 5 | 보조 온도(독립 열원/주변부) | mean_step1, mean_step5, min_step4, std_step1, std_step6 |
| C17 | 4 | 히터/척 온도(°C) ★필수(온도) | max_step4, min_step1, std_step1, std_step6 |
| C56 | 4 | RF Match Load/Tune 축 절대위치 | last_step4, max_step4, min_step4, std_step4 |
| C60 | 4 | 동일물리량 2채널 교대판독(멀티플렉스) ⚠️물리량 미확정 | mean_step4, std_step4, std_step6, std_step7 |
| C15 | 3 | Gas Flow A 실측 ★필수(가스) | last_step4, max_step1, max_step4 |
| C31 | 3 | RF 출력 실측(Vdc와 −0.978 동행) ★필수(RF) | max_step4, mean_step4, mean_step5 |
| C48 | 3 | Main Gas Flow 설정값 | last_step4, max_step1, mean_step5 |
| C57 | 3 | C58 제어축(밸브/유량 설정, 6~72 정수) | last_step4, mean_step4, mean_step5 |
| C61 | 3 | RF 파형 음(−)측 피크전압 후보 | last_step1, last_step7, std_step7 |
| C63 | 3 | step별 이산 운전점+상승 드리프트 ⚠️물리명 미확정 | min_step4, min_step7, std_step6 |
| C4 | 2 | 가스 유량 Setpoint A (설정값) | mean_step5, std_step6 |
| C16 | 2 | Main Gas Flow 변동신호(실측/과도) ★필수(가스) | max_step1, mean_step4 |
| C27 | 2 | 매칭 잔류 편차 | last_step4, max_step4 |
| C46 | 2 | Step 내 측정 순번(카운터) | std_step1, std_step6 |
| C54 | 2 | RF Match Load/Tune 축 절대위치 | min_step4, std_step4 |
| C12 | 1 | Step 단위 갱신 Vdc 연동 기준값 | mean_step1 |
| C32 | 1 | RF Reflected Power(디지털) | last_step4 |
| C49 | 1 | 이산 4단계 음수 상태값(등차 −11.5) | mean_step5 |
| C50 | 1 | Step4 종료 이벤트 카운트(Endpoint 후보) | std_step4 |

### A.3 준우승 셋 — Balanced-GA (108)

**Balanced-GA (n=108)** — GKF 70.262 / KFold 51.228 *(Colab 산, R6 상대비교 전용)* · floor {C17:2, C11:9, C31:2, C15:3, C16:3} · floor_ok ✅

| 카테고리 | 피처수 |
|---|---|
| 시간·레짐·PM(비센서 집계) | 8 |
| RF·플라즈마 | 54 |
| 가스유량 | 25 |
| 온도 | 7 |
| 장비상태·카운터 | 12 |
| 공정·시간(C46) | 2 |
| **합계** | **108** |

| 센서 | 수 | 물리 의미 (사전 v3, 등급 ?/??=유추 신뢰도) | 피처(통계_스텝) |
|---|---|---|---|
| C25 | 12 | 영점/베이스라인 드리프트(노후도 보조, 달력시간 corr −0.30) | last_step1, last_step4, last_step6, last_step7, max_step1, max_step6, mean_step4, min_step4, min_step7, std_step1, std_step4, std_step7 |
| C11 | 9 | DC Self-Bias 전압 Vdc ★필수(플라즈마) | last_step1, last_step6, last_step7, max_step4, max_step6, max_step7, min_step4, min_step6, std_step1 |
| C58 | 9 | He Backside 압력 후보(공정압력 기각) | last_step1, last_step4, last_step6, last_step7, max_step6, mean_step4, min_step6, std_step1, std_step4 |
| C62 | 8 | RF 전극 전압(Vpp 계열) | last_step4, max_step4, max_step6, mean_step1, mean_step4, min_step6, std_step1, std_step7 |
| C18 | 7 | RF 매칭 과도 편차 | last_step4, max_step4, max_step6, min_step1, min_step4, min_step7, std_step7 |
| C52 | 5 | 보조 온도(독립 열원/주변부) | max_step7, min_step4, std_step1, std_step4, std_step7 |
| C57 | 5 | C58 제어축(밸브/유량 설정, 6~72 정수) | last_step1, last_step4, mean_step1, mean_step4, min_step4 |
| C60 | 5 | 동일물리량 2채널 교대판독(멀티플렉스) ⚠️물리량 미확정 | mean_step4, min_step1, std_step1, std_step4, std_step7 |
| C59 | 4 | 동일물리량 2채널 교대판독(멀티플렉스) ⚠️물리량 미확정 | mean_step4, std_step1, std_step4, std_step7 |
| C61 | 4 | RF 파형 음(−)측 피크전압 후보 | last_step1, last_step7, min_step6, std_step7 |
| C15 | 3 | Gas Flow A 실측 ★필수(가스) | last_step4, max_step1, max_step4 |
| C16 | 3 | Main Gas Flow 변동신호(실측/과도) ★필수(가스) | last_step4, max_step1, mean_step4 |
| C27 | 3 | 매칭 잔류 편차 | last_step4, max_step4, std_step4 |
| C48 | 3 | Main Gas Flow 설정값 | last_step4, max_step1, max_step6 |
| C56 | 3 | RF Match Load/Tune 축 절대위치 | max_step4, min_step4, std_step4 |
| C63 | 3 | step별 이산 운전점+상승 드리프트 ⚠️물리명 미확정 | max_step6, min_step4, std_step4 |
| C4 | 2 | 가스 유량 Setpoint A (설정값) | max_step1, mean_step1 |
| C17 | 2 | 히터/척 온도(°C) ★필수(온도) | max_step4, min_step1 |
| C31 | 2 | RF 출력 실측(Vdc와 −0.978 동행) ★필수(RF) | max_step4, mean_step4 |
| C46 | 2 | Step 내 측정 순번(카운터) | std_step1, std_step7 |
| C50 | 2 | Step4 종료 이벤트 카운트(Endpoint 후보) | std_step1, std_step4 |
| C54 | 2 | RF Match Load/Tune 축 절대위치 | min_step4, std_step4 |
| C32 | 1 | RF Reflected Power(디지털) | last_step4 |
| C49 | 1 | 이산 4단계 음수 상태값(등차 −11.5) | mean_step1 |

### A.4 두 셋 비교 (물리 관점)

- **공통 69** · Cons 전용 30 · Bal 전용 39.
- Bal(+9)은 주로 **RF(+5)·가스(+6)**에서 늘고 특히 **C11(Vdc) last/max 계열을 더 담음**(9 vs 5). §1.3대로 Bal 전용의 상당수가 RFECV가 쳐낸 불안정 신호 → 슬림·안정 우선 규칙이 Cons를 택한 방향과 일치.
- **floor 유지(R10)**: Cons {C17:4, C11:5, C31:3, C15:3, C16:2} / Bal {C17:2, C11:9, C31:2, C15:3, C16:3} — 둘 다 5센서 ≥1 ✅.

<details><summary>Cons 전용 30개 / Bal 전용 39개 (펼치기)</summary>

**Cons 전용**: `C11_mean_step5`, `C11_min_step1`, `C12_mean_step1`, `C17_std_step1`, `C17_std_step6`, `C18_last_step1`, `C18_mean_step1`, `C18_std_step6`, `C25_mean_step5`, `C31_mean_step5`, `C46_std_step6`, `C48_mean_step5`, `C49_mean_step5`, `C4_mean_step5`, `C4_std_step6`, `C52_mean_step1`, `C52_mean_step5`, `C52_std_step6`, `C56_last_step4`, `C57_mean_step5`, `C58_mean_step1`, `C58_min_step7`, `C58_std_step7`, `C59_mean_step5`, `C59_std_step6`, `C60_std_step6`, `C62_max_step7`, `C62_std_step6`, `C63_min_step7`, `C63_std_step6`

**Bal 전용**: `C11_last_step1`, `C11_last_step6`, `C11_max_step4`, `C11_max_step7`, `C11_min_step6`, `C11_std_step1`, `C16_last_step4`, `C18_max_step6`, `C18_min_step1`, `C25_max_step1`, `C25_min_step7`, `C25_std_step7`, `C27_std_step4`, `C46_std_step7`, `C48_max_step6`, `C49_mean_step1`, `C4_max_step1`, `C4_mean_step1`, `C50_std_step1`, `C52_max_step7`, `C52_std_step4`, `C52_std_step7`, `C57_last_step1`, `C57_mean_step1`, `C57_min_step4`, `C58_last_step1`, `C58_last_step6`, `C58_last_step7`, `C58_max_step6`, `C58_min_step6`, `C58_std_step4`, `C60_min_step1`, `C60_std_step1`, `C61_min_step6`, `C62_min_step6`, `C62_std_step1`, `C62_std_step7`, `C63_max_step6`, `C63_std_step4`

</details>

### A.5 물리 판독 (강건 주석)

1. **RF·플라즈마가 절반** (Cons 49/99 · Bal 54/108). 식각이 RF 구동 공정이라 물리적으로 정합.
2. **두 셋 공통 최대 센서 = C25(영점/베이스라인 드리프트 = 노후도 보조, 달력시간 corr −0.30)** — Cons 10 / Bal 12. GA가 예측력으로 **가장 많이 집어온 게 공정 물리(온도·가스·RF출력)가 아니라 장기 드리프트/노후 지표**다. 이는 **M4.5 천장진단(잔차 78% Lot단위·센서 무상관 0.037)과 정합** — 남은 신호축이 Lot/레짐/노후이고 GA도 그 프록시(C25·C18 매칭과도·C58 He압력·C59/60 멀티플렉스)로 몰렸다.
3. **필수 5센서는 floor 앵커로 소수 존재** (Cons 17/99 · Bal 19/108). 물리 앵커는 유지되지만 **예측 기여의 bulk는 드리프트/보조 신호**다 — '센서가 실질 근거'라는 요건은 앵커 수준에서만 충족.
4. **⚠️ 물리 근거 리스크(B4 선행 안건)**: 고정 백본의 `C59/C60`(멀티플렉스)와 GA 상위 `C25·C63`이 **물리명 미확정**(사전 유추본, 등급 ?/??). 현업 설득력=물리 근거인데 **상위 기여 피처의 물리 정체가 미상**이라는 게 정직한 약점. 사전 v3가 지정한 멘토 확인 우선순위(**① C31 단위 ② C63 정체 ③ C59/C60 물리량**)를 **M6 B4 배터리에서 정식 안건**으로 올릴 것.
5. **종합**: 피처셋은 '물리 앵커(필수 5센서) + 레짐/노후 프록시 다수' 구조. 이 구조 자체가 M4.5·M7 캘리브레이션(파라미터 1순위, F1′ 교호는 낮은 기대)과 일치한다.
