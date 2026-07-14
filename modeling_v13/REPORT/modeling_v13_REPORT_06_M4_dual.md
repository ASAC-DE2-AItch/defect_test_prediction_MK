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
