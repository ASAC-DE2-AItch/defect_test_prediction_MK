# modeling_v13 REPORT 03 — M3 2차 선별 (RFECV vs GA × Conservative vs Balanced)

> **스냅샷 (고정)** · 작성 2026-07-13 · 노트북 `modeling_v13_select_*` (RFECV 로컬 / GA Colab)
> M3 시점 기록. 이후 마일스톤이 나와도 **수정하지 않는다.**
> 결과 출처: `select_result_<preset>_<method>.json` 4개.

---

## 0. 한 줄 요약

diet 두 프리셋에 RFECV·GA 2차 선별을 얹어 4조합 비교.
**GA가 RFECV를 명확히 앞선다** — GA는 피처를 줄이면서 KFold를 개선(예: Conservative 51.82→50.61),
RFECV는 거의 못 줄이고 개선도 미미. floor(5센서≥1)는 4조합 전부 유지.
**환경차 주의**: GA=Colab / RFECV=로컬이라 **GroupKFold 절대값은 직접 비교 불가**(KFold는 비교 가능).

---

## 1. 셋업

- **대상 4조합**: {Conservative, Balanced} × {RFECV, GA}.
- **고정(항상 포함)**: core10(10) + 필수 5센서 champion → floor 보장·백본. **가변(prunable)**: diet − 고정.
- **선별 목적함수**: `GroupKFold(C20)` OOF(프록시 LGBM 200라운드). **최종 보고**: 선택셋을 `M8_PARAMS`·705로 KFold + GroupKFold.
- **실행 환경**: RFECV = 로컬(사용자 venv), GA = Colab(자기완결 `colab_GA/`, 각 ~20분·CPU).

---

## 2. 결과 (`select_result_*.json`)

| 조합 | baseline n | base KF | base GKF | **선택 n** | KFold OOF | GroupKFold(C20) | floor |
|---|---|---|---|---|---|---|---|
| Conservative + RFECV | 151 | 51.817 | 71.192 | 146 | 51.671 | 71.194 | ✅ |
| Conservative + GA | 151 | 51.817 | 70.297 | **99** | **50.614** | 70.355 | ✅ |
| Balanced + RFECV | 136 | 51.648 | 71.254 | 81 | 51.339 | 71.445 | ✅ |
| Balanced + GA | 136 | 51.648 | 70.434 | **108** | 51.228 | **70.262** | ✅ |

> baseline = 해당 노트북의 diet∪core10(2차선별 없음), 같은 환경 기준.

---

## 3. ⚠️ 환경차 (해석의 전제)

- **baseline GKF가 방법별로 다르다**: RFECV(로컬) ≈71.2, GA(Colab) ≈70.3~70.4. 같은 diet∪core10인데도
  **~0.9pt 차이** → M2에서 본 로컬 vs 클라우드 GroupKFold 오프셋과 동일 원인(환경/버전차로 group 분할·부동소수 미세차).
- **KFold는 환경 무관**: `fold_kf5`(고정 컬럼) 사용 → M2에서 core10 KFold가 양쪽 40.387로 일치했듯 **교차 비교 가능**.
- **결론**: 조합 간 절대 순위는 **KFold(환경 robust)** 로, 정직-CV 개선 여부는 **각 노트북 baseline 대비 delta** 로 본다.

---

## 4. 해석

### 4.1 GA > RFECV (방법 우열, 명확)
- **KFold(환경 robust) 순위**: Conservative-GA **50.614** < Balanced-GA 51.228 < Balanced-RFECV 51.339 < Conservative-RFECV 51.671. → **GA 두 개가 RFECV 두 개를 모두 앞섬.**
- GA는 baseline 대비 KFold를 크게 낮춤(Cons 51.82→50.61 **−1.2**, Bal 51.65→51.23 −0.42)**면서 피처도 축소**(Cons 151→99, Bal 136→108).
- RFECV는 거의 못 줄이고(146/81) 개선도 미미 — gain 기반 후진제거의 greedy 한계.

### 4.2 정직-CV(GroupKFold) — baseline 대비 delta
- Conservative RFECV +0.002 / **GA +0.058**(소폭 악화)
- Balanced RFECV +0.191(악화) / **GA −0.172**(개선) ← **4조합 중 유일하게 정직-CV 개선**
- 즉 정직-CV 기준으로 "제 baseline을 이긴" 건 **Balanced-GA 뿐.**

### 4.3 GA 선택셋 특성
- Conservative-GA 선택 prunable 84 → 최종 99피처 / Balanced-GA 93 → 108피처.
- 두 GA 선택 prunable **교집합 54**(Cons 단독 30 · Bal 단독 39) → 코어 신호는 겹치나 세부는 다름(확정성 완벽하진 않음).

---

## 5. 판정 & 한계

- **방법**: **GA 채택**(RFECV 대비 전 지표 우위·더 슬림).
- **프리셋**: 아직 미확정 — 두 축이 갈린다.
  - **KFold(환경 robust)**: Conservative-GA(50.61, 99피처)가 최선·최다 개선.
  - **정직-CV(baseline 대비)**: Balanced-GA만 개선(−0.172), 최종 108피처.
- **한계 1(환경)**: GA끼리·RFECV끼리는 같은 환경이나 **GA vs RFECV 는 GKF 절대 비교 불가**. 최종 프리셋 확정 전
  **Conservative-GA·Balanced-GA 두 후보를 같은 환경에서 재평가** 필요.
- **한계 2(파라미터)**: 여전히 고정 M8_PARAMS(10피처용). 절대 성능은 승자 셋 고정 후 재튜닝에서 확정.

---

## 6. 산출물

| 파일 | 내용 |
|---|---|
| `modeling_v13_select_{Conservative,Balanced}_{RFECV,GA}.ipynb` | 4개 선별 노트북 |
| `colab_GA/` | GA Colab 자기완결 폴더(모듈+데이터+노트북) |
| `select_result_<preset>_<method>.json` × 4 | 선택 피처 + baseline 대비 KFold/GKF |

---

## 7. 다음 단계

1. **후보 2개 동일 환경 재평가** — Conservative-GA(99) · Balanced-GA(108)를 한 환경(로컬 또는 Colab)에서 KFold+GKF 재측정해 GKF 절대 비교 성립시키기.
2. **승자 고정 → 파라미터 재튜닝** — 그 셋에서 `num_leaves`·`num_boost_round` 등 Optuna(GroupKFold 목적)로 절대 성능 확정.
3. (선택) **GA 다중 시드** — seed 42/7/123로 선택셋 안정성(교집합) 확인.
