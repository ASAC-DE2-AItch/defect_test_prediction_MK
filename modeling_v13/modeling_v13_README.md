# modeling_v13 — FDC 집계 풀 (웨이퍼 5-fold OOF 학습 데이터)

> **목적**: `COMBINED_FULL_MODEL`(v12, valid 48.76)을 baseline으로 **피처 다이어트**를 진행하기 위한
> **후보 피처 풀**을 웨이퍼 단위 · 5-fold OOF 형태로 정리한 데이터셋.
> 이 파일은 *모델*이 아니라 다이어트의 **원료(candidate pool)** 다.
> **타깃 C65 = Defect Test 비트 수 → 낮을수록 좋음.**

---

## 1. 산출물

| 파일 | 내용 | 크기 |
|---|---|---|
| ~~`data/v13_fdc_pool_wf_oof.csv`~~ (폐기) | 플레인 CSV — 부분덤프 훼손 전력, **정본 아님**(현재 폴더에 없음). 재현 시 `build_fdc_pool.py` | — |
| `data/v13_fdc_pool_wf_oof.csv.gz` **(정본)** | 동일 데이터 (gzip, `pd.read_csv` 그대로 읽힘) | ~6.0 MB |
| `data/v13_fdc_pool_wf_oof.parquet` **(정본)** | 동일 데이터 (parquet, 로드 빠름) | ~4.7 MB |
| `build_fdc_pool.py` | 위 파일을 원본 train에서 **재현**하는 스크립트 | — |

재현: `python build_fdc_pool.py [train_data.csv 경로] [출력 디렉터리]`
(기본값 = `../문제1(하)/train_data.csv` → `./data`)

---

## 2. 데이터 명세

- **행**: 11,939 (train 웨이퍼 = C64, 1 WF = 1 행)
- **열**: 659 = 키/메타 3 + 피처 655 + 타깃 1

| 컬럼 | 역할 | 비고 |
|---|---|---|
| `C64` | 웨이퍼 ID (키) | 피처 아님 |
| `fold_kf5` | 5-fold OOF 배정 (0~4) | `KFold(5, shuffle=True, random_state=42)`, C64 오름차순 기준 |
| `C20` | **Lot ID (CV 메타데이터)** | GroupKFold(C20) 정직-CV용. **절대 피처로 쓰지 말 것** (프로젝트 원칙) |
| `<피처 655개>` | FDC 집계 피처 | 아래 §3 |
| `C65` | 타깃 (Defect Test) | WF 내 상수 |

### 피처 명명 규칙
`{센서}_{통계}_step{N}` — 예: `C17_mean_step1`, `C31_max_step6`, `C11_std_step4`
+ `C41_max_step{N}` (Step 소요시간)

---

## 3. 풀 정의 (27센서 → 수치 26 × 5통계 × 5스텝 + C41)

- **센서 27종** = 표준 pm_feature **23센서** + **필수 복원 4센서(C11·C15·C16·C31)**
  - 문자열 `C6`(레시피)는 수치 집계에서 자연 제외 → **수치 26종**
- **통계 5종**: `mean, std, max, min, last`
  - `last` = C40(시각) 정렬 후 그룹의 **가장 최근 측정값** (원본 파이프라인과 동일)
- **Step(C7) 5종**: {1, 4, 5, 6, 7} → pivot 으로 펼침
- **피처 수**: 26 × 5 × 5 = 650 + `C41_max_step`×5 = **655**

### 필수 5센서 보장 (v13 요구사항)
| 센서 | 물리 의미 | 피처 수 |
|---|---|---|
| C17 | 온도 (제일 중요) | 25 |
| C11 | 플라즈마 | 25 |
| C31 | RF | 25 |
| C15 | 가스 | 25 |
| C16 | 가스 | 25 |

→ 5센서 모두 풀에 존재. 다이어트 단계에서 **"각 센서 ≥ 1 피처"** floor 제약을 이 컬럼들 위에 걸면 됨.

### 결측 구조 (정상 — 트리계열이 자체 처리)
- `mean`/`max`/`min`/`last` 컬럼 평균 결측 **13.0%** → 대부분 **step5**(전체 WF의 34.8%만 보유) 탓.
- `std` 컬럼 평균 결측 **38.6%** → step5 부재 + **단일행 (WF,Step) 그룹의 std=NaN**.
- 결측은 **의미 있는 신호**(그 WF가 해당 Step을 안 거침)이므로 임의 대치하지 않음.

---

## 4. 자체 검증 (클라우드 미러)

- ✅ **fold 재현성**: 동일 규칙 재배정 시 `fold_kf5` 완전 일치. fold 크기 [2388,2388,2388,2388,2387].
- ✅ **결측 구조 정합**: `C17_mean_step5` 결측 65.2% ↔ step5 보유 WF 34.8% (합 100%).
- ✅ **모델-ready**: LGBM 5-fold OOF 정상 학습·예측.

| 구성 | LGBM 5-fold OOF RMSE | 해석 |
|---|---|---|
| 센서풀 단독 (655) | **74.40** | 시간/레짐 신호 **부재** — 이 풀은 원료일 뿐 |
| 센서풀 + C33(레짐 프록시 1개) | **62.96** (Δ **−11.45**) | 레짐 1개만 얹어도 급락 → **풀 정상 반응 확인** |

> **주의**: 74.40 은 "나쁜 모델"이 아니라 **"레짐/시간 피처가 빠진 순수 센서 풀"의 당연한 값**이다.
> 프로젝트 검증상 **레짐 신호가 압도적**(레짐 제거 시 CV 259)이고 센서의 순기여는 +3.91pt 수준.
> 따라서 이 풀은 다이어트에서 baseline(COMBINED_FULL)의 **시간/레짐 7피처와 결합**해 쓰는 것을 전제로 한다.

---

## 5. 사용법

```python
import pandas as pd
df = pd.read_parquet("data/v13_fdc_pool_wf_oof.parquet")   # 또는 read_csv(...csv.gz)

meta  = ["C64", "fold_kf5", "C20"]
feats = [c for c in df.columns if c not in meta + ["C65"]]  # 655
y     = df["C65"]

# 5-fold OOF 예시
import numpy as np, lightgbm as lgb
oof = np.zeros(len(df))
for k in range(5):
    tr, va = df["fold_kf5"] != k, df["fold_kf5"] == k
    m = lgb.LGBMRegressor(...).fit(df.loc[tr, feats], y[tr])
    oof[va] = m.predict(df.loc[va, feats])

# 정직-CV(신규 Lot 견고성 확인)가 필요하면 KFold 대신 GroupKFold(groups=df["C20"])
```

---

## 6. 누적 실험 로그

| 날짜 | 마일스톤 | 내용 | 상태 |
|---|---|---|---|
| 2026-07-13 | **M0 — 풀 빌드** | 27센서 FDC 집계 풀(655피처) 웨이퍼 5-fold OOF CSV 생성·검증 | ✅ |
| 2026-07-13 | **M1 — 다이어트** | 4단계 제거(Variance→결측→Corr→VIF) + 필수 5센서 floor. **Conservative 141 · Balanced 126** 확정. Aggressive 제외. | ✅ |
| 2026-07-13 | **M2 — 성능 비교** | 두 프리셋 × core10 결합, 고정 LGBM(M8_PARAMS) 2 CV. **로컬 확정**: 정직-CV diet 결합 71.2 < core10 단독 78.2(센서 +7pt), KFold는 core10 40.4 우위(다피처 과적합). Cons·Bal 초박빙 → **미확정** | ✅ (로컬 실행·재튜닝 대기) |
| 2026-07-13 | **M3 — 2차 선별** | Cons/Bal × RFECV/GA 4조합 완료(GA=Colab). **GA>RFECV**(KFold: Cons-GA 50.61 최선, 피처 151→99). 정직-CV baseline 대비 개선은 Bal-GA만(−0.17). GA vs RFECV는 환경차로 GKF 절대비교 불가 → 후보 2개 동일환경 재평가 필요 | ✅ (재평가·재튜닝 대기) |
| 2026-07-14 | **Phase 0 — 정리·사전등록** | v2.0 판정규칙 README 전재(§10) · 데이터 정본(gz/parquet) 명기 · M4 노트북 의존성·floor 재확인 | ✅(문서) · 환경버전/M4 실행 대기 |
| 2026-07-14 | **M4 — 챔피언 확정전+기준선 동결** | Cons-GA(99)/Bal-GA(108) 동일환경 재평가. |ΔGKF|=0.094<0.3 → **R4 규칙2 → 챔피언 Conservative-GA(99)**. **B1=71.366 동결**. 두 GA셋 모두 B0(71.212) 미달(Colab 개선=환경오프셋). R²=0.926 | ✅ (REPORT_04) |
| 2026-07-14 | **M4.5 — Lot 효과 진단(천장)** | C65 Lot-ICC **0.987**(98.7% Lot간). 챔피언 잔차 Lot-ICC **0.784**(여지 큼) 그러나 잔차–센서 최대상관 **0.037**(접근 난망). RMSE 71.366 정확 재현. → −0.5 불확실·**M5 파라미터 1순위** | ✅ (REPORT_05) |
| 2026-07-14 | **M4 듀얼+보조 — 4모델 정직축 맞대결 (v2.2 CV)** | `final_compare_dual`(LGBM·ET) + `compare_seg_xgb`(SegLGBM·XGBoost) 로컬 확정. **§4.2 G4 → 셋 Conservative-GA(99)·B1_LGBM 71.366 동결**(R² 0.926). ET 정직축 열세(+1.0, lot-mate 암기 = v8 M7 답) · SegLGBM 무변화 · **XGBoost −0.59/−0.62 정직축 우위(잠정, R7 다중시드 대기)**. v2.2 stable 폴드맵 로컬=미러 Δ0.000(XGB만 버전 드리프트 +0.2). | ✅ (REPORT_06) |
| 2026-07-14 | **XGB 다중시드 검증 (R7 확정)** | Cons-GA(99) 다중시드 lot-CV(seed 1/2/3)+stable — **XGBoost 가 4파티션 전부 LGBM 우세**. 시드 Δ(LGBM−XGB) 평균 **+0.590**·최악 **+0.426** → §6.2 기준(평균≥0.5 ∧ 최악≥0) 통과 → **XGB M5 챔피언 후보 편입 근거 성립**. LGBM 미러=로컬 Δ0.000·XGB 버전드리프트 +0.01~0.26. 게이트 70.712는 M5 튜닝 대상(원값 70.75 미달). | ✅ (REPORT_07) · §6.2 승인 대기 |
| 2026-07-15 | **PLAN v2.4 — §6.2 스코프 개정 (사용자 승인)** | M5 챔피언 후보 = **{LGBM, XGBoost}** 확정(ET 제외 → F5′ 앙상블 예비 동결). 결정 이력 8 추가. 산출물명 `tuned_params_v13_{lgbm,xgb}.json`으로 정정. G5·앵커(70.712)·확정 셋 Cons-GA(99)·CV(stable)는 불변. | ✅ (PLAN v2.4) |
| 2026-07-15 | **M5 튜닝 노트북 제작·미러검증** | `modeling_v13_m5_tuning.ipynb`: 목적=stable GKF OOF RMSE, **LGBM 80 / XGBoost 60 trials** Optuna(early stop+MedianPruner) → Stage B 다중시드 스크리닝(고정 rounds 정직수치) → 트랙별 대표·`tuned_params_v13_{lgbm,xgb}.json`. 미러 QUICK 스모크 10셀 무오류(floor·n=99·누수 assert 통과, xgb 3.2.0). | ✅ 제작 완료 → **사용자 실행 대기** (Colab ~1.5–3h) |
| 2026-07-15 | **core10 ablation (진단·실행 완료)** | `modeling_v13_ablation_no_core10.ipynb` 로컬 실행. **core10 제거 시 GKF: LGBM 71.366→79.295(+7.93) · XGB 70.517→78.922(+8.41)**. no_core10(89 센서) > core10 단독(78.2) = **레짐/시간 지배·센서 상보 확증**. LGBM full99 71.366 정확재현 / XGB full99 70.517(동결 70.773 −0.256=버전드리프트→P0-3 필요 재확인). floor core10 없이도 유지. **게이트·확정 셋 불변**(R1). | ✅ 완료 (REPORT_08) |

---

## 7. 노트북 사용법

### `modeling_v13_feature_diet.ipynb` (M1)
4단계 다이어트. 입력 `data/v13_fdc_pool_wf_oof.csv.gz`.
프리셋 컷 기준(`PRESETS` dict)만 바꾸면 강도 조절. 산출:
`feature_diet_summary.csv`, `feature_diet_selected.json`(프리셋별 선택 목록·champion). 런타임 ~10초, CPU.

### `modeling_v13_perf_compare.ipynb` (M2)
diet 선택셋 ∪ core10 → 고정 LGBM으로 KFold OOF + GroupKFold(C20) 평가.
**전제 파일**: `../modeling_v8/v8_timeline_common.py`, `../문제1(하)/train_data.csv`, `../pm_log.json`,
같은 폴더 `feature_diet_selected.json`. 산출 `perf_compare_results.csv`. **실측 런타임 ~46분(로컬)**, **CPU 전용**.

### `modeling_v13_select_<preset>_<method>.ipynb` × 4 (M3)
Cons/Bal × RFECV/GA 2차 선별. 공통 로직 `v13_select_common.py` import.
**고정** core10+champion, **가변** diet−고정을 `GroupKFold(C20)` 목적으로 선별 → 최종 KFold+GKF OOF.
산출 `select_result_<preset>_<method>.json`. 4개 독립·병렬 실행 가능.
⚠️ **fit 수 많음** — 로컬 실측(M2 fit당 ~100s) 기준 RFECV ~1시간+, GA 수시간 위험. 설정 경량화/빠른 CPU 권장.

### `modeling_v13_m5_tuning.ipynb` (M5 · v2.4)
확정 셋 **Cons-GA(99)** 고정, 목적함수 = **stable GKF(C20) 5-fold OOF RMSE**. **LGBM 80 / XGBoost 60 trials** Optuna(early stopping + MedianPruner) → **Stage B 다중시드 스크리닝**(top-K를 고정 rounds=평균 best_iter로 seed 1/2/3 재적합 = 정직 수치) → 트랙별 대표 선발.
**전제 파일**: `v13_fdc_pool_wf_oof.csv.gz` · `core10_meta_wf.csv` · `feature_diet_selected.json` · `select_result_Conservative_GA.json` (노트북 폴더 또는 `data/`·`colab_GA/`).
**산출**: `tuned_params_v13_{lgbm,xgb}.json` · `m5_stageB_summary.json` · `m5_stageB_results.csv` · `m5_study_{lgbm,xgb}.csv`.
**런타임**: Colab ~1.5–3h / 로컬 ~3–6h (CPU). 빠른 점검은 상수 셀 `QUICK=True`(또는 env `M5_QUICK=1`). XGB Colab GPU는 `XGB_DEVICE="cuda"`(수치는 로컬 CPU 재확인=공식, R6).
**확인 포인트**: 튜닝 전 baseline LGBM stable GKF ≈ **71.366**(Δ<0.05) self-check → 로더·폴드맵 무결. **챔피언/게이트(G5 ≤70.712)는 강건이 회신 숫자로 판정** — 노트북 검산 블록은 참고용. ⚠️ 노트북은 모델 챔피언을 스스로 확정하지 않음.

---

## 8. 스냅샷 인덱스 (REPORT)

| # | 마일스톤 | 파일 | 요지 |
|---|---|---|---|
| 01 | M1 다이어트 | `REPORT/modeling_v13_REPORT_01_M1.md` | 4단계 결과·floor 검증·champion·상수 220개 분석 |
| 02 | M2 성능비교 | `REPORT/modeling_v13_REPORT_02_M2.md` | core10 결합 2 CV 결과·정직-CV 센서 기여·재튜닝 필요 |
| 03 | M3 2차선별 | `REPORT/modeling_v13_REPORT_03_M3.md` | RFECV vs GA × 2프리셋. GA 우위·환경차 주의·후보 2개 재평가 |
| 04 | M4 챔피언확정 | `REPORT/modeling_v13_REPORT_04_M4.md` | 동일환경 재평가·R4 판정(챔피언 Cons-GA 99)·B1 71.366 동결·GA가 B0 미달(환경오프셋 확증)·게이트 앵커 상충 |
| 05 | M4.5 천장진단 | `REPORT/modeling_v13_REPORT_05_M4.5.md` | C65 98.7% Lot간·잔차 ICC 0.784이나 센서 무상관(0.037)·−0.5 난망·M5 파라미터 1순위·자기완결 로더 전환 |
| 06 | M4 듀얼+보조 | `REPORT/modeling_v13_REPORT_06_M4_dual.md` | 2셋×4모델(LGBM·ET·SegLGBM·XGB) 정직축. 셋 Cons-GA(99) 확정·B1_LGBM 71.366 동결·ET 열세(lot-mate)·**XGB −0.6 우위**(잠정) |
| 07 | XGB 다중시드 | `REPORT/modeling_v13_REPORT_07_xgb_multiseed.md` | Cons-GA(99) seed1/2/3 lot-CV. XGB가 4파티션 전부 LGBM 우세(평균 Δ+0.590·최악 +0.426) → §6.2 편입기준 통과(v2.4 근거) |
| 08 | core10 ablation | `REPORT/modeling_v13_REPORT_08_ablation_core10.md` | core10 제거 시 GKF +7.9~8.4 악화·센서만(89)은 core10 단독(78.2)보다 나쁨=레짐 지배 확증·XGB 버전드리프트 재확인. 진단(게이트 불변) |

---

## 9. 다음 단계 (v2.0 로드맵 — 정본 `modeling_v13_PLAN.md` §2)

**[지금 · v2.2] M4 듀얼(LGBM·ET) + SegLGBM·XGBoost 노트북 제작·미러검증 완료 → 로컬 실행 대기.** 미러 프리뷰(로컬 확정 전, 잠정): 셋=Cons-GA(99)·B1_LGBM 71.366 재확인, ET 정직축 열세(lot-mate 암기), **XGBoost 정직축 −0.8 주목**(다중시드 확인 대기). 정본 로드맵 = `modeling_v13_PLAN.md` v2.2 §2. (아래 1~5는 v2.0 골격 — M8/P7에서 현행화)

1. **M4 (Phase 1)** — `modeling_v13_final_compare.ipynb` 로컬 Restart & Run All → Cons-GA(99)/Bal-GA(108) **동일환경** KFold+GKF → **§10.3 규칙**으로 프리셋 확정 + **B1 동결** → `REPORT_04_M4`.
2. **M4.5 진단** — C65 Lot 간/내 분산분해 + OOF 잔차 lot-ICC → 정직축 천장 추정(사다리 기대치 캘리브레이션).
3. **M5 재튜닝** — 챔피언셋 고정, Optuna 목적 = **GroupKFold(C20)** → `tuned_params_v13.json` (게이트 G5: GKF ≤ B1−0.5).
4. **M6 현업 검증 배터리 B1~B5** — 다중시드 lot-CV·시간분할·레짐분해·물리근거·운영전제 (게이트 **G6′ = 본체**).
5. (미달 시) **M7 사다리** F1′→F5′ → **M8** 확정·문서화·핸드오프.

> 구 §9의 valid/test 대조는 대회축 폐기(PLAN 결정 이력 ③)로 **M8 부록 R 옵션 1회**로 강등. 의사결정 축 = GroupKFold(C20) 정직-CV 단일.

---

## 10. 판정 규칙 v2.3 (사전등록 · Phase 0 P0-2)

> **정본은 항상 `modeling_v13_PLAN.md`(v2.3)**. 이 절은 매 세션 참조할 판정 규칙을 README에 전재(사전등록)한 것. 결과를 본 뒤 규칙을 바꾸지 않는다(개정 = 사용자 승인 + PLAN v2.x). **v2.1**: LGBM·ET 듀얼 → 모델 챔피언(§6.2, PLAN). **v2.2**: 정직축 GKF = 동결 stable 폴드맵(§10.7). **v2.3**: 게이트 앵커 = **B − 0.5 = 70.712**(§10.2).

### 10.1 평가 축 (의사결정 언어)
| 축 | 정의 | 역할 |
|---|---|---|
| **① 현업축 (主)** | GroupKFold(C20) 5-fold OOF — 학습에 없던 Lot 예측 | 모든 선택·튜닝·게이트의 **유일** 기준 |
| **①′ 보조축** | (a) 다중시드 lot-CV: unique C20을 `KFold(5,shuffle,seed∈{1,2,3})`로 묶은 fold (b) 시간분할: C40 전기→후기(Lot 단위 컷) | 채택 확인 (B1·B2) |
| ② 참고축 | KFold(`fold_kf5`) · (옵션) valid/test | **기록만 — 의사결정 금지** |

**홀드아웃 규율**: 시간분할 후기 구간 조회 = 프로젝트 통산 **2회**(M6 1회 + M8 1회). valid/test는 M8 부록 R 옵션 1회. 그 외 조회 금지.

**CV 구현 (v2.2)**: GKF 는 동결 `stable_group_kfold`(argsort `kind='stable'`) — 환경독립, 로컬 동결값 정확 재현. 상세 §10.7.

### 10.2 기준선 B (동결)
| 후보 | GKF(로컬) | 상태 |
|---|---|---|
| B0 = diet∪core10 (Conservative 151) | **71.212** | ✅ 로컬 확정 (`perf_compare_results.csv`) |
| B1 = M4 챔피언 (Conservative-GA 99) | **71.366** | ✅ 동결 (`final_compare_results.json`) |

**공식 기준선 B = min(B0, B1) = 71.212** (B1=71.366 > B0 → **B = B0**). **목표 GKF ≤ B − 0.5 = 70.712 (v2.3 확정 · 사용자 승인 07-14)**. §0.3 축약문구 'B1−0.5'=70.866 은 **폐기**. 조건부 = 70.712~71.012 (B−0.5~B−0.2).
**가드레일**: 정직축 R² = 1 − (GKF/261.7)² ≥ 0.9 ⇔ RMSE ≤ 82.7 (현재 결합 ≈ 0.926 — 사실상 항상 통과).
> B0·B1 은 v2.2 stable 폴드맵으로 **동일 수치 재현**(환경독립). 게이트 비교도 stable 폴드맵 기준.

### 10.3 M4 판정 규칙 (사전등록 — 결과 보기 전 고정)
후보: **Conservative-GA(99) vs Balanced-GA(108)**, 한 노트북 = 한 환경 재평가.
```
1) |ΔGKF| ≥ 0.3 → GKF 낮은 쪽 승
2) |ΔGKF| < 0.3 → 피처 적은 쪽 승 (= Conservative-GA 99, 운영·해석 단순성 우선)
   ※ KFold 는 tie-break 에서 제외 (대회축 배제 일관성)
```
> ⚠️ 노트북(`modeling_v13_final_compare.ipynb`)이 출력·저장하는 `winner_by_groupkfold`는 **GKF 최소값 단독**이라 위 규칙과 어긋날 수 있다(격차<0.3이면 규칙은 슬림 쪽). **공식 판정은 저장된 4개 숫자에 위 규칙을 적용해 별도로 내린다.**
> Colab 예비치(탐색·상대비교 전용, R6): Cons-GA GKF 70.355 / Bal-GA GKF 70.262 (Colab, Δ≈0.093). 로컬 재측정 전 순위 미확정.
> **M4 로컬 확정(2026-07-14)**: Cons-GA **71.366** / Bal-GA 71.272 (Δ=0.094<0.3) → 규칙2 → **챔피언 Conservative-GA(99)**. 노트북 auto-winner('Balanced-GA', GKF-min)는 **무효**. 두 셋 모두 B0(71.212) 미달 → Colab '개선'은 환경 오프셋(R6).

### 10.4 게이트 요약
| 게이트 | 위치 | 조건 |
|---|---|---|
| G4 | M4 | §10.3 규칙으로 승자 유일 결정 + B1 동결 |
| G5 | M5 | 로컬 GKF ≤ B − 0.5 = 70.712 (R² ≥ 0.9 가드레일) |
| **G6′** | M6 | 다중시드 lot-CV 평균개선 ≥0.5 ∧ 최악시드 ≥0 ∧ 시간분할 우위 ∧ R²≥0.9 ∧ B3~B5 완비 |
| G7 | M7 | B1 프로토콜 평균개선 ≥0.5pt (2연속 무개선 중단) |
| G8 | M8 | 시간분할 최종 1회 + 문서 3종 + 운영 가이드 |

### 10.5 데이터 정본 (P0-1)
- **정본 = `data/v13_fdc_pool_wf_oof.csv.gz` / `.parquet`** (둘 다 11,939 × 659). 플레인 `.csv`는 부분덤프 훼손 전력 → 정본 아님(현재 폴더에 없음). 재현 필요 시 `python build_fdc_pool.py`.

### 10.6 로컬 환경 버전 (P0-3 · 기록 환경 = 로컬, R6)
> 기록 수치는 로컬 venv 산만 인정. 아래 표는 사용자 회신으로 동결.
> **v2.2: numpy·scikit-learn·lightgbm 버전 핀 필수** — GKF stable 폴드맵 재현·모델 결정성의 근거. requirements.txt 가 무핀이라 회신값을 명시 핀(무핀 시 업그레이드로 B1 조용히 붕괴 위험).

| 패키지 | 버전 |
|---|---|
| python | ⬜ |
| lightgbm | ⬜ |
| **xgboost** | ⬜ **(v2.4 필수 — XGB 정직축 절대값 버전 의존, 미러 3.2.0 대비 로컬 드리프트 +0.01~0.26 확인)** |
| scikit-learn | ⬜ |
| numpy | ⬜ |
| pandas | ⬜ |
| optuna | ⬜ |

캡처 커맨드(venv 활성 후): `python --version` 그리고 `python -m pip freeze`
> 미러(클라우드 검증) 환경 참고: python 3.10 · lightgbm 4.6.0 · xgboost 3.2.0 · scikit-learn 1.7.2 · optuna 4.9.0. **로컬 venv 값으로 이 표를 동결**(R6) — 특히 xgboost 는 로컬 실행 버전을 반드시 기입.

### 10.7 환경독립 CV (v2.2 — 사용자 승인 07-14)
정직축 GKF 는 **동결 `stable_group_kfold`** 를 쓴다(표준 `sklearn GroupKFold` 대체).

- **문제**: 표준 GroupKFold 의 lot→fold 배정이 **numpy `argsort` 기본정렬의 버전별 타이브레이크**에 의존 → 절대값이 환경마다 다름(미러 검증: 로컬 Cons 71.366/Bal 71.272 ↔ 컨테이너 71.510/71.629, **원순위 뒤집힘**). KFold(`fold_kf5`)는 고정 컬럼이라 정확 일치.
- **해소**: `np.argsort(cnt, kind='stable')[::-1]` 로 그룹을 크기 내림차순·안정 타이브레이크 배정 → 로컬 동결값을 **어느 환경에서도 Δ0.000 재현**. **B0/B1 수치 불변**(리베이스 없음).
- **적용**: M4~M8·다중시드 lot-CV·시간분할 전역. 헬퍼는 M4 노트북(`modeling_v13_final_compare_dual.ipynb`)에 인라인. 의존: numpy·sklearn·lightgbm 핀(§10.6, P0-3).

---

## 부록 A. C11·C15·C16·C31 이 표준 550풀에서 제외됐던 이유 (그리고 v13 복원 근거)

### A.1 제외 경위 (pm_feature `config.py`)
표준 풀은 FDC 후보 30센서에서 **다중공선성(상관 0.8+) 그룹**을 대표만 남기고 축소한 결과다:

```
MULTICOLLINEAR_GROUPS = {
  group1: [C1, C9, C17, C63],          # → 대표 C17, C63 유지 / C1, C9 제거
  group2: [C4, C5],                    # → 대표 C4 유지 / C5 제거
  group3: [C11, C15, C16, C31, C62],   # → 대표 C62 만 유지 / C11·C15·C16·C31 제거
  group4: [C46, C48],                  # → 다중공선성 아님, 둘 다 유지
}
COLS_REMOVE_MULTICOLLINEAR = [C1, C9, C5, C11, C15, C16, C31]   # 총 7종 제거
FDC_FEATURES = 30 − 7 = 23센서
```

즉 **C11·C15·C16·C31 은 group3(서로 상관 0.6~0.9)에서 대표 C62 하나로 대체**되며 제거됐다.
`config.py` 주석에는 특히 **"C16 중요도 0 확인 후 제거"** 로, LightGBM gain 기준 개별 기여가
낮았다는 점도 병기돼 있다. 목적은 *상관 높은 중복 축소 + 트리 분기 노이즈 감소*였다.

### A.2 문제점 — 물리 의미 소실
group3 은 공정상 **서로 다른 물리량**이다:

| 센서 | 물리 의미 |
|---|---|
| C11 | 플라즈마 |
| C31 | RF |
| C15 · C16 | 가스 |
| C62 | (group3 대표로만 잔존) |

상관이 높다고 대표 1개로 접으면, **PM 전후로 역할이 갈리는**(config 주석: "PM 전후 역할 다름)
개별 센서의 신호와 **원인 진단 가능성**이 사라진다. 실제로 프로젝트 실무 방향성 결정
(`real_model_direction.md`)에서 "센서가 실질 근거가 되어야 신규 Lot에 견고" 라는 요건이 확정됐고,
온도(C17)만 남은 표준 풀은 이 요건을 부분적으로만 충족한다.

### A.3 v13 결정 — 복원(단, group1/2 는 표준 유지)
- **복원**: `C11(플라즈마) · C31(RF) · C15·C16(가스)` — 물리적으로 구별되는 핵심 공정 신호.
- **표준 유지**: group1의 C1·C9, group2의 C5 는 이번 요구사항 밖이라 **표준대로 제외 유지**
  (필요 시 "29센서 완전 해제" 옵션으로 확장 가능 — 본 세션에서 27센서 선택).
- **결과**: 23 → **27센서**. 다중공선성은 트리계열에서 예측 성능을 크게 해치지 않으며(중복 분기일 뿐),
  다이어트 단계의 중요도 기반 선별 + 필수 floor 제약으로 최종셋을 정리한다.

> 요약: 표준 풀의 C11·C15·C16·C31 제외는 **다중공선성 축소(대표 C62 유지)** 목적이었고
> "C16 gain≈0" 도 근거였다. v13은 **물리적 의미·신규 Lot 견고성**을 위해 이 4센서를 되살리되,
> 통계적 중복은 다이어트에서 관리한다.
