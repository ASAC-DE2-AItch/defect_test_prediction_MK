# modeling_v8 — 안내서 & 실험 인덱스 (살아있는 문서)

> 이 문서는 modeling_v8의 **살아있는 인덱스**입니다 — 노트북 사용법 + **누적 실험 로그** + 각 마일스톤 **스냅샷 보고서로의 링크**. 코드·결과가 바뀌면 **이 문서만** 갱신하고, 마일스톤별 상세 분석은 고정 스냅샷 REPORT 파일에 있습니다.
> PLAN **v1.6**(verdict + 관찰 트랙) 반영. 현재 노트북 범위: Phase 3(피처) + M0a·M0b·M-T·M1·M2·M3·M4 + **M5a(모델 벤치)**.

## 📁 파일 구성 & 명명 규칙

```
modeling_v8/
├─ modeling_v8.ipynb              노트북 (피처 빌드 + M0a~M4 Cell 1~12 + M5a 벤치 Cell 13~15)
├─ outputs/                       M5a 산출물: model_bench.csv · model_bench_corr.csv · results_M5a.json
├─ modeling_v8_README.md          ← 이 파일: 살아있는 인덱스(사용법+누적 로그+인덱스)
├─ modeling_v8_PLAN.md            기획서 (PLAN v1.5)
├─ pkl_recovered_meta.json        원본 pkl 복원 메타(PARAMS·top-10·gain)
└─ REPORT/                        스냅샷 보고서 (마일스톤 배치별, 고정)
   ├─ modeling_v8_REPORT_01_M0.md    Phase 3 + M0a·M0b·M-T (G1)
   ├─ modeling_v8_REPORT_02_M1.md    M1 (+C23_te, 기각)
   ├─ modeling_v8_REPORT_03_M2.md    M2 (센서 풀 gain 재선별, 기각)
   ├─ modeling_v8_REPORT_04_M3.md    M3 (row-level 결합, 기각)
   ├─ modeling_v8_REPORT_05_M4.md    M4 (피처셋 확정, 🟢 G2)
   └─ modeling_v8_REPORT_06_M5a.md   M5a (모델 벤치 7종×3트랙, ExtraTrees 잠정선두·검증대기)
```

**명명 규칙** — `REPORT/modeling_v8_REPORT_<NN>_<마일스톤>.md`

- **위치**: 스냅샷 REPORT는 `REPORT/` 하위 폴더에 모은다. README(이 파일)·노트북·PLAN은 `modeling_v8/` 루트.
- `<NN>` = 순번(01, 02, 03…) → 파일 브라우저에서 항상 시간·논리 순서로 정렬.
- `<마일스톤>` = 그 배치의 태그(M0, M1, M2…) → 내용 즉시 식별.
- 각 REPORT는 **그 시점 분석의 스냅샷으로 고정**한다. 이후 마일스톤이 나와도 **옛 REPORT를 고치지 않고 새 번호 파일**을 만든다.
- **README(이 파일)만** 살아있는 문서로 계속 갱신한다 — 누적 로그·요약·인덱스가 여기 모인다.
- 다음 마일스톤 예시: M3 → `REPORT/modeling_v8_REPORT_04_M3.md`.

## 30초 요약 (누적)

| 항목 | 값 |
|------|---|
| 피처 테이블 | train 11,939×569 / valid·test 1,990×568 |
| **M0a** (pkl 재현) | valid **37.97** / test 39.04 — 🟢 G1(a) |
| **M0b** (재학습) | CV(wafer) **40.35** / valid **38.40** / test 38.91 — 🟢 **G1 완결** |
| **M-T** (시간-only 대조) | CV 44.26 / valid 44.62 — 센서 기여 **+3.91pt(CV)·+6.22pt(valid)** |
| **M1** (+C23_te) | CV 41.03 / valid 38.78 — **❌ 기각**(ΔCV +0.68, 레짐 프록시) |
| **M2** (센서 풀 재선별) | 최고 CV 43.94 — **❌ 기각**(ΔCV +3.59, 코어 10 못 넘음, C25 미부상 23위) |
| **M3** (row-level 결합) | CV **50.87** / valid 49.27 / test 48.88 — **❌ 기각**(ΔCV +10.52, WF-level 대비 큰 열세) |
| **M4** (피처셋 확정) | 레짐제거 +218.9·센서제거 +3.91(둘 다 필수) / 시간 dedup +8.0(7종 유지) / gain-greedy 최선 43.94(코어10 우위) — **🟢 G2 통과** |
| **M5a** (모델 벤치 7종) | F-C10: **ExtraTrees 38.00** > LGBM 40.35(=cv_m0b) > SegLGBM 40.84 > XGB 41.43 > CatB 41.66 ≫ HistGB 50.97 > Ridge 67.14. 코어10이 3모델 전부에서 트랙 우위(G2 견고). **ExtraTrees 잠정선두 −2.35pt — lot-mate 누수 의심, valid/test 확인(Cell 15)·M7 대기** |
| P1 스모크 | ✅ v1.5 3종(quiet/quiet-major/loud) 통과 — loud앵커로 가짜 스파이크 차단 |
| 비교 | v5 CV 60.5 / valid 61.4 → **v8 M0b CV 40.35 / valid 38.40** (약 −20pt) |

## 누적 실험 로그 (마일스톤마다 1행)

| ID | 변경점 | valid | test | CV(wafer) | 판정 | 스냅샷 |
|----|--------|-------|------|-----------|------|--------|
| **M0a** | 원본 pkl 재현 (피처 정합) | **37.97** | 39.04 | — | ✅ G1(a) | `_01_M0` |
| **M0b** | 동일 10피처 재학습 (wafer 5-fold) | **38.40** | 38.91 | **40.35** | ✅ **G1 완결** | `_01_M0` |
| **M-T** | 시간-only 대조군 (센서 3종 제외) | 44.62 | 44.15 | 44.26 | ✅ 센서 +3.91pt(CV) | `_01_M0` |
| **M1** | +C23_te (11피처, 중첩 OOF) | 38.78 | 39.20 | **41.03** | ❌ 기각 (ΔCV +0.68) | `_02_M1` |
| **M2** | +센서풀 gain 재선별 (최고 TOP_15) | 41.67 | 43.25 | **43.94** | ❌ 기각 (ΔCV +3.59) | `_03_M2` |
| **M3** | row-level 결합 (v5 프레임 + 그룹 A/B broadcast) | 49.27 | 48.88 | **50.87** | ❌ 기각 (ΔCV +10.52) | `_04_M3` |
| **M4** | 블록 ablation · 시간 dedup · TOP_N 스윕 | 38.40 | 38.91 | **40.35** | 🟢 **G2** (코어10 확정) | `_05_M4` |
| **M5a** | 모델 벤치 7종×3트랙 (코어10 동결) | — | — | **38.00** (ExtraTrees) | ⚠️ 잠정 (검증대기) | `_06_M5a` |

## 버전 비교표 (CV/valid, 낮을수록 좋음)

| 버전 | baseline | v2 | v3 | v4 | v5 | v6 | **v8 (M0b)** |
|------|---------|----|----|----|----|----|------|
| CV(wafer) | 61.15 | 61.25 | 60.51 | 61.19 | 60.52 | 66.04 | **40.35** |
| valid | — | — | — | — | 61.38 | — | **38.40** |
| test | — | — | 60.51 | — | 60.52 | — | **38.91** |

> v8 = **약 −20pt 도약**. v8 CV는 wafer 5-fold `KFold(shuffle,42)` — 프리어 버전과 스킴 차이로 ±1pt 있을 수 있으나 v8 **내부** 마일스톤은 동일 스킴 고정. M0a(pkl 재현) 앵커는 valid 37.97.

---

## 이 노트북은 무엇을 하나요?

`EDA/02_EDA_regime`이 "무엇을 쓸지"(레짐·시간 지배, 10개 피처)를 **정했다면**, 이 노트북은 그 결론을 **실제 학습용 피처 테이블로 제작**하고 검증합니다.

- **Phase 3(피처)**: 원본 `pm_feature/`를 이식 → 시간/레짐(그룹 A) + FDC 센서 집계(그룹 B + 풀) 테이블.
- **M0a(앵커)**: 만든 피처를 복원 pkl로 직접 예측 → valid ≈ 38.3 재현 시 "우리 피처 = 원본 피처"(G1a).
- **M0b(재학습·G1 완결)**: 코어 10 + 복원 파라미터로 재학습(wafer 5-fold OOF + full-train) → CV/valid 확정.
- **M-T(시간-only 대조)**: 센서 3종 뺀 7피처 → "센서 기여 = M-T − 모델" 기준선.
- **M1(+C23_te)**: v5의 C23 타깃인코딩을 얹어 레짐 통제 후 잔여 신호 판정(채택 게이트 ΔCV ≤ −0.3).
- **M2(센서 풀 gain 재선별)**: 코어 10 + 전체 센서 집계 풀(563개)로 2-pass gain 재선별 → 코어 10을 못 넘으면 기각.

## ⚠️ 레짐 피처 (v1.5 — 조용/요란 verdict)

`pm_log.json`은 **verdict 포맷** `[{"date","type","verdict"}]`을 씁니다. **`is_high_regime`**은 최근 PM의 `type`이 아니라 **verdict 상태기계**(loud→갱신, quiet→유지)로, **`high_regime_days`**는 **마지막 loud 이벤트 앵커** 마스크로 만듭니다 — quiet PM에서 dslp만 리셋되고 hrd는 연속돼 *가짜 신품 스파이크*를 막습니다. 데이터 내 유일 경계(12-24)가 loud+major라 **옛 type 규칙과 값 동일**(Cell 4 동치 assert) → M0a~M1 수치 전부 불변. 원본 pkl은 옛 이름 `is_post_pm`/`post_pm_days`로 학습됐고 현 데이터에선 값 동일 → M0a는 옛 이름으로 pkl에 먹입니다.

## 입력 · 출력

**입력** (상대경로, `modeling_v8/`에서 실행 기준)

| 파일 | 용도 |
|------|------|
| `../문제1(하)/{train_data,valid_X,test_X}.csv` | 피처 빌드 |
| `../pm_log.json` | **verdict 포맷** 정본 `[{"date":"2018-12-24","type":"major","verdict":"loud"}]` (verdict 없으면 major→loud 폴백) |
| `../pm_feature/lgbm_model.pkl` | M0a 예측 + 파라미터·피처명 복원 |
| `../문제1_하_answer/{valid,test}_Y_answer.csv` | RMSE 채점 (정합 검증 한정) |

**출력**: (현 단계) 화면 출력만 — 피처 테이블·M0a/M0b/M-T/M1 RMSE·센서 기여·스모크 결과. 제출 CSV·results.json은 M4(피처 확정) 이후 셀에서 생성 예정.

## 노트북 구성 (셀별)

| 셀 | 내용 |
|----|------|
| **0 (설명)** | 목적·레짐 피처 정의(v1.5)·게이트 |
| **1** | 설정·상수·`pm_log` **verdict 파서**·**pkl 로드**(top-10·파라미터) |
| **2** | 원본 train/valid/test 로드 |
| **3** | 빌더 — `preprocess` · `make_fdc_features`(23센서×5통계×step + C41_max) · `make_meta_features`(그룹 A, **verdict 상태기계·loud 앵커**) · `build_features` |
| **4** | 피처 테이블 + **v1.5 동치 assert**(verdict==type, 그림자 열 제거) + pkl 별칭 값-동일 |
| **5** | **M0a** — 복원 pkl 예측 → RMSE ≈ 38.3 → G1(a) |
| **6** | **P1 스모크 3종**(v1.5) — quiet/quiet-major/loud 피처 거동 assert |
| **7** | **M0b** — 코어 10 재학습(wafer 5-fold OOF + full-train 705) → G1 완결 |
| **8** | **M-T** — 시간 7피처 대조군 + 센서 기여(M-T−M0b) |
| **9** | **M1** — +C23_te(11, 중첩 OOF TE m=20) → ΔCV 채택 게이트 |
| **10** | **M2** — 센서 풀(563) gain 재선별: pass1 gain 랭킹 → TOP_N 스윕 + 코어10+K probe → ΔCV 게이트 |
| **11** | **M3** — row-level 결합: v5 빌더(행 센서+context+row_pos+C6/C7) + 그룹 A broadcast + hour_row(157피처) → M0b 동일 분할 OOF → 행→WF 평균 → ΔCV 게이트 |
| **12** | **M4** — 피처셋 확정: 블록 ablation(GA/GB) + 시간 dedup(최소셋 vs 7종) + TOP_N 스윕(M2 gain 랭킹) → **G2** 판정 |
| **12.5** | (선행) **M5 환경** — xgboost·catboost 설치 확인(미설치 시 벤치 자동 제외) |
| **13** | **M5a** 하니스 — 3트랙(F-C10·F-T15·F-P3) 정의 + `run_candidate`(동일 fold OOF) + 후보 7종 등록(LGBM/XGB/CatB/HistGB/ExtraTrees/Ridge+spline/SegLGBM) |
| **14** | **M5a** 실행 — F-C10 7종 → 상위3×트랙 → 벤치표 + OOF 오차상관(13×13) + 상위2 선정 + 트랙 valid/test 최초기록 → `outputs/` 저장 |
| **15** | **M5a 확인** — ExtraTrees valid/test 조회(M5b 진입 게이트): 랜덤 CV 우위가 일반화인지 lot-mate 암기인지 판정 |

## 실행 방법

1. **위치**: 반드시 `modeling_v8/` 폴더 안에서 실행 (상대경로 `../문제1(하)` 기준).
2. **커널**: `venv (Python 3.12)`.
3. **패키지**: `pandas numpy lightgbm scikit-learn`. 추가 설치 불필요.
4. Jupyter/VS Code에서 **Restart & Run All**. 약 12~20분(train 44MB 로드 + 피처 3세트 빌드 + pkl 예측 + M0b·M-T·M1·M2 재학습 + **M3 row-level 5-fold**). **M3(Cell 11)가 가장 김** — 123,614행 × 157피처 × 5-fold(그다음 M2 Cell 10).
5. 확인: Cell 4 `✅ v1.5 동치`, Cell 5 `🟢 G1(a) 통과`, Cell 6 `(a)(b)(c) ✅`, Cell 7 `🟢 G1 완결`(valid 38.40), Cell 8 `센서 기여 +3.91pt`, Cell 9 `❌ 기각`(ΔCV +0.68), Cell 10 `❌ 기각`(ΔCV +3.59), Cell 11 `❌ 기각`(ΔCV +10.52, M3 CV 50.87), Cell 12 `🟢 G2 통과`(레짐제거 +218.9·센서제거 +3.91·시간 dedup +8.0).

> ⚠️ **pkl 로드(R11)**: Cell 1이 원본 `lgbm_model.pkl`을 unpickle합니다. 로컬 lightgbm 버전이 원본과 크게 다르면 실패할 수 있습니다. 에러 시 lightgbm 버전 확인 또는 작성자에게 `Booster.save_model()` 텍스트 포맷 요청.

## 현재 상태 & 다음

- ✅ **M0a~M4 완료** — 피처 정합(G1a) + 재학습 확정(**G1**) + 센서 기여 +3.91pt + **M1·M2·M3 전부 기각** + **M4 🟢 G2 통과**. → **코어 10 WF-level 피처셋 동결**(CV 40.35 / valid 38.40 / test 38.91).
- ✅ **M5a 완료(모델 벤치)** — 후보 7종 × 3트랙. 부스팅 3종 근접(LGBM 40.35 최선)·선형/세그먼트 부적합. **코어 10이 3모델 전부에서 F-P3·F-T15 우위 → G2 견고**. 하니스 정합 확증(LGBM=cv_m0b, F-T15 valid/test=동결표). **⚠️ ExtraTrees 38.00으로 랜덤 CV −2.35pt 선두이나 lot-mate 누수 의심 → 검증 대기.**
- ⬜ **다음 (즉시)**: **Cell 15 확인** — ExtraTrees valid/test 조회로 일반화/암기 판정. → **M5b**(확정된 상위 2개 Optuna 경량, F-C10만 튜닝·트랙 전이 1회) → (M6 앙상블 — ExtraTrees가 부스팅과 최탈상관 0.76~0.79/Ridge 0.46) → **M7**(Lot-CV `GroupKFold(C20)`·time-split 정직성 — ExtraTrees 우위의 최종 판정) → **M8**(시간 재분할 시뮬).
- 상세 스냅샷: M0 → `REPORT/modeling_v8_REPORT_01_M0.md`, M1 → `…_02_M1.md`, M2 → `…_03_M2.md`, M3 → `…_04_M3.md`, M4 → `…_05_M4.md`, **M5a → `…_06_M5a.md`**.
