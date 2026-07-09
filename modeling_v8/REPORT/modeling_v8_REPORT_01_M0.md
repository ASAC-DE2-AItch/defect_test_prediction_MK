# modeling_v8 결과 보고서 01 — M0 배치 (Phase 3 + M0a · M0b · M-T)

> **프로젝트**: SK하이닉스 반도체 결함 예측 (FDC Trace → C65 회귀)
> **문서 성격**: **스냅샷 보고서**(고정) — 이 파일은 *M0 배치*(피처 구현 + 앵커 검증 + 재학습 + 시간-only 대조)의 결과를 기록하며 이후 마일스톤이 나와도 **수정하지 않는다**. 다음 배치는 새 번호 파일(`_02_M1`, `_03_M2`…). 누적 로그·버전표·인덱스는 `../modeling_v8_README.md` 참조.
> **범위**: Phase 3(피처 구현) + **M0a**(앵커) + **M0b**(재학습·G1 완결) + **M-T**(시간-only 대조). PLAN **v1.5**(verdict) 반영.
> **작성 기준(2026-07-09)**: `modeling_v8.ipynb` 실행 결과 + 원본 `lgbm_model.pkl` 복원. M0a는 2026-07-08 사용자 로컬 확인, M0b·M-T·v1.5는 클라우드 미러 재현(로컬 `Restart & Run All` 확인 대기). 수치는 재현 가능한 실제 실행값.

---

## 0. 배경 — v8 전략 한 줄

`EDA/02_EDA_regime`(Phase 1)이 규명했듯, 결함(C65)은 **PM 레짐(저불량↔loud-급등) + PM 이후 경과 시간**이 지배하고, v1~v7이 ~60에 갇힌 건 이 시간축을 버리고 센서를 프록시로 썼기 때문이다. FEATURE_SPEC은 시간/레짐 피처 10개로 valid ~38.3을 주장했다. **Phase 3는 그 10피처를 원본 코드 이식으로 실제 제작하고, M0a는 그것이 원본과 동일함을 pkl 재현으로 증명하며, M0b는 우리 손으로 재학습해 CV/valid를 확정한다(G1).**

### PLAN v1.5 (verdict) 반영

`is_high_regime`을 최근 PM의 `type`이 아니라 **조용/요란 verdict 상태기계**(loud→갱신, quiet→유지)로, `high_regime_days` 앵커를 **마지막 loud 이벤트**로 교체했다. 데이터 내 유일 경계(12-24)가 loud+major라 **전 수치 불변**(Cell 4 동치 assert 통과) — 그래서 아래 M0a~M-T 값은 v1.4와 동일하다.

---

## 1. Phase 3 — 피처 구현

원본 `pm_feature/preprocessing.py`·`feature_engineering.py`를 우리 경로로 함수 단위 이식:

- `preprocess` — C40 datetime 파싱 → 무의미 컬럼 23개 제거 → 시간 정렬. train/valid/test 동일 적용.
- `make_fdc_features` — FDC 23종 × {mean,std,max,min,last} × step pivot(`{센서}_{통계}_step{n}`) + `C41_max_step{n}`.
- `make_meta_features` — 그룹 A 시간/레짐 피처. `pm_log`(**verdict 포맷**)에서 **verdict 상태기계**로 `is_high_regime`(loud→갱신, quiet→유지)·`high_regime_days`(**마지막 loud 앵커** 마스크) 생성(+ pkl 재현용 `is_post_pm`·`post_pm_days` 별칭, last-PM 앵커).
- `is_special_recipe` = (C6=="C6_1"). / `C23`(레시피 세부 28종, str) — M1 재료로 실어둠.

**피처 테이블**: train `11,939 × 569` / valid·test `1,990 × 568` (열 = FDC 집계 풀 + 메타 + 타깃).

### 코어 10피처 (M0 앵커)

| 그룹 | 피처 |
|------|------|
| A 시간/레짐 (7) | `is_high_regime`, `high_regime_days`, `days_since_last_pm`, `C33`, `dslp_x_hour`, `hour`, `hour_x_c33` |
| B 센서/레시피 (3) | `C60_mean_step4`, `C59_mean_step4`, `is_special_recipe` |

### 정합 assert (통과)

- WF 수: train 11,939 / valid 1,990 / test 1,990 ✅
- 코어 10피처 전부 존재 ✅
- **v1.5 동치**: `is_high_regime(verdict) == is_high(type)`, `high_regime_days(loud앵커) == hrd(type)` — 전 세트 ✅ (유일 경계 12-24가 loud+major → 두 규칙 동치, 검증 후 그림자 열 제거)
- **pkl 별칭 값-동일**: `is_high_regime == is_post_pm`, `high_regime_days == post_pm_days` ✅ → 급등 8,247 / 저불량 3,692 WF

---

## 2. M0a — 앵커 검증 (복원 pkl로 직접 예측)

만든 피처 테이블을 **학습 없이** 원본 `lgbm_model.pkl`(복원 top-10 피처)로 valid/test 예측:

| 지표 | 값 | 기대 | 판정 |
|------|---|------|------|
| **valid RMSE** | **37.97** | ~38.3 (원본 주석) | ✅ 재현 |
| test RMSE | 39.04 | — | ✅ |

> **🟢 G1(a) 통과.** valid 37.97이 원본 ~38.3과 ±1 이내로 일치 → **우리가 이식한 피처 테이블 = 원본 피처 테이블**임이 증명됐다(피처 정의·경계·NaN 처리 모두 정합). v5 valid 61.38 대비 **약 −23pt**로, v8이 겨냥한 ~38 앵커가 우리 파이프라인에 실재함을 확인. (M0a는 원본 pkl의 **재현**이다. 우리 재학습 CV/valid는 아래 M0b에서 측정한다.)

---

## 3. M0b · M-T — 재학습(G1 완결) + 센서 기여

M0a로 "우리 피처 = 원본 피처"가 증명됐으니, 이제 **복원 파라미터(lr 0.029, num_leaves 175 등)로 직접 재학습**해 v8 파이프라인의 CV/valid를 확정한다.

**CV 스킴**: 피처 테이블은 **wafer 1행**(고유 C64 = 행수 11,939)이므로 wafer-level `KFold(5, shuffle, seed42)` OOF를 쓴다. 이 단계에선 각 wafer가 단일 그룹이라 `GroupKFold(C64)`와 목적이 동일하고(누수 없음), **그룹이 비자명해지는 row-level(M3+)부터** `GroupKFold(C64)`로 전환한다. full-train은 원본 pkl의 `best_iteration`(705라운드) 예산을 그대로 쓴다.

| 모델 | 피처 | CV(wafer, OOF) | valid | test |
|------|------|:---:|:---:|:---:|
| **M0b** (재학습) | 코어 10 | **40.35** | **38.40** | 38.91 |
| **M-T** (시간-only 대조) | 시간/레짐 7 (센서 3종 제외) | 44.26 | 44.62 | 44.15 |
| **센서 기여** (M-T − M0b) | `C60/C59_mean_step4` + `is_special_recipe` | **+3.91pt** | **+6.22pt** | +5.24pt |

> **🟢 G1 완결.** 재학습 valid **38.40 ≤ 42**, CV **40.35**(목표 ~41 이내) → v8 파이프라인의 CV/valid가 확정됐다. M0a(pkl 재현 37.97)와도 ±0.5로 일치해 재학습이 원본을 충실히 복제함을 보인다.
>
> **센서는 시간축 위에서도 유의미하게 기여한다.** 시간/레짐 7피처만으로 CV 44.26에 그치고, 센서 3종(C59·C60 step4 집계 + 레시피)을 더하면 CV −3.91pt·valid −6.22pt 개선된다. 이는 "시간축이 지배적이면 v5 센서를 합칠 의미가 없다"(PLAN §8.4)는 우려에 대한 반증 — **센서 결합(M2·M3)은 실익이 있다.** 이후 모든 마일스톤에서 "센서 증분 = 모델 − M-T"로 기여를 추적한다.

---

## 4. P1 — 프로덕션 안전 (스모크 테스트, v1.5 신세계관 3종)

`pm_log.json` 사본에 가상 이벤트 1줄(`date`+`type`+`verdict`)을 추가하고 피처를 재계산 — 3개 시나리오 전부 통과:

| 시나리오 | 가상 이벤트 | 기대·결과 |
|----------|------------|-----------|
| (a) 고불량 중 quiet | `{2019-01-20, minor, quiet}` | 이후 673 WF: `is_high` **1 유지** & `high_regime_days` **연속(리셋 X)** & `dslp`만 리셋 ✅ |
| (b) 저불량 중 quiet major | `{2018-12-15, major, quiet}` | 저불량 606 WF: `is_high` **0 유지**(type major여도 verdict quiet → 점프 없음) ✅ |
| (c) 신규 loud | `{2019-01-20, major, loud}` | 이후 673 WF: `is_high` 갱신 & `high_regime_days` **새 loud 앵커에서 0부터 재시작**(hrd==dslp) ✅ |

> 새 PM이 발생해도 **코드 수정 없이 pm_log 1줄 추가**만으로 전 피처가 자동 대응(P1). 특히 **(a)가 v1.5 loud-앵커 수정의 핵심**: quiet PM에서 dslp만 리셋되고 hrd는 연속되므로, 구 버전(dslp 앵커)이라면 hrd=0 → 모델이 "요란 D+0"으로 오독해 **가짜 ~1400 스파이크**를 뱉던 버그가 선제 차단된다. 예측-밴드 정밀검증은 M8(§9.4). 하드코딩 날짜는 pm_log·campaign_start 2곳뿐.

---

## 5. 이 배치 판정 요약

| ID | 변경점 | valid | test | CV(wafer) | 판정 |
|----|--------|-------|------|-----------|------|
| **M0a** | 원본 pkl 재현 (피처 정합) | **37.97** | 39.04 | — | ✅ G1(a) 통과 |
| **M0b** | 동일 10피처 재학습 (wafer 5-fold) | **38.40** | 38.91 | **40.35** | ✅ **G1 완결** |
| **M-T** | 시간-only 대조군 (센서 3종 제외) | 44.62 | 44.15 | 44.26 | ✅ 센서 +3.91pt(CV) |

> **v5 대비**: CV 60.5 → **40.35**, valid 61.4 → **38.40** (약 −20pt 도약). 전 마일스톤 누적 로그·버전 비교표는 `../modeling_v8_README.md`.

---

## 6. 다음 단계

- ✅ **M0 배치 완결** — 피처 정합(G1a) + 재학습 CV/valid 확정(**G1 완결**) + 센서 기여 +3.91pt(CV) 기준선.
- **→ M1**(`+C23_te`, 채택 게이트): `modeling_v8_REPORT_02_M1.md` 참조. 이후 M2(센서풀·C25)·M3(row-level)·M4(TOP_N·ablation, G2).

---

## 부록: 재현성

- 모든 수치는 `modeling_v8.ipynb` 실제 실행값. M0a는 사용자 로컬 확인(2026-07-08), M0b·M-T·v1.5는 클라우드 미러 재현(로컬 확인 대기).
- M0a는 원본 `lgbm_model.pkl`의 top-10 피처(`booster.feature_name()`)로 예측 — 피처명·순서가 원본과 정합.
- 피처 파이프라인은 원본 `pm_feature/`를 함수 단위 이식했고, valid 37.97 재현이 그 충실성을 증명한다. v1.5 verdict 교체 후에도 동치 assert로 수치 불변을 확인.
