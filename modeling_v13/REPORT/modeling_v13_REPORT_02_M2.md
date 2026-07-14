# modeling_v13 REPORT 02 — M2 Perf Compare (Conservative vs Balanced, core10 결합)

> **스냅샷 (고정)** · 작성 2026-07-13 · 노트북 `modeling_v13_perf_compare.ipynb`
> M2(성능 비교) 시점 기록. 이후 마일스톤이 나와도 **수정하지 않는다.**
> ✅ 아래 RMSE는 **로컬 실행 확정값**(`perf_compare_results.csv`, 2026-07-13). 결론은 클라우드 미러와 동일,
> GroupKFold 절대값만 환경차로 ~+0.7~1.0pt(예: core10 77.49→78.18) 이동 — 순위·해석 불변.

---

## 0. 한 줄 요약

M1 두 프리셋을 **core10 결합** 상태로 고정 LGBM에 태워 2개 CV로 비교.
**정직-CV(GroupKFold C20)에서 diet 결합(≈71.2)이 core10 단독(78.2)을 앞서** — 신규 Lot에서 센서가 기여.
Conservative vs Balanced는 **초박빙**(정직-CV Conservative 근소 우위, KFold Balanced 근소 우위).

---

## 1. 셋업

- **결합 규칙**: `프리셋 diet ∪ core10`(중복 제거).
- **core10**(modeling_v8 M4 동결) = 레짐/시간 7 (`is_high_regime, high_regime_days, days_since_last_pm, C33, dslp_x_hour, hour, hour_x_c33`) + 센서/레시피 3 (`C60_mean_step4, C59_mean_step4, is_special_recipe`).
- **시간/레짐 8피처는 원본 train에서 재빌드**(`v8_timeline_common.build`), 센서 2피처는 v13 풀에 존재 → C64 조인(손실 0).
- **모델(고정)**: `M8_PARAMS`(v8 복원 LGBM) · 705 rounds. 프리셋 간 동일 → 피처셋만의 차이 측정.
- **평가**: ① KFold 5-fold OOF(`fold_kf5`) · ② `GroupKFold(C20)` 정직-CV.

---

## 2. 결과 (로컬 실행 확정 · `perf_compare_results.csv`)

| 피처셋 | 피처수 | KFold OOF | GroupKFold(C20) |
|---|---|---|---|
| core10 단독 (참조) | 10 | **40.387** | 78.181 |
| Balanced + core10 | 136 | 51.564 | 71.416 |
| Conservative + core10 | 151 | 51.572 | **71.212** |

**충실성 확인**: core10 단독 KFold **40.387** ≈ v8 동결 CV 40.35 → 파이프라인 재현 정상.
(참고 클라우드 미러값: core10 GKF 77.49 / Balanced 70.45 / Conservative 70.11 — GroupKFold 절대값만 환경차, 순위 동일.)

---

## 3. 해석

- **두 CV가 정반대**를 가리킨다.
  - **KFold(랜덤)**: core10 단독(40.4)이 diet 결합(≈51.6)보다 우수. 원인 = `M8_PARAMS`가 **10피처 기준 튜닝** → 130+피처·705라운드에서 과적합.
  - **GroupKFold(C20) 정직-CV**: diet 결합(≈71.2)이 core10 단독(78.2)을 **≈7pt 앞섬**. 신규 Lot 일반화에서 **센서 피처 기여** → v13이 4센서(C11·C15·C16·C31) 복원한 취지와 정합.
- **Conservative vs Balanced**: 초박빙.
  - 정직-CV: Conservative 71.212 < Balanced 71.416 (Δ0.20, Conservative 근소 우위)
  - KFold: Balanced 51.564 < Conservative 51.572 (Δ0.008, Balanced 근소 우위)
  - Balanced가 더 슬림(136 vs 151).

---

## 4. 판정 & 한계

- **미확정**. 격차가 파라미터 노이즈 수준이고, 순위 전체가 **10피처용 고정 파라미터**에 의존.
- 확정 전 필수: 결합셋에서 `num_boost_round`·`num_leaves` 등 **재튜닝**(KFold 과적합 해소) 후 정직-CV 재비교.
- 프로젝트 원칙(신규 Lot 견고성 우선)상 최종 판정 지표는 **GroupKFold(C20)**로 둔다.

---

## 5. 산출물

| 파일 | 내용 |
|---|---|
| `modeling_v13_perf_compare.ipynb` | core10 결합 + 2 CV 평가 |
| `perf_compare_results.csv` | 결과 표(로컬 실행 확정값) |

> **실측 런타임(로컬)**: 총 **≈46분**(Cons 1063s + Bal 1033s + core10 656s, 30 LGBM fit).
> 클라우드 미러(~3–4분)보다 훨씬 느림 — 사용자 환경이 fit당 ~100s(단일스레드/CPU 성능차 추정).
> **CPU 전용**(LightGBM CPU 경로). 후속 선별 노트북(RFECV/GA)은 fit 수가 많아 **로컬에서 수시간** 소요 위험 → 설정 경량화 또는 빠른 CPU 권장.
