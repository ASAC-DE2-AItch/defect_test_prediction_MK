# modeling_v13 REPORT 01 — M1 Feature Diet (4단계)

> **스냅샷 (고정)** · 작성 2026-07-13 · 노트북 `modeling_v13_feature_diet.ipynb`
> 이 문서는 M1(피처 다이어트) 시점의 분석 기록이다. 이후 마일스톤이 나와도 **수정하지 않는다.**

---

## 0. 한 줄 요약

v13 FDC 풀(11,939 WF × **655 피처**)에 4단계 제거를 적용해
**Conservative 141 · Balanced 126** 피처로 축소. 필수 5센서(C17·C11·C31·C15·C16)는
champion 면제로 **전 단계 각 ≥1 보장** — floor 위반 0.

---

## 1. 방법 — 제거 순서 + floor 규칙

**순서**: ① Variance Threshold(상수 제거) → ② 결측률 필터 → ③ Correlation → ④ VIF

**Floor 규칙(필수 요구사항)**: `C17`(온도)·`C11`(플라즈마)·`C31`(RF)·`C15`·`C16`(가스)
각 센서는 최소 1개 피처 생존. 구현 = 센서별 **champion 1개**를 미리 뽑아 4단계 전체에서 **제거 면제**.
champion 선정: 비상수(`nunique>1`) 중 **결측률 최소 → |타깃 상관| 최대 → 이름순**.

**단계별 판정 기준**

| 단계 | 기준 | champion |
|---|---|---|
| ① Variance | exact 상수(`nunique≤1`) 제거 + min-max 정규화 분산 `nzv ≤ 임계` 제거(안전망) | 면제 |
| ② 결측률 | 결측률 `> miss` 제거 | 면제 |
| ③ Correlation | champion·|타깃상관| 내림차순 순회, 이미 keep된 것과 `|corr| > corr` 이면 드롭(타깃 관련 큰 쪽 생존) | 절대 keep |
| ④ VIF | 역상관행렬 `VIF=diag(inv(R))`, 임계 초과 non-champion 최댓값 반복 제거 | 절대 keep |

---

## 2. 프리셋 컷 기준

| 프리셋 | 결측률 컷 | \|상관\| 컷 | VIF 컷 | nzv |
|---|---|---|---|---|
| Conservative | > 0.70 | > 0.97 | > 10 | 1e-5 |
| Balanced | > 0.50 | > 0.95 | > 10 | 1e-5 |

> Aggressive(miss 0.40 / corr 0.90 / vif 5)는 개념 비교 후 **후속 대상에서 제외**(2026-07-13 결정).

---

## 3. 단계별 생존 피처 수

| 단계 | Conservative | Balanced |
|---|---|---|
| 시작 | 655 | 655 |
| ① Variance | 435 | 435 |
| ② 결측률 | 435 | 344 |
| ③ Correlation | 226 | 189 |
| ④ VIF | **141** | **126** |
| 최종 센서 수 | 25 | 26 |

---

## 4. 필수 5센서 최종 피처 수 (floor 검증)

| 프리셋 | C17 | C11 | C31 | C15 | C16 | floor |
|---|---|---|---|---|---|---|
| Conservative | 4 | 12 | 3 | 3 | 3 | ✅ |
| Balanced | 3 | 11 | 2 | 3 | 3 | ✅ |

**champion(양 프리셋 공통)**: C17→`C17_max_step4`, C11→`C11_min_step4`,
C31→`C31_mean_step4`, C15→`C15_max_step1`, C16→`C16_max_step1`.

---

## 5. 핵심 관찰

- **① Variance에서 빠진 220개는 전부 exact 상수**(`nunique≤1`). `nzv≤1e-5` 추가 제거는 **0** — 즉 이 데이터에선 상수 제거만 실질 작동(nzv는 재사용 대비 안전망).
- 상수 220개 내역: 통계별 std 62 / min 45 / max·last 39 / mean 35, 스텝별 step7 65 · step5 55 · step6 52 등. (setpoint성·희소 step 집계)
- 필수 센서 중 **C31·C15·C16은 25개 중 13~17개가 상수** — champion 면제가 없으면 floor가 깨질 수 있던 센서. 면제로 안전 확보.
- ② 결측 단계: Conservative(0.70)는 추가 제거 0, Balanced(0.50)는 91개 제거(step5·std 계열 결측 다수).

---

## 6. 산출물

| 파일 | 내용 |
|---|---|
| `modeling_v13_feature_diet.ipynb` | 4단계 파이프라인(2 프리셋) |
| `feature_diet_summary.csv` | 단계별 생존 수 요약 |
| `feature_diet_selected.json` | 프리셋별 선택 피처 목록 + champion + 센서별 카운트 + 컷 기준 |

---

## 7. 자체 검증 (클라우드 미러)

- ✅ 노트북 end-to-end 실행 무경고. floor assert 통과(양 프리셋).
- ✅ 결정론적: 모델 학습 없이 통계 기반 선별 → 로컬 재현 시 동일 수치.
- ✅ champion은 비상수 중 선정되어 상수 유입 없음.

> **주의**: 이 단계는 피처 *후보 축소*일 뿐, 성능 판정은 M2(`perf_compare`)에서 core10 결합으로 수행.

---

## 8. 프리셋 간 피처 중첩 (Conservative ↔ Balanced)

### 8.1 요약

| 구분 | 피처 수 |
|---|---|
| **공통** (교집합) | **108** |
| **Conservative 단독** | **33** |
| **Balanced 단독** | **18** |
| Conservative 합계 | 141 (= 108 + 33) |
| Balanced 합계 | 126 (= 108 + 18) |

- **Balanced는 Conservative의 부분집합이 아니다** — 양방향 차이 존재(Balanced 단독 18개).
- 두 셋의 **합집합 = 159**, **자카드 유사도 = 108/159 ≈ 0.68**. 공통 비율이 높아 코어 신호는 대체로 겹친다.
- **차이의 정체 = Step**. Conservative 단독 33개는 **step5(15) · step6(16)** 이 지배 — 결측 컷이 느슨(0.70)해 **step5(≈65% 결측) 집계가 생존**하고, corr/VIF 컷도 완화(0.97/10)돼 step6 std가 더 남았다. Balanced 단독 18개는 **step6(10) · step7(3) · step4(2) · step1(3)** — 결측 컷(0.50)이 step5를 걷어낸 뒤 corr/VIF 경로가 갈리며 다른 대표가 생존한 결과.
  - Conservative-only step 분포: `{step1:2, step5:15, step6:16}`
  - Balanced-only step 분포: `{step1:3, step4:2, step6:10, step7:3}`
- **필수 5센서**는 공통 교집합 안에 champion 포함해 모두 존재(floor는 두 셋 각각 §4에서 충족).

### 8.2 공통 피처 108개 (센서별)

- **C4** (2): `C4_max_step1`, `C4_mean_step1`
- **C11** (9): `C11_last_step1`, `C11_last_step6`, `C11_last_step7`, `C11_max_step1`, `C11_max_step4`, `C11_max_step6`, `C11_max_step7`, `C11_min_step4`, `C11_std_step7`
- **C12** (1): `C12_mean_step1`
- **C15** (3): `C15_last_step4`, `C15_max_step1`, `C15_max_step4`
- **C16** (2): `C16_max_step1`, `C16_mean_step4`
- **C17** (3): `C17_max_step4`, `C17_min_step1`, `C17_std_step1`
- **C18** (8): `C18_last_step1`, `C18_last_step4`, `C18_max_step4`, `C18_mean_step1`, `C18_min_step1`, `C18_min_step4`, `C18_min_step7`, `C18_std_step7`
- **C25** (12): `C25_last_step1`, `C25_last_step4`, `C25_last_step6`, `C25_last_step7`, `C25_max_step1`, `C25_max_step6`, `C25_mean_step4`, `C25_min_step4`, `C25_min_step7`, `C25_std_step1`, `C25_std_step4`, `C25_std_step7`
- **C27** (3): `C27_last_step4`, `C27_max_step4`, `C27_std_step4`
- **C31** (2): `C31_max_step4`, `C31_mean_step4`
- **C32** (2): `C32_last_step4`, `C32_min_step4`
- **C46** (2): `C46_std_step1`, `C46_std_step7`
- **C48** (3): `C48_last_step4`, `C48_max_step1`, `C48_max_step4`
- **C50** (2): `C50_std_step1`, `C50_std_step4`
- **C52** (8): `C52_last_step4`, `C52_last_step6`, `C52_max_step7`, `C52_mean_step1`, `C52_min_step4`, `C52_std_step1`, `C52_std_step4`, `C52_std_step7`
- **C54** (3): `C54_max_step4`, `C54_min_step4`, `C54_std_step4`
- **C56** (5): `C56_last_step4`, `C56_max_step4`, `C56_mean_step4`, `C56_min_step4`, `C56_std_step4`
- **C57** (5): `C57_last_step1`, `C57_last_step4`, `C57_mean_step1`, `C57_mean_step4`, `C57_min_step4`
- **C58** (11): `C58_last_step1`, `C58_last_step4`, `C58_last_step6`, `C58_last_step7`, `C58_mean_step1`, `C58_mean_step4`, `C58_min_step6`, `C58_min_step7`, `C58_std_step1`, `C58_std_step4`, `C58_std_step7`
- **C59** (3): `C59_std_step1`, `C59_std_step4`, `C59_std_step7`
- **C60** (4): `C60_min_step1`, `C60_std_step1`, `C60_std_step4`, `C60_std_step7`
- **C61** (5): `C61_last_step1`, `C61_last_step6`, `C61_last_step7`, `C61_min_step4`, `C61_std_step7`
- **C62** (7): `C62_last_step4`, `C62_max_step4`, `C62_max_step6`, `C62_max_step7`, `C62_mean_step1`, `C62_mean_step4`, `C62_std_step7`
- **C63** (3): `C63_min_step4`, `C63_min_step7`, `C63_std_step4`

### 8.3 Conservative 단독 33개 (센서별)

- **C4** (3): `C4_mean_step5`, `C4_min_step6`, `C4_std_step6`
- **C11** (3): `C11_mean_step5`, `C11_min_step1`, `C11_std_step6`
- **C16** (1): `C16_mean_step5`
- **C17** (1): `C17_std_step6`
- **C18** (3): `C18_mean_step5`, `C18_min_step6`, `C18_std_step6`
- **C25** (2): `C25_mean_step5`, `C25_std_step6`
- **C31** (1): `C31_mean_step5`
- **C32** (1): `C32_mean_step5`
- **C46** (1): `C46_std_step6`
- **C48** (2): `C48_mean_step5`, `C48_std_step6`
- **C49** (1): `C49_mean_step5`
- **C52** (2): `C52_mean_step5`, `C52_std_step6`
- **C57** (1): `C57_mean_step5`
- **C58** (2): `C58_mean_step5`, `C58_std_step6`
- **C59** (2): `C59_mean_step5`, `C59_std_step6`
- **C60** (1): `C60_std_step6`
- **C61** (3): `C61_mean_step5`, `C61_std_step1`, `C61_std_step6`
- **C62** (1): `C62_std_step6`
- **C63** (2): `C63_mean_step5`, `C63_std_step6`

### 8.4 Balanced 단독 18개 (센서별)

- **C4** (1): `C4_max_step6`
- **C11** (2): `C11_min_step6`, `C11_std_step1`
- **C16** (1): `C16_last_step4`
- **C18** (1): `C18_max_step6`
- **C25** (1): `C25_min_step6`
- **C41** (1): `C41_max_step7`
- **C48** (1): `C48_max_step6`
- **C49** (2): `C49_mean_step1`, `C49_mean_step7`
- **C54** (1): `C54_mean_step4`
- **C58** (1): `C58_max_step6`
- **C59** (1): `C59_max_step7`
- **C61** (1): `C61_min_step6`
- **C62** (2): `C62_min_step6`, `C62_std_step1`
- **C63** (2): `C63_max_step6`, `C63_min_step6`
