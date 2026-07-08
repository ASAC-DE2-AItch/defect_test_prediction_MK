# 02_EDA_regime.ipynb 안내서

> 이 문서는 **노트북 사용 설명서**입니다. 분석 "결과"가 궁금하면 같은 폴더의 `02_EDA_regime_REPORT.md`를 보세요.
> **버전 맥락**: modeling_v8 **Phase 1(검증 EDA)** 산출물. 기존 `01_EDA.ipynb`의 후속입니다.

## 이 노트북은 무엇을 하는 건가요?

`01_EDA`가 "센서 위주로 데이터를 처음 훑어본" 분석이었다면, 이 노트북은 **PM(예방 정비) 레짐**에 집중한 **검증 EDA**입니다.

결함 수치(C65)는 **저불량 레짐(~638)** 과 **급등 레짐(~1123)**, 두 상태를 오갑니다. 급등을 만드는 건 **major PM**(큰 정비 → 챔버 대교란)이고, 한 major 사이클 안에 **minor PM**(작은 정비 → 저불량으로 복귀)이 낍니다(패턴 **M m m M …**). 요리에 비유하면, 오븐을 **크게 뜯어 청소(major)** 하면 한동안 맛이 확 달라졌다가, 중간에 **가볍게 닦으면(minor)** 원래 맛으로 돌아오는 셈이죠. 이 노트북은 그 major PM 시점을 **X 데이터만으로 탐지**하고, 시간·레짐 신호가 유효한지 수치로 확인해 **Phase 3 피처 설계의 근거(pm_log)** 를 확정합니다.

## ⚠️ 대원칙 (누수 방지)

**PM 날짜 탐지에는 X 컬럼(C40 시각, C33 장비 카운터)만 씁니다. 정답(C65)은 탐지에 절대 쓰지 않고 사후 검증에만.** 그리고 **PM의 종류(major/minor)는 정비 기록 기반 입력값**(`pm_log.json`)이며, C65 급등을 보고 역추정하지 않습니다(그렇게 하면 타깃이 피처에 새어 들어감).

## 30초 요약 (핵심 결과 미리보기)

| 확인 항목 | 결론 |
|-----------|------|
| PM 이벤트 | **minor(데이터 시작 직전) + major(2018-12-24)** — C33 두 사이클(3~39 / 1~74)로 관측. X 탐지는 major만(minor 리셋은 데이터 전) |
| 레짐 점프 | 저불량 **637.5** → 급등 **1123.5** (약 2배) |
| 레짐 스위치 | **`is_high_regime`** = "가장 최근 PM이 major인가" 0/1. major에서만 1 → 저불량/급등 경계는 12-24 |
| 설명력 사다리 | 전체평균 262 → +레짐 134 → +주decay 76 → +hour 68 (레짐·시간만으로) |
| "최강 센서" C17의 정체 | 전체 상관 −0.80이지만 레짐 내부에선 −0.19/+0.10로 붕괴 → **레짐 프록시(confound)** |
| 게이트 | 🟢 **G0 통과** → Phase 3(피처 구현) 진입 가능 |

자세한 근거·그래프는 `02_EDA_regime_REPORT.md`에 있습니다.

## 입력 · 출력 (한눈 표)

**입력**

| 파일 | 용도 |
|------|------|
| `../문제1(하)/train_data.csv` | 레짐 탐지·통계·사다리 (X + C65는 사후검증만) |
| `../문제1(하)/valid_X.csv`, `test_X.csv` | major PM 독립 재탐지 (P1 증명, X만) |
| `../pm_log.json` | **타입 포맷 정본** `[{"date":"2018-12-24","type":"major"}]`. 탐지 날짜와 **일치 assert**, **종류(major/minor)를 여기서 읽음** |
| `../modeling_v5/outputs/valid_Y_submit.csv` | E10 v5 잔차 진단용 |
| `../문제1_하_answer/valid_Y_answer.csv` | E10 v5 잔차 진단용(진단 한정 사용) |

**출력**

| 파일 | 내용 |
|------|------|
| `../pm_log_meta.json` | PM 탐지 근거(규칙·시각·증거·**type**·교차검증). 정본 pm_log와 분리 |
| `assets/regime_decay_{ko,en}.png` | 급등 레짐 decay 곡선 (한글·영어) |
| `assets/regime_hour_{ko,en}.png` | 시간대·요일 프로파일 (한글·영어) |
| `assets/regime_c59c60_{ko,en}.png` | step4 C59-C60 상호배타 산점도 (한글·영어) |
| `assets/regime_confound_scan.csv` | 센서별 전체/저불량/급등 상관 전수표 (`corr_all`/`corr_low`/`corr_high`) |

## 배경 지식 (용어)

| 용어 | 쉬운 설명 |
|------|----------|
| major PM | 큰 정비 → 챔버 대교란 → C65 **급등**(저불량 대비 +486). 12-24 실측 |
| minor PM | 작은 정비 → 교란 작음 → **저불량 레짐으로 복귀**. 한 major 사이클에 2회 낌 |
| 레짐(regime) | 결함 수준의 상태. **저불량**(~638) / **급등**(~1123) |
| `is_high_regime` | "가장 최근 PM이 major인가" 0/1. major 뒤=1(급등), minor 뒤·PM 이전=0(저불량) |
| `days_since_last_pm` | 마지막 PM(major·minor 무관) 이후 경과일(소수). 매 PM마다 0으로 리셋 |
| `high_regime_days` | `days_since_last_pm × is_high_regime` — 급등 레짐일 때만 경과일 노출(급하강 곡선을 저불량과 분리 학습) |
| C33 | 장비 시간 카운터. 하루 ~1.6씩 증가하다 **PM 때 리셋** → 날짜 탐지 단서 |
| confound(교란) | 겉으론 A↔C가 관련 있어 보이나 사실은 공통 원인(레짐/시간) 탓인 착시 |
| within-레짐 상관 | 저불량 내부·급등 내부로 나눠 본 상관. confound를 걸러낸 "진짜 신호" |
| 설명력 사다리 | 예측기를 단계별로 쌓으며 RMSE가 얼마나 줄어드는지 보는 in-sample 진단 |

## 노트북 구성 (셀별 설명)

| 셀 | 내용 |
|----|------|
| **0 (설명)** | 목적·레짐 정의(저불량/급등, major/minor)·대원칙(C65로 탐지 금지)·G0 |
| **1** | 설정·로드·헬퍼. 경로, 한글 폰트, `CHART_LANGS` 토글, 키컬럼 object 강제, WF 테이블 헬퍼 |
| **2 (E1)** | C40 파싱 검증(NaT=0) + train/valid/test 기간 커버리지 표 |
| **3 (E2)** | **PM 날짜 탐지(X만)** → 타입 pm_log 파서(레거시=major 폴백) → 종류 읽기 → **`is_high_regime` 계산** → 루트 pm_log 일치 assert → valid/test 독립 재탐지 → `pm_log_meta.json` 저장. C33 두 사이클(저불량 3~39 / 급등 1~74) 확인 |
| **4 (E3)** | 레짐 통계(저불량/급등 평균·표준편차) + 설명력 사다리 |
| **5 (E4)** | 급등 레짐 decay 곡선 (일·주 단위) → 차트 저장 |
| **6 (E5)** | 시간대(hour) 프로파일 + 요일(dow) confound → 차트 저장 |
| **7 (E6)** | 특수 레시피 C6_1 효과 크기(경과일 ±3일 매칭 비교) |
| **8 (E7)** | 센서 confound 전수 스캔(전체/저불량/급등 상관) → CSV 저장 |
| **9 (E8)** | C59/C60 step4 구조(상호배타·비율·within 상관) → 차트 저장 |
| **10 (E9)** | C23×레짐 교차 / Lot 시간 스팬 |
| **11 (E10)** | v5 valid 잔차 vs 레짐 피처 상관(결합 전략 사전 진단) |
| **12** | **판정 요약표(E1~E10) + G0 선언** |

## 실행 방법

1. **위치**: 반드시 `EDA/` 폴더 안에서 `02_EDA_regime.ipynb`를 엽니다 (상대경로 `../문제1(하)` 기준).
2. **커널**: `venv (Python 3.12)` — 3.13/3.14 금지(ENVIRONMENT.md).
3. **패키지**: 기존과 동일(`pandas numpy matplotlib scikit-learn lightgbm`). 추가 설치 불필요.
4. 셀을 위에서부터 순서대로 실행(Shift+Enter). **약 1분 내** 완료.
5. 실행이 끝나면 `pm_log_meta.json`(루트)과 `assets/regime_*` 6개 그림 + `regime_confound_scan.csv`가 생성되고, 마지막 셀에 **🟢 G0 통과**가 출력됩니다.

### 차트 언어·한글 폰트

- 차트는 **한글본(`_ko`)·영어본(`_en`) 2가지가 동시에** 저장됩니다. 한쪽만 원하면 셀 1의 `CHART_LANGS = ["ko", "en"]`를 `["ko"]` 또는 `["en"]`로 바꾸세요.
- 셀 1이 한글 폰트를 자동 지정합니다(Windows 기본 **Malgun Gothic** 우선). 실행 시 `[font] 한글 폰트 적용: …`이 출력됩니다. 폰트 미발견 경고가 뜨면 영어본(`_en`)을 쓰면 됩니다(ASCII라 항상 정상).

## 이 분석 다음에는?

G0 통과 → **Phase 2(컬럼 정책, PLAN §6.2)** 는 확정 상태, 다음은 **Phase 3 — `modeling_v8/modeling_v8.ipynb`**:

```
원본 preprocessing/feature_engineering 이식 → 피처 빌드
  (그룹 A 시간/레짐: is_high_regime · high_regime_days · days_since_last_pm · C33 · hour · 교호 + 그룹 B 센서 3 + 집계 풀)
 → M0a: 복원한 pkl로 valid/test 정합 검증(≈38.3) → M0b: 재학습 재현(G1) → M1~M4
```

> **참고**: pkl 원본 모델의 피처명은 `is_post_pm`/`post_pm_days`지만, Phase 3부터는 major/minor를 구분하는 `is_high_regime`/`high_regime_days`로 대체합니다(high_regime_days·split은 현재 데이터 동일값, `is_post_pm`은 폐기). 결과 보고는 `02_EDA_regime_REPORT.md`(§상단 배경 정정 포함)를 참조하세요.
