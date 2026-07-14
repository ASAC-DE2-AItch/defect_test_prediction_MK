# colab_GA — GA 2차 선별 (Colab 실행용 자기완결 폴더)

로컬이 느려 **GA만 Colab에서** 돌리기 위한 폴더. 원본 train(43MB)·v8 모듈 의존을 없애고,
core10 시간/레짐 8피처를 **미리 계산한 CSV**로 대체해 **이 폴더만 업로드하면** 실행된다.

## 폴더 내용

| 파일 | 역할 | 크기 |
|---|---|---|
| `modeling_v13_select_Conservative_GA_colab.ipynb` | Conservative + GA 노트북 | ~6KB |
| `modeling_v13_select_Balanced_GA_colab.ipynb` | Balanced + GA 노트북 | ~6KB |
| `v13_select_colab.py` | 자기완결 모듈(상수 인라인, meta는 CSV 로드) | ~6KB |
| `v13_fdc_pool_wf_oof.csv.gz` | 웨이퍼 풀(diet 소스 + fold + C20 + C59/C60) | 5.8MB |
| `core10_meta_wf.csv` | C64 + 시간/레짐 8피처(미리 계산) | 0.9MB |
| `feature_diet_selected.json` | M1 diet 선택 목록 + champion | 7KB |

## 실행 방법

노트북 **첫 셋업 셀**이 지원 파일 4개(`v13_select_colab.py`, `v13_fdc_pool_wf_oof.csv.gz`,
`core10_meta_wf.csv`, `feature_diet_selected.json`)를 흔한 위치에서 **자동 탐색**하고,
못 찾으면 **업로드 창**을 띄운다. 아래 중 편한 방법을 쓰면 된다.

**A. 업로드 창 (가장 간단 · 권장)**
1. `..._GA_colab.ipynb` 를 Colab에서 연다.
2. **셋업 셀 실행** → 파일을 못 찾으면 업로드 버튼이 뜬다 → 4개 파일을 **한 번에 선택**해 올린다.
3. 이후 셀 순서대로 실행. (`select_result_*.json` 이 마지막에 생성됨)

**B. 파일 패널 드래그**
1. 왼쪽 **파일 패널**에 4개 파일을 `/content` 로 드래그.
2. 셋업 셀 실행 → 자동으로 `BASE="."` 잡힘.
   (서브폴더 `/content/colab_GA` 에 올려도 자동 탐색됨)

**C. Google Drive (세션 끊겨도 유지)**
1. 이 `colab_GA` 폴더를 Drive에 업로드.
2. 셋업 셀 앞에 `from google.colab import drive; drive.mount('/content/drive')` 실행.
   셋업 셀이 `/content/drive/MyDrive/colab_GA` 등을 자동 탐색한다(다른 경로면 셀의 `CANDIDATES` 에 추가).

> ❗ 업로드 누락 에러가 뜨면 = 셋업 셀 실행 시점에 4개 파일이 아직 `/content` 에 없다는 뜻.
> 업로드 창에서 올리거나 파일 패널에 드래그한 뒤 셋업 셀을 **다시 실행**하면 된다.

## 런타임 · 주의

- **런타임 유형 = CPU** 로 충분. 이 LightGBM 경로는 **GPU/T4 이득 없음**(GPU 켜도 빨라지지 않음).
- GA는 적합도 평가마다 프록시 LGBM OOF(3-fold)를 돌려 **fit 수가 수백 회** → Colab CPU 성능에 따라 수십 분~1시간+.
  - 너무 느리면 노트북의 `pop`·`gens`·`fit_splits` 를 낮추거나, `PROXY_ROUNDS`(모듈)를 200→120 으로.
- 두 노트북은 독립 → **Colab 세션 2개로 병렬** 가능.

## 산출물

각 노트북이 `select_result_<preset>_GA.json` 생성(선택 피처 + baseline 대비 KFold/GroupKFold OOF).
파일 패널에서 로컬로 내려받아 RFECV 결과와 함께 4개(Cons/Bal × RFECV/GA) 비교에 사용.

> ⚠️ 결과 해석: 선별·최종 모두 고정 M8_PARAMS(10피처용) 기준 → **상대 우열 비교용**.
> 절대 성능은 승자 셋 고정 후 파라미터 재튜닝에서 확정.
