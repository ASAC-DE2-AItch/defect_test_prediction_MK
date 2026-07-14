# 결합 모델 (FULL 192피처) 핸드오프 패키지

웨이퍼 불량(C65) 예측 결합 모델 — **row-level 센서 + WF 집계 + 시간/레짐 피처**.
이 배포판은 **다이어트 전 FULL 버전(192피처, valid RMSE 48.76)** 이다.
⚠️ `C65` = 불량 비트 수 → **낮을수록 좋음**.

## 이게 어떤 모델인가 (한눈에)

- **입력 단위:** 원본은 웨이퍼(C64)가 여러 step(row)으로 흩어진 형태.
- **예측 방식:** step(row) 단위로 예측 → 웨이퍼 내 평균 = 웨이퍼 예측값.
- **피처(192개):** 원본 센서 36 + WF 집계 144(센서×mean/std/min/max) + 구조 2(wf_nrows,row_pos)
  + 범주형 2(C6 레시피, C7 step) + 시간/레짐 7 + C23 타깃인코딩 1. 상세는 **`COMBINED_FULL_FEATURES.md`**.
- **모델 실체:** 5-fold GroupKFold(C64) LightGBM 앙상블 + fold별 C23 인코더를 `CombinedModel` 객체로 묶음.

## 폴더 구성

```
combined_full_handover/
├── README.md                       ← 이 파일
├── COMBINED_FULL_FEATURES.md       ← 192개 피처 목록·정체·생성식
├── _feature_list.json              ← 학습 순서 피처명 전체 (기계 판독용)
├── requirements.txt
├── src/
│   ├── config.py                   ← 컬럼 분류·경로 정의
│   ├── preprocessing.py            ← 결측/상수/중복 컬럼 제거 + datetime + 정렬
│   ├── feature_engineering.py      ← WF 시간/레짐 피처 생성 (pm_log 기반)
│   ├── combined_model.py           ← build_rows() + CombinedModel (FULL: DROP_DIET=[])
│   └── predict.py                  ← 예측 실행 스크립트
├── models/
│   └── combined_model_full.pkl     ← 학습 완료 모델 (30MB)
└── data/processed/
    ├── pm_log.json                 ← PM 이벤트 날짜 (시간피처 계산 입력)
    └── pm_bins.json                ← 하위호환용 구간 경계
```

## 실행 방법 (바로 예측)

원본 형식(train_data.csv 처럼 step 단위, C64/C40/센서 컬럼 포함)의 CSV만 있으면 된다.
C65(정답)는 있어도/없어도 됨.

```bash
pip install -r requirements.txt
python src/predict.py <입력csv> [출력csv]

# 예 (정답 포함 파일 → RMSE도 참고 출력)
python src/predict.py data/raw/train_data.csv predictions.csv
```

출력 `predictions.csv`: `wafer_id, predicted_C65`.

## 코드로 예측 (임베드)

```python
import sys; sys.path.insert(0, "src")
from predict import predict
out = predict("어떤_원본형식.csv", "out.csv")   # DataFrame 반환
```

## 주의 / 전제

- **입력 스키마**는 원본 train_data.csv와 동일해야 한다(같은 컬럼명 C1~C65, C40 타임스탬프 등).
  일부 컬럼(결측/상수/중복 23개)은 preprocessing에서 자동 제거되므로 있어도 무방.
- **시간피처**는 `data/processed/pm_log.json`의 PM 날짜를 기준으로 계산된다.
  새 PM이 있었으면 이 파일에 날짜를 추가(`{"date":"YYYY-MM-DD","type":"major"}`)하면 코드 수정 없이 반영.
- 이 모델은 **원본 대회 train으로 학습**됨. 처음부터 재학습하려면 원본 train 데이터 + WF 파이프라인 산출물이 필요(원 프로젝트 참고).
- 재현성: seed 고정(42). 같은 입력이면 같은 예측.

## 성능 참고

| 평가 | RMSE |
|------|------|
| 원본 valid (인터리브 분할) | 48.76 |
| 원본 test | 49.15 |

(참고: 원 프로젝트에는 센서 다이어트로 46피처까지 줄인 경량판 D4b(valid 44.30)도 있으나, 이 배포판은 FULL 192피처 버전이다.)
