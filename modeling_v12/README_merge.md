# combined_model_full_handover ⊕ modeling_v9 — 피처 결합 모듈

두 모델의 피처를 **다이어트 없이 그대로** 합치는 모듈. 두 세트의 granularity가
달라서 결합 뼈대를 두 가지로 제공한다.

## 파일

| 파일 | 역할 |
|---|---|
| `v9_features.py` | modeling_v9 의 테이블형 피처(266개)를 웨이퍼(C64) 단위로 재현. `V9FeatureBuilder`(fit/transform) 포함 — train에서 컬럼·결측대치값 확정 후 valid/test에 동일 적용. **1D-CNN·타깃clipping은 제외**(피처가 아니라 모델/타깃 처리). |
| `merge_features.py` | `merge_row_level()`(A안) / `merge_wafer_level()`(B안) 두 결합 함수. |

두 파일을 `combined_model_full_handover/src/` 안(= `config.py` 옆)에 두면 import가 바로 된다.

## 두 결합 방식

- **A) row 뼈대** — combined 의 row(step) 프레임에 v9 웨이퍼피처를 C64로 broadcast.
  combined 192개 **전부 보존** + v9 266개 추가 = **457피처**. combined 아키텍처(GroupKFold, row예측→WF평균) 유지.
- **B) 웨이퍼 뼈대** — v9 웨이퍼 테이블에 combined 의 웨이퍼단위 피처(WF집계·구조·시간/레짐·C6·C23)를 결합 = **419피처**.
  combined 의 row-level 원본센서 36·row_pos·C7 은 웨이퍼 단위값이 없어 자연 제외(정보는 `_wf_` 집계에 이미 포함 — 다이어트 아님).

## 사용 예

```python
import sys, json; sys.path.insert(0, "combined_model_full_handover/src")
import numpy as np, pandas as pd
from merge_features import make_wf_time, merge_row_level, merge_wafer_level
from v9_features import V9FeatureBuilder

pm_bins = np.array(json.load(open("combined_model_full_handover/data/processed/pm_bins.json")))
pm_log  = json.load(open("combined_model_full_handover/data/processed/pm_log.json"))

raw_tr = pd.read_csv("train_data.csv")
wt_tr  = make_wf_time(raw_tr, pm_bins, pm_log)

vb = V9FeatureBuilder()
# B안(웨이퍼 뼈대): 더 정확했던 결합
tbl_tr, cols = merge_wafer_level(raw_tr, wt_tr, vb, fit_v9=True, has_target=True)
# valid 는 같은 vb 로 transform (fit_v9=False)
```

빠른 확인: `python merge_features.py <train_csv>` → 두 뼈대의 (행 수, 피처 수) 출력.

## ⚠️ 정확도 결과 — 반드시 확인

공통 조건(LightGBM 5-fold, combined 하이퍼파라미터)으로 valid RMSE 비교:

| 방식 | valid RMSE | CV(OOF) |
|---|---|---|
| A0) combined FULL(192) — 기준 | **48.68** | 49.86 |
| A) combined + v9 (row 뼈대) | 67.44 | 35.01 |
| B0) v9 only (웨이퍼 뼈대) — 기준 | 108.51 | 42.95 |
| B) v9 + combined (웨이퍼 뼈대) | **65.10** | 37.02 |

**두 결합 중엔 B(웨이퍼 뼈대, 65.10)가 더 정확**하지만, **둘 다 combined 단독(48.68)보다 valid가 나쁘다.**
CV(35~37)와 valid(65~108)의 큰 격차 = 전형적 **lot-mate 누수** 신호.

원인: modeling_v9 의 **LOT 집계 피처(`LOT_mean_*`, `LOT_std_*`, C20 기반)**.
modeling_v9 주석 그대로 "LOT 집계 누수 허용(최고 점수 방식)" — 같은 LOT 웨이퍼 정보를 끌어와
CV만 낮추고 **신규 LOT(valid)엔 일반화가 안 된다.** 프로젝트 원칙(`CLAUDE.md`: "Lot ID(C20/C21/C22)는
피처로 사용하지 않음") 및 메모리의 "lot-mate 누수 미해소"와 정확히 일치.

## 정확도를 실제로 올리려면 (권장 후속)

1. **누수 피처만 제외 재비교** — v9에서 `LOT_mean_*`/`LOT_std_*` 12개(+lot기반 교호 검토)만 빼고
   나머지 254개는 그대로 결합. (다이어트 아님 — '누수 소스'만 격리)
2. **정직한 CV로 재평가** — 랜덤 KFold 대신 **GroupKFold(C20=LOT)** 로 CV를 돌려 valid와의 격차 확인.
3. 그래도 combined 48.68을 넘지 못하면, v9의 **Step별 집계(`s{step}_mean/std_*`)** 만 선별 결합하는 방향.

> 이 모듈 자체는 요청대로 **전량 단순 결합**을 지원한다. 위 후속은 '정확도 개선'이 목표일 때의 다음 단계다.
