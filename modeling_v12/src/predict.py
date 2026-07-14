"""
predict.py — 결합 모델(FULL 192피처)로 예측하기.

원본 형식(train_data.csv 처럼 step 단위, C64/C40/센서 컬럼 포함)의 CSV를 받아
전처리 → WF 시간피처 생성 → row-level 결합 피처 → 예측(WF 단위)까지 한 번에 수행.
C65(정답)는 있어도/없어도 됨(있으면 무시하고 예측만 출력).

사용:
  python src/predict.py <입력csv> [출력csv]
  예) python src/predict.py data/raw/train_data.csv predictions.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import ID_COL, TARGET_COL
from preprocessing import preprocess
from feature_engineering import build_features, _load_pm_bins, _load_pm_log
from combined_model import build_rows, CombinedModel


def predict(input_csv: str, output_csv: str = "predictions.csv") -> pd.DataFrame:
    print(f"[1/4] 입력 로드: {input_csv}")
    raw = pd.read_csv(input_csv)

    print("[2/4] 전처리 + WF 시간피처 생성...")
    clean = preprocess(raw.copy())
    pm_bins, pm_log = _load_pm_bins(), _load_pm_log()   # 학습 시 저장값 재사용
    wf_time, _ = build_features(clean, pm_bins=pm_bins, pm_log=pm_log)

    print("[3/4] row-level 결합 피처 빌드 + 예측...")
    rows, _ = build_rows(raw[[c for c in raw.columns if c != TARGET_COL]],
                         wf_time, has_target=False)
    model = CombinedModel.load()
    print(f"  모델: {len(model.use)}개 피처 (FULL). 예측 WF 수: {rows[ID_COL].nunique():,}")
    pred = model.predict_wf(rows).rename("predicted_C65")

    print("[4/4] 저장...")
    out = pred.reset_index().rename(columns={ID_COL: "wafer_id"})
    # 정답이 입력에 있으면 RMSE도 참고 출력
    if TARGET_COL in raw.columns:
        y = raw.groupby(ID_COL)[TARGET_COL].first()
        m = out.set_index("wafer_id").join(y.rename("actual"))
        rmse = float(np.sqrt(np.mean((m["actual"] - m["predicted_C65"]) ** 2)))
        print(f"  (참고) 입력에 정답 존재 → RMSE {rmse:.4f}")
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {output_csv} ({len(out):,} WF)")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python src/predict.py <입력csv> [출력csv]")
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "predictions.csv")
