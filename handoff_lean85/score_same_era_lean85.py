# -*- coding: utf-8 -*-
"""
score_same_era_lean85.py — lean-85 valid/test same-era 홀드아웃 채점

baseline_vanilla 와 **동일 축**(train 전체 학습 → valid/test 정답 직접 채점)에서
lean-85(XGBoost 동결 85피처)의 홀드아웃 점수를 산출한다.

⚠️ 축 주의: 이 수치는 same-era(후향) 홀드아웃이다. valid/test 는 train 과 같은 시기
   (신규 Lot 0.1%)라 lot-mate 로 낙관될 수 있다. **현업 인용 수치는 시간축 99.84** 이며
   본 수치는 baseline 대비용(기록/부록)으로만 쓴다.

사용: 프로젝트 venv(xgboost 3.3.0), handoff_lean85/ 에서
    python score_same_era_lean85.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import lean85_pipeline as lp

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "문제1(하)"
ANS = ROOT / "문제1_하_answer"
FMT = "%Y-%m-%d %H:%M:%S.%f"


def _restore_c40(raw):
    """test_X 의 C40 엑셀 손상('MM:SS.f') → C10(unix epoch)+9h(KST) 복원."""
    if raw["C40"].astype(str).str.match(r"\d{4}-\d\d-\d\d").mean() >= 0.5:
        return raw
    raw = raw.copy()
    c10 = pd.to_numeric(raw["C10"], errors="coerce")
    raw["C40"] = (pd.to_datetime(c10, unit="s") + pd.Timedelta(hours=9)).dt.strftime(FMT).str.slice(0, -3)
    return raw


def main():
    lean, params, rounds = lp.load_frozen()
    pm = lp.find_file("pm_log.json")
    sigma = lp.SIGMA_C65

    tr = lp.build_wafer_table(pd.read_csv(DATA / "train_data.csv", low_memory=False), pm, lean)
    model = lp.fit_lean85(tr, lean, params, rounds)
    print(f"lean-85 XGB {rounds}r · xgboost {lp.xgb.__version__} · train {len(tr)} WF\n")

    rows = []
    for split in ["valid", "test"]:
        raw = pd.read_csv(DATA / f"{split}_X.csv", low_memory=False)
        raw = raw.drop(columns=[c for c in ["C64.1", "C65"] if c in raw.columns], errors="ignore")
        t = lp.build_wafer_table(_restore_c40(raw), pm, lean)
        ans = pd.read_csv(ANS / f"{split}_Y_answer.csv")[["C64", "C65"]]
        t = t.merge(ans, on="C64", how="left", suffixes=("", "_a"))
        y = (t["C65_a"] if "C65_a" in t else t["C65"]).to_numpy(float)
        pred = model.predict(t[lean])
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        rows.append((split, len(t), rmse, 1 - (rmse / sigma) ** 2))
        print(f"{split}: n={len(t)} | 홀드아웃 RMSE {rmse:.3f} | honest R² {1 - (rmse/sigma)**2:.4f}")

    print("\n[축 주의] same-era 후향 홀드아웃 — baseline(test 85.0) 대비용. 현업 인용은 시간축 99.84.")
    pd.DataFrame(rows, columns=["split", "n", "rmse_holdout", "R2"]).to_csv(
        Path(__file__).resolve().parent / "lean85_same_era_holdout.csv", index=False)


if __name__ == "__main__":
    main()
