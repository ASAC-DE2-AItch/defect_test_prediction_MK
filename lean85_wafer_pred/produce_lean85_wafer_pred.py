# -*- coding: utf-8 -*-
"""
produce_lean85_wafer_pred.py — lean-85 wafer별 예측/잔차 산출 (다른 파트 전달용)

민지님 seg 방식 (2026-07-19 확정):
    · 병합 = train + valid_X⋈answer + test_X⋈answer = 15,919 WF, wafer 시각순 정렬
    · 경계 = C33 리셋(대PM, 2018-12-23) = wafer C64_9664 (seq 4904)
    · seg1(4,903, C65중앙 636) = 학습 지형 → 시간순 앞 80% TRAIN(3,922) / 뒤 20% VAL(981)
    · seg2(11,016, C65중앙 1082) = 홀드아웃(평가 전용, 불가침)

모델 = lean-85 (**XGBoost** 동결 85피처·246r·seed42 — 요청서의 'LightGBM'은 오기).
예측 2종 (요청: 홀드아웃·재학습 둘 다):
    · y_pred_holdout : TRAIN(3,922)로 1회 학습 → seg2 예측 (무재학습, seg2 순수 불가침)
    · y_pred_retrain : TRAIN 초기 → seg2 주간(7일) 확장창 재학습 (B2′ 롤링, 과거 seg2 편입=운영 시뮬)

시각 주의: test_X 의 C40 은 엑셀 손상('MM:SS.f') → C10(unix epoch)+9h(KST)로 복원.

출력: lean-85_wafer_pred.csv  [C64, y_true, y_pred_holdout, y_pred_retrain, is_onset]
    잔차는 수신 측에서 |y_true − y_pred_*| 로 계산.

사용: 프로젝트 venv(xgboost 3.3.0)에서
    python produce_lean85_wafer_pred.py
    (handoff_lean85/lean85_pipeline.py 를 자동 참조 — 같은 프로젝트 루트 하위)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "handoff_lean85"))
import lean85_pipeline as lp   # noqa: E402

DATA = ROOT / "문제1(하)"
ANS = ROOT / "문제1_하_answer"
BOUNDARY_WF = "C64_9664"       # C33 리셋 경계(seg2 첫 웨이퍼) — 민지님 방식
TRAIN_FRAC = 0.8               # seg1 시간순 앞 80% = TRAIN
H_DAYS = 7                     # 재학습 확장창 horizon(일)
ONSET_DAYS = 7                 # seg2 진입 후 7일 = 레짐-온셋 플래그
KST = pd.Timedelta(hours=9)
FMT = "%Y-%m-%d %H:%M:%S.%f"


def _restore_c40(raw: pd.DataFrame) -> pd.DataFrame:
    """C40 이 datetime 형식이 아니면(test_X 엑셀 손상) C10(epoch)+9h 로 복원."""
    looks_dt = raw["C40"].astype(str).str.match(r"\d{4}-\d\d-\d\d").mean()
    if looks_dt >= 0.5:
        return raw
    raw = raw.copy()
    c10 = pd.to_numeric(raw["C10"], errors="coerce")
    dt = pd.to_datetime(c10, unit="s") + KST
    raw["C40"] = dt.dt.strftime(FMT).str.slice(0, -3)   # us→ms 3자리
    return raw


def _split_table(split: str, is_train: bool) -> pd.DataFrame:
    """raw → lean-85 웨이퍼 피처 + C65(정답) + ts_epoch(C10 min, 시각정렬 기준)."""
    if is_train:
        raw = pd.read_csv(DATA / "train_data.csv", low_memory=False)
    else:
        raw = pd.read_csv(DATA / f"{split}_X.csv", low_memory=False)
        raw = raw.drop(columns=[c for c in ["C64.1", "C65"] if c in raw.columns], errors="ignore")
    raw = _restore_c40(raw)
    pm = lp.find_file("pm_log.json")
    tbl = lp.build_wafer_table(raw, pm, LEAN)                      # 85피처 + wf_ts (+C65 if train)

    c10 = pd.to_numeric(raw["C10"], errors="coerce")
    ts = raw.assign(_c10=c10).groupby(lp.ID_COL)["_c10"].min()
    tbl["ts_epoch"] = tbl[lp.ID_COL].map(ts)

    if not is_train:
        ans = pd.read_csv(ANS / f"{split}_Y_answer.csv")[[lp.ID_COL, "C65"]]
        tbl = tbl.merge(ans, on=lp.ID_COL, how="left", suffixes=("", "_a"))
        if "C65_a" in tbl:
            tbl["C65"] = tbl["C65_a"]
            tbl = tbl.drop(columns=["C65_a"])
    tbl["split"] = "train" if is_train else split
    return tbl


def main():
    global LEAN, PARAMS, ROUNDS
    LEAN, PARAMS, ROUNDS = lp.load_frozen()
    print(f"lean-85: {len(LEAN)}피처 · XGB {ROUNDS}r · xgboost {lp.xgb.__version__}")

    tr = _split_table("train", True)
    va = _split_table("valid", False)
    te = _split_table("test", False)
    allwf = pd.concat([tr, va, te], ignore_index=True)
    assert len(allwf) == 15919, f"병합 WF {len(allwf)} != 15919"
    allwf = allwf.sort_values(["ts_epoch", lp.ID_COL]).reset_index(drop=True)
    allwf["seq"] = np.arange(1, len(allwf) + 1)

    # seg 분할 — 경계 웨이퍼(C33 리셋) 위치 기준
    bpos = int(allwf.index[allwf[lp.ID_COL] == BOUNDARY_WF][0])
    seg1 = allwf.iloc[:bpos].reset_index(drop=True)
    seg2 = allwf.iloc[bpos:].reset_index(drop=True)
    print(f"seg1 {len(seg1)} (C65중앙 {seg1['C65'].median():.0f}) | "
          f"seg2 {len(seg2)} (C65중앙 {seg2['C65'].median():.0f}) | 경계 {BOUNDARY_WF} seq {bpos + 1}")
    assert len(seg1) == 4903 and len(seg2) == 11016, "seg 크기 불일치 — 방식 재확인"

    # seg1 시간순 80/20
    n_tr = int(len(seg1) * TRAIN_FRAC)
    TRAIN = seg1.iloc[:n_tr].reset_index(drop=True)
    VAL = seg1.iloc[n_tr:].reset_index(drop=True)
    print(f"TRAIN {len(TRAIN)} | VAL {len(VAL)} (VAL은 채점자, 예측대상은 seg2)")

    # (1) 무재학습 홀드아웃 — TRAIN 1회 학습 → seg2
    m_hold = lp.fit_lean85(TRAIN, LEAN, PARAMS, ROUNDS)
    pred_hold = m_hold.predict(seg2[LEAN])

    # (2) 재학습(B2′ 롤링) — TRAIN 초기 → seg2 주간 확장창(과거 seg2 편입)
    H = H_DAYS * 86400
    pred_re = np.full(len(seg2), np.nan)
    tsr = seg2["ts_epoch"].to_numpy()
    t0, tend = tsr.min(), tsr.max()
    cur = t0
    n_fold = 0
    while cur <= tend:
        m = (tsr >= cur) & (tsr < cur + H)
        if m.sum() > 0:
            past = seg2[seg2["ts_epoch"] < cur]
            trainset = pd.concat([TRAIN, past], ignore_index=True)
            mdl = lp.fit_lean85(trainset, LEAN, PARAMS, ROUNDS)
            pred_re[m] = mdl.predict(seg2.loc[m, LEAN])
            n_fold += 1
        cur += H
    assert not np.isnan(pred_re).any(), "재학습 예측 미커버 웨이퍼 존재"
    print(f"재학습 fold {n_fold}개")

    is_onset = (seg2["ts_epoch"] < t0 + ONSET_DAYS * 86400).astype(int)
    out = pd.DataFrame({
        lp.ID_COL: seg2[lp.ID_COL],
        "y_true": seg2["C65"].astype(float),
        "y_pred_holdout": np.round(pred_hold, 4),
        "y_pred_retrain": np.round(pred_re, 4),
        "is_onset": is_onset.to_numpy(),
    })
    out.to_csv(HERE / "lean-85_wafer_pred.csv", index=False, encoding="utf-8-sig")

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))
    print(f"\n저장: {HERE / 'lean-85_wafer_pred.csv'} ({len(out)} 행)")
    print(f"  홀드아웃 seg2 RMSE {_rmse(out.y_true, out.y_pred_holdout):.2f} | "
          f"재학습 seg2 RMSE {_rmse(out.y_true, out.y_pred_retrain):.2f}")
    print(f"  (온셋 제외 정착) 홀드아웃 {_rmse(out.y_true[out.is_onset==0], out.y_pred_holdout[out.is_onset==0]):.2f} | "
          f"재학습 {_rmse(out.y_true[out.is_onset==0], out.y_pred_retrain[out.is_onset==0]):.2f}")


if __name__ == "__main__":
    main()
