"""
combined_lotte.py — combined(192) + Lot 타깃인코딩(C20_te)  [대회 점수용]
──────────────────────────────────────────────────────────────────────
combined_model_full_handover 의 row-level 피처(192개)에 **Lot 단위 타깃인코딩**
C20_te(그 Lot 의 평균 C65) 1개만 추가한다. valid/test 가 train 과 Lot 을
99.9~100% 공유하는 대회 구조를 직접 활용해 점수를 크게 낮춘다.

검증 결과 (이 저장소 데이터):
    combined(192)          : OOF 49.86 / valid 48.68 / test 48.65
    combined + C20_te      : OOF 34.87 / valid 34.29 / test 34.33   ← −14pt

⚠️ 누수 주의 — 이 점수는 valid/test 가 train 과 Lot 을 공유해서 나온다.
   신규 Lot(GroupKFold C20) 기준으론 70~100대이며 실무 일반화가 안 된다.
   프로젝트 원칙(Lot 피처 금지)과 배치되는 '대회 점수 전용' 변형이다.

이 파일은 combined_model_full_handover/src/ 안(= config.py 옆)에 둔다.
데이터 4경로만 아래 PATHS 에 맞춰주면 바로 실행:  python src/combined_lotte.py
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent                      # combined_model_full_handover/
sys.path.insert(0, str(SRC))

from preprocessing import preprocess
from feature_engineering import build_features
from combined_model import (build_rows, PARAMS, CAT_COLS, N_FOLDS,
                            EARLY_STOPPING, TE_COL)
from config import ID_COL, TARGET_COL
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

# ── 데이터 경로 (환경에 맞게 수정) ─────────────────────────────────
PATHS = {
    "train":     ROOT / "data" / "raw" / "train_data.csv",
    "valid_X":   ROOT.parent / "문제1(하)" / "valid_X.csv",
    "valid_ans": ROOT.parent / "문제1_하_answer" / "valid_Y_answer.csv",
    "test_X":    ROOT.parent / "문제1(하)" / "test_X.csv",
    "test_ans":  ROOT.parent / "문제1_하_answer" / "test_Y_answer.csv",
}
PM_BINS = np.array(json.load(open(ROOT / "data/processed/pm_bins.json")))
PM_LOG  = json.load(open(ROOT / "data/processed/pm_log.json"))


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def rows_of(path, has_t):
    """원본 CSV → combined row-level 피처 + Lot(C20) 부착."""
    raw = pd.read_csv(path)
    clean = preprocess(raw.copy())
    wt, _ = build_features(clean, pm_bins=PM_BINS, pm_log=PM_LOG)
    x = raw if has_t else raw[[c for c in raw.columns if c != TARGET_COL]]
    rows, feats = build_rows(x, wt, has_target=has_t)
    lot = raw.groupby(ID_COL)["C20"].first().rename("C20")
    rows = rows.merge(lot, on=ID_COL, how="left")   # build_rows 는 C20 제외 → 따로 부착
    return rows, feats


def _te_fit(frame, col, m=20):
    prior = frame[TARGET_COL].mean()
    agg = frame.groupby(col)[TARGET_COL].agg(["mean", "count"])
    enc = (agg["mean"] * agg["count"] + prior * m) / (agg["count"] + m)
    return enc, prior


def train_eval(use_lot_te: bool):
    rows_tr, feats = rows_of(PATHS["train"], True)
    rows_va, _ = rows_of(PATHS["valid_X"], False)
    rows_te, _ = rows_of(PATHS["test_X"], False)
    use = feats + [TE_COL + "_te"] + (["C20_te"] if use_lot_te else [])
    y = rows_tr[TARGET_COL].values
    gr = rows_tr[ID_COL].astype("category").cat.codes.values
    for r in (rows_tr, rows_va, rows_te):
        for c in CAT_COLS:
            r[c] = r[c].astype("category")

    oof = np.zeros(len(rows_tr))
    va_p = np.zeros(len(rows_va)); te_p = np.zeros(len(rows_te))
    for tri, vai in GroupKFold(N_FOLDS).split(rows_tr, y, gr):
        tf, vf = rows_tr.iloc[tri].copy(), rows_tr.iloc[vai].copy()
        vae, tee = rows_va.copy(), rows_te.copy()
        # C23 타깃인코딩 (combined 기본)
        e23, p23 = _te_fit(tf, TE_COL)
        for d in (tf, vf, vae, tee):
            d[TE_COL + "_te"] = d[TE_COL].map(e23).fillna(p23)
        # Lot(C20) 타깃인코딩 (대회 점수 레버)
        if use_lot_te:
            e20, p20 = _te_fit(tf, "C20")
            for d in (tf, vf, vae, tee):
                d["C20_te"] = d["C20"].map(e20).fillna(p20)
        mdl = lgb.LGBMRegressor(**PARAMS)
        mdl.fit(tf[use], tf[TARGET_COL], eval_set=[(vf[use], vf[TARGET_COL])],
                categorical_feature=CAT_COLS,
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                           lgb.log_evaluation(0)])
        oof[vai] = mdl.predict(vf[use])
        va_p += mdl.predict(vae[use]) / N_FOLDS
        te_p += mdl.predict(tee[use]) / N_FOLDS

    def wf(rows, p):
        return pd.DataFrame({ID_COL: rows[ID_COL].values, "p": p}).groupby(ID_COL)["p"].mean()
    oof_wf = wf(rows_tr, oof); y_wf = rows_tr.groupby(ID_COL)[TARGET_COL].first()
    va_wf = wf(rows_va, va_p); te_wf = wf(rows_te, te_p)
    ansv = pd.read_csv(PATHS["valid_ans"]).set_index("C64")["C65"]
    anst = pd.read_csv(PATHS["test_ans"]).set_index("C64")["C65"]
    iv = ansv.index.intersection(va_wf.index); it = anst.index.intersection(te_wf.index)
    return (rmse(y_wf.loc[oof_wf.index], oof_wf),
            rmse(ansv.loc[iv], va_wf.loc[iv]),
            rmse(anst.loc[it], te_wf.loc[it]),
            va_wf, te_wf)


if __name__ == "__main__":
    print("combined(192) + Lot 타깃인코딩(C20_te):")
    o, v, t, va_wf, te_wf = train_eval(use_lot_te=True)
    print(f"  OOF={o:.3f}  valid={v:.3f}  test={t:.3f}")
    out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    va_wf.rename("predicted_C65").reset_index().to_csv(out / "valid_lotte_submit.csv", index=False)
    te_wf.rename("predicted_C65").reset_index().to_csv(out / "test_lotte_submit.csv", index=False)
    print(f"  제출 저장: {out}/valid_lotte_submit.csv, test_lotte_submit.csv")
