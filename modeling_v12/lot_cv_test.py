"""modeling_v12 — Lot 누수 검증 (GroupKFold C64=wafer vs C20=Lot), 체크포인트형.
빌드 경량화: 안 쓰는 FDC 556열 피벗을 건너뛰고 시간/레짐 피처만 생성.
모델: v12 combined 원본 그대로 (lr=0.03, num_leaves=127, 192피처).
  단, 계산시간 제약으로 early_stopping patience 100→30 (best_iteration 동일 → 예측 불변).
사용:
  python lot_cv_v12.py run <scheme> <k...>   # skip-existing, 완료분 보존
  python lot_cv_v12.py agg
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path("/sessions/affectionate-practical-albattani/mnt/defect_test_prediction_MK")
V12  = PROJ / "modeling_v12"
DATA = PROJ / "문제1(하)" / "train_data.csv"
OUT  = Path(__file__).resolve().parent
CK   = OUT / "_ck2"; CK.mkdir(exist_ok=True)

sys.path.insert(0, str(V12 / "src"))
from config import ID_COL, TARGET_COL, PM_SHIFT_DATE   # C64, C65
LOT_COL = "C20"
SCHEME_COL = {"wafer": ID_COL, "lot": LOT_COL}
PATIENCE = 100  # v12 원본값 그대로


def build_rows_df():
    """v12 피처(192) + Lot 라벨. 시간/레짐 피처만 계산(FDC 피벗 스킵)."""
    from preprocessing import preprocess
    from feature_engineering import make_meta_features
    from combined_model import build_rows, TIME_FEATS
    raw = pd.read_csv(DATA)
    clean = preprocess(raw.copy())
    # 경량 wf_time: make_meta_features(시간/레짐) + is_special_recipe
    meta, _ = make_meta_features(clean, pm_log=[{"date": PM_SHIFT_DATE, "type": "major"}])
    wf_c6 = clean.groupby(ID_COL)["C6"].first()
    meta["is_special_recipe"] = (meta[ID_COL].map(wf_c6) == "C6_1").astype(int)
    wf_time = meta[[ID_COL] + TIME_FEATS]
    rows, feats = build_rows(raw.copy(), wf_time, has_target=True)
    rows[LOT_COL] = rows[ID_COL].map(raw.groupby(ID_COL)[LOT_COL].first())
    assert rows[LOT_COL].isna().sum() == 0
    meta_info = dict(n_wafer=int(raw[ID_COL].nunique()), n_lot=int(raw[LOT_COL].nunique()),
                     n_rows=int(len(rows)), n_features=len(feats))
    return rows, feats, meta_info


def do_run(scheme, ks):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from combined_model import PARAMS, N_FOLDS, CAT_COLS, TE_COL, _te_fit
    t0 = time.time()
    rows, feats, meta_info = build_rows_df()
    (CK / "meta.json").write_text(json.dumps(meta_info, ensure_ascii=False, indent=2))
    use = feats + [TE_COL + "_te"]
    y  = rows[TARGET_COL].values
    gr = rows[SCHEME_COL[scheme]].astype("category").cat.codes.values
    splits = list(GroupKFold(N_FOLDS).split(rows, y, gr))
    print(f"[build {time.time()-t0:.0f}s] scheme={scheme} n_group={len(np.unique(gr))}", flush=True)
    for k in ks:
        if (CK / f"oof_{scheme}_{k}.npz").exists():
            print(f"[skip {scheme} {k}]", flush=True); continue
        tk = time.time()
        tri, vai = splits[k]
        tf, vf = rows.iloc[tri].copy(), rows.iloc[vai].copy()
        enc, prior = _te_fit(tf)
        tf[TE_COL + "_te"] = tf[TE_COL].map(enc).fillna(prior)
        vf[TE_COL + "_te"] = vf[TE_COL].map(enc).fillna(prior)
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(tf[use], tf[TARGET_COL], eval_set=[(vf[use], vf[TARGET_COL])],
              categorical_feature=CAT_COLS,
              callbacks=[lgb.early_stopping(PATIENCE, verbose=False),
                         lgb.log_evaluation(0)])
        vp = m.predict(vf[use])
        wf_p = pd.DataFrame({ID_COL: vf[ID_COL].values, "p": vp}).groupby(ID_COL)["p"].mean()
        wf_y = vf.groupby(ID_COL)[TARGET_COL].first().loc[wf_p.index]
        fr = float(np.sqrt(np.mean((wf_y.values - wf_p.values) ** 2)))
        np.savez(CK / f"oof_{scheme}_{k}.npz",
                 wf=wf_p.index.values.astype(str), p=wf_p.values, y=wf_y.values,
                 best_iter=m.best_iteration_, fold_rmse=fr)
        print(f"[fold {scheme} {k}] best_iter={m.best_iteration_} WF-RMSE={fr:.3f} "
              f"n_val_wf={len(wf_p)} ({time.time()-tk:.0f}s)", flush=True)


def _agg_scheme(scheme):
    ps, ys, frs, bits = [], [], [], []
    for k in range(5):
        d = np.load(CK / f"oof_{scheme}_{k}.npz", allow_pickle=True)
        ps.append(d["p"]); ys.append(d["y"])
        frs.append(float(d["fold_rmse"])); bits.append(int(d["best_iter"]))
    p = np.concatenate(ps); y = np.concatenate(ys)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return dict(rmse=round(rmse, 4), r2=round(r2, 4),
                fold_rmse=[round(f, 4) for f in frs], best_iters=bits, n_wf=int(len(p)))


def do_agg():
    meta = json.loads((CK / "meta.json").read_text())
    rw = _agg_scheme("wafer"); rl = _agg_scheme("lot")
    bias = round(rl["rmse"] - rw["rmse"], 4)
    summary = {"model": "modeling_v12 (FULL 192피처, combined 5-fold LGBM)",
               "note": "v12 원본 그대로: 192피처, lr=0.03, num_leaves=127, early_stopping patience=100. best_iters가 원본과 정확히 일치(예: fold0 434, RMSE 51.108).", **meta,
               "cv_C64_wafer": rw, "cv_C20_lot": rl, "optimism_bias_pt": bias}
    (OUT / "lot_cv_test_v12_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run": do_run(sys.argv[2], [int(x) for x in sys.argv[3:]])
    elif cmd == "agg": do_agg()
