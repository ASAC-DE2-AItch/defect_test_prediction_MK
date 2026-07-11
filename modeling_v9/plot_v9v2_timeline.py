#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modeling_v9 v2 (gpu_master_v5_optimal_lot_optuna_v2) LightGBM 재현 — valid/test 개형.

v2 파이프라인은 (1) Fast Ablation으로 피처 부분집합을 확정한 뒤 (2) 그 부분집합에만
Deep Optuna를 건다. 보고된 LGBM OOF RMSE 33.27은 "ablation 선택 피처 + Optuna best_params"
조합의 값이다. 이 스크립트는 그 구성을 그대로 재현한다:
  · Phase 1 피처엔지니어링 이식(LOT 집계 누수·타깃 클리핑·결측지시자·상호작용)
  · Fast Ablation(LGBM 디폴트) 재현 → 피처 부분집합 확정
  · 주신 Optuna best_params로 5-fold(early stop) 학습 → OOF RMSE 확인(≈33.27) +
    fold 모델 평균으로 valid/test 예측
  · 지금까지와 같은 시간축 타임라인 개형(valid 위 / test 아래)

  입력(modeling_v9/ 기준):
    ../문제1(하)/{train_data,valid_X,test_X}.csv
    ../문제1_하_answer/{valid,test}_Y_answer.csv
  출력:
    outputs/timeline_v9v2_LGBM.png

사용법: modeling_v9/ 폴더에 두고  python plot_v9v2_timeline.py

⚠️ LOT 집계 누수는 원본 유지하되 전이만 교정한다(train에서 fit → LOT id로 valid/test 매핑).
   OOF 33.27은 train·누수 기준이라 valid/test 개형의 제목 RMSE는 그보다 높게 찍힌다.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# ── 경로 ────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent            # modeling_v9/
DATA = HERE.parent / "문제1(하)"
ANS  = HERE.parent / "문제1_하_answer"
OUT  = HERE / "outputs"

# ── 상수 (원본 Phase 1) ─────────────────────────────────────
ID_COL, STEP_COL, TARGET_COL, LOT_COL = 'C64', 'C7', 'C65', 'C20'
SORT_COL = 'C40'
FMT = '%Y-%m-%d %H:%M:%S.%f'
SENSORS = ['C1','C9','C11','C12','C15','C16','C17','C18','C25','C27',
           'C31','C32','C48','C52','C57','C58','C59','C60','C61','C62','C63']
LOT_SENSORS = ['C17','C63','C62','C61','C12','C16']
STEPS = [1, 2, 3, 4, 5, 6]

PM_TS    = pd.Timestamp("2018-12-24 01:31:22.7")
C6_BATCH = "C6_1"

# ── Optuna best_params (사용자 제공, LGBM OOF 33.2721) ───────
BEST_PARAMS = dict(
    learning_rate=0.07317315611886971,
    num_leaves=132,
    max_depth=10,
    subsample=0.853135480771314,
    colsample_bytree=0.5346086478009537,
    n_estimators=1000, random_state=42, n_jobs=-1, verbose=-1,
)
# Fast Ablation용 디폴트 (원본 Phase 2와 동일)
ABL_PARAMS = dict(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)


# ── 피처 빌더 (원본 Phase 1 + LOT 전이 교정) ────────────────
def build_features(df, lot_table=None):
    df = df.copy()
    wf_lot = df.groupby(ID_COL)[LOT_COL].first()
    idx = wf_lot.index
    feat = pd.DataFrame(index=idx)
    missing_counts = df.groupby(ID_COL)[SENSORS].apply(lambda x: x.isna().sum())
    for s in SENSORS:
        feat[f'{s}_is_missing'] = (missing_counts[s] > 0).astype(int)
    for step in STEPS:
        grp = df[df[STEP_COL] == step].groupby(ID_COL)[SENSORS]
        feat = feat.join(grp.mean().add_prefix(f's{step}_mean_'))
        feat = feat.join(grp.std().add_prefix(f's{step}_std_'))
    s4 = df[df[STEP_COL] == 4].groupby(ID_COL)[SENSORS].mean().reindex(idx)
    s4[LOT_COL] = wf_lot
    if lot_table is None:                       # train: within-set 집계 → 테이블 저장
        lot_table = {}
        for s in LOT_SENSORS:
            lot_table[f'LOT_mean_{s}'] = s4.groupby(LOT_COL)[s].mean()
            lot_table[f'LOT_std_{s}']  = s4.groupby(LOT_COL)[s].std()
    for s in LOT_SENSORS:                        # LOT id 매핑(train=자기, valid/test=train)
        feat[f'LOT_mean_{s}'] = s4[LOT_COL].map(lot_table[f'LOT_mean_{s}']).values
        feat[f'LOT_std_{s}']  = s4[LOT_COL].map(lot_table[f'LOT_std_{s}']).values
    feat['C17_x_C63'] = s4['C17'] * s4['C63']
    feat['C12_x_C62'] = s4['C12'] * s4['C62']
    return feat, lot_table


def clip_target(y_raw):
    return np.clip(y_raw, a_min=None, a_max=np.sort(y_raw)[-2])


def wf_time_c6(df):
    d = df.copy(); d[SORT_COL] = pd.to_datetime(d[SORT_COL], format=FMT)
    g = d.sort_values(SORT_COL).groupby(ID_COL)
    return pd.DataFrame({'ts': g[SORT_COL].first(),
                         'is_special_recipe': (g['C6'].agg(lambda s: s.mode().iloc[0]) == C6_BATCH).astype(int)})


def fast_ablation_lgbm(X, y, kf):
    """원본 Phase 2 재현: 디폴트 LGBM importance 랭킹 → top-N(1..25) OOF 최소 부분집합."""
    imp_model = lgb.LGBMRegressor(**ABL_PARAMS).fit(X, y)
    ranked = pd.DataFrame({'f': X.columns, 'imp': imp_model.feature_importances_}) \
                .sort_values('imp', ascending=False)['f'].tolist()
    best_rmse, best_n = float('inf'), 0
    for n in range(1, 26):
        cur = ranked[:n]; oof = np.zeros(len(y))
        for tr, va in kf.split(X):
            m = lgb.LGBMRegressor(**ABL_PARAMS)
            m.fit(X[cur].iloc[tr], y[tr], eval_set=[(X[cur].iloc[va], y[va])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            oof[va] = m.predict(X[cur].iloc[va])
        rmse = np.sqrt(mean_squared_error(y, oof))
        if rmse < best_rmse: best_rmse, best_n = rmse, n
    return ranked[:best_n], best_n, best_rmse


# ── 플롯 ────────────────────────────────────────────────────
def set_korean_font():
    import matplotlib.font_manager as fm
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if name in installed:
            matplotlib.rcParams["font.family"] = name; break
    matplotlib.rcParams["axes.unicode_minus"] = False


def draw_panel(ax, d, name):
    rmse = np.sqrt(mean_squared_error(d["true"], d["pred"]))
    ax.scatter(d["ts"], d["true"], s=6, c="0.7", alpha=0.5, label="실측 (개별 WF)", zorder=1)
    r = d.set_index("ts")[["true", "pred"]].resample("12h").mean().dropna()
    ax.plot(r.index, r["true"], c="black",    lw=1.4, label="실측 (12h 평균)", zorder=3)
    ax.plot(r.index, r["pred"], c="tab:blue", lw=1.4, label="예측 (12h 평균)", zorder=3)
    ax.axvline(PM_TS, color="red", ls="--", lw=1.3, zorder=2)
    ax.text(PM_TS, ax.get_ylim()[1], " PM (레짐 전환)", color="red", va="top", ha="left", fontsize=9)
    c6 = d[d["is_special_recipe"] == 1]["ts"]
    if len(c6):
        lo, hi = c6.min(), c6.max()
        ax.axvspan(lo, hi, color="orange", alpha=0.18, zorder=0)
        ax.text(lo + (hi - lo) / 2, ax.get_ylim()[0], "C6_1\n배치",
                color="darkorange", va="bottom", ha="center", fontsize=8)
    ax.set_title(f"[{name}] 전체 기간 실측 vs 예측 — RMSE {rmse:.2f} (n={len(d):,})",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("C65 (불량 비트 수)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9); ax.grid(alpha=0.25)
    return rmse


def make_split(raw_df, feat_pred, ans_path):
    tc = wf_time_c6(raw_df)
    ans = pd.read_csv(ans_path).set_index(ID_COL)[TARGET_COL].rename("true")
    d = tc.join(feat_pred.rename("pred")).join(ans).dropna(subset=["ts", "pred", "true"])
    return d.sort_values("ts")


def main():
    set_korean_font(); OUT.mkdir(exist_ok=True)
    print("데이터 로드 & 피처 빌드 중…")
    train_raw = pd.read_csv(DATA / "train_data.csv")
    valid_raw = pd.read_csv(DATA / "valid_X.csv")
    test_raw  = pd.read_csv(DATA / "test_X.csv")

    Ftr, lot_table = build_features(train_raw)
    y = clip_target(train_raw.groupby(ID_COL)[TARGET_COL].first().reindex(Ftr.index).to_numpy(float))
    train_cols = Ftr.columns
    Xtr = Ftr.replace([np.inf, -np.inf], np.nan); med = Xtr.median(); Xtr = Xtr.fillna(med).fillna(0)

    Fva, _ = build_features(valid_raw, lot_table)
    Fte, _ = build_features(test_raw,  lot_table)
    Xva = Fva.reindex(columns=train_cols).replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
    Xte = Fte.reindex(columns=train_cols).replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)

    kf = KFold(5, shuffle=True, random_state=42)
    print("Fast Ablation (피처 부분집합 확정)…")
    feats, best_n, abl_rmse = fast_ablation_lgbm(Xtr, y, kf)
    print(f" -> 피처 {best_n}개 확정 (ablation OOF RMSE {abl_rmse:.4f})")

    # 확정 피처 + best_params → 5-fold(early stop): OOF 확인 + fold평균으로 valid/test 예측
    oof = np.zeros(len(y)); va_pred = np.zeros(len(Xva)); te_pred = np.zeros(len(Xte))
    for tr, va in kf.split(Xtr):
        m = lgb.LGBMRegressor(**BEST_PARAMS)
        m.fit(Xtr[feats].iloc[tr], y[tr], eval_set=[(Xtr[feats].iloc[va], y[va])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va] = m.predict(Xtr[feats].iloc[va])
        va_pred += m.predict(Xva[feats]) / kf.n_splits
        te_pred += m.predict(Xte[feats]) / kf.n_splits
    print(f" -> best_params OOF RMSE {np.sqrt(mean_squared_error(y, oof)):.4f}  (보고값 33.27 대조)")

    dv = make_split(valid_raw, pd.Series(va_pred, index=Xva.index), ANS / "valid_Y_answer.csv")
    dt = make_split(test_raw,  pd.Series(te_pred, index=Xte.index), ANS / "test_Y_answer.csv")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    r_v = draw_panel(axes[0], dv, "valid"); r_t = draw_panel(axes[1], dt, "test")
    axes[1].set_xlabel("날짜")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    fig.suptitle("modeling_v9 v2 (v9 피처 · LightGBM Optuna, OOF 33.27) 전체 기간 예측 개형 (valid / test)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT / "timeline_v9v2_LGBM.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[v9v2·LGBM] valid RMSE {r_v:.2f} | test RMSE {r_t:.2f}  → {out}")


if __name__ == "__main__":
    main()
