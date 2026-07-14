# -*- coding: utf-8 -*-
"""modeling_v13 — GA 2차 선별 (Colab 자기완결 버전).

로컬 `v13_select_common.py` 와 동일 로직이되, **원본 train / v8 모듈 의존을 제거**했다.
core10 시간·레짐 8피처는 미리 계산된 `core10_meta_wf.csv` 에서 읽는다.

같은 폴더에 필요한 파일 (모두 이 폴더에 함께 업로드):
  - v13_fdc_pool_wf_oof.csv.gz   (웨이퍼 풀; diet 소스 + fold + C20 + C59/C60 센서)
  - core10_meta_wf.csv           (C64 + 시간/레짐 8피처; 미리 계산됨)
  - feature_diet_selected.json   (M1 diet 선택 목록 + champion)

설계: 고정(core10+champion) 항상 포함 → floor(5센서≥1) 보장·백본. 가변(diet−고정)만 GA 선별.
목적함수 = GroupKFold(C20) OOF RMSE. 최종 보고 = 선택셋을 M8_PARAMS·705로 KFold+GKF.
"""
import os, re, json
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

# ── v8 에서 가져온 동결 상수 (인라인) ─────────────────────────
ID_COL, TARGET_COL = "C64", "C65"
CORE10 = ["is_high_regime", "high_regime_days", "days_since_last_pm", "C33",
          "dslp_x_hour", "hour", "hour_x_c33",
          "C60_mean_step4", "C59_mean_step4", "is_special_recipe"]
M8_PARAMS = dict(objective="regression", metric="rmse",
    learning_rate=0.029017547696366934, num_leaves=175, min_child_samples=5,
    feature_fraction=0.6324704159196377, bagging_fraction=0.864012693783303, bagging_freq=7,
    lambda_l1=5.04154328625296, lambda_l2=0.024814259264649002,
    min_split_gain=0.2573073648505903, verbose=-1, seed=42)
BEST_ROUNDS = 705

PROTECTED = ["C17", "C11", "C31", "C15", "C16"]

# 선별 루프용 경량 프록시 (빠름). 최종 보고는 M8_PARAMS/705.
PROXY_PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05,
                    num_leaves=63, min_child_samples=20, feature_fraction=0.8,
                    bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)
PROXY_ROUNDS = 200


def _rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def sensor_of(c):
    m = re.match(r"(C\d+)_", c)
    return m.group(1) if m else c


def floor_ok(feats):
    have = {s: sum(1 for c in feats if sensor_of(c) == s) for s in PROTECTED}
    return all(v >= 1 for v in have.values()), have


def load(preset, base="."):
    """df(pool + core10 meta 조인), y, groups, 피처 그룹 반환. base = 파일들이 있는 폴더."""
    pool = pd.read_csv(os.path.join(base, "v13_fdc_pool_wf_oof.csv.gz"))
    pool[ID_COL] = pool[ID_COL].astype(str)

    meta = pd.read_csv(os.path.join(base, "core10_meta_wf.csv"))
    meta[ID_COL] = meta[ID_COL].astype(str)
    df = pool.merge(meta, on=ID_COL, how="inner")

    sel = json.loads(open(os.path.join(base, "feature_diet_selected.json"), encoding="utf-8").read())
    diet = sel["selected"][preset]
    champions = list(sel["champions"][preset].values())

    fixed = [f for f in dict.fromkeys(CORE10 + champions) if f in df.columns]
    prunable = [f for f in diet if f not in set(fixed) and f in df.columns]

    y = df[TARGET_COL].to_numpy(float)
    groups = df["C20"].to_numpy()
    return df, y, groups, dict(fixed=fixed, prunable=prunable, champions=champions,
                               core10=CORE10, diet=diet)


def _oof_group(df, feats, y, groups, params, rounds, n_splits=5):
    oof = np.zeros(len(df)); imps = np.zeros(len(feats))
    for tr, va in GroupKFold(n_splits=n_splits).split(df, y, groups=groups):
        m = lgb.train(params, lgb.Dataset(df.iloc[tr][feats], y[tr]), num_boost_round=rounds)
        oof[va] = m.predict(df.iloc[va][feats])
        imps += m.feature_importance(importance_type="gain")
    return _rmse(y, oof), imps / n_splits


def _oof_kfold(df, feats, y, params, rounds):
    oof = np.zeros(len(df))
    for k in range(5):
        tr, va = df["fold_kf5"] != k, df["fold_kf5"] == k
        m = lgb.train(params, lgb.Dataset(df.loc[tr, feats], y[tr]), num_boost_round=rounds)
        oof[va] = m.predict(df.loc[va, feats])
    return _rmse(y, oof)


# ── GA (hand-rolled, fixed 강제 포함 · GroupKFold 목적) ───────
def ga_select(df, y, groups, fixed, prunable, pop=16, gens=10, cx=0.6, mut=0.08,
              elite=2, fit_splits=3, seed=42, verbose=True):
    rng = np.random.default_rng(seed)
    P = len(prunable); cache = {}

    def fitness(mask):
        key = mask.tobytes()
        if key in cache:
            return cache[key]
        chosen = [prunable[i] for i in range(P) if mask[i]]
        rmse, _ = _oof_group(df, fixed + chosen, y, groups, PROXY_PARAMS, PROXY_ROUNDS, fit_splits)
        cache[key] = rmse
        return rmse

    popm = (rng.random((pop, P)) < 0.5); popm[0, :] = True     # 개체 0 = diet 전체
    scores = np.array([fitness(m) for m in popm])
    for g in range(gens):
        idx = np.argsort(scores); popm, scores = popm[idx], scores[idx]
        if verbose:
            print(f"  GA gen {g:2d}  best_GKF={scores[0]:.3f}  |mask|={int(popm[0].sum())}")
        new = [popm[i].copy() for i in range(elite)]
        while len(new) < pop:
            a = popm[min(rng.integers(0, pop, 3))]
            b = popm[min(rng.integers(0, pop, 3))]
            child = np.where(rng.random(P) < cx, a, b)
            child = np.logical_xor(child, rng.random(P) < mut)
            if child.sum() == 0:
                child[rng.integers(0, P)] = True
            new.append(child)
        popm = np.array(new); scores = np.array([fitness(m) for m in popm])
    j = int(np.argmin(scores))
    chosen = [prunable[i] for i in range(P) if popm[j][i]]
    return dict(best_rmse=float(scores[j]), best_subset=chosen,
                n_selected=len(chosen), n_eval=len(cache))


def final_report(df, y, groups, fixed, selected_prunable, label):
    feats = fixed + list(selected_prunable)
    ok, have = floor_ok(feats)
    kf = _oof_kfold(df, feats, y, M8_PARAMS, BEST_ROUNDS)
    gk, _ = _oof_group(df, feats, y, groups, M8_PARAMS, BEST_ROUNDS, 5)
    return dict(label=label, n_total=len(feats), n_fixed=len(fixed),
                n_selected=len(selected_prunable), floor_ok=bool(ok), protected=have,
                KFold_OOF=round(kf, 3), GroupKFold_C20=round(gk, 3), features=feats)
