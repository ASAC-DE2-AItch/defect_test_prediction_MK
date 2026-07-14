# -*- coding: utf-8 -*-
"""modeling_v13 M3 — 2차 피처 선별 공통 헬퍼 (RFECV · GA).

각 노트북(Conservative/Balanced × RFECV/GA)이 이 모듈을 import 해서
PRESET·METHOD 만 지정하고 실행한다.

설계 원칙
---------
- **고정(fixed)**: core10(10) + 필수 5센서 champion  → 항상 모델에 포함. floor(5센서≥1) 자동 보장·백본 유지.
- **가변(prunable)**: diet 선택 피처 − 고정  → RFECV/GA가 이 안에서 부분집합을 고른다.
- **선별 목적함수**: `GroupKFold(C20)` OOF RMSE (프로젝트 결정지표 = 신규 Lot 정직-CV).
- **최종 보고**: 고른 셋을 `M8_PARAMS`·705라운드로 KFold OOF + GroupKFold(C20) 둘 다.

전제 파일: ../modeling_v8/v8_timeline_common.py, ../문제1(하)/train_data.csv, ../pm_log.json,
          ./feature_diet_selected.json, ./data/v13_fdc_pool_wf_oof.csv.gz
"""
import os, sys, json, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "modeling_v8"))
import v8_timeline_common as tl              # 빌더 + CORE10 + M8_PARAMS + BEST_ROUNDS
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

PROTECTED = ["C17", "C11", "C31", "C15", "C16"]

# 선별 루프용 경량 프록시 LGBM (빠름). 최종 보고는 tl.M8_PARAMS / tl.BEST_ROUNDS 사용.
PROXY_PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05,
                    num_leaves=63, min_child_samples=20, feature_fraction=0.8,
                    bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)
PROXY_ROUNDS = 200


def _rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


# ─────────────────────────────────────────────────────────────
# 데이터 로드 & 피처 그룹
# ─────────────────────────────────────────────────────────────
def load(preset):
    """df(pool+core10 조인), y, groups, 그리고 fixed/prunable/champions 반환."""
    pool = pd.read_csv(os.path.join(HERE, "data", "v13_fdc_pool_wf_oof.csv.gz"))
    pool["C64"] = pool["C64"].astype(str)

    pm = tl.parse_pm_log(json.loads(open(os.path.join(ROOT, "pm_log.json"), encoding="utf-8").read()))
    raw = pd.read_csv(os.path.join(ROOT, "문제1(하)", "train_data.csv"))
    dfp = tl.preprocess(raw)
    meta = tl.make_meta_features(dfp, pm)
    meta = meta.merge(dfp.groupby(tl.ID_COL)["C6"].first().reset_index(), on=tl.ID_COL)
    meta["is_special_recipe"] = (meta["C6"] == "C6_1").astype(int)
    META8 = ["is_high_regime", "high_regime_days", "days_since_last_pm", "C33",
             "dslp_x_hour", "hour", "hour_x_c33", "is_special_recipe"]
    meta[tl.ID_COL] = meta[tl.ID_COL].astype(str)
    df = pool.merge(meta[[tl.ID_COL] + META8], on="C64", how="inner")

    sel = json.loads(open(os.path.join(HERE, "feature_diet_selected.json"), encoding="utf-8").read())
    diet = sel["selected"][preset]
    champions = list(sel["champions"][preset].values())
    core10 = list(tl.CORE10)

    fixed = list(dict.fromkeys(core10 + champions))              # 항상 포함
    fixed = [f for f in fixed if f in df.columns]
    prunable = [f for f in diet if f not in set(fixed) and f in df.columns]

    y = df[tl.TARGET_COL].to_numpy(float)
    groups = df["C20"].to_numpy()
    return df, y, groups, dict(fixed=fixed, prunable=prunable, champions=champions,
                               core10=core10, diet=diet)


def sensor_of(c):
    import re
    m = re.match(r"(C\d+)_", c)
    return m.group(1) if m else c


def floor_ok(feats):
    have = {s: sum(1 for c in feats if sensor_of(c) == s) for s in PROTECTED}
    return all(v >= 1 for v in have.values()), have


# ─────────────────────────────────────────────────────────────
# OOF 평가 (프록시 / 최종 공용)
# ─────────────────────────────────────────────────────────────
def _oof_group(df, feats, y, groups, params, rounds, n_splits=5):
    oof = np.zeros(len(df))
    gkf = GroupKFold(n_splits=n_splits)
    imps = np.zeros(len(feats))
    for tr, va in gkf.split(df, y, groups=groups):
        d = lgb.Dataset(df.iloc[tr][feats], y[tr])
        m = lgb.train(params, d, num_boost_round=rounds)
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


def eval_group(df, feats, y, groups, proxy=True, n_splits=5):
    p, r = (PROXY_PARAMS, PROXY_ROUNDS) if proxy else (tl.M8_PARAMS, tl.BEST_ROUNDS)
    rmse, _ = _oof_group(df, feats, y, groups, p, r, n_splits)
    return rmse


# ─────────────────────────────────────────────────────────────
# ① RFECV (수동, fixed 강제 포함 · GroupKFold 목적)
# ─────────────────────────────────────────────────────────────
def rfe_cv(df, y, groups, fixed, prunable, step=5, min_keep=5, n_splits=5, verbose=True):
    """least-gain prunable 를 step 개씩 제거하며 GroupKFold OOF 곡선 기록 → 최적 크기 선택."""
    cur = list(prunable)
    history = []                       # (n_prunable, rmse, subset)
    while True:
        feats = fixed + cur
        rmse, imps = _oof_group(df, feats, y, groups, PROXY_PARAMS, PROXY_ROUNDS, n_splits)
        history.append((len(cur), rmse, list(cur)))
        if verbose:
            print(f"  RFE  prunable={len(cur):3d}  GKF_RMSE={rmse:.3f}")
        if len(cur) <= min_keep:
            break
        # prunable 부분의 gain 으로 하위 step 제거
        pim = imps[len(fixed):]
        order = np.argsort(pim)                    # 오름차순(gain 낮은 것 먼저)
        drop = set(np.array(cur)[order[:min(step, len(cur) - min_keep)]])
        cur = [c for c in cur if c not in drop]
    best = min(history, key=lambda h: h[1])        # RMSE 최소 지점
    return dict(best_n=best[0], best_rmse=best[1], best_subset=best[2], history=history)


# ─────────────────────────────────────────────────────────────
# ② GA (hand-rolled, fixed 강제 포함 · GroupKFold 목적)
# ─────────────────────────────────────────────────────────────
def ga_select(df, y, groups, fixed, prunable, pop=16, gens=10, cx=0.6, mut=0.08,
              elite=2, fit_splits=3, seed=42, verbose=True):
    rng = np.random.default_rng(seed)
    P = len(prunable)
    cache = {}

    def fitness(mask):
        key = mask.tobytes()
        if key in cache:
            return cache[key]
        chosen = [prunable[i] for i in range(P) if mask[i]]
        feats = fixed + chosen
        rmse, _ = _oof_group(df, feats, y, groups, PROXY_PARAMS, PROXY_ROUNDS, fit_splits)
        cache[key] = rmse
        return rmse

    # 초기 개체군: 절반 확률로 켜짐 (+ 전체 켜짐 1개 = diet 그대로)
    popm = (rng.random((pop, P)) < 0.5)
    popm[0, :] = True
    scores = np.array([fitness(m) for m in popm])
    for g in range(gens):
        idx = np.argsort(scores)
        popm, scores = popm[idx], scores[idx]      # 오름차순(좋을수록 앞)
        if verbose:
            best_n = int(popm[0].sum())
            print(f"  GA   gen {g:2d}  best_GKF={scores[0]:.3f}  |mask|={best_n}")
        new = [popm[i].copy() for i in range(elite)]   # 엘리트
        while len(new) < pop:
            # 토너먼트 선택 2명
            a = popm[min(rng.integers(0, pop, 3))]
            b = popm[min(rng.integers(0, pop, 3))]
            # 균등 교차
            child = np.where(rng.random(P) < cx, a, b)
            # 변이
            flip = rng.random(P) < mut
            child = np.logical_xor(child, flip)
            if child.sum() == 0:                        # 공집합 방지
                child[rng.integers(0, P)] = True
            new.append(child)
        popm = np.array(new)
        scores = np.array([fitness(m) for m in popm])
    j = int(np.argmin(scores))
    chosen = [prunable[i] for i in range(P) if popm[j][i]]
    return dict(best_rmse=float(scores[j]), best_subset=chosen,
                n_selected=len(chosen), n_eval=len(cache))


# ─────────────────────────────────────────────────────────────
# 최종 보고 (선택셋 → KFold + GroupKFold, M8_PARAMS·705)
# ─────────────────────────────────────────────────────────────
def final_report(df, y, groups, fixed, selected_prunable, label):
    feats = fixed + list(selected_prunable)
    ok, have = floor_ok(feats)
    kf = _oof_kfold(df, feats, y, tl.M8_PARAMS, tl.BEST_ROUNDS)
    gk, _ = _oof_group(df, feats, y, groups, tl.M8_PARAMS, tl.BEST_ROUNDS, 5)
    return dict(label=label, n_total=len(feats), n_fixed=len(fixed),
                n_selected=len(selected_prunable), floor_ok=bool(ok),
                protected=have, KFold_OOF=round(kf, 3), GroupKFold_C20=round(gk, 3),
                features=feats)
