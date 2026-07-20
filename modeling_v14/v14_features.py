# -*- coding: utf-8 -*-
"""
v14_features.py — 드리프트-겨냥 신규 피처 인과 빌더 (V1 생성 · V2 평가 공용 단일 소스)

설계 원칙 (강건/PLAN §3, R11):
- 절대 센서값이 아니라 '상대값'. P1~P3 은 각 웨이퍼가 자기보다 시각(wf_ts)이
  '엄격히 앞선' 데이터만 참조하는 expanding 방식으로 계산 → 미래/타깃 누수 0.
  (타깃 C65 는 P1~P3 계산에 절대 미사용. 사용 컬럼은 센서 집계·PM·레짐·시각뿐)
- P4(타깃 분해)는 fold 안에서 stage1(core10) 재학습이 필요 → 정적 컬럼화 불가.
  본 모듈엔 fold-level 규약 함수(p4_two_stage_fold)만 두고, 실제 채점은 V2 롤링.
- floor(R10): 필수 5센서(C17·C11·C31·C15·C16) 각 ≥1 을 산출물마다 assert.

인과 규약 요약:
  * 입력 프레임은 반드시 wf_ts 오름차순 정렬(_time_sort) 후 넘긴다.
  * 모든 통계(mean/std/median)는 'cumulative − 자기자신' = 엄격 과거만.
  * 과거가 없으면(초기 구간) 중립값 0 으로 채운다(신호 없음 = 편차 0).
"""
import re
import numpy as np
import pandas as pd

ID_COL, TARGET_COL, C20_COL = "C64", "C65", "C20"
CORE10 = ["is_high_regime", "high_regime_days", "days_since_last_pm", "C33",
          "dslp_x_hour", "hour", "hour_x_c33", "C60_mean_step4", "C59_mean_step4", "is_special_recipe"]
PROTECTED = ["C17", "C11", "C31", "C15", "C16"]           # floor 필수 5센서
PM_BIN_EDGES = [-np.inf, 1, 3, 7, 14, 30, np.inf]          # days_since_last_pm 위상(고정, 데이터 미참조)


# ---------- 공통 헬퍼 ----------
def sensor_of(c):
    m = re.match(r"(C\d+)_", c)
    return m.group(1) if m else c


def floor_ok(feat_cols):
    have = {s: sum(1 for c in feat_cols if sensor_of(c) == s) for s in PROTECTED}
    return all(v >= 1 for v in have.values()), have


def assert_floor(feat_cols, tag=""):
    ok, have = floor_ok(feat_cols)
    assert ok, f"[floor 위반 {tag}] 5센서 각≥1 실패: {have}"
    return have


def sensor_base_cols(all_cols):
    """P1~P3 입력 = 센서 집계 컬럼(Cxx_stat_stepN). core10 시간피처·ID·타깃 제외."""
    bad = set(CORE10) | {ID_COL, TARGET_COL, C20_COL, "fold_kf5", "wf_ts", "lot_ts"}
    base = [c for c in all_cols if c not in bad and re.match(r"C\d+_", c)]
    # 안전: 어떤 상황에도 타깃/ID 가 섞이지 않음
    assert TARGET_COL not in base and ID_COL not in base and C20_COL not in base
    return base


def _time_sort(df):
    assert "wf_ts" in df.columns, "wf_ts 필요(V0 로더로 병합)"
    return df.sort_values(["wf_ts", ID_COL], kind="mergesort").reset_index(drop=True)


def _past_stats(values, group):
    """time-정렬된 프레임 가정. 각 행의 '엄격 과거(같은 group)' mean/std 를 벡터화 계산.
    cumulative − 자기자신 = 과거합/과거제곱합, cumcount = 과거개수."""
    v = np.asarray(values, float)
    g = pd.Series(np.asarray(group))
    d = pd.DataFrame({"v": v, "v2": v * v, "g": g.values})
    past_sum = d.groupby("g")["v"].cumsum().to_numpy() - v
    past_sq = d.groupby("g")["v2"].cumsum().to_numpy() - v * v
    past_cnt = d.groupby("g").cumcount().to_numpy().astype(float)   # 과거 개수(자기 제외)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(past_cnt >= 1, past_sum / past_cnt, np.nan)
        var = np.where(past_cnt >= 2, past_sq / past_cnt - mean * mean, np.nan)
    std = np.sqrt(np.clip(var, 0, None))
    return mean, std


# ---------- P1: PM 위상-상대 z-score ----------
def build_P1_pm_relative(df, sensor_cols, pm_col="days_since_last_pm"):
    """센서 집계를 days_since_last_pm 위상 구간(고정 edge)별로, '과거만'의 mean/std 로 z-score.
    → PM 사이클 위상 고정 → 노후 드리프트 상쇄. 반환: C64 + <col>__p1pmz."""
    df = _time_sort(df)
    pm = pd.to_numeric(df[pm_col], errors="coerce").fillna(-1).to_numpy()
    pm_bin = pd.cut(pm, PM_BIN_EDGES, labels=False, include_lowest=True)
    pm_bin = pd.Series(pm_bin).fillna(-1).astype(int).to_numpy()
    out = {ID_COL: df[ID_COL].to_numpy()}
    for c in sensor_cols:
        m, s = _past_stats(df[c].to_numpy(), pm_bin)
        z = (df[c].to_numpy() - m) / s
        out[f"{c}__p1pmz"] = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    res = pd.DataFrame(out)
    assert_floor([c for c in res.columns if c != ID_COL], "P1")
    return res


# ---------- P2: 레짐-조건부 baseline 차감 ----------
def build_P2_regime_conditional(df, sensor_cols, regime_col="is_high_regime"):
    """is_high_regime 별 '과거만' baseline(mean) 을 차감 → 레짐 레벨 제거, 레짐 내 편차만.
    반환: C64 + <col>__p2reg."""
    df = _time_sort(df)
    reg = pd.to_numeric(df[regime_col], errors="coerce").fillna(-1).astype(int).to_numpy()
    out = {ID_COL: df[ID_COL].to_numpy()}
    for c in sensor_cols:
        m, _ = _past_stats(df[c].to_numpy(), reg)
        dev = df[c].to_numpy() - m
        out[f"{c}__p2reg"] = np.nan_to_num(dev, nan=0.0, posinf=0.0, neginf=0.0)
    res = pd.DataFrame(out)
    assert_floor([c for c in res.columns if c != ID_COL], "P2")
    return res


# ---------- P3: 직전 k-Lot median 대비 편차(detrend) ----------
def build_P3_stationary(df, sensor_cols, k_lots=5):
    """각 Lot(C20) 대표시각 기준 '직전 k Lot' median 대비 편차 → 달력 추세 제거.
    baseline 은 엄격 과거 Lot 만(R11). 반환: C64 + <col>__p3dt."""
    df = _time_sort(df)
    # Lot 대표시각 = Lot 내 wf_ts median (V0 규약과 동일)
    lot_ts = df.groupby(C20_COL)["wf_ts"].transform("median")
    dfx = df.assign(_lot_ts=lot_ts)
    out = {ID_COL: df[ID_COL].to_numpy()}
    # Lot 단위 집계 → 직전 k Lot median → 웨이퍼로 broadcast
    lots = (dfx.groupby(C20_COL)
              .agg(_lt=("_lot_ts", "first"))
              .reset_index().sort_values("_lt").reset_index(drop=True))
    for c in sensor_cols:
        lot_val = dfx.groupby(C20_COL)[c].median()
        lots["_v"] = lots[C20_COL].map(lot_val).to_numpy()
        base = np.full(len(lots), np.nan)
        vals = lots["_v"].to_numpy()
        for i in range(len(lots)):
            if i >= 1:
                base[i] = np.median(vals[max(0, i - k_lots):i])   # 직전 k Lot(엄격 과거)
        lots["_base"] = base
        base_map = dict(zip(lots[C20_COL], lots["_base"]))
        dev = dfx[c].to_numpy() - dfx[C20_COL].map(base_map).to_numpy()
        out[f"{c}__p3dt"] = np.nan_to_num(dev.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    res = pd.DataFrame(out)
    assert_floor([c for c in res.columns if c != ID_COL], "P3")
    return res


# ---------- R11 인과 spot-check (표본 재계산 대조) ----------
def assert_causal_P2(df, feat_df, sensor_col, regime_col="is_high_regime", n=50, tol=1e-6, rng=0):
    """저장된 P2 피처가 '엄격 과거만' 산출인지 표본 행에서 직접 재계산해 대조."""
    df = _time_sort(df)
    reg = pd.to_numeric(df[regime_col], errors="coerce").fillna(-1).astype(int).to_numpy()
    ts = df["wf_ts"].to_numpy()
    x = df[sensor_col].to_numpy()
    fmap = dict(zip(feat_df[ID_COL], feat_df[f"{sensor_col}__p2reg"]))
    ids = df[ID_COL].to_numpy()
    r = np.random.default_rng(rng)
    idxs = r.choice(np.arange(1, len(df)), size=min(n, len(df) - 1), replace=False)
    viol = 0
    for i in idxs:
        past = (ts < ts[i]) & (reg == reg[i])     # 엄격 과거 & 같은 레짐
        recomputed = (x[i] - x[past].mean()) if past.sum() >= 1 else 0.0
        recomputed = 0.0 if not np.isfinite(recomputed) else recomputed
        if abs(fmap[ids[i]] - recomputed) > tol + 1e-6 * abs(recomputed):
            viol += 1
    assert viol == 0, f"R11 위반: P2[{sensor_col}] 표본 {viol}/{len(idxs)} 불일치(미래참조 의심)"
    return len(idxs)


# ---------- P4: 타깃 분해 (fold-level 규약 — V2 롤링 전용) ----------
def p4_two_stage_fold(train_idx, test_idx, X_core, X_sensor, y, make_model):
    """P4 두단계(인과, fold 내부):
      stage1 = core10 를 train 으로 학습 → level 예측(train/test)
      stage2 = 센서로 (y − level) '잔차'를 학습 → test 잔차 예측
      최종 test 예측 = level_test + resid_test
    stage1 은 train 만 보고, test 타깃은 어디에도 미사용 → 누수 0.
    반환: (pred_test, level_test). V2 rolling 루프에서 fold 마다 호출."""
    m1 = make_model(); m1.fit(X_core.iloc[train_idx], y[train_idx])
    level_tr = m1.predict(X_core.iloc[train_idx]); level_te = m1.predict(X_core.iloc[test_idx])
    resid_tr = y[train_idx] - level_tr
    m2 = make_model(); m2.fit(X_sensor.iloc[train_idx], resid_tr)
    resid_te = m2.predict(X_sensor.iloc[test_idx])
    return level_te + resid_te, level_te
