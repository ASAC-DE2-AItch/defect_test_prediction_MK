# -*- coding: utf-8 -*-
"""
lean85_pipeline.py — lean-85 최종 운영 파이프라인 (handoff 단일 소스)
════════════════════════════════════════════════════════════════════════
확정 모델 (2026-07-19 사용자 확정):
    lean-85 = Conservative-GA(99) − 시간불안정 14  (v13 REPORT_12)
    학습기  = XGBoost (v13 M5 튜닝 파라미터 동결, 246 rounds, seed 42)
    운영    = **정기 재학습 전제** (주간 또는 요란 PM 기입 시 즉시)

정직 수치 (현업 인용 규칙 — REPORT_01 §5):
    시간축(미래 1주, 재학습): pooled RMSE 99.84 · honest R² 0.8545   ← 현업 인용
    lot-CV(GKF C20, same-era): stable 66.83 / seed_mean 66.956      ← same-era 전용
    무재학습: 254.9 (배포 불가 → 재학습은 선택이 아니라 필수)

규율 승계:
    R2  누수 금지: C64/C20/fold 등 식별자는 피처에 절대 미포함 (assert)
    R6  공식 판정 = 사용자 로컬 venv. 미러/타 환경 수치는 상대비교 전용
    R10 floor: 필수 5센서(C17·C11·C31·C15·C16) 각 ≥1 (assert)
    인과: 모든 피처는 웨이퍼 시점 기지값(집계·시각·pm_log 과거 이벤트)만 사용

사용 (요약 — 자세한 건 handoff_lean85_README.md):
    import lean85_pipeline as lp
    lean, params, rounds = lp.load_frozen()
    raw   = pd.read_csv("train_data.csv")
    vdir, model, mani = lp.retrain(raw, lp.find_file("pm_log.json"), out_dir="models")
    new   = pd.read_csv("valid_X.csv")
    preds = lp.predict_lean85(model, lp.build_wafer_table(new, pm_log, lean), lean)
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# ──────────────────────────────────────────────────────────────────────
# 상수 (동결)
# ──────────────────────────────────────────────────────────────────────
PKG_DIR = Path(__file__).resolve().parent
FROZEN_DIR = PKG_DIR / "frozen"

ID_COL, STEP_COL, TARGET_COL, LOT_COL = "C64", "C7", "C65", "C20"
TIME_COL, FMT = "C40", "%Y-%m-%d %H:%M:%S.%f"
RECIPE_COL, C33_COL = "C6", "C33"

REF_DATE = pd.Timestamp("2018-12-01")      # 캠페인 시작일 — pre-PM 경과일 기준점 (v12 동결)
SIGMA_C65 = 261.7                          # honest R² 분모
AGG_FUNCS = ["mean", "std", "max", "min", "last"]

CORE10 = ["is_high_regime", "high_regime_days", "days_since_last_pm", "C33",
          "dslp_x_hour", "hour", "hour_x_c33", "C60_mean_step4", "C59_mean_step4",
          "is_special_recipe"]
PROTECTED = ["C17", "C11", "C31", "C15", "C16"]      # R10 필수 5센서
META_COLS = ["is_high_regime", "high_regime_days", "days_since_last_pm", "C33",
             "dslp_x_hour", "hour", "hour_x_c33", "is_special_recipe"]

ONSET_DAYS = 7.0        # 요란 PM 후 이 기간 = 레짐-온셋 → 예측 신뢰 낮음 플래그 (B2′ 근거)
RETRAIN_DAYS = 7        # 정기 재학습 주기(일)
BENCH_POOLED = 99.840   # B2′/v14 V0 lean-85 시간축 벤치마크 (수용 검사 기준)
BENCH_TOL = 0.5         # 환경차 허용 오차 (R6 — 공식 판정은 로컬 venv)

XGB_DEVICE = os.environ.get("XGB_DEVICE", "cpu")


# ──────────────────────────────────────────────────────────────────────
# 공통 헬퍼
# ──────────────────────────────────────────────────────────────────────
def _rmse(a, b) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def r2_honest(rmse: float) -> float:
    return round(1 - (rmse / SIGMA_C65) ** 2, 4)


def sensor_of(col: str) -> str:
    m = re.match(r"(C\d+)_", col)
    return m.group(1) if m else col


def floor_ok(feat_cols):
    have = {s: sum(1 for c in feat_cols if sensor_of(c) == s) for s in PROTECTED}
    return all(v >= 1 for v in have.values()), have


def find_file(name: str, extra_dirs=()) -> Path:
    """패키지·프로젝트 표준 위치에서 파일 탐색 (노트북/CLI 공용)."""
    root = PKG_DIR.parent
    dirs = [PKG_DIR, FROZEN_DIR, root, root / "문제1(하)",
            root / "modeling_v13" / "data", root / "modeling_v13" / "colab_GA",
            root / "modeling_v14" / "data", *map(Path, extra_dirs)]
    for d in dirs:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"'{name}' 을 찾지 못함. 탐색 위치: {[str(d) for d in dirs]}")


def load_frozen():
    """동결 스펙 로드: (lean_features 85, xgb_params, n_estimators)."""
    lean = json.load(open(FROZEN_DIR / "lean85_features.json", encoding="utf-8"))["lean_features"]
    xgj = json.load(open(FROZEN_DIR / "tuned_params_xgb.json", encoding="utf-8"))
    params, rounds = xgj["params"], int(xgj["n_estimators"])

    assert len(lean) == 85, f"lean-85 아님: {len(lean)}"
    ok, have = floor_ok(lean)
    assert ok, f"R10 floor 위반: {have}"
    banned = {ID_COL, LOT_COL, TARGET_COL, "fold_kf5", "wf_ts", "lot_ts"}
    assert not (set(lean) & banned), f"R2 위반: 식별자 유입 {set(lean) & banned}"
    return lean, params, rounds


def parse_pm_log(pm_log):
    """pm_log(list | 경로) → 시간순 [(Timestamp, type)]. 요란(loud) PM만 기입하는 규약.
    레거시 ["2018-12-24"] / 신규 [{"date": ..., "type": ...}] 모두 지원."""
    if isinstance(pm_log, (str, Path)):
        pm_log = json.load(open(pm_log, encoding="utf-8"))
    out = []
    for e in pm_log:
        if isinstance(e, dict):
            out.append((pd.Timestamp(e["date"]), e.get("type", "major")))
        else:
            out.append((pd.Timestamp(e), "major"))
    return sorted(out, key=lambda x: x[0])


# ──────────────────────────────────────────────────────────────────────
# 피처 빌드 (raw 트레이스 → 웨이퍼 1행)  — v13 build_fdc_pool / v12 메타와 동일 스킴
# ──────────────────────────────────────────────────────────────────────
def _lean_sensor_cols(lean):
    """lean-85 중 센서 집계 컬럼(Cxx_stat_stepN)과 그 원천 센서 목록."""
    sens_cols = [c for c in lean if re.match(r"C\d+_(mean|std|max|min|last)_step\d+$", c)]
    sensors = sorted({sensor_of(c) for c in sens_cols}, key=lambda s: int(s[1:]))
    return sens_cols, sensors


def _fdc_aggregate(raw: pd.DataFrame, sensors, expected_cols):
    """전역 C40(datetime 파싱) 정렬 → groupby(C64,C7) 5통계 → pivot → 기대 컬럼 reindex.
    (정렬·집계 스킴은 v13 build_fdc_pool.py 와 동일 — 'last'=시간순 마지막 유효값.
     ⚠️ 문자열 정렬 금지: %f 자릿수 차이로 순서가 틀어질 수 있어 반드시 파싱 후 정렬)"""
    df = raw.copy()
    df["_ts_sort"] = pd.to_datetime(df[TIME_COL], format=FMT)   # 동결본과 동일: 엄격 파싱
    df = df.sort_values("_ts_sort").reset_index(drop=True)
    for s in sensors:
        df[s] = pd.to_numeric(df[s], errors="coerce")
    agg = df.groupby([ID_COL, STEP_COL])[sensors].agg(AGG_FUNCS)
    agg.columns = ["_".join(c) for c in agg.columns]
    wide = agg.reset_index().pivot(index=ID_COL, columns=STEP_COL)
    wide.columns = [f"{c0}_step{int(c1)}" for c0, c1 in wide.columns]
    # 신규 배치에 특정 Step 이 없으면 해당 컬럼 자체가 사라짐 → NaN 으로 복원(XGB 자체 처리)
    wide = wide.reindex(columns=expected_cols)
    return wide.reset_index()


def _meta_features(raw: pd.DataFrame, pm_log):
    """core 메타 8종 — v12 feature_engineering(2026-07-09 A안) 재현.
    is_high_regime = 요란 PM 이후 one-way 플래그 (구명 유지 — lean-85 컬럼명과 일치)."""
    df = raw.copy()
    df["_ts"] = pd.to_datetime(df[TIME_COL], format=FMT)   # 엄격 파싱(포맷 이탈 시 즉시 실패)
    wf = df.groupby(ID_COL)

    meta = wf[C33_COL].first().reset_index()
    meta["wf_ts"] = wf["_ts"].min().to_numpy()
    meta[LOT_COL] = wf[LOT_COL].first().to_numpy()          # CV/정렬 메타 — 피처 아님
    meta["hour"] = pd.Series(meta["wf_ts"]).dt.hour

    events = parse_pm_log(pm_log)
    days, flag = [], []
    for d in meta["wf_ts"]:
        past = [e for e in events if e[0] <= d]
        if past:
            days.append((d - past[-1][0]).total_seconds() / 86400.0)
            flag.append(1)
        else:
            days.append((d - REF_DATE).total_seconds() / 86400.0)
            flag.append(0)
    meta["days_since_last_pm"] = days
    meta["is_high_regime"] = flag
    meta["high_regime_days"] = meta["days_since_last_pm"] * meta["is_high_regime"]
    meta["dslp_x_hour"] = meta["days_since_last_pm"] * meta["hour"]
    meta["hour_x_c33"] = meta["hour"] * meta[C33_COL]

    rec = wf[RECIPE_COL].first().reset_index()
    meta = meta.merge(rec, on=ID_COL)
    meta["is_special_recipe"] = (meta[RECIPE_COL] == "C6_1").astype(int)
    return meta.drop(columns=[RECIPE_COL])


def build_wafer_table(raw: pd.DataFrame, pm_log, lean) -> pd.DataFrame:
    """raw 트레이스 → 웨이퍼 1행 테이블 [C64, C20, wf_ts, lean-85 피처, (C65)].
    train(타깃 有)·신규 X(타깃 無) 공용. 인과: 웨이퍼 시점 기지값만 사용."""
    sens_cols, sensors = _lean_sensor_cols(lean)
    missing = [s for s in sensors if s not in raw.columns]
    assert not missing, f"raw 에 센서 누락: {missing}"

    fdc = _fdc_aggregate(raw, sensors, sens_cols)
    meta = _meta_features(raw, pm_log)
    tbl = fdc.merge(meta, on=ID_COL, how="inner")

    if TARGET_COL in raw.columns:
        tgt = raw.groupby(ID_COL)[TARGET_COL].first().reset_index()
        tbl = tbl.merge(tgt, on=ID_COL, how="left")

    absent = [c for c in lean if c not in tbl.columns]
    assert not absent, f"lean-85 컬럼 누락: {absent[:5]}"
    ok, have = floor_ok(lean)
    assert ok, f"R10 floor 위반: {have}"
    assert LOT_COL not in lean and ID_COL not in lean, "R2 위반"

    order = [ID_COL, LOT_COL, "wf_ts"] + lean + ([TARGET_COL] if TARGET_COL in tbl.columns else [])
    return tbl[order].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# 모델 (동결 파라미터)
# ──────────────────────────────────────────────────────────────────────
def make_model(params, rounds) -> xgb.XGBRegressor:
    p = dict(params)
    p.update(objective="reg:squarederror", tree_method="hist", device=XGB_DEVICE,
             random_state=42, n_estimators=int(rounds))
    return xgb.XGBRegressor(**p)


def fit_lean85(table: pd.DataFrame, lean, params, rounds) -> xgb.XGBRegressor:
    assert TARGET_COL in table.columns, "학습에는 C65 필요"
    m = table[TARGET_COL].notna()
    model = make_model(params, rounds)
    model.fit(table.loc[m, lean], table.loc[m, TARGET_COL].to_numpy(float))
    return model


def predict_lean85(model, table: pd.DataFrame, lean) -> pd.DataFrame:
    """예측 + 레짐-온셋 low_confidence 플래그.
    low_confidence=1: 요란 PM 후 ONSET_DAYS 이내 — B2′ 실증상 어떤 모델도 오차 급증
    (온셋 주 RMSE 147/113/109). 이 구간 예측은 참고용으로만."""
    pred = model.predict(table[lean])
    low = ((table["is_high_regime"] == 1) &
           (table["days_since_last_pm"] <= ONSET_DAYS)).astype(int)
    return pd.DataFrame({ID_COL: table[ID_COL], "pred_C65": pred,
                         "low_confidence": low.to_numpy()})


def evaluate_rmse(model, table: pd.DataFrame, lean):
    """타깃 보유 테이블의 RMSE (주간 모니터링용)."""
    m = table[TARGET_COL].notna()
    rmse = _rmse(table.loc[m, TARGET_COL].to_numpy(float),
                 model.predict(table.loc[m, lean]))
    return rmse, r2_honest(rmse)


# ──────────────────────────────────────────────────────────────────────
# 재학습 (핵심 SOP) — 전체 누적 데이터로 재적합 → 버전 폴더에 저장
# ──────────────────────────────────────────────────────────────────────
def retrain(raw_all: pd.DataFrame, pm_log, out_dir="models", tag: str | None = None,
            lean=None, params=None, rounds=None):
    """누적 raw 전체 + 최신 pm_log 로 lean-85 재적합 후 버전 저장.

    반환: (버전 폴더 Path, 학습된 모델, manifest dict)
    저장: lean85_model.json (XGB 이식 포맷) + manifest.json (재현 메타)
    """
    if lean is None:
        lean, params, rounds = load_frozen()
    table = build_wafer_table(raw_all, pm_log, lean)
    model = fit_lean85(table, lean, params, rounds)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vdir = Path(out_dir) if os.path.isabs(str(out_dir)) else PKG_DIR / out_dir
    vdir = vdir / (f"lean85_{stamp}" + (f"_{tag}" if tag else ""))
    vdir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(vdir / "lean85_model.json"))

    events = parse_pm_log(pm_log)
    m = table[TARGET_COL].notna()
    manifest = {
        "package": "handoff_lean85 v1.0",
        "model": "lean-85 XGBoost (동결 파라미터, v13 M5)",
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "n_wafers_train": int(m.sum()),
        "train_period": [str(table.loc[m, "wf_ts"].min()), str(table.loc[m, "wf_ts"].max())],
        "pm_log_dates": [str(d.date()) for d, _ in events],
        "features_n": len(lean),
        "features_sha1": hashlib.sha1(",".join(lean).encode()).hexdigest()[:12],
        "n_estimators": int(rounds),
        "params": params,
        "onset_days_flag": ONSET_DAYS,
        "env": {"python": platform.python_version(), "xgboost": xgb.__version__,
                "pandas": pd.__version__, "numpy": np.__version__},
        "benchmark": {"time_axis_pooled_rmse": BENCH_POOLED, "honest_R2": 0.8545,
                      "lot_cv_seed_mean": 66.956,
                      "note": "시간축 수치만 현업 인용 (REPORT_01 §5)"},
    }
    json.dump(manifest, open(vdir / "manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return vdir, model, manifest


def load_model(version_dir) -> tuple[xgb.XGBRegressor, dict]:
    vdir = Path(version_dir)
    model = xgb.XGBRegressor()
    model.load_model(str(vdir / "lean85_model.json"))
    manifest = json.load(open(vdir / "manifest.json", encoding="utf-8"))
    return model, manifest


def should_retrain(manifest: dict, pm_log, now=None):
    """재학습 트리거 판정 → (bool, 사유 리스트).
    ① 정기: 마지막 학습 후 RETRAIN_DAYS 경과   ② 이벤트: 신규 요란 PM 기입."""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    reasons = []
    last = pd.Timestamp(manifest["created_at_local"])
    if (now - last).total_seconds() >= RETRAIN_DAYS * 86400:
        reasons.append(f"정기: 마지막 재학습 후 {RETRAIN_DAYS}일 경과")
    known = set(manifest.get("pm_log_dates", []))
    new_pm = [str(d.date()) for d, _ in parse_pm_log(pm_log) if str(d.date()) not in known]
    if new_pm:
        reasons.append(f"이벤트: 신규 요란 PM 기입 {new_pm} → 즉시 재학습")
    return bool(reasons), reasons


# ──────────────────────────────────────────────────────────────────────
# 모니터링 — 입력 PSI 드리프트
# ──────────────────────────────────────────────────────────────────────
def psi(expected, actual, bins=10):
    """Population Stability Index. 관례: <0.1 안정 / 0.1~0.25 주의 / ≥0.25 경보."""
    e = np.asarray(expected, float); a = np.asarray(actual, float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e) < 10 or len(a) < 10:
        return np.nan
    qs = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return np.nan
    qs[0], qs[-1] = -np.inf, np.inf
    pe = np.histogram(e, qs)[0] / len(e)
    pa = np.histogram(a, qs)[0] / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def drift_report(ref_table: pd.DataFrame, new_table: pd.DataFrame, lean, top=15):
    """피처별 PSI 상위 top — 재학습 보조 트리거·원인 추적용."""
    rows = [{"feature": c, "psi": psi(ref_table[c], new_table[c])} for c in lean]
    rep = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    rep["level"] = np.select([rep["psi"] >= 0.25, rep["psi"] >= 0.1],
                             ["경보", "주의"], default="안정")
    return rep.head(top)


# ──────────────────────────────────────────────────────────────────────
# 수용 검사 — B2′ 롤링-재학습 재현 (환경/코드 변경 후 1회 실행 권장)
# ──────────────────────────────────────────────────────────────────────
def walkforward_acceptance(table: pd.DataFrame, lean, params, rounds,
                           init_weeks=4, h_days=7, min_test=30, verbose=True):
    """B2′ 프로토콜(동결) lean-85 arm 재현: 확장창 · Lot 시간정렬 · 주간 재학습.
    기대 pooled = 99.840 ± 0.5 (R6: 공식 판정은 사용자 로컬 venv)."""
    df = table.copy()
    lot_ts = df.groupby(LOT_COL)["wf_ts"].transform("median")
    df["lot_ts"] = lot_ts
    y = df[TARGET_COL].to_numpy(float)

    t0 = df["wf_ts"].min()
    T = t0 + pd.Timedelta(weeks=init_weeks)
    end = df["wf_ts"].max()
    H = pd.Timedelta(days=h_days)

    folds, ys, ps = [], [], []
    while T < end:
        te = ((df["lot_ts"] > T) & (df["lot_ts"] <= T + H)).to_numpy()
        if te.sum() < min_test:
            T += H
            continue
        tr = (df["lot_ts"] <= T).to_numpy()
        m = make_model(params, rounds)
        m.fit(df.loc[tr, lean], y[tr])
        p = m.predict(df.loc[te, lean])
        r = _rmse(y[te], p)
        folds.append({"cut": str(T)[:10], "n": int(te.sum()),
                      "train_n": int(tr.sum()), "rmse": round(r, 3)})
        ys.append(y[te]); ps.append(p)
        if verbose:
            print(f"  {str(T)[:10]}  test {te.sum():>4} | train {tr.sum():>5} | RMSE {r:7.2f}")
        T += H

    pooled = _rmse(np.concatenate(ys), np.concatenate(ps))
    out = {"pooled_rmse": round(pooled, 3), "R2_honest": r2_honest(pooled),
           "n_folds": len(folds), "folds": folds,
           "benchmark": BENCH_POOLED, "delta": round(pooled - BENCH_POOLED, 3),
           "pass": bool(abs(pooled - BENCH_POOLED) <= BENCH_TOL)}
    if verbose:
        print(f"pooled {pooled:.3f} (기준 {BENCH_POOLED}±{BENCH_TOL}) → "
              f"{'✅ 수용' if out['pass'] else '❌ 기준 이탈 — 환경/코드 점검'}")
    return out
