#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modeling_v8 타임라인 개형 — 공통 헬퍼 (피처 빌더 · 모델 학습 · 플롯).

이 파일 하나에 v8 노트북(Cell 1·3·7·13·15)의 재료를 모아두고,
세 개의 얇은 스크립트가 이 모듈을 import 해서 그림만 다르게 그린다:

  compare_tracks_lgbm.py       LightGBM 기준 F-T15 vs F-P3 비교
  compare_tracks_et.py         ExtraTrees 기준 F-T15 vs F-P3 비교
  compare_models_by_track.py   트랙별 LightGBM vs ExtraTrees 비교

★ 4개 파일(이 모듈 + 스크립트 3개)을 modeling_v8/ 폴더에 함께 두고 실행할 것.

색 규칙(전 스크립트 공통): 첫 곡선 = 파란 실선, 둘째 곡선 = 초록 실선, 실측 = 검정.
  · 트랙 비교:  F-T15=파랑, F-P3=초록
  · 모델 비교:  LightGBM=파랑, ExtraTrees=초록
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error

# ── 경로 (스크립트 위치 기준) ───────────────────────────────
HERE = Path(__file__).resolve().parent            # modeling_v8/
DATA = HERE.parent / "문제1(하)"
ANS  = HERE.parent / "문제1_하_answer"
PM_LOG = HERE.parent / "pm_log.json"
OUT  = HERE / "outputs"

# ── 상수 (Cell 1) ───────────────────────────────────────────
COLS_ALL_NULL = ['C2','C13','C26','C37','C43','C47','C53','C55']
COLS_CONSTANT = ['C3','C8','C14','C19','C21','C24','C28','C29','C30','C44','C45','C51']
COLS_DUPLICATE= ['C36','C35','C38']
COLS_TO_DROP  = COLS_ALL_NULL + COLS_CONSTANT + COLS_DUPLICATE
FDC_FEATURES  = ['C4','C6','C12','C17','C18','C25','C27','C32','C42','C46','C48','C49',
                 'C50','C52','C54','C56','C57','C58','C59','C60','C61','C62','C63']
AGG_FUNCS = ['mean','std','max','min','last']
ID_COL, STEP_COL, TARGET_COL, SORT_COL = 'C64','C7','C65','C40'
FMT = '%Y-%m-%d %H:%M:%S.%f'
REF = pd.Timestamp('2018-12-01')

# ── 플롯 상수 ───────────────────────────────────────────────
PM_TS    = pd.Timestamp("2018-12-24 01:31:22.7")   # major·loud 전환 (pm_log_meta.json)
C6_BATCH = "C6_1"
BLUE, GREEN = "tab:blue", "tab:green"

# ── 트랙 (Cell 13 §8.5 동결 상수) ───────────────────────────
CORE10 = ['is_high_regime','high_regime_days','days_since_last_pm','C33','dslp_x_hour',
          'hour','hour_x_c33','C60_mean_step4','C59_mean_step4','is_special_recipe']
TRACKS = {
    'F-T15': ['is_high_regime','high_regime_days','days_since_last_pm','C12_mean_step6','C33',
              'dslp_x_hour','C12_mean_step7','hour','C4_mean_step4','C4_max_step1','hour_x_c33',
              'C12_max_step6','C61_mean_step1','C60_std_step4','C62_mean_step1'],
    'F-P3' : CORE10 + ['C12_mean_step6','C12_mean_step7','C4_mean_step4'],
    'F-C10': CORE10,
}

# ── LightGBM 파라미터 (Cell 7 M8_PARAMS = 복원 pkl) ─────────
M8_PARAMS = dict(objective='regression', metric='rmse',
    learning_rate=0.029017547696366934, num_leaves=175, min_child_samples=5,
    feature_fraction=0.6324704159196377, bagging_fraction=0.864012693783303, bagging_freq=7,
    lambda_l1=5.04154328625296, lambda_l2=0.024814259264649002,
    min_split_gain=0.2573073648505903, verbose=-1, seed=42)
BEST_ROUNDS = 705
ET_N_ESTIMATORS, ET_RANDOM_STATE = 500, 42


# ── pm_log 파서 (Cell 1) ────────────────────────────────────
def parse_pm_log(raw):
    out = []
    for e in raw:
        if isinstance(e, dict):
            ts = pd.Timestamp(e["date"]); ty = e.get("type", "major")
            vd = e.get("verdict", "loud" if ty == "major" else "quiet")
        else:
            ts = pd.Timestamp(e); ty = "major"; vd = "loud"
        out.append((ts, ty, vd))
    return sorted(out, key=lambda x: x[0])


# ── 피처 빌더 (Cell 3) ──────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df[SORT_COL] = pd.to_datetime(df[SORT_COL], format=FMT)
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    return df.sort_values(SORT_COL).reset_index(drop=True)


def make_fdc_features(df):
    numeric = [c for c in FDC_FEATURES if pd.api.types.is_numeric_dtype(df[c])]
    string  = [c for c in FDC_FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    step_dur = df.groupby([ID_COL, STEP_COL])['C41'].max().rename('C41_max').reset_index()
    agg = df.groupby([ID_COL, STEP_COL])[numeric].agg(AGG_FUNCS)
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg = agg.reset_index().merge(step_dur, on=[ID_COL, STEP_COL])
    wide = agg.pivot(index=ID_COL, columns=STEP_COL)
    wide.columns = [f'{c[0]}_step{int(c[1])}' for c in wide.columns]
    wide = wide.reset_index()
    if string:
        wide = wide.merge(df.groupby(ID_COL)[string].first().reset_index(), on=ID_COL)
    return wide


def _time_feats(dates, pm_events):
    dslp, is_post, is_high, hrd = [], [], [], []
    for d in dates:
        past = [(t, ty, vd) for t, ty, vd in pm_events if t <= d]
        if past:
            last_ts = past[-1][0]
            dslp.append((d - last_ts).total_seconds() / 86400); is_post.append(1)
            state, last_loud = 0, None
            for t, ty, vd in past:
                if vd == "loud": state, last_loud = 1, t
            is_high.append(state)
            hrd.append((d - last_loud).total_seconds() / 86400 if state == 1 and last_loud is not None else 0.0)
        else:
            dslp.append((d - REF).total_seconds() / 86400)
            is_post.append(0); is_high.append(0); hrd.append(0.0)
    return pd.DataFrame({'days_since_last_pm': dslp, 'is_post_pm': is_post,
                         'is_high_regime': is_high, 'high_regime_days': hrd}, index=dates.index)


def make_meta_features(df, pm_events):
    wf = df.groupby(ID_COL)
    meta = wf['C33'].first().reset_index()
    wd = wf[SORT_COL].first().reset_index().rename(columns={SORT_COL: '_date'})
    tf = _time_feats(wd['_date'], pm_events)
    wd = pd.concat([wd, tf], axis=1); wd['hour'] = wd['_date'].dt.hour
    cols = [ID_COL, 'days_since_last_pm', 'is_post_pm', 'is_high_regime',
            'high_regime_days', 'hour', '_date']
    meta = meta.merge(wd[cols], on=ID_COL)
    meta['post_pm_days'] = meta['days_since_last_pm'] * meta['is_post_pm']
    meta['dslp_x_hour']  = meta['days_since_last_pm'] * meta['hour']
    meta['hour_x_c33']   = meta['hour'] * meta['C33']
    return meta


def build_features(df_raw, pm_events):
    df  = preprocess(df_raw)
    res = make_fdc_features(df).merge(make_meta_features(df, pm_events), on=ID_COL)
    res['is_special_recipe'] = (res['C6'] == 'C6_1').astype(int)
    if TARGET_COL in df.columns:
        res = res.merge(df.groupby(ID_COL)[TARGET_COL].first().reset_index(), on=ID_COL)
    return res  # '_date'(WF 대표 시각, KST) 보존


def load_tables():
    """train/valid/test 피처 테이블 + y + pm_events 반환."""
    pm = parse_pm_log(json.loads(PM_LOG.read_text(encoding="utf-8")))
    print("pm_events:", [(str(t.date()), ty, vd) for t, ty, vd in pm])
    print("피처 빌드 중… (train 로드+집계, 수십 초)")
    Xtr = build_features(pd.read_csv(DATA / "train_data.csv"), pm)
    Xva = build_features(pd.read_csv(DATA / "valid_X.csv"),   pm)
    Xte = build_features(pd.read_csv(DATA / "test_X.csv"),    pm)
    y = Xtr[TARGET_COL].to_numpy(float)
    print(f"피처 테이블: train {Xtr.shape} / valid {Xva.shape} / test {Xte.shape}")
    return Xtr, Xva, Xte, y, pm


# ── 모델 학습 → predict_fn(Xdf)->예측배열 ───────────────────
def fit_lgbm(Xtr, feats, y):
    """Cell 7 full_train — M8_PARAMS, 705 rounds."""
    import lightgbm as lgb
    model = lgb.train(M8_PARAMS, lgb.Dataset(Xtr[feats], y), num_boost_round=BEST_ROUNDS)
    return lambda Xdf: model.predict(Xdf[feats])


def fit_et(Xtr, feats, y):
    """Cell 15 et_valid_test — median 대치(train fit) + ExtraTrees 500그루."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy='median').fit(Xtr[feats])
    model = ExtraTreesRegressor(n_estimators=ET_N_ESTIMATORS, n_jobs=-1,
                                random_state=ET_RANDOM_STATE).fit(imp.transform(Xtr[feats]), y)
    return lambda Xdf: model.predict(imp.transform(Xdf[feats]))


# ── 플롯 ────────────────────────────────────────────────────
def set_korean_font():
    import matplotlib.font_manager as fm
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if name in installed:
            matplotlib.rcParams["font.family"] = name; break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _base(Xdf, ans_path):
    """WF 단위 ts·true·is_special_recipe·C64 (ts 정렬)."""
    ans = pd.read_csv(ans_path); ans[ID_COL] = ans[ID_COL].astype(str)
    d = Xdf[[ID_COL, '_date', 'is_special_recipe']].copy()
    d[ID_COL] = d[ID_COL].astype(str)
    d = d.merge(ans.rename(columns={TARGET_COL: 'true'}), on=ID_COL)
    return d.rename(columns={'_date': 'ts'}).sort_values('ts')


def _panel(ax, name, base, Xdf, curves):
    """base=_base(...), curves=[dict(label,color,predict_fn)]. 여러 곡선을 겹쳐 그린다."""
    frame = base[['ts', 'true']].copy()
    rmses = []
    for c in curves:
        pr = pd.Series(c['predict_fn'](Xdf), index=Xdf[ID_COL].astype(str))
        vals = base[ID_COL].map(pr).to_numpy(float)
        frame[c['label']] = vals
        rmses.append((c['label'], np.sqrt(mean_squared_error(base['true'], vals))))
    # 개별 WF 실측
    ax.scatter(base['ts'], base['true'], s=6, c='0.7', alpha=0.5, label='실측 (개별 WF)', zorder=1)
    # 12h 평균선
    r = frame.set_index('ts').resample('12h').mean().dropna()
    ax.plot(r.index, r['true'], c='black', lw=1.4, label='실측 (12h 평균)', zorder=4)
    for c in curves:
        ax.plot(r.index, r[c['label']], c=c['color'], lw=1.4,
                label=f"{c['label']} 예측 (12h 평균)", zorder=3)
    # PM 전환선 + C6_1 배치 음영
    ax.axvline(PM_TS, color='red', ls='--', lw=1.3, zorder=2)
    ax.text(PM_TS, ax.get_ylim()[1], " PM (레짐 전환)", color='red', va='top', ha='left', fontsize=9)
    c6 = base[base['is_special_recipe'] == 1]['ts']
    if len(c6):
        lo, hi = c6.min(), c6.max()
        ax.axvspan(lo, hi, color='orange', alpha=0.18, zorder=0)
        ax.text(lo + (hi - lo) / 2, ax.get_ylim()[0], "C6_1\n배치",
                color='darkorange', va='bottom', ha='center', fontsize=8)
    rmse_txt = "  ·  ".join(f"{lab} RMSE {rm:.2f}" for lab, rm in rmses)
    ax.set_title(f"[{name}]  {rmse_txt}  (n={len(base):,})", fontsize=12, fontweight="bold")
    ax.set_ylabel("C65 (불량 비트 수)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9); ax.grid(alpha=0.25)


def render_figure(out_path, suptitle, splits, curves):
    """splits=[('valid',Xva,ans),('test',Xte,ans)], curves=[dict(label,color,predict_fn)].
    같은 curves를 valid(위)/test(아래) 두 패널에 겹쳐 그린다."""
    set_korean_font()
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    for ax, (name, Xdf, ans_path) in zip(axes, splits):
        _panel(ax, name, _base(Xdf, ans_path), Xdf, curves)
    axes[1].set_xlabel("날짜")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out_path}")


def splits_of(Xva, Xte):
    return [('valid', Xva, ANS / "valid_Y_answer.csv"),
            ('test',  Xte, ANS / "test_Y_answer.csv")]
