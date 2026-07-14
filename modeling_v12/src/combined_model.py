"""
combined_model.py
─────────────────────────────────────────────────────────────────
[프로젝트 기본 모델 (2026-07-08 채택)]
  팀원 row-level 센서 프레임 + 우리 시간/레짐 피처(v2) 결합 모델.

구조:
  row(=step) 단위 예측 → WF 내 평균 집계
  피처 = 핵심 센서 7종 × (raw+집계4) 35 + 구조/범주형 4
         + C23 타깃인코딩 1 + 시간/레짐 7 (통합명) = 46 (+TE 47)
         (다이어트 이력 192→127→46 — 아래 DROP_DIET 참고. 정체 미상 피처 0개)
  학습 = GroupKFold(C64) 5-fold 앙상블 (C23 TE는 fold별 재계산 — 누수 차단)
  예측 = 5개 fold 모델 평균 → WF 평균

채택 근거 (docs/COMBINED_MODEL.md 상세):
  - 새 사이클 위상 강건성 (m→m δ=22: lot-aware 23.5 vs WF-level 30.4)
  - 센서 앵커: 시간 피처가 보정 안 된 구간에서 챔버 상태를 직접 읽음 (7종 전부 정체 식별)
  - 운영(FC lot-aware)은 동급 (73.5 vs 74.0)
  - 트레이드오프: 정적 valid 44.30 (우리 WF-level 37.94 대비 -6.4) 감수

사용:
  from combined_model import build_rows, CombinedModel
  학습 스크립트: python src/train_combined.py
─────────────────────────────────────────────────────────────────
"""

import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import ID_COL, TARGET_COL   # 'C64', 'C65'

# ─────────────────────────────────────────
# 피처 정의 (팀원 modeling_v5 원본 규칙 유지)
# ─────────────────────────────────────────
DROP_IDS   = ["C34", "C35", "C38"]
DROP_LOT   = ["C20", "C21", "C22"]
DROP_CONST = ["C14", "C24", "C30"]
DROP_EXCL  = ["C26", "C28", "C29", "C37"]
DROP_ALLNA = ["C2", "C13", "C43", "C47", "C53", "C55"]
DROP_TIME  = ["C10", "C39", "C40", "C41"]
DROP_DUP   = ["C36"]

# ── 이 배포판은 FULL(다이어트 전, 192피처) 구성 ──
# 원 프로젝트에서는 센서 다이어트로 46피처까지 줄였으나(D4b), 이 핸드오프는
# 다이어트 전 전체 센서를 쓰는 FULL 모델(valid 48.76)이다. DROP_DIET=[]로 고정.
# (다이어트 상세는 원 프로젝트 COMBINED_FEATURES.md 참고 — 이 배포판엔 미포함)
DROP_DIET = []

DROP_ALL = (DROP_IDS + DROP_LOT + DROP_CONST + DROP_EXCL + DROP_ALLNA
            + DROP_TIME + DROP_DUP + DROP_DIET)

CAT_COLS = ["C6", "C7"]      # LightGBM native categorical (레시피, step)
TE_COL   = "C23"             # out-of-fold 타깃인코딩 대상 (28종 recipe)
TE_SMOOTH = 20

# 우리 시간/레짐 피처 — 통합명 (2026-07-09 요란/조용 규약 A안: is_post_loud_pm 체계).
# wf_features.csv 등 WF-level 파이프라인 산출물에서 C64 기준 merge.
# is_special_recipe는 스윕상 잉여(C6 범주형과 중복)지만 무해 + 전 측정치와 일관 → 유지.
TIME_FEATS = ["is_post_loud_pm", "days_since_last_pm", "hour",
              "dslp_x_hour", "hour_x_c33", "post_pm_days", "is_special_recipe"]

# 하이퍼파라미터 (팀원 v5 원본 — 결합 상태에서의 재튜닝은 미실시)
PARAMS = dict(objective="regression", metric="rmse", boosting_type="gbdt",
              learning_rate=0.03, num_leaves=127, max_depth=-1, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
              reg_alpha=0.5, reg_lambda=1.0, n_estimators=4000,
              random_state=42, n_jobs=-1, verbose=-1)
N_FOLDS = 5
EARLY_STOPPING = 100

MODEL_PATH = ROOT / "models" / "combined_model_full.pkl"


# ─────────────────────────────────────────
# row-level 피처 빌드
# ─────────────────────────────────────────

def build_rows(raw: pd.DataFrame, wf_time: pd.DataFrame,
               has_target: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """원본(step 단위) + WF-level 시간피처 → row-level 피처 프레임.

    raw     : 원본 CSV (train_data.csv / valid_X.csv / test_X.csv) 그대로
    wf_time : WF-level 파이프라인 산출물 (wf_features.csv 등) — TIME_FEATS 소스
    반환    : (rows DataFrame, 피처 컬럼 목록)

    ⚠️ raw는 read_csv 직후의 RangeIndex여야 한다 (row_pos 계산이 인덱스 정렬에 의존).
    """
    df = raw.copy()
    excl = set([ID_COL, TARGET_COL] + DROP_ALL + CAT_COLS + [TE_COL])
    sensors = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]

    # WF 전역 context: 주요 센서의 WF 집계를 각 row에 broadcast
    g = df.groupby(ID_COL)
    ctx = g[sensors].agg(["mean", "std", "min", "max"])
    ctx.columns = [f"{c}_wf_{s}" for c, s in ctx.columns]
    ctx["wf_nrows"] = g.size()
    df = df.merge(ctx, on=ID_COL, how="left")
    df["row_pos"] = g.cumcount()

    feat_cols = sensors + list(ctx.columns) + ["row_pos"] + CAT_COLS
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    out = df[[ID_COL] + feat_cols].copy()
    out[TE_COL] = df[TE_COL].values
    if has_target:
        out[TARGET_COL] = df[TARGET_COL].values

    out = out.merge(wf_time[[ID_COL] + TIME_FEATS], on=ID_COL, how="left")
    assert out[TIME_FEATS].isna().sum().sum() == 0, "시간피처 merge 누락 — wf_time에 없는 WF 존재"
    return out, feat_cols + TIME_FEATS


def _te_fit(frame: pd.DataFrame, col: str = TE_COL, m: int = TE_SMOOTH):
    """스무딩 타깃인코딩 (fold 학습분만 사용 — 호출측에서 보장)."""
    prior = frame[TARGET_COL].mean()
    agg = frame.groupby(col)[TARGET_COL].agg(["mean", "count"])
    enc = (agg["mean"] * agg["count"] + prior * m) / (agg["count"] + m)
    return enc, prior


# ─────────────────────────────────────────
# 결합 모델 (5-fold 앙상블)
# ─────────────────────────────────────────

class CombinedModel:
    """5-fold GroupKFold 앙상블 + fold별 C23 TE 인코더를 하나로 묶은 모델."""

    def __init__(self):
        self.models: list = []
        self.encs:   list = []   # (enc Series, prior) per fold
        self.use:    list[str] | None = None

    def fit(self, rows: pd.DataFrame, feats: list[str], verbose: bool = True):
        self.models, self.encs = [], []
        self.use = feats + [TE_COL + "_te"]
        y  = rows[TARGET_COL].values
        gr = rows[ID_COL].astype("category").cat.codes.values
        oof = np.zeros(len(rows))
        for k, (tri, vai) in enumerate(GroupKFold(N_FOLDS).split(rows, y, gr), 1):
            tf, vf = rows.iloc[tri].copy(), rows.iloc[vai].copy()
            enc, prior = _te_fit(tf)
            tf[TE_COL + "_te"] = tf[TE_COL].map(enc).fillna(prior)
            vf[TE_COL + "_te"] = vf[TE_COL].map(enc).fillna(prior)
            m = lgb.LGBMRegressor(**PARAMS)
            m.fit(tf[self.use], tf[TARGET_COL],
                  eval_set=[(vf[self.use], vf[TARGET_COL])],
                  categorical_feature=CAT_COLS,
                  callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                             lgb.log_evaluation(0)])
            oof[vai] = m.predict(vf[self.use])
            self.models.append(m)
            self.encs.append((enc, prior))
            if verbose:
                print(f"  fold{k}/{N_FOLDS} best_iter={m.best_iteration_}")
        # OOF(WF 단위) 기록
        oof_wf = pd.DataFrame({ID_COL: rows[ID_COL].values, "p": oof}).groupby(ID_COL)["p"].mean()
        y_wf   = rows.groupby(ID_COL)[TARGET_COL].first()
        self.oof_rmse_ = float(np.sqrt(np.mean(
            (y_wf.loc[oof_wf.index].values - oof_wf.values) ** 2)))
        return self

    def predict_rows(self, rows: pd.DataFrame) -> np.ndarray:
        """row 단위 예측 (fold 앙상블 평균)."""
        p = np.zeros(len(rows))
        for m, (enc, prior) in zip(self.models, self.encs):
            te = rows[TE_COL].map(enc).fillna(prior)
            X = rows[self.use[:-1]].copy()
            X[TE_COL + "_te"] = te.values
            p += m.predict(X) / len(self.models)
        return p

    def predict_wf(self, rows: pd.DataFrame) -> pd.Series:
        """WF 단위 예측 (row 예측 → WF 평균). index=C64."""
        return (pd.DataFrame({ID_COL: rows[ID_COL].values, "p": self.predict_rows(rows)})
                .groupby(ID_COL)["p"].mean())

    def save(self, path=MODEL_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=MODEL_PATH) -> "CombinedModel":
        with open(path, "rb") as f:
            return pickle.load(f)
