"""
SK Hynix Defect Test Prediction — 학습 파이프라인
FDC Trace → C65 (Defect Test) 회귀 예측 | 평가: RMSE

사용법:
    python train.py                # 기본 파라미터로 학습
    python train.py --tune         # Optuna 하이퍼파라미터 튜닝 후 학습
    python train.py --tune --n_trials 200   # trial 수 지정
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# ─── 경로 설정 ───────────────────────────────────────────────
DATA_DIR = Path("문제1(하)")
ANS_DIR = Path("문제1_하_answer")
OUTPUT_DIR = Path("outputs")

WF_ID = "C64"
TARGET = "C65"
N_FOLDS = 5

DROP_IDS = ["C34", "C35", "C38"]
DROP_LOT_IDS = ["C20", "C21", "C22"]
DROP_CONST = ["C14", "C24"]
DROP_EXCLUDE = ["C26", "C28", "C29", "C37"]
DROP_ALLNA = ["C2", "C13", "C43", "C47", "C53", "C55"]
DROP_TIME_RAW = ["C10", "C39", "C40"]
DROP_ALL = DROP_IDS + DROP_LOT_IDS + DROP_CONST + DROP_EXCLUDE + DROP_ALLNA + DROP_TIME_RAW

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}


# ─── 피처 엔지니어링 ─────────────────────────────────────────
def compute_slope(series):
    y = series.dropna().values
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope


def extract_wf_features(df, has_target=True):
    meta_cols = [WF_ID] + DROP_ALL + ([TARGET] if has_target else [])
    cat_cols = ["C6", "C7"]
    all_exclude = set(meta_cols + cat_cols)
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in all_exclude
    ]

    groups = df.groupby(WF_ID)

    agg_df = groups[numeric_cols].agg(["mean", "std", "min", "max", "median"])
    agg_df.columns = [f"{c}_{fn}" for c, fn in agg_df.columns]

    range_df = groups[numeric_cols].apply(lambda x: x.max() - x.min())
    range_df.columns = [f"{c}_range" for c in numeric_cols]

    delta_df = groups[numeric_cols].apply(lambda x: x.iloc[-1] - x.iloc[0])
    delta_df.columns = [f"{c}_delta" for c in numeric_cols]

    slope_df = groups[numeric_cols].apply(lambda x: x.apply(compute_slope))
    slope_df.columns = [f"{c}_slope" for c in numeric_cols]

    meta_df = pd.DataFrame(index=groups.groups.keys())
    meta_df["n_rows"] = groups.size()
    meta_df["C41_total"] = groups["C41"].max()
    meta_df["C33_first"] = groups["C33"].first()
    meta_df["C33_max"] = groups["C33"].max()

    if "C6" in df.columns:
        c6_dummies = pd.get_dummies(
            df[[WF_ID, "C6"]].set_index(WF_ID)["C6"], prefix="C6"
        )
        c6_feat = c6_dummies.groupby(level=0).mean()
    else:
        c6_feat = pd.DataFrame(index=groups.groups.keys())

    if "C7" in df.columns:
        c7_dummies = pd.get_dummies(
            df[[WF_ID, "C7"]].astype(str).set_index(WF_ID)["C7"], prefix="C7"
        )
        c7_feat = c7_dummies.groupby(level=0).mean()
    else:
        c7_feat = pd.DataFrame(index=groups.groups.keys())

    features = pd.concat(
        [agg_df, range_df, delta_df, slope_df, meta_df, c6_feat, c7_feat], axis=1
    )
    features.index.name = WF_ID
    features = features.reset_index()

    if has_target:
        target_df = groups[TARGET].first().reset_index()
        features = features.merge(target_df, on=WF_ID)

    return features


# ─── 데이터 로드 & 전처리 ────────────────────────────────────
def load_data():
    print("데이터 로드 중...")
    train = pd.read_csv(DATA_DIR / "train_data.csv")
    valid_X = pd.read_csv(DATA_DIR / "valid_X.csv")
    test_X = pd.read_csv(DATA_DIR / "test_X.csv")

    valid_Y_prob = pd.read_csv(DATA_DIR / "valid_Y_problem.csv")
    test_Y_prob = pd.read_csv(DATA_DIR / "test_Y_problem.csv")

    valid_Y_ans = pd.read_csv(ANS_DIR / "valid_Y_answer.csv")
    test_Y_ans = pd.read_csv(ANS_DIR / "test_Y_answer.csv")

    train_cols_no_target = [c for c in train.columns if c != TARGET]
    valid_X = valid_X[train_cols_no_target]
    test_X = test_X[train_cols_no_target]

    print(f"  train: {train.shape}, valid_X: {valid_X.shape}, test_X: {test_X.shape}")
    return train, valid_X, test_X, valid_Y_prob, test_Y_prob, valid_Y_ans, test_Y_ans


def build_features(train, valid_X, test_X, valid_Y_ans, test_Y_ans):
    print("피처 추출 중 (WF 단위 집계)...")
    t0 = time.time()

    train_feat = extract_wf_features(train, has_target=True)
    valid_feat = extract_wf_features(valid_X, has_target=False)
    test_feat = extract_wf_features(test_X, has_target=False)

    feature_cols = [c for c in train_feat.columns if c not in [WF_ID, TARGET]]
    common_cols = sorted(
        set(feature_cols) & set(valid_feat.columns) & set(test_feat.columns)
    )

    X_train = train_feat[common_cols].values
    y_train = train_feat[TARGET].values
    wf_train = train_feat[WF_ID].values

    X_valid = valid_feat[common_cols].values
    X_test = test_feat[common_cols].values
    wf_valid = valid_feat[WF_ID].values
    wf_test = test_feat[WF_ID].values

    valid_answer = valid_Y_ans.set_index("C64").loc[wf_valid, "C65"].values
    test_answer = test_Y_ans.set_index("C64").loc[wf_test, "C65"].values

    wf_to_int = {wf: i for i, wf in enumerate(np.unique(wf_train))}
    groups = np.array([wf_to_int[wf] for wf in wf_train])

    elapsed = time.time() - t0
    print(f"  피처 수: {len(common_cols)}, 소요: {elapsed:.1f}s")

    return (
        X_train, y_train, groups, wf_train,
        X_valid, X_test, wf_valid, wf_test,
        valid_answer, test_answer, common_cols,
    )


# ─── 학습 & 평가 ─────────────────────────────────────────────
def train_and_evaluate(params, X_train, y_train, groups, X_valid, X_test,
                       valid_answer, test_answer, verbose=True):
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.zeros(len(X_train))
    valid_preds = np.zeros(len(X_valid))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[val_idx], y_train[val_idx])],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        oof[val_idx] = model.predict(X_train[val_idx])
        valid_preds += model.predict(X_valid) / N_FOLDS
        test_preds += model.predict(X_test) / N_FOLDS

        fold_rmse = np.sqrt(mean_squared_error(y_train[val_idx], oof[val_idx]))
        fold_scores.append(fold_rmse)
        if verbose:
            print(f"  Fold {fold+1}: RMSE={fold_rmse:.4f}, best_iter={model.best_iteration_}")

    oof_rmse = np.sqrt(mean_squared_error(y_train, oof))
    valid_rmse = np.sqrt(mean_squared_error(valid_answer, valid_preds))
    test_rmse = np.sqrt(mean_squared_error(test_answer, test_preds))

    return oof_rmse, valid_rmse, test_rmse, valid_preds, test_preds, model


# ─── Optuna 튜닝 ─────────────────────────────────────────────
def run_optuna(X_train, y_train, groups, n_trials=100):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": 42,
            "n_jobs": -1,
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }

        gkf = GroupKFold(n_splits=N_FOLDS)
        oof = np.zeros(len(X_train))

        for _, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            model = lgb.LGBMRegressor(**p)
            model.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            oof[val_idx] = model.predict(X_train[val_idx])

        return np.sqrt(mean_squared_error(y_train, oof))

    print(f"\nOptuna 튜닝 시작 (n_trials={n_trials})...")
    study = optuna.create_study(direction="minimize", study_name="lgbm_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  최적 CV RMSE: {study.best_value:.4f}")
    print(f"  최적 파라미터:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    best_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
        "n_estimators": 3000,
        **study.best_params,
    }
    return best_params, study


# ─── 제출 파일 생성 ───────────────────────────────────────────
def save_submissions(wf_valid, valid_preds, wf_test, test_preds,
                     valid_Y_prob, test_Y_prob):
    OUTPUT_DIR.mkdir(exist_ok=True)

    valid_submit = valid_Y_prob.copy()
    valid_submit["C65"] = valid_submit["C64"].map(dict(zip(wf_valid, valid_preds)))
    valid_submit.to_csv(OUTPUT_DIR / "valid_Y_submit.csv", index=False)

    test_submit = test_Y_prob.copy()
    test_submit["C65"] = test_submit["C64"].map(dict(zip(wf_test, test_preds)))
    test_submit.to_csv(OUTPUT_DIR / "test_Y_submit.csv", index=False)

    print(f"\n제출 파일 저장:")
    print(f"  {OUTPUT_DIR / 'valid_Y_submit.csv'} — {valid_submit.shape}")
    print(f"  {OUTPUT_DIR / 'test_Y_submit.csv'}  — {test_submit.shape}")


def save_results(params, oof_rmse, valid_rmse, test_rmse):
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = {
        "oof_rmse": round(oof_rmse, 4),
        "valid_rmse": round(valid_rmse, 4),
        "test_rmse": round(test_rmse, 4),
        "params": params,
    }
    path = OUTPUT_DIR / "results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  결과 저장: {path}")


# ─── 메인 ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SK Hynix Defect Test Prediction")
    parser.add_argument("--tune", action="store_true", help="Optuna 하이퍼파라미터 튜닝 실행")
    parser.add_argument("--n_trials", type=int, default=100, help="Optuna trial 수 (기본: 100)")
    args = parser.parse_args()

    train, valid_X, test_X, valid_Y_prob, test_Y_prob, valid_Y_ans, test_Y_ans = load_data()

    (
        X_train, y_train, groups, wf_train,
        X_valid, X_test, wf_valid, wf_test,
        valid_answer, test_answer, common_cols,
    ) = build_features(train, valid_X, test_X, valid_Y_ans, test_Y_ans)

    baseline_rmse = np.sqrt(
        mean_squared_error(valid_answer, np.full_like(valid_answer, y_train.mean()))
    )
    print(f"\n베이스라인 RMSE (평균 예측): {baseline_rmse:.2f}")

    if args.tune:
        params, study = run_optuna(X_train, y_train, groups, n_trials=args.n_trials)
    else:
        params = DEFAULT_PARAMS

    print(f"\n{'='*50}")
    print("최종 학습 시작...")
    oof_rmse, valid_rmse, test_rmse, valid_preds, test_preds, model = train_and_evaluate(
        params, X_train, y_train, groups, X_valid, X_test, valid_answer, test_answer
    )

    print(f"\n{'='*50}")
    print(f"베이스라인 (평균)   RMSE : {baseline_rmse:.4f}")
    print(f"CV OOF              RMSE : {oof_rmse:.4f}")
    print(f"Valid (리더보드)    RMSE : {valid_rmse:.4f}")
    print(f"Test  (최종)        RMSE : {test_rmse:.4f}")
    print(f"{'='*50}")

    gap = abs(oof_rmse - valid_rmse)
    print(f"CV↔Valid 격차: {gap:.4f}", "✓ 안정적" if gap < 30 else "⚠ 과적합 의심")

    save_submissions(wf_valid, valid_preds, wf_test, test_preds, valid_Y_prob, test_Y_prob)
    save_results(params, oof_rmse, valid_rmse, test_rmse)


if __name__ == "__main__":
    main()
