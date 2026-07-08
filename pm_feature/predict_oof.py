"""
predict_oof.py
─────────────────────────────────────────────────────────────────
[이 파일이 하는 일]
  train 데이터(11,939 WF)에 대해 OOF(Out-Of-Fold) 예측을 생성하고,
  R2R 분석용 CSV를 저장한다.

    wafer_id(C64) / C33 / predicted_c65(OOF) / actual_c65(C65)

[왜 OOF인가?]
  모델이 자기가 학습한 웨이퍼를 예측하면 답을 '외워서' 잔차가
  비현실적으로 작아진다(과적합). 그 잔차로 R2R bias를 계산하면
  실제 운영에서 나올 오차를 과소평가한다.

  OOF는 이를 피한다:
    - train을 K개 fold로 나눔
    - 각 fold를 '빼고' 나머지로 학습 → 그 fold를 예측
    - K번 반복 → 모든 웨이퍼가 '자기를 안 본 모델'의 정직한 예측을 가짐
  → 이렇게 얻은 잔차(실측 − 예측)가 진짜 일반화 오차를 반영한다.

[학습 설정]
  train.py와 동일한 top-10 피처 + 동일한 Optuna 하이퍼파라미터를 사용한다.
  fold 분할: 5-fold, seed 42 (프로젝트 CV 관례와 동일).
  각 fold의 early stopping은 held-out fold 기준(표준 OOF 관례).

[검증]
  출력되는 OOF RMSE는 CV RMSE(~41) 근처여야 정상이다.
  ~15처럼 낮으면 leak, valid RMSE(~38)와 같으면 다른 데이터를 본 것.

사용:
  python src/predict_oof.py
  → outputs/oof_predictions.csv 저장
─────────────────────────────────────────────────────────────────
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import PROCESSED_DATA_PATH, ID_COL, TARGET_COL
from train import PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS, TOP_N_FEATURES

FEATURE_IMPORTANCE_PATH = ROOT / "outputs" / "feature_importance.csv"
OUT_PATH                = ROOT / "outputs" / "oof_predictions.csv"

N_FOLDS = 5
SEED    = 42
ORDER_COL = 'C33'   # 처리 순서 대용 (FDC 시간 카운터, PM마다 리셋)


def _get_top_features() -> list[str]:
    """train.py와 동일하게 feature_importance.csv에서 gain 상위 N개를 가져온다."""
    fi = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return fi[fi['gain'] > 0].head(TOP_N_FEATURES)['feature'].tolist()


def generate_oof() -> pd.DataFrame:
    print("[1/3] 데이터 로드 중...")
    train = pd.read_csv(ROOT / PROCESSED_DATA_PATH)
    feats = _get_top_features()
    print(f"  train: {len(train):,}개 WF / 피처 {len(feats)}개")
    print(f"  피처: {feats}")

    X = train[feats]
    y = train[TARGET_COL].to_numpy(dtype=float)

    oof = np.full(len(train), np.nan)   # OOF 예측을 담을 배열

    print(f"\n[2/3] {N_FOLDS}-fold OOF 예측 생성 중 (seed={SEED})...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for i, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
        dvalid = lgb.Dataset(X.iloc[va_idx], label=y[va_idx], reference=dtrain)
        model = lgb.train(
            params=PARAMS,
            train_set=dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        # held-out fold(=이 모델이 본 적 없는 웨이퍼)만 예측 → OOF
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        fold_rmse = np.sqrt(np.mean((y[va_idx] - oof[va_idx]) ** 2))
        print(f"  fold {i}/{N_FOLDS}: best_iter={model.best_iteration:>4}  "
              f"fold RMSE={fold_rmse:.4f}")

    oof_rmse = np.sqrt(np.mean((y - oof) ** 2))
    print(f"\n  ▶ 전체 OOF RMSE: {oof_rmse:.4f}  "
          f"(CV값 ~41 근처면 정상 / ~15면 leak 의심)")

    print("\n[3/3] 결과 정리 중...")
    out = pd.DataFrame({
        'wafer_id':      train[ID_COL].to_numpy(),
        'C33':           train[ORDER_COL].to_numpy(),
        'predicted_c65': oof,
        'actual_c65':    y,
    })
    # 처리 순서 대용으로 정렬: PM 이전→이후, 각 구간 내 C33 오름차순(≈시간순)
    if 'is_post_pm' in train.columns:
        out['_is_post_pm'] = train['is_post_pm'].to_numpy()
        out = out.sort_values(['_is_post_pm', 'C33']).drop(columns='_is_post_pm')
    else:
        out = out.sort_values('C33')
    out = out.reset_index(drop=True)
    return out


def main():
    out = generate_oof()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n[완료] 저장: {OUT_PATH.relative_to(ROOT)}  ({len(out):,}행)")
    print(out.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
