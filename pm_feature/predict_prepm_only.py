"""
predict_prepm_only.py
─────────────────────────────────────────────────────────────────
[이 파일이 하는 일]
  PM 이전(pre-PM) 데이터로만 학습한 뒤 PM 이후(post-PM)를 예측하고,
  predict_oof.py와 동일한 4컬럼 CSV를 저장한다.

    wafer_id(C64) / C33 / predicted_c65 / actual_c65   (post-PM 웨이퍼만)

[무엇을 시뮬레이션하나 — COLD 스타트 / 크로스-사이클]
  "이전 사이클(pre-PM)만 본 모델이 새 사이클(post-PM)에 진입"하는 상황.
  train(pre-PM)과 예측대상(post-PM)이 완전히 분리돼 leak이 불가능하다.
  → OOF처럼 K-fold를 쓸 필요 없이, pre-PM 학습 → post-PM 예측 한 번.

[예상 결과 — 큰 오차는 정상이다]
  is_post_pm / post_pm_days 는 pre-PM에서 전부 0(상수)이라 모델이
  레짐 점프(post-PM은 C65가 +486 높음)를 배울 방법이 없다.
  → post-PM을 pre-PM 레벨(~638)로 과소예측 → raw RMSE가 매우 큼(~700대).
  이것이 R2R bias 보정이 필요한 이유를 보여주는 원재료다(문서 COLD: 748→R2R 90).

[학습 설정]
  predict_oof.py / train.py 와 동일한 top-10 피처 + Optuna 파라미터.
  early stopping은 pre-PM 내부 홀드아웃(15%)으로만 수행 → post-PM 미접촉.

사용:
  python src/predict_prepm_only.py
  → outputs/prepm_to_postpm_predictions.csv 저장
─────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

try:
    sys.stdout.reconfigure(encoding='utf-8')   # Windows 콘솔(cp949) 유니코드 크래시 방지
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import PROCESSED_DATA_PATH, ID_COL, TARGET_COL
from train import PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS, TOP_N_FEATURES

FEATURE_IMPORTANCE_PATH = ROOT / "outputs" / "feature_importance.csv"
OUT_PATH                = ROOT / "outputs" / "prepm_to_postpm_predictions.csv"

SEED      = 42
ORDER_COL = 'C33'
REGIME_COL = 'is_post_pm'   # 0 = pre-PM, 1 = post-PM


def _get_top_features() -> list[str]:
    fi = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return fi[fi['gain'] > 0].head(TOP_N_FEATURES)['feature'].tolist()


def run() -> pd.DataFrame:
    print("[1/3] 데이터 로드 및 pre/post 분리 중...")
    df = pd.read_csv(ROOT / PROCESSED_DATA_PATH)
    feats = _get_top_features()

    pre  = df[df[REGIME_COL] == 0]    # PM 이전 (학습용)
    post = df[df[REGIME_COL] == 1]    # PM 이후 (예측 대상)
    print(f"  pre-PM  (학습): {len(pre):,}개 WF")
    print(f"  post-PM (예측): {len(post):,}개 WF")
    print(f"  피처 {len(feats)}개: {feats}")

    X_pre,  y_pre  = pre[feats],  pre[TARGET_COL].to_numpy(dtype=float)
    X_post, y_post = post[feats], post[TARGET_COL].to_numpy(dtype=float)

    print("\n[2/3] pre-PM 학습 중 (early stopping은 pre-PM 내부 홀드아웃)...")
    # post-PM을 절대 건드리지 않도록 early stopping용 검증셋도 pre-PM에서만 분리
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_pre, y_pre, test_size=0.15, random_state=SEED)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_es, label=y_es, reference=dtrain)
    model = lgb.train(
        params=PARAMS,
        train_set=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    print(f"  best_iteration: {model.best_iteration}")

    print("\n[3/3] post-PM 예측 중...")
    pred_post = model.predict(X_post, num_iteration=model.best_iteration)
    raw_rmse  = np.sqrt(np.mean((y_post - pred_post) ** 2))
    bias      = float(np.mean(pred_post - y_post))
    print(f"  ▶ post-PM raw RMSE: {raw_rmse:.2f}  (큰 값이 정상 — 레짐 점프 미학습)")
    print(f"  ▶ 평균 bias(예측−실측): {bias:+.1f}  (음수 = 과소예측, R2R이 이 레벨을 흡수)")

    out = pd.DataFrame({
        'wafer_id':      post[ID_COL].to_numpy(),
        'C33':           post[ORDER_COL].to_numpy(),
        'predicted_c65': pred_post,
        'actual_c65':    y_post,
    }).sort_values('C33').reset_index(drop=True)   # post-PM 내부 C33 오름차순 ≈ 시간순
    return out


def main():
    out = run()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n[완료] 저장: {OUT_PATH.relative_to(ROOT)}  ({len(out):,}행)")
    print(out.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
