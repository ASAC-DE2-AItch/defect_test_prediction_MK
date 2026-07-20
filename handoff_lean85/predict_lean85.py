# -*- coding: utf-8 -*-
"""
predict_lean85.py — lean-85 신규 데이터 예측 CLI

사용 (handoff_lean85/ 에서):
    python predict_lean85.py --model models/lean85_<stamp> --data <신규_X.csv>
    python predict_lean85.py --model models/lean85_<stamp> --data <신규_X.csv> --out preds.csv

출력 CSV: [C64, pred_C65, low_confidence]
    low_confidence=1 : 요란 PM 후 7일 이내(레짐-온셋) — 오차 급증 구간, 참고용으로만.
주의: pm_log 는 반드시 **최신 운영 로그**를 쓸 것 (오래된 로그 = days/레짐 피처 왜곡).
"""
import argparse

import pandas as pd

import lean85_pipeline as lp


def main():
    ap = argparse.ArgumentParser(description="lean-85 예측")
    ap.add_argument("--model", required=True, help="모델 버전 폴더 (예: models/lean85_20260719_090000)")
    ap.add_argument("--data", required=True, help="신규 raw 트레이스 CSV (valid_X 형식)")
    ap.add_argument("--pm-log", default=None, help="최신 pm_log.json (기본: 프로젝트 루트 자동 탐색)")
    ap.add_argument("--out", default="preds_lean85.csv", help="예측 저장 경로")
    a = ap.parse_args()

    model, mani = lp.load_model(a.model)
    lean, _, _ = lp.load_frozen()
    assert mani["features_n"] == len(lean) == 85, "모델-피처 스펙 불일치"
    pm_p = a.pm_log or lp.find_file("pm_log.json")

    print(f"[1/3] 모델 {a.model} (학습 {mani['created_at_local'][:10]}, "
          f"웨이퍼 {mani['n_wafers_train']:,})")
    raw = pd.read_csv(a.data)
    print(f"[2/3] 피처 빌드: rows={len(raw):,} | pm_log={pm_p}")
    table = lp.build_wafer_table(raw, pm_p, lean)

    ok, reasons = lp.should_retrain(mani, pm_p)
    if ok:
        print("⚠️  재학습 권고 상태의 모델입니다:")
        for r in reasons:
            print(f"    - {r}")

    preds = lp.predict_lean85(model, table, lean)
    preds.to_csv(a.out, index=False)
    n_low = int(preds["low_confidence"].sum())
    print(f"[3/3] 저장: {a.out} | 웨이퍼 {len(preds):,} | "
          f"low_confidence {n_low:,} ({n_low / max(len(preds), 1):.1%})")
    if n_low:
        print("    ⚠️ 레짐-온셋(요란 PM 후 7일) 웨이퍼 포함 — 해당 예측은 참고용.")


if __name__ == "__main__":
    main()
