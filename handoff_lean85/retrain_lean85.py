# -*- coding: utf-8 -*-
"""
retrain_lean85.py — lean-85 재학습 CLI (운영 SOP 진입점)

사용 (프로젝트 venv 활성화 후, handoff_lean85/ 에서):
    python retrain_lean85.py                          # 기본 경로 자동 탐색
    python retrain_lean85.py --data <누적_raw.csv> --pm-log <pm_log.json> --tag weekly
    python retrain_lean85.py --acceptance             # 재학습 + B2′ 수용검사(수 분 소요)

트리거 (README §재학습 SOP):
    ① 정기 주간(7일)   ② 요란 PM 기입 즉시   ③ PSI 경보/주간 RMSE 급등 시 수시
"""
import argparse
import json

import pandas as pd

import lean85_pipeline as lp


def main():
    ap = argparse.ArgumentParser(description="lean-85 재학습")
    ap.add_argument("--data", default=None, help="누적 raw 트레이스 CSV (기본: train_data.csv 자동 탐색)")
    ap.add_argument("--pm-log", default=None, help="pm_log.json 경로 (기본: 프로젝트 루트 자동 탐색)")
    ap.add_argument("--out", default="models", help="모델 버전 저장 폴더 (기본: handoff_lean85/models)")
    ap.add_argument("--tag", default=None, help="버전 태그 (예: weekly, postPM)")
    ap.add_argument("--acceptance", action="store_true", help="재학습 후 B2′ 수용검사 실행")
    a = ap.parse_args()

    data_p = a.data or lp.find_file("train_data.csv")
    pm_p = a.pm_log or lp.find_file("pm_log.json")
    print(f"[1/3] 데이터 로드: {data_p}")
    raw = pd.read_csv(data_p)
    print(f"      rows={len(raw):,} | pm_log={pm_p}")

    print("[2/3] 피처 빌드 + 재학습 (동결 파라미터)")
    vdir, model, mani = lp.retrain(raw, pm_p, out_dir=a.out, tag=a.tag)
    print(f"      웨이퍼 {mani['n_wafers_train']:,} | 기간 {mani['train_period'][0][:10]}"
          f" ~ {mani['train_period'][1][:10]}")
    print(f"      저장: {vdir}")

    if a.acceptance:
        print("[3/3] B2′ 수용검사 (기대 99.840±0.5)")
        lean, params, rounds = lp.load_frozen()
        table = lp.build_wafer_table(raw, pm_p, lean)
        acc = lp.walkforward_acceptance(table, lean, params, rounds)
        mani["acceptance"] = acc
        json.dump(mani, open(vdir / "manifest.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    else:
        print("[3/3] 수용검사 생략 (--acceptance 로 실행 가능)")

    print("\n완료. 예측:")
    print(f"  python predict_lean85.py --model \"{vdir}\" --data <신규_X.csv>")


if __name__ == "__main__":
    main()
