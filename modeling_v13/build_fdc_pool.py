# -*- coding: utf-8 -*-
"""
build_fdc_pool.py  ―  modeling_v13
────────────────────────────────────────────────────────────────────────────
COMBINED_FULL_MODEL 다이어트용 **FDC 집계 풀(웨이퍼 5-fold OOF 학습 데이터)** 생성.

■ 풀 정의
    27센서 = 표준 pm_feature 23센서 + 필수 복원 4센서(C11·C15·C16·C31)
    이 중 수치 26종  ×  [mean, std, max, min, last] 5통계  ×  Step(C7) pivot
    (+ C41_max_step = Step 소요시간)   →  655 피처
    * C6(레시피)는 문자열이라 수치 집계에서 자연 제외(다이어트 아님).

■ 스킴(원본 파이프라인과 동일: pm_feature/feature_engineering.make_fdc_features)
    1) C40(측정시각) 파싱 → 전체 시간정렬  (※ 'last' 통계가 '가장 최근 측정값'이 되도록)
    2) groupby([C64, C7]) 로 5통계 집계 → 다중레벨 컬럼 평탄화
    3) C41 은 groupby([C64,C7]).max() → C41_max
    4) pivot(index=C64, columns=C7) → {센서}_{통계}_step{N} 열로 펼침
       (WF가 갖지 않은 Step 칸은 NaN, 단일행 그룹의 std 도 NaN → 트리계열이 자체 처리)

■ 출력(웨이퍼=C64 1행 단위)
    [C64, fold_kf5, C20, <655 피처>, C65]
      · fold_kf5 : KFold(n_splits=5, shuffle=True, random_state=42),
                   C64 오름차순 정렬 순서 기준 배정  (프로젝트 표준 웨이퍼 CV)
      · C20      : Lot ID (**CV 메타데이터** — GroupKFold(C20) 정직-CV용. 절대 피처로 쓰지 말 것)
      · C65      : 타깃(Defect Test) — 낮을수록 좋음

사용:  python build_fdc_pool.py [train_data.csv 경로] [출력 디렉터리]
       기본 경로 = ../문제1(하)/train_data.csv , 출력 = ./data
────────────────────────────────────────────────────────────────────────────
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# ── 경로 (인자로 덮어쓰기 가능) ──
HERE = Path(__file__).resolve().parent
RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "문제1(하)" / "train_data.csv"
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL, STEP_COL, TARGET_COL, SORT_COL, LOT_COL = "C64", "C7", "C65", "C40", "C20"
DUR_COL = "C41"
AGG_FUNCS = ["mean", "std", "max", "min", "last"]

# 표준 pm_feature 23센서 (다중공선성 필터 후) — 부록 참조
FDC_STD23 = ["C4","C6","C12","C17","C18","C25","C27","C32","C42","C46","C48",
             "C49","C50","C52","C54","C56","C57","C58","C59","C60","C61","C62","C63"]
# v13 필수 복원 (group3 다중공선성으로 표준풀서 제외됐던 물리 센서)
RESTORE = ["C11", "C15", "C16", "C31"]     # 플라즈마 · 가스 · 가스 · RF
REQUIRED5 = ["C17", "C11", "C31", "C15", "C16"]
POOL_SENSORS = FDC_STD23 + RESTORE          # 27 (C6 문자열 포함)


def main():
    print(f"[1/5] load + 시간정렬 : {RAW}")
    df = pd.read_csv(RAW)
    df[SORT_COL] = pd.to_datetime(df[SORT_COL], format="%Y-%m-%d %H:%M:%S.%f")
    df = df.sort_values(SORT_COL).reset_index(drop=True)
    numeric = [c for c in POOL_SENSORS if pd.api.types.is_numeric_dtype(df[c])]
    print(f"      raw {df.shape[0]:,}행, WF={df[ID_COL].nunique():,}, 수치센서 {len(numeric)}종")

    print("[2/5] WF+Step 5통계 집계 → pivot")
    fdc = df.groupby([ID_COL, STEP_COL])[numeric].agg(AGG_FUNCS)
    fdc.columns = ["_".join(c) for c in fdc.columns]
    fdc = fdc.reset_index()
    dur = df.groupby([ID_COL, STEP_COL])[DUR_COL].max().rename("C41_max").reset_index()
    fdc = fdc.merge(dur, on=[ID_COL, STEP_COL])
    wide = fdc.pivot(index=ID_COL, columns=STEP_COL)
    wide.columns = [f"{c0}_step{int(c1)}" for c0, c1 in wide.columns]
    wide = wide.reset_index()
    feat_cols = [c for c in wide.columns if c != ID_COL]
    print(f"      피처 {len(feat_cols)}개")

    print("[3/5] 타깃(C65)·Lot(C20 메타) 결합")
    tgt = df.groupby(ID_COL)[TARGET_COL].first().reset_index()
    lot = df.groupby(ID_COL)[LOT_COL].first().reset_index()
    tbl = wide.merge(lot, on=ID_COL).merge(tgt, on=ID_COL)

    print("[4/5] 웨이퍼 5-fold 배정 (KFold5, shuffle, seed42; C64 오름차순)")
    tbl = tbl.sort_values(ID_COL).reset_index(drop=True)
    tbl["fold_kf5"] = -1
    for k, (_, vi) in enumerate(KFold(5, shuffle=True, random_state=42).split(tbl)):
        tbl.loc[vi, "fold_kf5"] = k
    assert (tbl["fold_kf5"] >= 0).all()

    tbl = tbl[[ID_COL, "fold_kf5", LOT_COL] + feat_cols + [TARGET_COL]]
    fc = tbl.select_dtypes(include=[np.number]).columns.drop(["fold_kf5"])
    tbl[fc] = tbl[fc].apply(lambda s: s.map(lambda v: float(f"{v:.6g}") if pd.notna(v) else v))

    print("[5/5] 저장")
    tbl.to_csv(OUT_DIR / "v13_fdc_pool_wf_oof.csv", index=False)
    tbl.to_csv(OUT_DIR / "v13_fdc_pool_wf_oof.csv.gz", index=False, compression="gzip")
    try:
        tbl.to_parquet(OUT_DIR / "v13_fdc_pool_wf_oof.parquet", index=False)
    except Exception as e:
        print("      parquet skip(설치 시 생성):", e)

    print(f"\n[완료] {tbl.shape[0]:,} WF × {tbl.shape[1]} 열 (피처 {len(feat_cols)})")
    print(f"       fold: {tbl['fold_kf5'].value_counts().sort_index().tolist()}")
    for s in REQUIRED5:
        print(f"       필수센서 {s}: {sum(c.startswith(s+'_') for c in feat_cols)}개 피처")


if __name__ == "__main__":
    main()
