"""
merge_features.py — combined_model_full_handover ⊕ modeling_v9  피처 결합 모듈
──────────────────────────────────────────────────────────────────────
'피처 다이어트 없이' 두 모델의 피처를 그대로 합친다. 두 세트의 granularity가
다르므로 두 가지 결합 뼈대를 모두 제공한다.

  · merge_row_level(...)   : combined 의 row(step) 뼈대에 v9 웨이퍼피처를
                            C64 로 broadcast (combined 192 전부 보존 + v9 추가)
  · merge_wafer_level(...) : v9 의 웨이퍼 뼈대에 combined 의 웨이퍼단위 피처
                            (WF집계·구조·시간/레짐·C23)만 결합

의존:
  - combined_model_full_handover/src (config, preprocessing, feature_engineering, combined_model)
  - v9_features.py  (같은 폴더)
사용 예는 파일 하단 __main__ 참고.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 이 파일은 combined_model_full_handover/src/ 안에 두는 것을 전제로 한다
# (config.py, preprocessing.py, feature_engineering.py, combined_model.py 와 같은 폴더).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ID_COL, TARGET_COL                       # 'C64', 'C65'
from preprocessing import preprocess
from feature_engineering import build_features
from combined_model import build_rows, TIME_FEATS, TE_COL, CAT_COLS
from v9_features import V9FeatureBuilder, build_v9_wafer_features


# ── WF 시간/레짐 피처(combined) 산출 ────────────────────────────────
def make_wf_time(raw: pd.DataFrame, pm_bins, pm_log) -> pd.DataFrame:
    clean = preprocess(raw.copy())
    wf_time, _ = build_features(clean, pm_bins=pm_bins, pm_log=pm_log)
    return wf_time


# ── (A) row 뼈대 결합 ──────────────────────────────────────────────
def merge_row_level(raw: pd.DataFrame, wf_time: pd.DataFrame,
                    v9_builder: V9FeatureBuilder, fit_v9: bool,
                    has_target: bool = True):
    """combined row-level 프레임 + v9 웨이퍼피처(C64 broadcast).

    반환: (rows DataFrame, feature_cols list)
      rows 에는 ID/타깃/C23 도 포함. 모델 입력은 feature_cols(+C23_te) 사용.
    """
    x = raw if has_target else raw[[c for c in raw.columns if c != TARGET_COL]]
    rows, feats = build_rows(x, wf_time, has_target=has_target)
    v9 = v9_builder.fit_transform(raw) if fit_v9 else v9_builder.transform(raw)
    rows = rows.merge(v9, on=ID_COL, how="left")
    return rows, feats + list(v9.columns)


# ── (B) 웨이퍼 뼈대 결합 ───────────────────────────────────────────
def merge_wafer_level(raw: pd.DataFrame, wf_time: pd.DataFrame,
                      v9_builder: V9FeatureBuilder, fit_v9: bool,
                      has_target: bool = True):
    """v9 웨이퍼 테이블 + combined 의 웨이퍼단위 피처(WF집계·구조·시간/레짐·C6·C23).

    combined 의 row-level 원본센서 36개·row_pos·C7 은 웨이퍼 단위값이 없어 제외
    (그 정보는 이미 WF집계 _wf_ 로 들어있음). '다이어트'가 아니라 단위 정합상 자연 제외.

    반환: (table DataFrame[index=C64], feature_cols list)  — C23_te 는 학습 시 fold별 부여.
    """
    x = raw if has_target else raw[[c for c in raw.columns if c != TARGET_COL]]
    rows, feats = build_rows(x, wf_time, has_target=has_target)
    wl_cols = [c for c in feats if ("_wf_" in c) or c in ("wf_nrows", "C6") or c in TIME_FEATS]
    keep = wl_cols + [TE_COL] + ([TARGET_COL] if has_target else [])
    tbl = rows.groupby(ID_COL, observed=True)[keep].first()

    v9 = v9_builder.fit_transform(raw) if fit_v9 else v9_builder.transform(raw)
    tbl = tbl.join(v9, how="left")
    return tbl, wl_cols + list(v9.columns)


if __name__ == "__main__":
    import json
    PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
    pm_bins = np.array(json.load(open(PROC / "pm_bins.json")))
    pm_log = json.load(open(PROC / "pm_log.json"))
    raw = pd.read_csv(sys.argv[1])
    wt = make_wf_time(raw, pm_bins, pm_log)
    vb = V9FeatureBuilder()
    rows, cols = merge_row_level(raw, wt, vb, fit_v9=True, has_target=(TARGET_COL in raw.columns))
    print(f"[row 뼈대] rows={len(rows):,}  피처={len(cols)}개")
    vb2 = V9FeatureBuilder()
    tbl, cols2 = merge_wafer_level(raw, wt, vb2, fit_v9=True, has_target=(TARGET_COL in raw.columns))
    print(f"[웨이퍼 뼈대] wafers={len(tbl):,}  피처={len(cols2)}개")
