"""
v9_features.py
─────────────────────────────────────────────
modeling_v9 (gpu_master_v5_optimal_lot_optuna_v2.py) 의 '테이블형 피처'를
웨이퍼(C64) 단위 한 줄로 재현하는 모듈.

원본 스크립트의 피처 생성부(비급 1·3, Step 집계, LOT 집계, 교호)를
그대로 옮겨오되, train에서 확정한 (컬럼 목록 + 결측 대치값)을 valid/test에
동일 적용할 수 있게 fit/transform 형태로 감쌌다.

⚠️ 1D-CNN 은 '피처'가 아니라 별도 모델이라 제외한다(요청: 피처 결합).
⚠️ 타깃(C65) 관련 처리(clipping 등)도 제외 — 여긴 X(피처)만 만든다.
"""
import numpy as np
import pandas as pd

# 원본 modeling_v9 의 센서 목록 그대로
V9_SENSORS = ['C1', 'C9', 'C11', 'C12', 'C15', 'C16', 'C17', 'C18', 'C25', 'C27',
              'C31', 'C32', 'C48', 'C52', 'C57', 'C58', 'C59', 'C60', 'C61', 'C62', 'C63']
ID_COL = 'C64'
LOT_COL = 'C20'
STEP_COL = 'C7'
LOT_SENSORS = ['C17', 'C63', 'C62', 'C61', 'C12', 'C16']


def build_v9_wafer_features(df: pd.DataFrame) -> pd.DataFrame:
    """raw(step 단위) DataFrame → 웨이퍼(C64) 단위 modeling_v9 피처 테이블.

    반환: index=C64 인 DataFrame (대치 전 원시값; inf만 NaN 처리).
    """
    wf_index = df.groupby(ID_COL).size().index
    wf_lot = df.groupby(ID_COL)[LOT_COL].first()
    feat = pd.DataFrame(index=wf_index)

    # 비급 3: Missing Indicator (센서별 결측이 하나라도 있으면 1)
    missing_counts = df.groupby(ID_COL)[V9_SENSORS].apply(lambda x: x.isna().sum())
    for s in V9_SENSORS:
        if missing_counts[s].sum() > 0:
            feat[f'{s}_is_missing'] = (missing_counts[s] > 0).astype(int)

    # Step(1~6)별 mean / std 집계
    for step in [1, 2, 3, 4, 5, 6]:
        sub = df[df[STEP_COL] == step]
        grp = sub.groupby(ID_COL)[V9_SENSORS]
        feat = feat.join(grp.mean().add_prefix(f's{step}_mean_'))
        feat = feat.join(grp.std().add_prefix(f's{step}_std_'))

    # LOT 집계 (누수 허용 방식 — 센서값을 LOT 단위로 broadcast; 타깃 미사용)
    s4 = df[df[STEP_COL] == 4].groupby(ID_COL)[V9_SENSORS].mean()
    s4 = s4.reindex(wf_index)
    s4[LOT_COL] = wf_lot
    for s in LOT_SENSORS:
        feat[f'LOT_mean_{s}'] = s4.groupby(LOT_COL)[s].transform('mean')
        feat[f'LOT_std_{s}'] = s4.groupby(LOT_COL)[s].transform('std')

    # 교호(interaction)
    feat['C17_x_C63'] = s4['C17'] * s4['C63']
    feat['C12_x_C62'] = s4['C12'] * s4['C62']

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat.index.name = ID_COL
    return feat


class V9FeatureBuilder:
    """train 에서 (컬럼 목록 + 결측 대치값)을 학습해 valid/test 에 동일 적용."""

    def __init__(self):
        self.columns_ = None
        self.medians_ = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = build_v9_wafer_features(df)
        self.columns_ = list(feat.columns)
        self.medians_ = feat.median(numeric_only=True)
        return feat.fillna(self.medians_).fillna(0)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.columns_ is not None, "fit_transform 을 먼저 호출하세요."
        feat = build_v9_wafer_features(df)
        # train 과 동일한 컬럼 집합으로 정렬 (없는 건 0, 추가된 건 버림)
        feat = feat.reindex(columns=self.columns_)
        return feat.fillna(self.medians_).fillna(0)
