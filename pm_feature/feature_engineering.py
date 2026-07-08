"""
feature_engineering.py
─────────────────────────────────────────────
전처리된 행 단위 데이터를 WF 한 줄로 집계하는 파일.

집계 방법 (2단계):
  1단계 — WF+Step 단위로 통계값 계산
    · FDC 센서 24개 × [mean, std, max, min, last] 5가지
    · C41(Step 소요시간) → Step별 max(C41)
  2단계 — WF 한 줄로 펼치기 (pivot)
    · 컬럼명 형식: {센서}_{통계}_step{번호}
    · 예) C17_mean_step1, C17_max_step2

추가로 파생 피처(pm_phase, wf_row_count 등)를 만들어 합친다.
출력: data/processed/wf_features.csv  (train.py 가 읽어감)
─────────────────────────────────────────────
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import (
    CLEAN_DATA_PATH,
    PROCESSED_DATA_PATH,
    VALID_X_PATH,
    VALID_FEATURES_PATH,
    TEST_X_PATH,
    TEST_FEATURES_PATH,
    PM_BINS_PATH,
    PM_LOG_PATH,
    FDC_FEATURES,
    AGG_FUNCS,
    ID_COL,       # 'C64'  — WF ID
    STEP_COL,     # 'C7'   — 공정 Step 번호
    TARGET_COL,   # 'C65'  — 불량 비트 수 (예측 대상)
    SORT_COL,     # 'C40'  — 날짜 컬럼
    PM_SHIFT_DATE,  # '2018-12-24' — 이 시점 기준 C65 레벨 급변
)

# 시간 피처 기준 앵커 (train/valid/test 공통 고정값)
_REF_DATE  = pd.Timestamp('2018-12-01')      # 캠페인 시작일 = PM 이전 경과일 기준점

# ─────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────

def load_clean() -> pd.DataFrame:
    """전처리된 CSV(train_clean.csv)를 불러온다."""
    path = ROOT / CLEAN_DATA_PATH
    df = pd.read_csv(path, parse_dates=[SORT_COL])
    print(f"  로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")
    return df


# ─────────────────────────────────────────
# 2. FDC 센서 집계 (핵심 단계)
# ─────────────────────────────────────────

def make_fdc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    FDC 센서를 WF+Step 단위로 집계한 뒤 WF 한 줄로 펼친다.

    [1단계] groupby(WF, Step) → agg(mean, std, max, min, last)
      예) WF_1의 Step1에서 C17 측정값 [0.82, 0.83, 0.81]
          → C17_mean=0.82, C17_max=0.83, C17_min=0.81,
            C17_std=0.01, C17_last=0.81

    [2단계] pivot → WF 한 줄
      컬럼명: C17_mean_step1, C17_max_step1, C17_mean_step2, ...
      해당 Step이 없는 WF의 칸은 NaN (LightGBM이 자체 처리)

    주의: C6는 문자열 타입이라 수치 집계 불가.
          WF 내에서 값이 고정이므로 WF 단위 first()로 별도 처리.
    """
    # 수치 집계 가능한 FDC 컬럼과 문자열 FDC 컬럼 분리
    numeric_fdc = [c for c in FDC_FEATURES if pd.api.types.is_numeric_dtype(df[c])]
    string_fdc  = [c for c in FDC_FEATURES if not pd.api.types.is_numeric_dtype(df[c])]

    # --- C41: Step별 최대 경과시간 = Step 소요시간 대리 지표 ---
    step_duration = (
        df.groupby([ID_COL, STEP_COL])['C41']
        .max()
        .rename('C41_max')
        .reset_index()
    )

    # --- 수치 FDC 센서를 WF+Step 단위로 5가지 통계 집계 ---
    fdc_agg = (
        df.groupby([ID_COL, STEP_COL])[numeric_fdc]
        .agg(AGG_FUNCS)
    )

    # 다중 레벨 컬럼 평탄화: (C17, mean) → C17_mean
    fdc_agg.columns = ['_'.join(col) for col in fdc_agg.columns]
    fdc_agg = fdc_agg.reset_index()

    # C41_max 합치기
    fdc_agg = fdc_agg.merge(step_duration, on=[ID_COL, STEP_COL])

    # --- WF+Step → WF 한 줄로 피벗 ---
    fdc_wide = fdc_agg.pivot(index=ID_COL, columns=STEP_COL)

    # 컬럼명 변환: (C17_mean, 1.0) → C17_mean_step1
    fdc_wide.columns = [
        f'{col[0]}_step{int(col[1])}' for col in fdc_wide.columns
    ]
    fdc_wide = fdc_wide.reset_index()

    # --- 문자열 FDC 컬럼: WF 내 고정값이므로 WF 단위 first()로 추출 ---
    if string_fdc:
        str_wf = df.groupby(ID_COL)[string_fdc].first().reset_index()
        fdc_wide = fdc_wide.merge(str_wf, on=ID_COL)

    n_feat = fdc_wide.shape[1] - 1
    print(f"  FDC 집계 완료: {n_feat}개 피처 "
          f"(수치 {len(numeric_fdc)}개 × {len(AGG_FUNCS)}통계 × Step수 "
          f"+ C41_max + 문자열 {len(string_fdc)}개)")
    return fdc_wide


# ─────────────────────────────────────────
# 3. 메타 파생 피처 생성
# ─────────────────────────────────────────

def _compute_time_features(
    wf_dates: pd.Series,
    pm_log: list[str],
) -> pd.DataFrame:
    """
    WF 측정 날짜와 PM 이벤트 로그로부터 시간 피처를 계산한다.

    days_since_last_pm: 가장 최근 PM으로부터 경과일.
                        PM 이전 구간은 캠페인 시작일(_REF_DATE) 기준으로 경과일 계산.
                        PM이 발생할 때마다 0으로 리셋 → 복수 PM 주기 지원.
    pm_cycle_count:     지금까지 발생한 PM 횟수 (pre-PM=0, post-PM1=1, post-PM2=2, ...).
                        after_pm_shift(단순 0/1)보다 일반적 — PM이 반복돼도 구분 가능.
    """
    pm_timestamps = sorted([pd.Timestamp(d) for d in pm_log])

    days_vals  = []
    cycle_vals = []

    for d in wf_dates:
        past_pms = [p for p in pm_timestamps if p <= d]
        if past_pms:
            last_pm = past_pms[-1]
            days_vals.append((d - last_pm).total_seconds() / 86400)
            cycle_vals.append(len(past_pms))
        else:
            # PM 이전: 캠페인 시작일(_REF_DATE) 기준 경과일 (pre-PM 노화 트렌드 포착)
            days_vals.append((d - _REF_DATE).total_seconds() / 86400)
            cycle_vals.append(0)

    return pd.DataFrame(
        {'days_since_last_pm': days_vals, 'pm_cycle_count': cycle_vals},
        index=wf_dates.index,
    )


def make_meta_features(
    df: pd.DataFrame,
    pm_bins: np.ndarray | None = None,
    pm_log: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    WF 단위 메타 파생 피처를 생성한다.

    pm_bins=None  → train 모드: 데이터에서 구간 경계 계산 후 반환
    pm_bins=array → valid/test 모드: 전달된 경계를 그대로 적용
    pm_log=None   → train 모드: config의 PM_SHIFT_DATE 사용
    pm_log=list   → valid/test 모드: 저장된 pm_log 재사용

    반환: (meta DataFrame, pm_bins array)
    """
    wf = df.groupby(ID_COL)

    meta = wf['C33'].first().reset_index()

    if pm_bins is None:
        _, pm_bins = pd.qcut(meta['C33'], q=4, retbins=True, duplicates='drop')

    if pm_log is None:
        pm_log = [PM_SHIFT_DATE]

    # --- 시간 파생 피처 ---
    #   days_since_last_pm: PM 주기 내 경과일 (PM마다 리셋, pre-PM은 캠페인 시작 기준)
    #   pm_cycle_count    : PM 발생 횟수 (0=pre-PM, 1=1차 PM 이후, ...)
    #   hour              : 측정 시작 시각(0~23) — 3교대 근무·장비 열적 안정성 반영
    wf_date = wf[SORT_COL].first().reset_index().rename(columns={SORT_COL: '_date'})
    time_feats = _compute_time_features(wf_date['_date'], pm_log)
    wf_date    = pd.concat([wf_date, time_feats], axis=1)
    wf_date['hour'] = wf_date['_date'].dt.hour

    meta = meta.merge(
        wf_date[[ID_COL, 'days_since_last_pm', 'pm_cycle_count', 'hour']],
        on=ID_COL,
    )

    # --- 레짐 플래그 ---
    #   is_post_pm: PM 레짐 플래그 (0=pre-PM, 1=PM 이후).
    #   pm_cycle_count(숫자)는 cycle 3 예측 시 미학습 값(3)이 되므로 모델 피처로
    #   쓰지 않는다. "몇 번째 cycle인지"는 R2R/재학습이 다루는 외부 상태이고,
    #   모델에는 레짐 구분만 플래그로 제공한다 (cycle 간 교환가능성 확보).
    #   pm_cycle_count 컬럼 자체는 메타데이터로 계속 생성한다.
    meta['is_post_pm'] = (meta['pm_cycle_count'] > 0).astype(int)

    # --- 교호(interaction) 피처 ---
    #   시간 관련 상호작용이 C65 예측에 강한 신호를 준다 (실험으로 검증, -4.6 RMSE).
    #   구성 요소가 모두 프로덕션-안전 피처(하드코딩 날짜 없음)라 PM 반복에도 확장된다.
    #   dslp_x_hour : PM 이후 경과일에 따라 3교대(hour) 패턴의 효과가 달라지는 정도
    #   hour_x_c33  : 장비 시간 카운터(C33) 상태와 측정 시각의 복합 효과
    #   post_pm_days: PM 이후 구간의 경과일만 남기는 '마스크' (pre-PM=0).
    #                 pm_cycle_count를 배율로 곱하지 않고 (>0) 게이팅만 하므로
    #                 PM이 2회 이상 발생해도 값이 배로 왜곡되지 않는다(경과일 그대로 유지).
    meta['dslp_x_hour']  = meta['days_since_last_pm'] * meta['hour']
    meta['hour_x_c33']   = meta['hour']              * meta['C33']
    meta['post_pm_days'] = meta['days_since_last_pm'] * (meta['pm_cycle_count'] > 0)

    print(f"  메타 파생 완료: {meta.shape[1] - 1}개 피처")
    return meta, pm_bins


# ─────────────────────────────────────────
# 4. 타겟값 추출
# ─────────────────────────────────────────

def make_target(df: pd.DataFrame) -> pd.DataFrame | None:
    """WF 단위 타겟값(C65)을 추출한다. valid_X처럼 C65가 없으면 None 반환."""
    if TARGET_COL not in df.columns:
        return None
    return df.groupby(ID_COL)[TARGET_COL].first().reset_index()


# ─────────────────────────────────────────
# 5. 공통 빌드 함수
# ─────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    pm_bins: np.ndarray | None = None,
    pm_log: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    FDC 집계 + 메타 파생 + 타겟을 합쳐 WF 단위 피처 테이블을 만든다.
    train / valid / test 모두 이 함수를 사용한다.

    반환: (result DataFrame, pm_bins)
    """
    fdc           = make_fdc_features(df)
    meta, pm_bins = make_meta_features(df, pm_bins, pm_log)
    target        = make_target(df)

    result = fdc.merge(meta, on=ID_COL)

    # --- 레시피 플래그 ---
    #   is_special_recipe: 레시피 ID(C6)가 특수 레시피(C6_1)인지 여부.
    #   특수 레시피는 C65가 ~280 낮은 진짜 신호 (주력 C6_0 대비. baseline §21).
    #   C12(Vdc setpoint) 원값은 PM 레짐 confound라 노이즈 → 플래그만 취한다
    #   (CV -0.59, valid 39.25→38.34). 초기엔 C12>-300 임계값으로 정의했으나
    #   C6와 100% 일치 확인 후 직접 소스인 레시피 ID 기반으로 교체 (매직넘버 제거).
    #   레시피 ID는 프로덕션에서 항상 알려진 값이라 실시간 안전.
    #   ⚠️ 신규 레시피(C6_2 등) 등장 시: 자동으로 0(주력 취급, 보수적 폴백).
    #      특수형 신규 레시피라면 이 매핑을 갱신해야 한다.
    #   R2R 관점: 레시피 배치를 모델이 인지하므로 R2R bias가 배치에 오염되지 않는다.
    result['is_special_recipe'] = (result['C6'] == 'C6_1').astype(int)

    if target is not None:
        result = result.merge(target, on=ID_COL)

    return result, pm_bins


def _save_pm_bins(bins: np.ndarray) -> None:
    """pm_phase 구간 경계를 JSON으로 저장한다 (valid/test 재사용 목적)."""
    out = ROOT / PM_BINS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(bins.tolist(), f)
    print(f"  pm_bins 저장: {out.relative_to(ROOT)}")


def _load_pm_bins() -> np.ndarray:
    """저장된 pm_phase 구간 경계를 불러온다."""
    with open(ROOT / PM_BINS_PATH) as f:
        return np.array(json.load(f))


def _save_pm_log(pm_log: list[str]) -> None:
    """PM 이벤트 날짜 목록을 JSON으로 저장한다 (valid/test 재사용 목적).
    프로덕션에서 새 PM이 발생하면 이 파일에 날짜를 추가하면 된다."""
    out = ROOT / PM_LOG_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(pm_log, f, indent=2)
    print(f"  pm_log 저장: {out.relative_to(ROOT)}")


def _load_pm_log() -> list[str]:
    """저장된 PM 이벤트 날짜 목록을 불러온다."""
    with open(ROOT / PM_LOG_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────
# 6. 전체 실행
# ─────────────────────────────────────────

def run() -> pd.DataFrame:
    """train 피처 엔지니어링 파이프라인을 실행하고 저장한다."""
    print("[1/4] 전처리 데이터 로드 중...")
    df = load_clean()

    print("[2/4] 피처 빌드 중 (FDC 집계 + 메타 파생)...")
    pm_log = [PM_SHIFT_DATE]   # 알려진 PM 날짜 목록 (프로덕션에서는 PM 발생 시 추가)
    result, pm_bins = build_features(df, pm_log=pm_log)

    print("[3/4] pm_bins 및 pm_log 저장 중...")
    _save_pm_bins(pm_bins)
    _save_pm_log(pm_log)

    print("[4/4] 피처 테이블 저장 중...")
    out = ROOT / PROCESSED_DATA_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  저장 완료: {out.relative_to(ROOT)} ({size_mb:.1f} MB)")

    print(f"\n[완료] {result.shape[0]:,}행(WF 수) × {result.shape[1]}열")
    return result


def run_split(input_path: str, output_path: str) -> pd.DataFrame:
    """valid / test 데이터에 동일한 피처 엔지니어링을 적용한다.

    train의 pm_bins를 재사용하므로 run() 이후에 호출해야 한다.
    """
    from preprocessing import preprocess   # 순환 import 방지를 위해 지역 import

    name = Path(input_path).name
    print(f"[1/3] {name} 로드 및 전처리 중...")
    df = pd.read_csv(input_path)
    df = preprocess(df)
    print(f"  완료: {df.shape[0]:,}행 × {df.shape[1]}열")

    print("[2/3] 피처 빌드 중 (train pm_bins, pm_log 재사용)...")
    pm_bins = _load_pm_bins()
    pm_log  = _load_pm_log()
    result, _ = build_features(df, pm_bins=pm_bins, pm_log=pm_log)

    print("[3/3] 저장 중...")
    out = ROOT / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  저장 완료: {out.relative_to(ROOT)} ({size_mb:.1f} MB)")

    print(f"\n[완료] {result.shape[0]:,}행(WF 수) × {result.shape[1]}열")
    return result


if __name__ == "__main__":
    import sys as _sys
    mode = _sys.argv[1] if len(_sys.argv) > 1 else "train"
    if mode == "valid":
        run_split(VALID_X_PATH, VALID_FEATURES_PATH)
    elif mode == "test":
        run_split(TEST_X_PATH, TEST_FEATURES_PATH)
    else:
        run()
