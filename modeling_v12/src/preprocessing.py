"""
preprocessing.py
─────────────────────────────────────────────
원본 CSV를 불러와서 기본 정리를 수행하는 파일.
train / valid / test 모두 동일한 로직을 적용한다.

핵심 함수:
  preprocess(df)              # 어떤 DataFrame이든 전처리 (datetime 파싱 → 컬럼 제거 → 정렬)
  run()                       # train_data 전용 실행
  run_split(input, output)    # valid / test 용 실행
─────────────────────────────────────────────
"""

import pandas as pd
from pathlib import Path
import sys

# 어느 디렉터리에서 실행해도 config.py 를 찾을 수 있도록 src/ 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    COLS_TO_DROP,
    SORT_COL,
    TIME_FORMAT,
    VALID_X_PATH,
    VALID_FEATURES_PATH,
    TEST_X_PATH,
    TEST_FEATURES_PATH,
)


def drop_useless_cols(df: pd.DataFrame) -> pd.DataFrame:
    """100% 결측·상수·완전 중복 컬럼 23개를 제거한다."""
    to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"  컬럼 제거: {len(to_drop)}개 → 잔여 {df.shape[1]}열")
    return df


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """SORT_COL(C40) 기준 오름차순 정렬한다.

    시간순 정렬을 해두지 않으면 train/validation 을 나눌 때
    미래 데이터가 학습에 섞여 들어가는 데이터 누수가 발생한다.
    """
    df = df.sort_values(SORT_COL).reset_index(drop=True)
    start = df[SORT_COL].min().strftime("%Y-%m-%d")
    end   = df[SORT_COL].max().strftime("%Y-%m-%d")
    print(f"  시간 정렬 완료: {start} ~ {end}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """핵심 전처리 — train / valid / test 에 동일하게 적용한다.

    1. C40 문자열 → datetime 변환
    2. 불필요 컬럼 23개 제거
    3. 시간순 정렬
    """
    df[SORT_COL] = pd.to_datetime(df[SORT_COL], format=TIME_FORMAT)
    df = drop_useless_cols(df)
    df = sort_by_time(df)
    return df


def _save(df: pd.DataFrame, out_path: str) -> None:
    """DataFrame을 CSV로 저장한다."""
    out = ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  저장 완료: {out.relative_to(ROOT)}  ({size_mb:.1f} MB)")


def run() -> pd.DataFrame:
    """train_data 전처리 파이프라인을 실행한다."""
    print("[1/3] 원본 데이터 로드 중...")
    df = pd.read_csv(ROOT / RAW_DATA_PATH)
    print(f"  로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")

    print("[2/3] 전처리 중...")
    df = preprocess(df)

    print("[3/3] 저장 중...")
    _save(df, CLEAN_DATA_PATH)

    print(f"\n[완료] {df.shape[0]:,}행 × {df.shape[1]}열")
    return df


def run_split(input_path: str, output_path: str) -> pd.DataFrame:
    """valid / test 데이터에 동일한 전처리를 적용하고 저장한다."""
    name = Path(input_path).name
    print(f"[1/3] {name} 로드 중...")
    df = pd.read_csv(input_path)
    print(f"  로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")

    print("[2/3] 전처리 중...")
    df = preprocess(df)

    print("[3/3] 저장 중...")
    _save(df, output_path)

    print(f"\n[완료] {df.shape[0]:,}행 × {df.shape[1]}열")
    return df


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "valid":
        run_split(VALID_X_PATH, VALID_FEATURES_PATH.replace("features", "clean"))
    elif mode == "test":
        run_split(TEST_X_PATH, TEST_FEATURES_PATH.replace("features", "clean"))
    else:
        run()
