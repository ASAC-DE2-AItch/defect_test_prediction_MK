"""
config.py
─────────────────────────────────────────────
프로젝트 전체에서 공통으로 쓰는 컬럼 분류 정보를 모아둔 파일.

다른 코드(preprocessing.py, feature_engineering.py, train.py)는
전부 여기서 정의한 목록을 가져다 쓴다.
나중에 컬럼 분류가 바뀌면 이 파일 하나만 고치면 된다.

작성 기준: EDA 및 다중공선성 분석 결과 (선택지 B - 최소 제거안)
─────────────────────────────────────────────
"""

from pathlib import Path

# 프로젝트 루트 (model/) 와 경진대회 데이터 루트 자동 계산
_PROJ_ROOT = Path(__file__).resolve().parent.parent          # model/
_COMP_ROOT = _PROJ_ROOT.parent                               # 최종 프로젝트 자료/
_COMP_DATA = _COMP_ROOT / "Data" / "문제1(하)" / "문제1(하)"
_COMP_ANS  = _COMP_ROOT / "Data" / "문제1_answer" / "문제1_하_answer"

# ─────────────────────────────────────────
# 1. 데이터 파일 경로
# ─────────────────────────────────────────

# train (model/ 기준 상대경로)
RAW_DATA_PATH       = "data/raw/train_data.csv"
CLEAN_DATA_PATH     = "data/processed/train_clean.csv"    # preprocessing.py 출력
PROCESSED_DATA_PATH = "data/processed/wf_features.csv"   # feature_engineering.py 출력

# valid / test (경진대회 폴더에 있어서 절대경로 사용)
VALID_X_PATH        = str(_COMP_DATA / "valid_X.csv")
VALID_Y_PATH        = str(_COMP_ANS  / "valid_Y_answer.csv")
TEST_X_PATH         = str(_COMP_DATA / "test_X.csv")
TEST_Y_PATH         = str(_COMP_ANS  / "test_Y_answer.csv")

# valid / test 피처 파일 (model/ 기준 상대경로)
VALID_FEATURES_PATH = "data/processed/valid_features.csv"
TEST_FEATURES_PATH  = "data/processed/test_features.csv"
PM_BINS_PATH        = "data/processed/pm_bins.json"
PM_LOG_PATH         = "data/processed/pm_log.json"

# ─────────────────────────────────────────
# 2. 제거 컬럼 (총 23개)
# ─────────────────────────────────────────

# 2-1. 완전 결측 (100% 비어있음, 8개)
COLS_ALL_NULL = ['C2', 'C13', 'C26', 'C37', 'C43', 'C47', 'C53', 'C55']

# 2-2. 상수값 (분산 0, 12개)
COLS_CONSTANT = ['C3', 'C8', 'C14', 'C19', 'C21', 'C24',
                  'C28', 'C29', 'C30', 'C44', 'C45', 'C51']

# 2-3. 완전 중복 (다른 컬럼과 100% 동일, 3개)
COLS_DUPLICATE = [
    'C36',  # C7과 100% 동일 (공정 Step 번호)
    'C35',  # C34와 100% 동일 (슬롯 번호)
    'C38',  # C64와 100% 동일 (WF ID)
]

# 제거 컬럼 전체 (총 23개)
COLS_TO_DROP = COLS_ALL_NULL + COLS_CONSTANT + COLS_DUPLICATE

# ─────────────────────────────────────────
# 3. 역할별 보존 컬럼 (총 42개)
# ─────────────────────────────────────────

# 3-1. 타겟 (Y) - 예측해야 할 값. Defect Test 결과
TARGET_COL = 'C65'

# 3-2. WF ID - groupby 기준. 모델 입력(X)에서는 제외(학습시 필요없는 데이터)
ID_COL = 'C64'

# 3-3. 시간 관련 - 정렬/검증 split 기준. 일부(C41)만 피처로 사용
#   C39 : Unix 타임스탬프 (float, 초 단위)
#   C40 : 날짜 문자열 'YYYY-MM-DD HH:MM:SS.mmm' → 정렬 기준으로 사용
#   C41 : Step 내 경과 시간 (float, 초 단위) → 피처로 사용
#   C10 : Unix 타임스탬프 (float, C39와 동일 스케일)
TIME_COLS = ['C39', 'C40', 'C41', 'C10']

# 시간 기반 train/validation split 시 정렬 기준 컬럼과 파싱 포맷
SORT_COL = 'C40'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

# 3-4. 공정 Step 번호 - groupby 기준. 모델 입력(X)에서는 제외(가공해서 넣기 위함)
STEP_COL = 'C7'

# 3-5. 메타 정보 - Lot/캐리어/슬롯/PM 카운터
#      C33(PM 카운터)는 추후 파생피처로 승격하여 모델 입력에 사용
META_COLS = ['C20', 'C22', 'C23', 'C33', 'C34']

# ─────────────────────────────────────────
# 4. 다중공선성 분석 결과
# ─────────────────────────────────────────
# 상관계수 0.8 이상으로 묶인 그룹들 (df[group].corr() 기준)

MULTICOLLINEAR_GROUPS = {
    'group1': ['C1', 'C9', 'C17', 'C63'],          # 서로 0.6~0.9, PM 전후 역할 다름
    'group2': ['C4', 'C5'],                          # 상관계수 0.95, 거의 동일 정보
    'group3': ['C11', 'C15', 'C16', 'C31', 'C62'],  # 서로 0.6~0.9, PM 전후 역할 다름
    'group4': ['C46', 'C48'],                        # 상관계수 0.81이지만
                                                       # 다중공선성 아님(혼재변수) → 둘 다 보존
}

# 다중공선성으로 인해 모델 입력에서 제외할 컬럼
# (group1, group3는 PM 전/후 역할 분담을 위해 대표 2개씩 남기고 나머지 제거
#  group2는 1개만 남기고 제거 / group4는 다중공선성이 아니므로 제거 없음)
COLS_REMOVE_MULTICOLLINEAR = [
    'C1', 'C9',                  # group1 중 C17, C63만 남김
    'C5',                        # group2 중 C4만 남김
    'C11', 'C15', 'C16', 'C31',  # group3 중 C62만 남김 (C16 중요도 0 확인 후 제거)
]

# ─────────────────────────────────────────
# 5. FDC 센서 피처 (다중공선성 제거 후)
# ─────────────────────────────────────────
# FDC 센서 변수 30개 중 다중공선성으로 확정된 중복만 제거하고
# 나머지는 EDA로 직접 검증 안 된 변수까지 전부 포함.
# → LightGBM의 feature_importance로 추후 2차 선별 예정

FDC_COLS_ALL = [
    'C1', 'C4', 'C5', 'C6', 'C9', 'C11', 'C12', 'C15', 'C16', 'C17',
    'C18', 'C25', 'C27', 'C31', 'C32', 'C42', 'C46', 'C48', 'C49',
    'C50', 'C52', 'C54', 'C56', 'C57', 'C58', 'C59', 'C60', 'C61',
    'C62', 'C63',
]

FDC_FEATURES = [c for c in FDC_COLS_ALL if c not in COLS_REMOVE_MULTICOLLINEAR]
# 결과: 23개 (FDC 30개 - 다중공선성 제거 7개 = 23개)
# ['C4','C6','C12','C17','C18','C25','C27','C32','C42','C46','C48',
#  'C49','C50','C52','C54','C56','C57','C58','C59','C60','C61','C62','C63']

# ─────────────────────────────────────────
# 5-1. 메타/시간에서 파생할 피처 (PM 시프트 등 EDA 핵심 발견 반영)
# ─────────────────────────────────────────
# 주의: C64(WF ID), C39/C40/C10(원본 타임스탬프), C7(Step)은
#       절대값 자체로는 의미가 없어 모델 입력에서 제외.
#       대신 아래처럼 "가공된 형태"로만 모델에 들어간다.
#
#   C33(PM 카운터)      → C33, C33_log
#   C41(Step내 경과시간) → Step별 max(C41) = Step별 소요시간
#
# 제거된 파생 피처 (중요도 확인 후):
#   C33_inv      → gain=0
#   C34          → 추가 시 +0.22 악화 (슬롯 번호가 노이즈로 작용)
#   wf_row_count → 순위 #260/263 (≈0%)
#   step5_exists → gain=0
#
# feature_engineering.py 에서 아래 이름으로 생성해 MODEL_FEATURES에 합친다.
META_DERIVED_FEATURES = [
    'C33', # C33_log 제거 (LightGBM 트리에서 단조 변환은 동일 정보) / pm_phase: pm_bins 하위 호환용
]
# Step별 소요시간은 Step별 집계 시 'C41_max' 형태 컬럼명으로 자동 생성됨

# ─────────────────────────────────────────
# 5-2. 최종 모델 입력 피처 (FDC + 메타파생)
# ─────────────────────────────────────────
MODEL_FEATURES = FDC_FEATURES + META_DERIVED_FEATURES
# 23개(FDC) + 3개(메타파생) = 26개 (+ Step별 집계 시 통계량만큼 추가로 늘어남)

# ─────────────────────────────────────────
# 6. 범주형 변수 (LightGBM categorical_feature 파라미터에 전달)
# ─────────────────────────────────────────
# C6      : 공정 레시피 코드 (문자열 카테고리)
# C46     : 장비 챔버 번호 1~8 (순서 의미 없음)
#
# 제외 이유:
#   C34 → 추가 시 RMSE +0.22 악화 (노이즈), 제거
#   C42 → 0/1 이진값. LightGBM이 수치로 처리해도 동일함
#   C49 → 값이 -35/-23/-12/0으로 크기 순서가 의미 있는 수치형
CATEGORICAL_COLS = ['C6', 'C46']

# ─────────────────────────────────────────
# 7. PM(예방정비) 관련 설정
# ─────────────────────────────────────────
PM_SHIFT_DATE = '2018-12-24'  # 이 날짜를 기준으로 C65 레벨이 급변함

# ─────────────────────────────────────────
# 8. WF 단위 집계 시 사용할 통계량
# ─────────────────────────────────────────
# EDA에서 mean보다 max/min이 더 강한 신호를 보인 경우가 많았음 (C17_max, C63_min 등)
AGG_FUNCS = ['mean', 'std', 'max', 'min', 'last']

# ─────────────────────────────────────────
# 9. 검증 (이 파일이 깨지지 않았는지 자체 점검)
# ─────────────────────────────────────────
def _validate():
    all_cols = (
        COLS_TO_DROP + [TARGET_COL, ID_COL, STEP_COL] +
        TIME_COLS + META_COLS + FDC_COLS_ALL
    )

    # 각 목록 내부에 중복이 없어야 함
    for name, lst in [
        ('COLS_TO_DROP', COLS_TO_DROP),
        ('FDC_COLS_ALL', FDC_COLS_ALL),
        ('MODEL_FEATURES', MODEL_FEATURES),
        ('CATEGORICAL_COLS', CATEGORICAL_COLS),
    ]:
        seen = set()
        dups = [c for c in lst if c in seen or seen.add(c)]
        assert not dups, f"{name} 내부에 중복 컬럼 있음: {dups}"

    # 제거 목록과 피처 목록이 겹치면 안 됨
    overlap = set(COLS_TO_DROP) & set(FDC_COLS_ALL)
    assert not overlap, f"COLS_TO_DROP과 FDC_COLS_ALL이 겹침: {overlap}"

    # 범주형 컬럼은 모두 모델 피처에 포함돼야 함
    missing_cat = set(CATEGORICAL_COLS) - set(MODEL_FEATURES)
    assert not missing_cat, f"CATEGORICAL_COLS 항목이 MODEL_FEATURES에 없음: {missing_cat}"

    # 전체 컬럼 수 확인 (중복 제거 후)
    n_unique = len(set(all_cols))
    assert n_unique == len(all_cols), (
        f"컬럼 목록 간 중복 있음. 총 {len(all_cols)}개 중 고유 {n_unique}개"
    )

    print(f"[OK] config.py 검증 통과 (전체 {n_unique}개 컬럼, "
          f"제거 {len(COLS_TO_DROP)}개, 모델 입력 피처 {len(MODEL_FEATURES)}개)")


if __name__ == "__main__":
    _validate()