# CLAUDE.md — SK Hynix Defect Test Prediction

## 프로젝트 개요
- FDC Trace 데이터를 이용한 Defect Test(C65) 회귀 예측 (난이도 下)
- 평가지표: RMSE
- 데이터: `문제1(하)/` (train, valid, test), 정답: `문제1_하_answer/`

## 성능 목표
- **목표 RMSE: ~40** (현재 ~62 → 약 20pt 추가 개선 필요)
- 이 목표에 도달할 때까지 피처 엔지니어링, 하이퍼파라미터 튜닝, 앙상블 등 개선 작업을 계속할 것

## 모델링 원칙
- Lot ID(C20/C21/C22)는 피처로 사용하지 않음 — 실제 현업 신규 데이터에 일반화 불가
- WF ID(C64/C34/C35/C38)도 피처에서 제외 — 물리적 의미 없는 식별자
- 물리적 의미가 있는 FDC 센서, Recipe(C6), Step(C7), 시간, PM(C33)만 피처로 사용
- train_data로만 학습, valid/test 정답은 평가용으로만 사용
- GroupKFold(C64 기준)로 CV — row 누수 방지

## 코드 포맷
- 기본: `.ipynb` (Jupyter Notebook)
- 보조: `.py` (유틸리티, 스크립트)

## 작업 방식
- ipynb 파일은 직접 수정하지 말고, 수정할 부분만 텍스트로 안내
- 사용자가 직접 노트북 셀을 수정하는 방식 선호
- **버전 관리**: 피처 엔지니어링, 모델, EDA 방식 등이 변경될 때(코드 버그 수정 제외) 기존 파일을 수정하지 않고 새 파일을 생성할 것 (예: `modeling_v2.ipynb`, `modeling_v3.ipynb`)
- **완료 후 자동 작업**: 사용자가 "컴파일 완료 됐어!"라고 보내면, (1) 해당 버전의 README와 REPORT 파일을 생성하고, (2) 다음에 무엇을 왜 어떻게 할 것인지 텍스트로 제안할 것
- **README / REPORT 유지 규칙**: README(노트북 사용 설명서)는 노트북을 고칠 때마다 함께 갱신하는 "살아있는 문서"로 유지한다. REPORT(해당 버전 분석의 결과 기록)는 그 시점 분석의 스냅샷으로 고정하고, 코드가 바뀌었다고 옛 REPORT를 고치지 않으며 분석이 바뀌면 새 버전에 새 REPORT를 작성한다.

## 컬럼 역할 (명세 기반)
| 구분 | 컬럼 | 처리 |
|---|---|---|
| WF ID (그룹키) | C64, C34, C35, C38 | C64만 그룹키, 나머지 drop |
| Recipe ID | C6(2종), C23, C30 | C6만 사용 |
| Step 번호 | C7(5종), C36 | C7 사용 |
| Lot ID | C20, C21, C22 | drop (일반화 불가) |
| Chamber | C24 | 상수 → drop |
| 장비 | C14 | 상수 → drop |
| 시간 | C10, C39, C40, C41 | C10/C39 → datetime 변환, C41 경과시간 |
| PM time | C33 | drift 지표 활용 |
| 제외 지정 | C26, C28, C29, C37 | drop |
| 전부 결측 | C2, C13, C43, C47, C53, C55 | drop |
| X (FDC 센서) | 나머지 수치형 | WF 단위 집계 피처 |
| Y | C65 | 타깃 (WF 내 상수) |

## 환경
- Python 가상환경: `venv/`
- 의존성: `requirements.txt`
- Windows Long Path 활성화 필요 (OneDrive 한글 경로)
