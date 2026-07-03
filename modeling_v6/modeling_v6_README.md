# modeling_v6.ipynb — 노트북 안내서

> 시퀀스 모델(1D-CNN, step 시퀀스 직접 학습). 이전: v3(최선 tabular), v5(row-level)
> **결과: Test RMSE 66.04 (딥러닝 실패, 방향 종료).** 결과·오차분석은 `modeling_v6_REPORT.md` 참조
>
> 이 문서는 **노트북 사용법**입니다. 실험 결과·해석은 같은 폴더의 REPORT를 보세요.

---

## 실행 전 준비

1. **커널**: `venv`
2. **의존성**: `python -m pip install torch --break-system-packages` (GPU 있으면 CUDA 빌드 권장, CPU도 수 분 내 학습)
3. **폴더**: 이 노트북은 `modeling_v6/` 안에서 실행 (데이터 경로 `../문제1(하)` 상대경로)
4. **실행 시간**: CPU 기준 5~15분 (5-Fold × 최대 60ep, early stopping)

---

## 셀 구성

| 셀 | 내용 |
|----|------|
| 1 | imports & 설정 (MAX_LEN=16, 센서/스텝 목록, device) |
| 2 | 데이터 로드 |
| 3 | 표준화 통계(train) + C23 인덱스 매핑 |
| 4 | WF → 3D 텐서 변환기 (시간정렬·padding·mask) |
| 5 | 1D-CNN 모델 정의 |
| 6 | GroupKFold 5-Fold 학습 루프 |
| 7 | WF-level 평가 + 제출/results.json 저장 |
| 8 | (선택) LSTM 변형 코드 |

---

## 설계 핵심

- 입력: WF당 (16 step, 41채널) = 36 센서 표준화 + C7 one-hot 5
- C23(28종 Recipe): 임베딩(dim 8)으로 CNN pooled feature에 concat
- masked mean pooling으로 가변 step 수(9~16) 처리
- 타깃 표준화 후 학습 → 예측 시 역변환
- GroupKFold(C64), fold별 early stopping

---

## 결과 요약

| 모델 | Test RMSE |
|------|-----------|
| v3 / v5 (tabular) | 60.5 |
| **v6 (1D-CNN)** | **66.0** (악화) |

딥러닝은 표본 부족(11,939 WF)·짧은 시퀀스로 GBDT에 패배. 상세 분석과 다음 방향은 `modeling_v6_REPORT.md` 참조.

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `outputs/valid_Y_submit.csv` | Valid 예측 |
| `outputs/test_Y_submit.csv` | Test 예측 |
| `outputs/results.json` | OOF / Valid / Test RMSE |
