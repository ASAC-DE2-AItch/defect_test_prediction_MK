# modeling_v5_confidence.ipynb — 노트북 안내서

> v5 예측에 신뢰도(예측별 신뢰구간 + 불확실성 점수)를 추가하고 valid로 검증하는 companion 노트북
>
> 원본: `modeling_v5.ipynb` (예측 모델)

---

## 실행 전 준비

1. **커널**: `venv (3.12)` — Python 3.12 (3.14 사용 금지, `ENVIRONMENT.md` 참조)
2. **위치**: `modeling_v5/` 폴더 안에서 실행 (데이터 경로 `../문제1(하)` 상대경로)
3. **실행 방법**: VS Code에서 커널 선택 후 실행, 또는 EFAULT 오류 시 `jupyter lab`로 브라우저 실행
4. **실행 시간**: 약 5~10분 (v5와 동일한 5-Fold 학습 + 신뢰도 계산)

---

## 셀 구성

| 셀 | 내용 |
|----|------|
| 1 | imports |
| 2 | 데이터 로드 (키 컬럼 object dtype 강제 — pandas 3.0 호환) |
| 3 | v5 row-level 피처 (원본 동일) |
| 4 | GroupKFold 학습 — OOF + **fold별** 예측 수집 |
| 5 | WF 집계 + 불확실성 σ = fold 불일치 (행수 assert) |
| 6 | Split Conformal 신뢰구간 보정 (train OOF) |
| 7 | valid 커버리지 검증 + 신뢰도 곡선 |
| 8 | 모델 신뢰도 진단 (예측vs실제, 구간별 RMSE, σ 유효성) |
| 9 | wafer별 신뢰도 테이블 저장 (저장 후 행수 재확인 assert) |

---

## 산출물

- 각 wafer에 `pred`, `sigma`(불확실성), `[lower, upper]`(90% 신뢰구간), `confidence`(OK/LOW).
- **cov90 = 0.904**로 구간 신뢰성 확인, **σ–오차 상관 0.34**로 불확실성 신호 유효성 확인.
- 상세 수치·해석은 `modeling_v5_confidence_REPORT.md`.

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `outputs/valid_confidence.csv` | valid wafer별 신뢰도 테이블 (1990행) |
| `outputs/test_confidence.csv` | test wafer별 신뢰도 테이블 (1990행) |
| `outputs/confidence_report.json` | 커버리지·σ상관 요약 |

---

## 트러블슈팅 메모

- **커널 실행 실패(EFAULT)**: VS Code Jupyter 소켓 문제. `jupyter lab`로 브라우저 실행하거나 `netsh winsock reset` 후 재부팅 (`ENVIRONMENT.md` 참조).
- **CSV 행수가 1990 미만**: Cell 9의 assert가 잡아냄. pandas 3.0 문자열 dtype 이슈 → Cell 2의 object 강제가 해결.
