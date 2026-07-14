# 결합 모델 (FULL 192피처) — 피처 목록

> **모델:** row-level 센서 + WF 집계 + 시간/레짐 피처 결합 (다이어트 전 FULL 버전)
> **성능:** valid RMSE **48.7612** (원본 train 학습 / 원본 valid 평가)
> **총 192개** = 원본 센서 36 + WF 집계 144 + 구조 2 + 범주형 2 + 시간/레짐 7 + 타깃인코딩 1
> ⚠️ 타겟: `C65` = 불량 비트 수 → **낮을수록 좋음**
> 예측 단위: row(step)별 예측 → 웨이퍼(C64) 내 평균 = WF 예측값

---

## [1] 원본 센서 — 36개 (step 단위 원본값)
```
C1, C3, C4, C5, C8, C9, C11, C12, C15, C16, C17, C18, C19, C25, C27, C31, C32, C33, C42, C44, C45, C46, C48, C49, C50, C51, C52, C54, C56, C57, C58, C59, C60, C61, C62, C63
```

## [2] WF context 집계 — 144개 (위 36종 × 4통계)
각 센서를 웨이퍼(C64) 단위로 집계해 그 WF의 모든 row에 broadcast.
이름 규칙: `{센서}_wf_{mean|std|min|max}` (예: `C17_wf_mean`)
```
C1, C3, C4, C5, C8, C9, C11, C12, C15, C16, C17, C18, C19, C25, C27, C31, C32, C33, C42, C44, C45, C46, C48, C49, C50, C51, C52, C54, C56, C57, C58, C59, C60, C61, C62, C63
```

## [3] 구조/범주형 — 4개
| 피처 | 내용 |
|------|------|
| `wf_nrows` | WF 내 row(step) 수 = `groupby(C64).size()` |
| `row_pos`  | WF 안에서 몇 번째 row인지 = `groupby(C64).cumcount()` |
| `C6` | 레시피 코드 (LightGBM native categorical) |
| `C7` | 공정 Step 번호 (native categorical) |

## [4] 시간/레짐 피처 — 7개
| 피처 | 생성식 | 의미 |
|------|--------|------|
| `is_post_loud_pm` | `(pm_log에 PM 존재)` → 0/1 | PM(요란) 이후 레짐 플래그 |
| `days_since_last_pm` | 최근 PM부터 경과일 (PM마다 리셋) | 감쇠 곡선 위치 |
| `hour` | 측정 시작 시각 (0~23) | 3교대 패턴 |
| `dslp_x_hour` | `days_since_last_pm × hour` | 교호 |
| `hour_x_c33` | `hour × C33` | 교호 |
| `post_pm_days` | `days_since_last_pm × is_post_loud_pm` | PM 이후 경과일 마스크 |
| `is_special_recipe` | `(C6 == 'C6_1')` → 0/1 | 특수 레시피 플래그 |

> PM 이벤트는 `data/processed/pm_log.json`에서 읽음 (`[{"date","type"}]`). 새 PM 발생 시 날짜 추가.

## [5] 타깃인코딩 — 1개
| 피처 | 생성식 |
|------|--------|
| `C23_te` | C23(28종 recipe) out-of-fold 타깃인코딩 (스무딩 m=20, fold별 재계산, 누수 차단) |

---
*학습 순서 기준 전체 피처명은 `_feature_list.json` 참고. 모델은 5-fold 앙상블 + fold별 C23 인코더를 `CombinedModel` 객체로 묶어 저장.*
