#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modeling_v10 / v11 — valid/test 예측 개형 (좌: 개별 WF 시간 산포 / 우: 12h 평균선).

노트북 변수에 의존하지 않고, 저장된 산출물만 읽어 그린다. 이 스크립트는 자기 위치를
기준으로 동작하므로 v10·v11 폴더 어디에 두든 동일하게 실행된다:
  - 예측: <이 폴더>/outputs/{split}_Y_submit.csv   (컬럼: wafer_id, predicted_C65)
  - 정답: ../문제1_하_answer/{split}_Y_answer.csv    (컬럼: C64, C65)
  - 시각: ../문제1(하)/{split}_X.csv 의 C10(Unix초→KST)

레이아웃(2×2): 행=valid/test.
  · 왼쪽 = 개별 WF 산포 — 실측(회색)·예측(파랑) 점을 측정 시각 위에 겹쳐 표시
  · 오른쪽 = 12h 평균선 — 실측(검정)·예측(파랑) 선만

사용법: 이 파일을 modeling_v10/ (또는 modeling_v11/) 폴더에 두고
    python plot_timeline.py
결과: <이 폴더>/outputs/valid_test_timeline.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from sklearn.metrics import mean_squared_error

# ── 경로 (이 스크립트 위치 기준) ─────────────────────────────
HERE       = Path(__file__).resolve().parent            # modeling_v10/ 또는 v11/
DATA_DIR   = HERE.parent / "문제1(하)"
ANS_DIR    = HERE.parent / "문제1_하_answer"
OUTPUT_DIR = HERE / "outputs"
MODEL_NAME = HERE.name                                   # 제목용

# ── 상수 ────────────────────────────────────────────────────
KST        = pd.Timedelta(hours=9)                  # C10은 Unix(UTC초) → KST
PM_TS      = pd.Timestamp("2018-12-24 01:31:22.7")  # major·loud(요란) 전환
TICK_START = pd.Timestamp("2018-12-01")             # x축 눈금 시작(10일 간격)


def set_korean_font():
    """Windows(Malgun) → Nanum → AppleGothic 순으로 잡는다."""
    import matplotlib.font_manager as fm
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def wf_time(split):
    """원본 X에서 WF별 대표 시각(C10 중앙값, KST)을 뽑는다."""
    df = pd.read_csv(DATA_DIR / f"{split}_X.csv", usecols=["C64", "C10"])
    ts = pd.to_datetime(df.groupby("C64")["C10"].median(), unit="s") + KST
    return ts.rename("ts")


def load_split(split):
    """예측(outputs, wafer_id/predicted_C65) + 정답(answer) + 시각(원본)을 WF 단위로 결합."""
    sub = pd.read_csv(OUTPUT_DIR / f"{split}_Y_submit.csv")
    pred = sub.set_index("wafer_id")["predicted_C65"].rename("pred")
    pred.index.name = "C64"
    true = pd.read_csv(ANS_DIR / f"{split}_Y_answer.csv").set_index("C64")["C65"].rename("true")
    d = (pd.DataFrame(wf_time(split))
         .join(pred).join(true)
         .dropna(subset=["ts", "pred", "true"])
         .sort_values("ts"))
    return d


def _pm_line(ax):
    ax.axvline(PM_TS, color="red", ls="--", lw=1.3, zorder=5)
    ax.text(PM_TS, ax.get_ylim()[1], " 요란 PM", color="red", va="top", ha="left", fontsize=9)


def _time_axis(ax):
    ticks = pd.date_range(TICK_START, pd.Timestamp("2019-02-11"), freq="10D")
    ax.xaxis.set_major_locator(mticker.FixedLocator(mdates.date2num(ticks)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))


def draw_scatter(ax, d, name):
    """왼쪽 — 개별 WF 산포: 실측(회색)·예측(파랑) 점을 시간 위에 겹침."""
    rmse = np.sqrt(mean_squared_error(d["true"], d["pred"]))
    ax.scatter(d["ts"], d["true"], s=8, c="0.45", alpha=0.45, label="실측 (개별 WF)", zorder=1)
    ax.scatter(d["ts"], d["pred"], s=8, c="tab:blue", alpha=0.35, label="예측 (개별 WF)", zorder=2)
    ax.set_title(f"[{name}] 산포 (개별 WF) — RMSE {rmse:.2f} (n={len(d):,})",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("C65 (불량 비트 수)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9); ax.grid(alpha=0.25)
    _pm_line(ax); _time_axis(ax)
    return rmse


def draw_mean_lines(ax, d, name):
    """오른쪽 — 12h 평균선: 실측(검정)·예측(파랑) 선만."""
    rmse = np.sqrt(mean_squared_error(d["true"], d["pred"]))
    r = d.set_index("ts")[["true", "pred"]].resample("12h").mean().dropna()
    ax.plot(r.index, r["true"], c="black",    lw=1.4, label="실측 (12h 평균)", zorder=3)
    ax.plot(r.index, r["pred"], c="tab:blue", lw=1.4, label="예측 (12h 평균)", zorder=3)
    ax.set_title(f"[{name}] 12h 평균선 — RMSE {rmse:.2f} (n={len(d):,})",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("C65 (불량 비트 수)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9); ax.grid(alpha=0.25)
    _pm_line(ax); _time_axis(ax)
    return rmse


def main():
    set_korean_font()
    OUTPUT_DIR.mkdir(exist_ok=True)

    dv, dt = load_split("valid"), load_split("test")
    # 2×2: 행=valid/test, 왼쪽=개별 WF 산포, 오른쪽=12h 평균선 (시간축 공유)
    fig, axes = plt.subplots(2, 2, figsize=(20, 11), sharex=True)
    draw_scatter(axes[0, 0], dv, "valid")
    r_valid = draw_mean_lines(axes[0, 1], dv, "valid")
    draw_scatter(axes[1, 0], dt, "test")
    r_test  = draw_mean_lines(axes[1, 1], dt, "test")
    axes[1, 0].set_xlabel("측정 시각"); axes[1, 1].set_xlabel("측정 시각")

    fig.suptitle(f"{MODEL_NAME} — (좌) 개별 WF 산포  /  (우) 12h 평균선",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out = OUTPUT_DIR / "valid_test_timeline.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[{MODEL_NAME}] valid RMSE {r_valid:.2f} | test RMSE {r_test:.2f}")
    print(f"saved -> {out}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
