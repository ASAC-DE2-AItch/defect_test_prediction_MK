#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modeling_v5 — valid/test 전체 기간 예측 개형 플롯.

노트북 변수(va_wf 등)에 의존하지 않고, 저장된 산출물만 읽어 그린다:
  - 예측: modeling_v5/outputs/{split}_Y_submit.csv
  - 정답: 문제1_하_answer/{split}_Y_answer.csv
  - 시각·배치: 문제1(하)/{split}_X.csv 의 C10(Unix초→KST), C6

사용법: 이 파일을 modeling_v5/ 폴더에 두고
    python plot_timeline.py
결과: modeling_v5/outputs/valid_test_timeline.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error

# ── 경로 (이 스크립트 위치 기준) ─────────────────────────────
HERE       = Path(__file__).resolve().parent            # modeling_v5/
DATA_DIR   = HERE.parent / "문제1(하)"
ANS_DIR    = HERE.parent / "문제1_하_answer"
OUTPUT_DIR = HERE / "outputs"

# ── 상수 ────────────────────────────────────────────────────
KST      = pd.Timedelta(hours=9)                    # C10/C39는 Unix(UTC초) → KST
PM_TS    = pd.Timestamp("2018-12-24 01:31:22.7")    # major·loud 전환 (pm_log_meta.json)
C6_BATCH = "C6_1"                                   # 음영 표시 배치 (post-only)


def set_korean_font():
    """Windows(Malgun) → Nanum → AppleGothic 순으로 잡는다."""
    import matplotlib.font_manager as fm
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def wf_meta(split):
    """원본 X에서 WF별 대표 시각(C10 중앙값, KST)과 C6 배치를 뽑는다."""
    df = pd.read_csv(DATA_DIR / f"{split}_X.csv", usecols=["C64", "C10", "C6"])
    g = df.groupby("C64")
    ts = pd.to_datetime(g["C10"].median(), unit="s") + KST
    c6 = g["C6"].agg(lambda s: s.mode().iloc[0])
    return pd.DataFrame({"ts": ts, "c6": c6})


def load_split(split):
    """예측(outputs) + 정답(answer) + 시각/배치(원본)을 WF 단위로 결합."""
    pred = pd.read_csv(OUTPUT_DIR / f"{split}_Y_submit.csv").set_index("C64")["C65"]
    true = pd.read_csv(ANS_DIR / f"{split}_Y_answer.csv").set_index("C64")["C65"]
    d = (wf_meta(split)
         .join(pred.rename("pred"))
         .join(true.rename("true"))
         .dropna(subset=["ts", "pred", "true"])
         .sort_values("ts"))
    return d


def draw_panel(ax, d, name):
    rmse = np.sqrt(mean_squared_error(d["true"], d["pred"]))
    # 개별 WF 실측 (scatter)
    ax.scatter(d["ts"], d["true"], s=6, c="0.7", alpha=0.5,
               label="실측 (개별 WF)", zorder=1)
    # 12h 평균선 (실측·예측)
    r = d.set_index("ts")[["true", "pred"]].resample("12h").mean().dropna()
    ax.plot(r.index, r["true"], c="black",    lw=1.4, label="실측 (12h 평균)", zorder=3)
    ax.plot(r.index, r["pred"], c="tab:blue", lw=1.4, label="예측 (12h 평균)", zorder=3)
    # PM 레짐 전환선
    ax.axvline(PM_TS, color="red", ls="--", lw=1.3, zorder=2)
    ax.text(PM_TS, ax.get_ylim()[1], " PM (레짐 전환)",
            color="red", va="top", ha="left", fontsize=9)
    # C6_1 배치 음영
    c6 = d[d["c6"] == C6_BATCH]["ts"]
    if len(c6):
        lo, hi = c6.min(), c6.max()
        ax.axvspan(lo, hi, color="orange", alpha=0.18, zorder=0)
        ax.text(lo + (hi - lo) / 2, ax.get_ylim()[0], "C6_1\n배치",
                color="darkorange", va="bottom", ha="center", fontsize=8)
    ax.set_title(f"[{name}] 전체 기간 실측 vs 예측 — RMSE {rmse:.2f} (n={len(d):,})",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("C65 (불량 비트 수)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    return rmse


def main():
    set_korean_font()
    OUTPUT_DIR.mkdir(exist_ok=True)

    dv, dt = load_split("valid"), load_split("test")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    r_valid = draw_panel(axes[0], dv, "valid")
    r_test  = draw_panel(axes[1], dt, "test")

    axes[1].set_xlabel("날짜")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    fig.suptitle("modeling_v5 전체 기간 예측 개형 (valid / test)",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = OUTPUT_DIR / "valid_test_timeline.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"valid RMSE {r_valid:.2f} | test RMSE {r_test:.2f}")
    print(f"saved -> {out}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
