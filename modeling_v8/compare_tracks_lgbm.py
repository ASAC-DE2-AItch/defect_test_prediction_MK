#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modeling_v8 — LightGBM 기준 트랙 비교 (F-T15 vs F-P3) 타임라인 개형.

한 그림 안에 두 트랙의 LightGBM 예측을 겹쳐 비교한다.
  F-T15 = 파란 실선, F-P3 = 초록 실선, 실측 = 검정.
출력: outputs/timeline_LGBM_tracks.png  (valid 위 / test 아래)

사용법: modeling_v8/ 폴더(= v8_timeline_common.py 와 같은 위치)에서
    python compare_tracks_lgbm.py
"""
import v8_timeline_common as C


def main():
    Xtr, Xva, Xte, y, _ = C.load_tables()
    curves = [
        dict(label='F-T15', color=C.BLUE,  predict_fn=C.fit_lgbm(Xtr, C.TRACKS['F-T15'], y)),
        dict(label='F-P3',  color=C.GREEN, predict_fn=C.fit_lgbm(Xtr, C.TRACKS['F-P3'],  y)),
    ]
    C.render_figure(
        C.OUT / "timeline_LGBM_tracks.png",
        "modeling_v8 · LightGBM — 트랙 비교 F-T15 vs F-P3 (valid / test)",
        C.splits_of(Xva, Xte), curves)


if __name__ == "__main__":
    main()
