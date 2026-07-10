#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modeling_v8 — 트랙별 모델 비교 (LightGBM vs ExtraTrees) 타임라인 개형.

각 트랙마다 그림 1장씩, LightGBM과 ExtraTrees 예측을 겹쳐 비교한다.
  LightGBM = 파란 실선, ExtraTrees = 초록 실선, 실측 = 검정.
출력: outputs/timeline_LGBMvsET_F-T15.png , outputs/timeline_LGBMvsET_F-P3.png

사용법: modeling_v8/ 폴더(= v8_timeline_common.py 와 같은 위치)에서
    python compare_models_by_track.py            # F-T15, F-P3 둘 다
    python compare_models_by_track.py F-P3       # 특정 트랙만
"""
import sys
import v8_timeline_common as C


def main():
    tracks = [a for a in sys.argv[1:] if a in C.TRACKS] or ['F-T15', 'F-P3']
    Xtr, Xva, Xte, y, _ = C.load_tables()
    splits = C.splits_of(Xva, Xte)
    for tname in tracks:
        feats = C.TRACKS[tname]
        curves = [
            dict(label='LightGBM',   color=C.BLUE,  predict_fn=C.fit_lgbm(Xtr, feats, y)),
            dict(label='ExtraTrees', color=C.GREEN, predict_fn=C.fit_et(Xtr, feats, y)),
        ]
        C.render_figure(
            C.OUT / f"timeline_LGBMvsET_{tname}.png",
            f"modeling_v8 · 트랙 {tname} — LightGBM vs ExtraTrees (valid / test)",
            splits, curves)


if __name__ == "__main__":
    main()
