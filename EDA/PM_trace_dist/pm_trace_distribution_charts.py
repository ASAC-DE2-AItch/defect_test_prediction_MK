# ============================================================================
# PM 전후 FDC 트레이스 분포 변화 차트 생성기
#   출력: out/01_overview_{ko,en}.png  (종합: pre/post 분포 박스쌍 + KS 막대)
#         out/02_trend_{ko,en}.png     (센서별 변동추이 격자, C6 레시피비율 포함)
#   실행: python pm_trace_distribution_charts.py   (train_data.csv 자동 탐색)
#   필요: pandas, numpy, scipy, matplotlib
# ============================================================================
# -*- coding: utf-8 -*-
"""PM 전후 FDC 트레이스 분포 변화 (v2)
   ① 종합: 전체표준화(윈저) pre/post 분포 박스쌍 + KS 막대  ② 센서별 변동추이 격자
   한글/영어 2버전. Noto Sans CJK 폰트, axes.unicode_minus=False."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import font_manager as fm
import pandas as pd, numpy as np, math, os
from scipy import stats

# 한글 폰트 자동 탐색 (Linux Noto / Windows Malgun / Mac AppleGothic)
import os as _os
_FONTC=['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        'C:/Windows/Fonts/malgun.ttf','/System/Library/Fonts/AppleGothic.ttf']
KRFONT='DejaVu Sans'
for _fp in _FONTC:
    if _os.path.exists(_fp):
        try:
            fm.fontManager.addfont(_fp); KRFONT=fm.FontProperties(fname=_fp).get_name(); break
        except Exception: pass
plt.rcParams['axes.unicode_minus']=False

import glob as _glob
# 데이터 자동 탐색: 이 스크립트가 어디서 실행되든 train_data.csv를 찾음
_CANDS=["../../문제1(하)/train_data.csv","../문제1(하)/train_data.csv",
        "문제1(하)/train_data.csv","train_data.csv",
        "/mnt/user-data/uploads/defect_test_prediction_MK/문제1(하)/train_data.csv"]
SRC=next((p for p in _CANDS if os.path.exists(p)), _CANDS[0])
OUT="./out"; os.makedirs(OUT,exist_ok=True)
PM=pd.Timestamp('2018-12-24')
SENSORS=['C4','C12','C17','C18','C25','C27','C32','C42','C46','C48','C49','C50',
         'C52','C54','C56','C57','C58','C59','C60','C61','C62','C63']

C_LINE='#3A4A5A'; C_BAND='#C3D0DE'; C_PRE='#2C6FB3'; C_POST='#D9822B'
C_PM='#3A3A3A'; BG_PRE='#EFF4FA'; BG_POST='#FCF4EA'; GRID='#E6E6E6'

TXT={
 'ko':{'sup1':'PM 전후 FDC 트레이스 분포 비교 — 종합',
   'main':'PM 이전 vs 이후 값 분포  (센서별 전체표준화 · 윈저 1–99%)',
   'ks':'분포 차이\n(KS 통계량)','xl':'표준화 값 (σ)',
   'cap1':'FDC 센서 22종 · 박스=IQR(25–75%), 점=중앙값, 수염=10–90% · 정렬: KS 내림차순 · pre 38,460행(~12-23)/post 85,154행(12-24~)',
   'lgpre':'PM 이전','lgpost':'PM 이후',
   'sup2':'센서별 PM 이전·이후 변동 추이  (일별 중앙값 · IQR 25–75%)',
   'lg':['일별 중앙값','IQR (25–75%)','PM 이전 중앙값','PM 이후 중앙값','PM 시점 (12-24)'],
   'cap2':'x축 2018-12-01~2019-02-08 · y축 로버스트 범위(2–98%)로 극단 스파이크 절단 · KS 내림차순 정렬 · C6은 범주형이라 값 대신 레시피 C6_1 비율(%)로 표시(맨 끝 패널)'},
 'en':{'sup1':'FDC Trace Distribution: Pre vs Post PM — Overview',
   'main':'Value distribution pre vs post PM  (per-sensor global z · winsorized 1–99%)',
   'ks':'Distribution\ndiff. (KS)','xl':'Standardized value (σ)',
   'cap1':'22 FDC sensors · box=IQR(25–75%), dot=median, whisker=10–90% · sorted by KS desc · pre 38,460 / post 85,154 rows',
   'lgpre':'Pre-PM','lgpost':'Post-PM',
   'sup2':'Per-sensor variation trend before/after PM  (daily median · IQR 25–75%)',
   'lg':['Daily median','IQR (25–75%)','Pre-PM median','Post-PM median','PM (12-24)'],
   'cap2':'x 2018-12-01–2019-02-08 · y clipped to robust 2–98% range · sorted by KS desc · C6 shown as recipe C6_1 share (%) in the last panel (categorical)'},
}
def setfont(lang): plt.rcParams['font.family']=KRFONT if lang=='ko' else 'DejaVu Sans'

print("loading...")
df=pd.read_csv(SRC,low_memory=False,usecols=['C40','C6']+SENSORS)
df['_dt']=pd.to_datetime(df['C40'],format='%Y-%m-%d %H:%M:%S.%f',errors='coerce')
df=df.dropna(subset=['_dt']).sort_values('_dt')
df['_pre']=df['_dt']<PM
d0=df['_dt'].dt.normalize().min(); d1=df['_dt'].dt.normalize().max()

# ---- per-sensor stats ----
rng=np.random.default_rng(0)
stat={}
for s in SENSORS:
    v=df[s].values; pre=v[df['_pre'].values]; post=v[~df['_pre'].values]
    lo,hi=np.percentile(v,[1,99])
    vw=np.clip(v,lo,hi); mu,sd=vw.mean(), vw.std()
    if sd==0: sd=1.0
    zpre=(np.clip(pre,lo,hi)-mu)/sd; zpost=(np.clip(post,lo,hi)-mu)/sd
    def qs(a): return np.percentile(a,[10,25,50,75,90])
    sa=pre if len(pre)<=20000 else rng.choice(pre,20000,replace=False)
    sb=post if len(post)<=20000 else rng.choice(post,20000,replace=False)
    ks=stats.ks_2samp(sa,sb).statistic
    stat[s]=dict(pre=qs(zpre),post=qs(zpost),ks=ks,
                 dirn=np.sign(np.clip(post,lo,hi).mean()-np.clip(pre,lo,hi).mean()))
order=sorted(SENSORS,key=lambda s:stat[s]['ks'],reverse=True)
n=len(order)
pre_med=df[df['_pre']][SENSORS].median(); post_med=df[~df['_pre']][SENSORS].median()

# ================= FIGURE 1 =================
def fig_overview(lang):
    setfont(lang); T=TXT[lang]
    fig=plt.figure(figsize=(15,11))
    gs=fig.add_gridspec(1,2,width_ratios=[3.6,1.0],wspace=0.04,
                        left=0.075,right=0.965,top=0.9,bottom=0.09)
    axM=fig.add_subplot(gs[0]); axK=fig.add_subplot(gs[1],sharey=axM)
    off=0.19
    for i,s in enumerate(order):
        y=i
        for arr,col,dy in [(stat[s]['pre'],C_PRE,-off),(stat[s]['post'],C_POST,+off)]:
            w10,q1,med,q3,w90=arr
            axM.plot([w10,w90],[y+dy]*2,color=col,lw=1.1,alpha=0.55,solid_capstyle='round',zorder=2)
            axM.plot([q1,q3],[y+dy]*2,color=col,lw=6.5,alpha=0.92,solid_capstyle='round',zorder=3)
            axM.plot([med],[y+dy],'o',color='white',ms=6,zorder=4)
            axM.plot([med],[y+dy],'o',color=col,ms=3.4,zorder=5)
        if i%2==1: axM.axhspan(y-0.5,y+0.5,color='#FAFAFA',zorder=0)
    axM.axvline(0,color='#BFBFBF',lw=0.9,zorder=1)
    axM.set_ylim(n-0.5,-0.5); axM.set_yticks(range(n)); axM.set_yticklabels(order,fontsize=10.5)
    axM.set_xlabel(T['xl'],fontsize=11); axM.set_title(T['main'],fontsize=12.5,pad=8,loc='left')
    axM.grid(axis='x',color=GRID,lw=0.8); axM.set_axisbelow(True)
    for sp in ['top','right','left']: axM.spines[sp].set_visible(False)
    axM.tick_params(axis='y',length=0)
    # KS bar
    for i,s in enumerate(order):
        col=C_POST if stat[s]['dirn']>=0 else C_PRE
        axK.barh(i,stat[s]['ks'],height=0.62,color=col,alpha=0.9,edgecolor='white',lw=0.5)
        axK.text(stat[s]['ks']+0.015,i,f"{stat[s]['ks']:.2f}",va='center',ha='left',fontsize=8.6,color='#333')
    axK.set_xlim(0,1.12); axK.set_title(T['ks'],fontsize=11,pad=8)
    axK.tick_params(axis='y',left=False,labelleft=False); axK.tick_params(axis='x',labelsize=9)
    axK.grid(axis='x',color=GRID,lw=0.8); axK.set_axisbelow(True)
    for sp in ['top','right','left']: axK.spines[sp].set_visible(False)
    hd=[Line2D([],[],color=C_PRE,lw=6.5,solid_capstyle='round'),
        Line2D([],[],color=C_POST,lw=6.5,solid_capstyle='round')]
    fig.legend(hd,[T['lgpre'],T['lgpost']],loc='upper right',ncol=2,fontsize=11.5,
               frameon=False,bbox_to_anchor=(0.965,0.965))
    fig.suptitle(T['sup1'],fontsize=17,fontweight='bold',x=0.075,ha='left',y=0.96)
    fig.text(0.075,0.022,T['cap1'],fontsize=8.7,color='#666',ha='left')
    p=f"{OUT}/01_overview_{lang}.png"; fig.savefig(p,dpi=150); plt.close(fig); print("saved",p); return p

# ================= FIGURE 2 =================
def fig_trend(lang):
    setfont(lang); T=TXT[lang]
    order23=order+['C6']; n2=len(order23)
    ncols=5; nrows=math.ceil(n2/ncols)
    fig,axes=plt.subplots(nrows,ncols,figsize=(20,3.05*nrows),sharex=False); axes=axes.ravel()
    g=df.set_index('_dt')[SENSORS]
    med=g.resample('D').median(); q1=g.resample('D').quantile(.25); q3=g.resample('D').quantile(.75)
    for k,s in enumerate(order23):
        ax=axes[k]
        if s=='C6':
            share=(df.assign(_c6=(df['C6']=='C6_1').astype(float)).set_index('_dt')['_c6'].resample('D').mean()*100)
            ax.axvspan(d0,PM,color=BG_PRE,zorder=0); ax.axvspan(PM,d1+pd.Timedelta(days=1),color=BG_POST,zorder=0)
            ax.plot(share.index,share.values,color=C_LINE,lw=1.3,zorder=3)
            ax.hlines(0.0,d0,PM,color=C_PRE,lw=1.8,ls=(0,(4,2)),zorder=4)
            ax.hlines(1.39,PM,d1+pd.Timedelta(days=1),color=C_POST,lw=1.8,ls=(0,(4,2)),zorder=4)
            ax.axvline(PM,color=C_PM,lw=1.3,ls='--',zorder=5)
            ttl='C6 · 레시피 C6_1 비율(%)' if lang=='ko' else 'C6 · recipe C6_1 share (%)'
            ax.set_title(ttl,fontsize=11,fontweight='bold',loc='left',pad=3,color='#8A4B08')
            ax.grid(True,color=GRID,lw=0.7); ax.set_axisbelow(True); ax.tick_params(labelsize=8.5)
            for sp in ['top','right']: ax.spines[sp].set_visible(False)
            for sp in ['left','bottom']: ax.spines[sp].set_color('#BBB')
            ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.set_xlim(d0,d1+pd.Timedelta(days=1)); ax.tick_params(axis='x',labelbottom=True)
            continue
        ax.axvspan(d0,PM,color=BG_PRE,zorder=0); ax.axvspan(PM,d1+pd.Timedelta(days=1),color=BG_POST,zorder=0)
        ax.fill_between(med.index,q1[s],q3[s],color=C_BAND,alpha=0.7,lw=0,zorder=1)
        ax.plot(med.index,med[s],color=C_LINE,lw=1.3,zorder=3)
        ax.hlines(pre_med[s],d0,PM,color=C_PRE,lw=1.8,ls=(0,(4,2)),zorder=4)
        ax.hlines(post_med[s],PM,d1+pd.Timedelta(days=1),color=C_POST,lw=1.8,ls=(0,(4,2)),zorder=4)
        ax.axvline(PM,color=C_PM,lw=1.3,ls='--',zorder=5)
        # robust y-lim: 1–99% of daily median, include pre/post med
        dm=med[s].dropna().values
        if len(dm):
            plo,phi=np.percentile(dm,[2,98])
        else: plo,phi=0,1
        ylo=min(plo,pre_med[s],post_med[s]); yhi=max(phi,pre_med[s],post_med[s])
        if yhi-ylo<1e-9: ylo,yhi=ylo-1,yhi+1
        pad=(yhi-ylo)*0.12; ax.set_ylim(ylo-pad,yhi+pad)
        ax.set_title(s,fontsize=12,fontweight='bold',loc='left',pad=3)
        ax.grid(True,color=GRID,lw=0.7); ax.set_axisbelow(True); ax.tick_params(labelsize=8.5)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        for sp in ['left','bottom']: ax.spines[sp].set_color('#BBB')
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.set_xlim(d0,d1+pd.Timedelta(days=1)); ax.tick_params(axis='x',labelbottom=True)
    for j in range(n2,len(axes)): axes[j].set_visible(False)
    handles=[Line2D([],[],color=C_LINE,lw=1.6),Patch(facecolor=C_BAND,alpha=0.7),
             Line2D([],[],color=C_PRE,lw=1.8,ls=(0,(4,2))),Line2D([],[],color=C_POST,lw=1.8,ls=(0,(4,2))),
             Line2D([],[],color=C_PM,lw=1.3,ls='--')]
    fig.legend(handles,T['lg'],loc='upper right',ncol=5,fontsize=11,frameon=False,bbox_to_anchor=(0.995,0.988))
    fig.suptitle(T['sup2'],fontsize=17,fontweight='bold',x=0.01,ha='left',y=0.992)
    fig.text(0.01,0.004,T['cap2'],fontsize=9.5,color='#666',ha='left')
    fig.tight_layout(rect=[0,0.012,1,0.965])
    p=f"{OUT}/02_trend_{lang}.png"; fig.savefig(p,dpi=140); plt.close(fig); print("saved",p); return p

paths=[]
for lang in ['ko','en']:
    paths.append(fig_overview(lang)); paths.append(fig_trend(lang))
print("KS order:", order)
print("DONE")
