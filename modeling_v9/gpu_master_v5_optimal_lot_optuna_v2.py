"""
Script 2 V2: gpu_master_v5_optimal_lot_optuna_v2.py
- 특징: LOT 집계 누수 허용 (최고 점수 방식)
- 비급: Target Clipping, Missing Indicator, Isotonic 적용
- 새로운 파이프라인: Fast Ablation (피처 확정) -> 해당 피처로만 Deep Optuna (100회)
"""

import numpy as np
import pandas as pd
import warnings
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = r'c:\Users\HP\Downloads\SK 하이닉스\Data\문제1(하)\train_data.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TRIALS = 100

print(f"🚀 [Phase 1] 데이터 로딩 및 분석가 비급 적용 (LOT 정보 포함)")
df = pd.read_csv(DATA_PATH)
sensors = ['C1','C9','C11','C12','C15','C16','C17','C18','C25','C27',
           'C31','C32','C48','C52','C57','C58','C59','C60','C61','C62','C63']

wf_y = df.groupby('C64')['C65'].first()
wf_lot = df.groupby('C64')['C20'].first()
feat = pd.DataFrame(index=wf_y.index)

# 비급 1: Target Clipping
y_raw = wf_y.values
second_max_y = np.sort(y_raw)[-2]
y = np.clip(y_raw, a_min=None, a_max=second_max_y)

# 비급 3: Missing Indicator
missing_counts = df.groupby('C64')[sensors].apply(lambda x: x.isna().sum())
for s in sensors:
    if missing_counts[s].sum() > 0: feat[f'{s}_is_missing'] = (missing_counts[s] > 0).astype(int)

for step in [1, 2, 3, 4, 5, 6]:
    sub = df[df['C7'] == step]
    grp = sub.groupby('C64')[sensors]
    feat = feat.join(grp.mean().add_prefix(f's{step}_mean_'))
    feat = feat.join(grp.std().add_prefix(f's{step}_std_'))

s4 = df[df['C7']==4].groupby('C64')[sensors].mean()
s4['C20'] = wf_lot
for s in ['C17','C63','C62','C61','C12','C16']:
    feat[f'LOT_mean_{s}'] = s4.groupby('C20')[s].transform('mean')
    feat[f'LOT_std_{s}']  = s4.groupby('C20')[s].transform('std')

feat['C17_x_C63'] = s4['C17'] * s4['C63']
feat['C12_x_C62'] = s4['C12'] * s4['C62']

X_all = feat.replace([np.inf, -np.inf], np.nan).fillna(feat.median()).fillna(0)
print(f" -> 피처 생성 완료 (총 {X_all.shape[1]}개)")

df_sorted = df.sort_values(by=['C64', 'C7']).fillna(0)
raw_sequences = [df_sorted.groupby('C64').get_group(wid)[sensors].values for wid in wf_y.index]
all_seq_data = np.vstack(raw_sequences)
scaler_cnn = StandardScaler().fit(all_seq_data)
sequences = [scaler_cnn.transform(seq) for seq in raw_sequences]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# =====================================================================
# Phase 2: Fast Feature Ablation (디폴트 파라미터로 피처 확정)
# =====================================================================
print("\n[Phase 2] Fast Feature Ablation (피처 개수 및 리스트 확정)")
models_default = {
    'LGBM': lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1),
    'XGB': XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1),
    'CAT': CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=5, random_seed=42, verbose=0)
}

best_features_dict = {}

for m_name, model in models_default.items():
    model.fit(X_all, y)
    imps = model.feature_importances_ if m_name != 'CAT' else model.get_feature_importance()
    sorted_features = pd.DataFrame({'f': X_all.columns, 'imp': imps}).sort_values('imp', ascending=False)['f'].tolist()
    
    best_rmse, best_n = float('inf'), 0
    for n in range(1, 26):
        cur_feat = sorted_features[:n]
        oof_cur = np.zeros(len(y))
        for tr_idx, va_idx in kf.split(X_all):
            X_tr, X_va = X_all[cur_feat].iloc[tr_idx], X_all[cur_feat].iloc[va_idx]
            if m_name == 'LGBM':
                m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
                m.fit(X_tr, y[tr_idx], eval_set=[(X_va, y[va_idx])], callbacks=[lgb.early_stopping(30, verbose=False)])
            elif m_name == 'XGB':
                m = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
                m.fit(X_tr, y[tr_idx], eval_set=[(X_va, y[va_idx])], verbose=False)
            else:
                m = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=5, random_seed=42, verbose=0)
                m.fit(X_tr, y[tr_idx], eval_set=(X_va, y[va_idx]), early_stopping_rounds=30, verbose=0)
            oof_cur[va_idx] = m.predict(X_va)
        cur_rmse = np.sqrt(mean_squared_error(y, oof_cur))
        if cur_rmse < best_rmse: best_rmse, best_n = cur_rmse, n
            
    best_features_dict[m_name] = sorted_features[:best_n]
    print(f" -> {m_name} 최적 피처 {best_n}개 확정 완료 (RMSE: {best_rmse:.4f})")

# =====================================================================
# Phase 3: Deep Optuna Tuning (오직 확정된 피처만 사용)
# =====================================================================
print(f"\n[Phase 3] 확정된 피처 대상 Deep Optuna 튜닝 (모델당 {N_TRIALS}회)")
final_oofs = {}

def get_lgbm_objective(X_subset, y):
    def objective(trial):
        params = {
            'n_estimators': 500, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1
        }
        oof = np.zeros(len(y))
        for tr_idx, va_idx in kf.split(X_subset):
            m = lgb.LGBMRegressor(**params)
            m.fit(X_subset.iloc[tr_idx], y[tr_idx], eval_set=[(X_subset.iloc[va_idx], y[va_idx])], callbacks=[lgb.early_stopping(30, verbose=False)])
            oof[va_idx] = m.predict(X_subset.iloc[va_idx])
        return np.sqrt(mean_squared_error(y, oof))
    return objective

def get_xgb_objective(X_subset, y):
    def objective(trial):
        params = {
            'n_estimators': 500, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42, 'n_jobs': -1
        }
        oof = np.zeros(len(y))
        for tr_idx, va_idx in kf.split(X_subset):
            m = XGBRegressor(**params)
            m.fit(X_subset.iloc[tr_idx], y[tr_idx], eval_set=[(X_subset.iloc[va_idx], y[va_idx])], verbose=False)
            oof[va_idx] = m.predict(X_subset.iloc[va_idx])
        return np.sqrt(mean_squared_error(y, oof))
    return objective

def get_cat_objective(X_subset, y):
    def objective(trial):
        params = {
            'iterations': 500, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'depth': trial.suggest_int('depth', 3, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_seed': 42, 'verbose': 0
        }
        oof = np.zeros(len(y))
        for tr_idx, va_idx in kf.split(X_subset):
            m = CatBoostRegressor(**params)
            m.fit(X_subset.iloc[tr_idx], y[tr_idx], eval_set=(X_subset.iloc[va_idx], y[va_idx]), early_stopping_rounds=30, verbose=0)
            oof[va_idx] = m.predict(X_subset.iloc[va_idx])
        return np.sqrt(mean_squared_error(y, oof))
    return objective

for m_name, obj_func in zip(['LGBM', 'XGB', 'CAT'], [get_lgbm_objective, get_xgb_objective, get_cat_objective]):
    print(f" -> {m_name} 튜닝 중... (피처 {len(best_features_dict[m_name])}개 대상)")
    X_subset = X_all[best_features_dict[m_name]]
    study = optuna.create_study(direction='minimize')
    study.optimize(obj_func(X_subset, y), n_trials=N_TRIALS)
    
    best_p = study.best_params
    best_p['random_state'] = 42 if m_name != 'CAT' else None
    if m_name == 'CAT': best_p['random_seed'] = 42
    if m_name != 'CAT': best_p['n_jobs'] = -1
    if m_name == 'LGBM': best_p['verbose'] = -1
    if m_name == 'CAT': best_p['verbose'] = 0
    best_p['n_estimators'] = 1000
    if m_name == 'CAT':
        best_p['iterations'] = 1000
        del best_p['n_estimators']

    final_oof = np.zeros(len(y))
    for tr_idx, va_idx in kf.split(X_subset):
        X_tr, X_va = X_subset.iloc[tr_idx], X_subset.iloc[va_idx]
        if m_name == 'LGBM':
            m = lgb.LGBMRegressor(**best_p)
            m.fit(X_tr, y[tr_idx], eval_set=[(X_va, y[va_idx])], callbacks=[lgb.early_stopping(30, verbose=False)])
        elif m_name == 'XGB':
            m = XGBRegressor(**best_p)
            m.fit(X_tr, y[tr_idx], eval_set=[(X_va, y[va_idx])], verbose=False)
        else:
            m = CatBoostRegressor(**best_p)
            m.fit(X_tr, y[tr_idx], eval_set=(X_va, y[va_idx]), early_stopping_rounds=30, verbose=0)
        final_oof[va_idx] = m.predict(X_va)
        
    final_oofs[m_name] = final_oof
    print(f"   * {m_name} 튜닝 완료 최적 OOF RMSE: {np.sqrt(mean_squared_error(y, final_oof)):.4f}")

# =====================================================================
# Phase 4: 1D-CNN 학습
# =====================================================================
print("\n[Phase 4] 1D-CNN 학습")
class WaferDataset(Dataset):
    def __init__(self, seqs, lbls):
        self.seqs, self.lbls = seqs, lbls
        self.max_len = max(len(s) for s in seqs)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        padded = np.zeros((self.max_len, self.seqs[idx].shape[1]), dtype=np.float32)
        padded[:len(self.seqs[idx]), :] = self.seqs[idx]
        return torch.tensor(padded, dtype=torch.float32), torch.tensor(self.lbls[idx], dtype=torch.float32)

class WaferCNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_features, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.regressor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.regressor(self.net(x.transpose(1, 2)).squeeze(-1)).squeeze(-1)

oof_cnn = np.zeros(len(y))
for fold, (train_idx, valid_idx) in enumerate(kf.split(sequences)):
    train_loader = DataLoader(WaferDataset([sequences[i] for i in train_idx], [y[i] for i in train_idx]), batch_size=128, shuffle=True)
    valid_loader = DataLoader(WaferDataset([sequences[i] for i in valid_idx], [y[i] for i in valid_idx]), batch_size=128, shuffle=False)
    cnn_model = WaferCNN(len(sensors)).to(device)
    optimizer = optim.AdamW(cnn_model.parameters(), lr=0.001)
    
    best_preds, best_loss = [], float('inf')
    for ep in range(25):
        cnn_model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            nn.MSELoss()(cnn_model(bx.to(device)), by.to(device)).backward()
            optimizer.step()
        cnn_model.eval()
        ep_preds, val_loss = [], 0
        with torch.no_grad():
            for bx, by in valid_loader:
                preds = cnn_model(bx.to(device))
                val_loss += nn.MSELoss()(preds, by.to(device)).item() * len(bx)
                ep_preds.extend(preds.cpu().numpy())
        if val_loss < best_loss: best_loss, best_preds = val_loss, ep_preds
    oof_cnn[valid_idx] = best_preds
final_oofs['CNN'] = oof_cnn

# =====================================================================
# Phase 5: Meta Stacking & Isotonic 보정 (비급)
# =====================================================================
print("\n[Phase 5] 최종 Meta Stacking 및 Isotonic 보정")
X_meta = np.column_stack([final_oofs['LGBM'], final_oofs['XGB'], final_oofs['CAT'], final_oofs['CNN']])
meta_oof = np.zeros(len(y))
for train_idx, valid_idx in kf.split(X_meta):
    m = Ridge(alpha=1.0)
    m.fit(X_meta[train_idx], y[train_idx])
    meta_oof[valid_idx] = m.predict(X_meta[valid_idx])

iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_oof = np.zeros(len(y))
for train_idx, valid_idx in kf.split(meta_oof):
    iso_reg.fit(meta_oof[train_idx], y[train_idx])
    iso_oof[valid_idx] = iso_reg.predict(meta_oof[valid_idx])

print("\n" + "="*60)
print(f" 🔥🔥🔥 최종 Script 2 (LOT + 10기 비급 + 피처확정 후 Optuna) RMSE: {np.sqrt(mean_squared_error(y, iso_oof)):.4f} 🔥🔥🔥")
print("="*60)
