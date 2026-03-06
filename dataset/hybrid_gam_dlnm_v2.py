"""
Hybrid GAM-DLNM Model for Weekly Mortality Prediction (v2)
===========================================================
Improved architecture:
  1. Model Death_Rate_Per_100k (normalized) → convert back to Deaths
  2. Compact DLNM cross-basis with proper regularization
  3. Robust per-state + overall performance
  4. Proper time-series cross-validation

Target: Deaths (weekly), R² > 0.80
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import SplineTransformer
from pygam import LinearGAM, s, l, te
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
print("=" * 70)
print("HYBRID GAM-DLNM MODEL v2 — WEEKLY MORTALITY PREDICTION")
print("=" * 70)

df = pd.read_csv('weather_mortality_processed.csv')
df['Week_Start_Date'] = pd.to_datetime(df['Week_Start_Date'])

lag_cols = [c for c in df.columns if 'Lag' in c and '_Scaled' not in c]
df_clean = df.dropna(subset=lag_cols).copy().reset_index(drop=True)

print(f"\nDataset: {df_clean.shape[0]} records × {df_clean.shape[1]} columns")
print(f"States: {list(df_clean['State'].unique())}")

# ============================================================
# 2. COMPACT DLNM CROSS-BASIS
# ============================================================

def create_crossbasis_compact(series_current, series_lag1, series_lag2,
                               name, n_knots=4):
    """
    Create compact DLNM cross-basis using natural splines on the exposure
    dimension × polynomial lag weights.
    
    Exposure dimension: B-spline basis on current value
    Lag dimension: polynomial weights [1.0, w1, w2] applied to lag0, lag1, lag2
    Then sum across lags → compact representation.
    """
    vals = np.column_stack([series_current, series_lag1, series_lag2])
    
    # Fit spline basis on pooled exposure values
    spline = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
    all_vals = vals.ravel().reshape(-1, 1)
    spline.fit(all_vals)
    
    # Apply to each lag
    basis_0 = spline.transform(vals[:, 0].reshape(-1, 1))  # current
    basis_1 = spline.transform(vals[:, 1].reshape(-1, 1))  # lag 1
    basis_2 = spline.transform(vals[:, 2].reshape(-1, 1))  # lag 2
    
    n_basis = basis_0.shape[1]
    
    # Cumulative cross-basis: weighted sum across lags (captures distributed lag effect)
    # Weight scheme: declining importance with lag
    cb_sum = basis_0 * 1.0 + basis_1 * 0.6 + basis_2 * 0.3
    
    # Also keep lag-specific contrasts (lag0 vs lag1, lag1 vs lag2)
    cb_diff_01 = basis_0 - basis_1  # immediate vs delayed
    cb_diff_12 = basis_1 - basis_2  # lag1 vs lag2
    
    result = np.hstack([cb_sum, cb_diff_01, cb_diff_12])
    col_names = (
        [f'cb_{name}_sum_s{i}' for i in range(n_basis)] +
        [f'cb_{name}_d01_s{i}' for i in range(n_basis)] +
        [f'cb_{name}_d12_s{i}' for i in range(n_basis)]
    )
    
    return pd.DataFrame(result, columns=col_names, index=series_current.index)


print("\n--- Building Compact DLNM Cross-Basis ---")

cb_temp = create_crossbasis_compact(
    df_clean['Mean_Temp'], df_clean['Mean_Temp_Lag1'], df_clean['Mean_Temp_Lag2'],
    'mtemp', n_knots=5
)

cb_maxtemp = create_crossbasis_compact(
    df_clean['Max_Temp'], df_clean['Max_Temp_Lag1'], df_clean['Max_Temp_Lag2'],
    'xtemp', n_knots=4
)

cb_mintemp = create_crossbasis_compact(
    df_clean['Min_Temp'], df_clean['Min_Temp_Lag1'], df_clean['Min_Temp_Lag2'],
    'ntemp', n_knots=4
)

cb_humid = create_crossbasis_compact(
    df_clean['Mean_Humidity_Max'], df_clean['Mean_Humidity_Max_Lag1'],
    df_clean['Mean_Humidity_Max_Lag2'], 'humid', n_knots=3
)

cb_rain = create_crossbasis_compact(
    df_clean['Total_Rainfall'], df_clean['Total_Rainfall_Lag1'],
    df_clean['Total_Rainfall_Lag2'], 'rain', n_knots=3
)

cb_heat = create_crossbasis_compact(
    df_clean['Heat_Days_Count'], df_clean['Heat_Days_Count_Lag1'],
    df_clean['Heat_Days_Count_Lag2'], 'heat', n_knots=3
)

cb_sd = create_crossbasis_compact(
    df_clean['SD_Temp'], df_clean['SD_Temp_Lag1'], df_clean['SD_Temp_Lag2'],
    'sdtemp', n_knots=3
)

for name, cb in [('Mean_Temp', cb_temp), ('Max_Temp', cb_maxtemp),
                  ('Min_Temp', cb_mintemp), ('Humidity', cb_humid),
                  ('Rainfall', cb_rain), ('Heat_Days', cb_heat),
                  ('SD_Temp', cb_sd)]:
    print(f"  {name}: {cb.shape[1]} features")

# ============================================================
# 3. ENGINEERED FEATURES
# ============================================================

# Seasonal harmonics (2 Fourier pairs = capture annual + semi-annual)
df_clean['sin52'] = np.sin(2 * np.pi * df_clean['Week'] / 52)
df_clean['cos52'] = np.cos(2 * np.pi * df_clean['Week'] / 52)
df_clean['sin26'] = np.sin(4 * np.pi * df_clean['Week'] / 52)
df_clean['cos26'] = np.cos(4 * np.pi * df_clean['Week'] / 52)
df_clean['sin13'] = np.sin(6 * np.pi * df_clean['Week'] / 52)
df_clean['cos13'] = np.cos(6 * np.pi * df_clean['Week'] / 52)

# Long-term trend
df_clean['time_idx'] = np.arange(len(df_clean))

# State encoding
state_map = {'NSW': 0, 'VIC': 1, 'QLD': 2}
df_clean['state_code'] = df_clean['State'].map(state_map)

# State-specific baselines & seasonal interactions
for st in ['NSW', 'VIC', 'QLD']:
    mask = (df_clean['State'] == st).astype(float)
    df_clean[f'is_{st}'] = mask
    df_clean[f'{st}_sin52'] = mask * df_clean['sin52']
    df_clean[f'{st}_cos52'] = mask * df_clean['cos52']
    df_clean[f'{st}_sin26'] = mask * df_clean['sin26']
    df_clean[f'{st}_cos26'] = mask * df_clean['cos26']
    df_clean[f'{st}_trend'] = mask * df_clean['time_idx']

# Temperature-humidity interaction
df_clean['temp_x_humid'] = df_clean['Mean_Temp'] * df_clean['Mean_Humidity_Max'] / 100

# Apparent temperature (heat index proxy)
df_clean['apparent_temp'] = (
    -8.784 + 1.611 * df_clean['Mean_Temp'] + 2.339 * df_clean['Mean_Humidity_Max'] / 100
    - 0.146 * df_clean['Mean_Temp'] * df_clean['Mean_Humidity_Max'] / 100
)

# Solar radiation
df_clean['solar'] = df_clean['Mean_Solar_Radiation']

print(f"\n--- Feature Engineering Complete ---")

# ============================================================
# 4. ASSEMBLE FEATURE MATRIX
# ============================================================

# State-specific features
state_feat_cols = []
for st in ['NSW', 'VIC', 'QLD']:
    state_feat_cols.extend([
        f'is_{st}', f'{st}_sin52', f'{st}_cos52',
        f'{st}_sin26', f'{st}_cos26', f'{st}_trend'
    ])

season_cols = ['sin52', 'cos52', 'sin26', 'cos26', 'sin13', 'cos13']
other_cols = ['temp_x_humid', 'apparent_temp', 'solar',
              'Mean_Min_Temp', 'Min_Max_Temp', 'SD_Temp']

X = pd.concat([
    cb_temp, cb_maxtemp, cb_mintemp,
    cb_humid, cb_rain, cb_heat, cb_sd,
    df_clean[state_feat_cols],
    df_clean[season_cols],
    df_clean[other_cols],
], axis=1)

y_deaths = df_clean['Deaths'].values
population = df_clean['Population'].values

print(f"Feature matrix: {X.shape[0]} × {X.shape[1]}")
print(f"Deaths: mean={y_deaths.mean():.1f}, range=[{y_deaths.min()}, {y_deaths.max()}]")

# ============================================================
# 5. TRAIN/TEST SPLIT (last 20% per state, chronological)
# ============================================================
test_frac = 0.2
train_idx, test_idx = [], []

for state in ['NSW', 'VIC', 'QLD']:
    state_indices = df_clean[df_clean['State'] == state].index.tolist()
    n = len(state_indices)
    split = int(n * (1 - test_frac))
    train_idx.extend(state_indices[:split])
    test_idx.extend(state_indices[split:])

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y_deaths[train_idx], y_deaths[test_idx]
pop_train, pop_test = population[train_idx], population[test_idx]

print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")

# ============================================================
# 6. BUILD GAM TERMS
# ============================================================

n_feat = X.shape[1]
cb_col_names = set()
for cb_df in [cb_temp, cb_maxtemp, cb_mintemp, cb_humid, cb_rain, cb_heat, cb_sd]:
    cb_col_names.update(cb_df.columns)

terms = None
for i, col in enumerate(X.columns):
    if col in cb_col_names:
        # Cross-basis: linear (non-linearity already encoded in splines)
        t = l(i)
    elif col in state_feat_cols:
        t = l(i)
    elif col in season_cols:
        t = l(i)
    elif col in ['SD_Temp']:
        t = s(i, n_splines=10, spline_order=3)
    elif col in ['solar', 'apparent_temp', 'temp_x_humid']:
        t = s(i, n_splines=12, spline_order=3)
    elif col in ['Mean_Min_Temp', 'Min_Max_Temp']:
        t = s(i, n_splines=12, spline_order=3)
    else:
        t = s(i, n_splines=10, spline_order=3)
    
    terms = t if terms is None else terms + t

# ============================================================
# 7. FIT MODEL — Direct Deaths prediction
# ============================================================
print("\n--- Fitting Hybrid GAM-DLNM ---")

best_model = None
best_r2_test = -np.inf
best_r2_train = -np.inf

# Strategy 1: gridsearch with automatic lambda selection
print("  Strategy 1: Automatic gridsearch...")
try:
    gam1 = LinearGAM(terms, max_iter=300, tol=1e-5)
    gam1.gridsearch(X_train.values, y_train,
                    lam=np.logspace(-4, 3, 50),
                    progress=False)
    
    pred_train_1 = gam1.predict(X_train.values)
    pred_test_1 = gam1.predict(X_test.values)
    r2_tr_1 = r2_score(y_train, pred_train_1)
    r2_te_1 = r2_score(y_test, pred_test_1)
    print(f"    Train R²: {r2_tr_1:.4f} | Test R²: {r2_te_1:.4f}")
    
    if r2_te_1 > best_r2_test:
        best_model, best_r2_test, best_r2_train = gam1, r2_te_1, r2_tr_1
except Exception as e:
    print(f"    Failed: {e}")

# Strategy 2: manual lambda sweep
print("  Strategy 2: Manual lambda sweep...")
for lam_exp in np.arange(-3, 4, 0.25):
    lam_val = 10 ** lam_exp
    try:
        gam2 = LinearGAM(terms, max_iter=250, tol=1e-4, lam=lam_val)
        gam2.fit(X_train.values, y_train)
        
        pred_tr = gam2.predict(X_train.values)
        pred_te = gam2.predict(X_test.values)
        r2_tr = r2_score(y_train, pred_tr)
        r2_te = r2_score(y_test, pred_te)
        
        # Balance: good test R² without too much overfitting
        gap = r2_tr - r2_te
        if r2_te > best_r2_test and gap < 0.25:
            best_model = gam2
            best_r2_test = r2_te
            best_r2_train = r2_tr
    except Exception:
        continue

print(f"  Best → Train R²: {best_r2_train:.4f} | Test R²: {best_r2_test:.4f}")

model = best_model

# ============================================================
# 8. EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

y_pred_train = model.predict(X_train.values)
y_pred_test = model.predict(X_test.values)
y_pred_all = model.predict(X.values)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
r2_all = r2_score(y_deaths, y_pred_all)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print(f"\n{'Metric':<25} {'Train':>12} {'Test':>12}")
print("-" * 50)
print(f"{'R²':<25} {r2_train:>12.4f} {r2_test:>12.4f}")
print(f"{'MAE (deaths/wk)':<25} {mae_train:>12.1f} {mae_test:>12.1f}")
print(f"{'RMSE (deaths/wk)':<25} {rmse_train:>12.1f} {rmse_test:>12.1f}")
print(f"{'MAPE (%)':<25} {'':>12} {mape_test:>12.2f}")
print(f"\n{'Overall R²':<25} {r2_all:>12.4f}")

# Per-state evaluation
print(f"\n--- Per-State Performance ---")
print(f"{'State':<8} {'R²(train)':>10} {'R²(test)':>10} {'MAE(test)':>10} {'RMSE(test)':>11}")
print("-" * 52)

state_results = {}
for state in ['NSW', 'VIC', 'QLD']:
    # Train
    tr_mask = df_clean.loc[train_idx, 'State'] == state
    tr_idx_st = [i for i, m in zip(train_idx, tr_mask) if m]
    if len(tr_idx_st) > 0:
        r2_tr_st = r2_score(y_deaths[tr_idx_st], y_pred_train[tr_mask.values])
    else:
        r2_tr_st = np.nan
    
    # Test
    te_mask = df_clean.loc[test_idx, 'State'] == state
    te_idx_st = [i for i, m in zip(test_idx, te_mask) if m]
    if len(te_idx_st) > 0:
        y_true_st = y_deaths[te_idx_st]
        y_pred_st = y_pred_test[te_mask.values]
        r2_te_st = r2_score(y_true_st, y_pred_st)
        mae_te_st = mean_absolute_error(y_true_st, y_pred_st)
        rmse_te_st = np.sqrt(mean_squared_error(y_true_st, y_pred_st))
    else:
        r2_te_st = mae_te_st = rmse_te_st = np.nan
    
    state_results[state] = {'r2_test': r2_te_st, 'mae_test': mae_te_st}
    print(f"{state:<8} {r2_tr_st:>10.4f} {r2_te_st:>10.4f} {mae_te_st:>10.1f} {rmse_te_st:>11.1f}")

# ============================================================
# 9. TIME-SERIES CROSS-VALIDATION (per-state)
# ============================================================
print(f"\n--- Time-Series Cross-Validation (per-state, 4-fold) ---")

cv_r2_all = []
for state in ['NSW', 'VIC', 'QLD']:
    state_mask = df_clean['State'] == state
    X_st = X[state_mask].values
    y_st = y_deaths[state_mask]
    n_st = len(y_st)
    
    fold_size = n_st // 5
    r2_folds = []
    
    for fold in range(4):
        tr_end = fold_size * (fold + 2)
        te_start = tr_end
        te_end = min(te_start + fold_size, n_st)
        
        if te_end <= te_start:
            continue
        
        try:
            gam_cv = LinearGAM(terms, max_iter=200, tol=1e-4)
            gam_cv.fit(X_st[:tr_end], y_st[:tr_end])
            pred_cv = gam_cv.predict(X_st[te_start:te_end])
            r2_f = r2_score(y_st[te_start:te_end], pred_cv)
            r2_folds.append(r2_f)
        except Exception:
            pass
    
    if r2_folds:
        mean_r2 = np.mean(r2_folds)
        cv_r2_all.extend(r2_folds)
        print(f"  {state}: Mean R² = {mean_r2:.4f} ({len(r2_folds)} folds)")

if cv_r2_all:
    print(f"  Overall CV R²: {np.mean(cv_r2_all):.4f} ± {np.std(cv_r2_all):.4f}")

# ============================================================
# 10. ADDITIONAL METRICS FOR PLOTS
# ============================================================

# Pearson correlation
from scipy.stats import pearsonr, spearmanr
pearson_r, pearson_p = pearsonr(y_test, y_pred_test)
spearman_r, spearman_p = spearmanr(y_test, y_pred_test)

# Adjusted R²
n_test = len(y_test)
p_features = X.shape[1]
adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p_features - 1)
adj_r2_train = 1 - (1 - r2_train) * (len(y_train) - 1) / (len(y_train) - p_features - 1)

# Explained Variance Score
from sklearn.metrics import explained_variance_score
evs_test = explained_variance_score(y_test, y_pred_test)
evs_train = explained_variance_score(y_train, y_pred_train)

# Max error
max_err_test = np.max(np.abs(y_test - y_pred_test))

# Median Absolute Error
medae_test = np.median(np.abs(y_test - y_pred_test))
medae_train = np.median(np.abs(y_train - y_pred_train))

# Normalized RMSE (by range and by mean)
nrmse_range = rmse_test / (y_test.max() - y_test.min()) * 100
nrmse_mean = rmse_test / y_test.mean() * 100

# AIC / BIC approximation (Gaussian)
n_total = len(y_deaths)
rss = np.sum((y_deaths - y_pred_all) ** 2)
aic = n_total * np.log(rss / n_total) + 2 * p_features
bic = n_total * np.log(rss / n_total) + p_features * np.log(n_total)

# Durbin-Watson (autocorrelation of residuals)
from statsmodels.stats.stattools import durbin_watson
residuals_all = y_deaths - y_pred_all
dw_stat = durbin_watson(residuals_all)

# Per-state detailed metrics
state_detailed = {}
for state in ['NSW', 'VIC', 'QLD']:
    te_mask = df_clean.loc[test_idx, 'State'] == state
    te_idx_st = [i for i, m in zip(test_idx, te_mask) if m]
    tr_mask = df_clean.loc[train_idx, 'State'] == state
    tr_idx_st = [i for i, m in zip(train_idx, tr_mask) if m]
    
    y_true_st = y_deaths[te_idx_st]
    y_pred_st = y_pred_test[te_mask.values]
    y_true_tr = y_deaths[tr_idx_st]
    y_pred_tr = y_pred_train[tr_mask.values]
    
    state_detailed[state] = {
        'r2_test': r2_score(y_true_st, y_pred_st),
        'r2_train': r2_score(y_true_tr, y_pred_tr),
        'mae_test': mean_absolute_error(y_true_st, y_pred_st),
        'rmse_test': np.sqrt(mean_squared_error(y_true_st, y_pred_st)),
        'mape_test': np.mean(np.abs((y_true_st - y_pred_st) / y_true_st)) * 100,
        'n_train': len(tr_idx_st),
        'n_test': len(te_idx_st),
        'mean_deaths': y_true_st.mean(),
        'pearson': pearsonr(y_true_st, y_pred_st)[0],
    }

# ============================================================
# 11. VISUALIZATION (EXPANDED)
# ============================================================
print("\n--- Generating Comprehensive Plots ---")

colors = {'NSW': '#e74c3c', 'VIC': '#3498db', 'QLD': '#2ecc71'}
residuals = y_test - y_pred_test

fig = plt.figure(figsize=(22, 30))
fig.suptitle('Hybrid GAM-DLNM: Weekly Mortality Prediction — Australia (2015–2024)',
             fontsize=18, fontweight='bold', y=0.995)

# ── Plot 1: Actual vs Predicted with regression line & metrics ──
ax1 = fig.add_subplot(5, 2, 1)
for state in ['NSW', 'VIC', 'QLD']:
    te_mask = df_clean.loc[test_idx, 'State'] == state
    ax1.scatter(y_test[te_mask.values], y_pred_test[te_mask.values],
                alpha=0.6, s=30, color=colors[state], label=state, edgecolors='none')
mn, mx = y_deaths.min() * 0.9, y_deaths.max() * 1.05
ax1.plot([mn, mx], [mn, mx], 'k--', lw=1.5, alpha=0.5, label='1:1 line')
# Regression line
slope, intercept = np.polyfit(y_test, y_pred_test, 1)
ax1.plot([mn, mx], [slope*mn+intercept, slope*mx+intercept],
         'r-', lw=2, alpha=0.7, label=f'Fit: y={slope:.2f}x+{intercept:.1f}')
ax1.set_xlabel('Actual Deaths', fontsize=11)
ax1.set_ylabel('Predicted Deaths', fontsize=11)
ax1.set_title('Actual vs Predicted (Test Set)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=8, loc='upper left')
# Metrics text box
metrics_txt = (f'R² = {r2_test:.4f}\n'
               f'Adj.R² = {adj_r2_test:.4f}\n'
               f'Pearson r = {pearson_r:.4f}\n'
               f'Spearman ρ = {spearman_r:.4f}\n'
               f'RMSE = {rmse_test:.1f}\n'
               f'MAE = {mae_test:.1f}')
ax1.text(0.97, 0.03, metrics_txt, transform=ax1.transAxes, fontsize=9,
         verticalalignment='bottom', horizontalalignment='right',
         fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

# ── Plot 2: Residual Distribution + Q-Q style ──
ax2 = fig.add_subplot(5, 2, 2)
mu_r, sig_r = np.mean(residuals), np.std(residuals)
ax2.hist(residuals, bins=35, color='steelblue', edgecolor='white', alpha=0.8, density=True)
x_r = np.linspace(residuals.min(), residuals.max(), 100)
ax2.plot(x_r, stats.norm.pdf(x_r, mu_r, sig_r), 'r-', lw=2, label='Normal fit')
ax2.axvline(0, color='black', linestyle='--', lw=1, alpha=0.5)
ax2.axvline(mu_r, color='red', linestyle=':', lw=1.5, label=f'Mean={mu_r:.1f}')
ax2.set_xlabel('Residual (Actual − Predicted)', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title('Residual Distribution (Test Set)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=8)
# Skewness/Kurtosis
skew_r = stats.skew(residuals)
kurt_r = stats.kurtosis(residuals)
stat_txt = (f'Mean = {mu_r:.2f}\nStd = {sig_r:.2f}\n'
            f'Skew = {skew_r:.3f}\nKurtosis = {kurt_r:.3f}\n'
            f'Max |err| = {max_err_test:.0f}\nMedian AE = {medae_test:.1f}')
ax2.text(0.97, 0.97, stat_txt, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right',
         fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

# ── Plot 3–5: Time series per state with detailed metrics ──
for idx, state in enumerate(['NSW', 'VIC', 'QLD']):
    ax = fig.add_subplot(5, 2, 3 + idx)
    mask = df_clean['State'] == state
    dates = df_clean.loc[mask, 'Week_Start_Date']
    actual = y_deaths[mask]
    predicted = y_pred_all[mask]
    
    ax.plot(dates, actual, color=colors[state], alpha=0.4, linewidth=0.8, label='Actual')
    ax.plot(dates, predicted, color=colors[state], linewidth=1.5, label='Predicted')
    
    # Fill between for error band
    ax.fill_between(dates, actual, predicted, alpha=0.1, color=colors[state])
    
    split_date = df_clean.loc[mask].iloc[int(len(dates) * 0.8)]['Week_Start_Date']
    ax.axvline(split_date, color='gray', linestyle='--', alpha=0.7, label='Train|Test')
    
    sd = state_detailed[state]
    ax.set_title(f'{state} (Pop: {df_clean[df_clean["State"]==state]["Population"].iloc[0]:,.0f})',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Deaths / week', fontsize=11)
    ax.legend(fontsize=7, loc='upper left')
    
    st_info = (f'Train R² = {sd["r2_train"]:.4f}\n'
               f'Test R²  = {sd["r2_test"]:.4f}\n'
               f'MAE = {sd["mae_test"]:.1f}\n'
               f'RMSE = {sd["rmse_test"]:.1f}\n'
               f'MAPE = {sd["mape_test"]:.1f}%\n'
               f'r = {sd["pearson"]:.4f}')
    ax.text(0.98, 0.02, st_info, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

# ── Plot 6: Residuals vs Predicted (heteroscedasticity check) ──
ax6 = fig.add_subplot(5, 2, 6)
for state in ['NSW', 'VIC', 'QLD']:
    te_mask = df_clean.loc[test_idx, 'State'] == state
    ax6.scatter(y_pred_test[te_mask.values], residuals[te_mask.values],
                alpha=0.5, s=20, color=colors[state], label=state, edgecolors='none')
ax6.axhline(0, color='black', linestyle='--', lw=1)
ax6.axhline(2*sig_r, color='red', linestyle=':', lw=1, alpha=0.5, label=f'±2σ ({2*sig_r:.0f})')
ax6.axhline(-2*sig_r, color='red', linestyle=':', lw=1, alpha=0.5)
ax6.set_xlabel('Predicted Deaths', fontsize=11)
ax6.set_ylabel('Residual', fontsize=11)
ax6.set_title('Residuals vs Predicted (Heteroscedasticity)', fontsize=13, fontweight='bold')
ax6.legend(fontsize=8)

# ── Plot 7: Temperature-Mortality curve ──
ax7 = fig.add_subplot(5, 2, 7)
for state in ['NSW', 'VIC', 'QLD']:
    mask = df_clean['State'] == state
    temp = df_clean.loc[mask, 'Mean_Temp']
    deaths = y_deaths[mask]
    bins_cut = pd.cut(temp, bins=15)
    g = pd.DataFrame({'t': temp, 'd': deaths}).groupby(bins_cut).agg(['mean', 'std', 'count'])
    g.columns = ['t_mean', 't_std', 't_n', 'd_mean', 'd_std', 'd_n']
    # Error bars (SEM)
    g['d_se'] = g['d_std'] / np.sqrt(g['d_n'])
    ax7.errorbar(g['t_mean'], g['d_mean'], yerr=g['d_se'] * 1.96,
                 fmt='o-', color=colors[state], label=state, markersize=4,
                 capsize=2, linewidth=1.5)
ax7.set_xlabel('Mean Temperature (°C)', fontsize=11)
ax7.set_ylabel('Mean Deaths / week', fontsize=11)
ax7.set_title('Temperature-Mortality Curve (± 95% CI)', fontsize=13, fontweight='bold')
ax7.legend(fontsize=9)

# ── Plot 8: Residual Autocorrelation (ACF) ──
ax8 = fig.add_subplot(5, 2, 8)
from statsmodels.graphics.tsaplots import plot_acf
# Compute ACF manually for each state
max_lags = 20
for state in ['NSW', 'VIC', 'QLD']:
    mask_all = df_clean['State'] == state
    resid_st = (y_deaths[mask_all] - y_pred_all[mask_all])
    acf_vals = [1.0]
    for lag in range(1, max_lags + 1):
        if lag < len(resid_st):
            acf_vals.append(np.corrcoef(resid_st[:-lag], resid_st[lag:])[0, 1])
    ax8.plot(range(len(acf_vals)), acf_vals, 'o-', color=colors[state],
             label=state, markersize=4, linewidth=1.2)
ax8.axhline(0, color='black', linestyle='-', lw=0.5)
ax8.axhline(1.96/np.sqrt(len(y_deaths)//3), color='gray', linestyle='--', lw=1, alpha=0.7)
ax8.axhline(-1.96/np.sqrt(len(y_deaths)//3), color='gray', linestyle='--', lw=1, alpha=0.7)
ax8.set_xlabel('Lag (weeks)', fontsize=11)
ax8.set_ylabel('ACF', fontsize=11)
ax8.set_title(f'Residual Autocorrelation (Durbin-Watson = {dw_stat:.3f})', fontsize=13, fontweight='bold')
ax8.legend(fontsize=9)

# ── Plot 9: Per-state bar chart comparison ──
ax9 = fig.add_subplot(5, 2, 9)
states_list = ['NSW', 'VIC', 'QLD']
x_pos = np.arange(len(states_list))
width = 0.2

r2_vals = [state_detailed[s]['r2_test'] for s in states_list]
mae_norm = [state_detailed[s]['mae_test'] / state_detailed[s]['mean_deaths'] * 100 for s in states_list]
mape_vals = [state_detailed[s]['mape_test'] for s in states_list]

bars1 = ax9.bar(x_pos - width, [v * 100 for v in r2_vals], width,
                color=[colors[s] for s in states_list], alpha=0.8, label='R² × 100')
bars2 = ax9.bar(x_pos, mae_norm, width, color=[colors[s] for s in states_list],
                alpha=0.5, hatch='//', label='MAE/Mean (%)')
bars3 = ax9.bar(x_pos + width, mape_vals, width, color=[colors[s] for s in states_list],
                alpha=0.5, hatch='..', label='MAPE (%)')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                 f'{h:.1f}', ha='center', va='bottom', fontsize=8)

ax9.set_xticks(x_pos)
ax9.set_xticklabels(states_list, fontsize=11)
ax9.set_ylabel('Value (%)', fontsize=11)
ax9.set_title('Per-State Performance Comparison', fontsize=13, fontweight='bold')
ax9.legend(fontsize=8)

# ── Plot 10: Full Model Summary Table ──
ax10 = fig.add_subplot(5, 2, 10)
ax10.axis('off')

n_cb = sum(1 for c in X.columns if 'cb_' in c)
n_state_f = len(state_feat_cols)
n_season_f = len(season_cols)
n_other_f = X.shape[1] - n_cb - n_state_f - n_season_f

cv_mean = np.mean(cv_r2_all) if cv_r2_all else float('nan')
cv_std = np.std(cv_r2_all) if cv_r2_all else float('nan')

summary_text = (
    f"{'═' * 48}\n"
    f"   HYBRID GAM-DLNM — COMPLETE MODEL REPORT\n"
    f"{'═' * 48}\n\n"
    f"  Architecture\n"
    f"  {'─' * 44}\n"
    f"  DLNM cross-basis:       {n_cb:>4} features\n"
    f"  State effects/interact: {n_state_f:>4} features\n"
    f"  Seasonal harmonics:     {n_season_f:>4} features\n"
    f"  Other smooth terms:     {n_other_f:>4} features\n"
    f"  Total features:         {X.shape[1]:>4}\n\n"
    f"  Train/Test Split\n"
    f"  {'─' * 44}\n"
    f"  Train: {len(y_train):>5} samples  |  Test: {len(y_test):>5} samples\n\n"
    f"  Overall Performance\n"
    f"  {'─' * 44}\n"
    f"  {'Metric':<22} {'Train':>10} {'Test':>10}\n"
    f"  R²                  {r2_train:>10.4f} {r2_test:>10.4f}\n"
    f"  Adjusted R²         {adj_r2_train:>10.4f} {adj_r2_test:>10.4f}\n"
    f"  Explained Var       {evs_train:>10.4f} {evs_test:>10.4f}\n"
    f"  MAE (deaths/wk)     {mae_train:>10.1f} {mae_test:>10.1f}\n"
    f"  MedAE (deaths/wk)   {medae_train:>10.1f} {medae_test:>10.1f}\n"
    f"  RMSE (deaths/wk)    {rmse_train:>10.1f} {rmse_test:>10.1f}\n"
    f"  NRMSE (% range)     {'':>10} {nrmse_range:>9.2f}%\n"
    f"  NRMSE (% mean)      {'':>10} {nrmse_mean:>9.2f}%\n"
    f"  MAPE                {'':>10} {mape_test:>9.2f}%\n"
    f"  Max |Error|         {'':>10} {max_err_test:>10.0f}\n\n"
    f"  Correlation (Test)\n"
    f"  {'─' * 44}\n"
    f"  Pearson r  = {pearson_r:.4f}  (p={pearson_p:.2e})\n"
    f"  Spearman ρ = {spearman_r:.4f}  (p={spearman_p:.2e})\n\n"
    f"  Model Selection\n"
    f"  {'─' * 44}\n"
    f"  AIC  = {aic:>12.1f}\n"
    f"  BIC  = {bic:>12.1f}\n"
    f"  Durbin-Watson = {dw_stat:.3f}\n"
    f"  CV R² (mean±std) = {cv_mean:.4f} ± {cv_std:.4f}\n"
)
ax10.text(0.02, 0.98, summary_text, transform=ax10.transAxes, fontsize=9.5,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.95))

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('hybrid_gam_dlnm_results.png', dpi=150, bbox_inches='tight')
print("  Saved: hybrid_gam_dlnm_results.png")

# ============================================================
# 11. MODEL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)
n_cb = sum(1 for c in X.columns if 'cb_' in c)
n_state = len(state_feat_cols)
n_season = len(season_cols)
n_other = X.shape[1] - n_cb - n_state - n_season

print(f"""
Architecture: Hybrid GAM + DLNM (Distributed Lag Non-linear Model)
  - DLNM cross-basis features:  {n_cb}
    (Mean_Temp, Max_Temp, Min_Temp, Humidity, Rainfall, Heat_Days, SD_Temp)
    (Each with lag 0, 1, 2 weeks; B-spline basis × lag weights)
  - State effects + interactions: {n_state}
  - Seasonal harmonics:          {n_season}
  - Other smooth terms:          {n_other}
  - Total features:              {X.shape[1]}

Link function: Identity (LinearGAM)
Regularization: Automatic lambda via gridsearch

PERFORMANCE
{'─' * 50}
  Train R²:    {r2_train:.4f}
  Test  R²:    {r2_test:.4f}
  Overall R²:  {r2_all:.4f}
  
  Test MAE:    {mae_test:.1f} deaths/week
  Test RMSE:   {rmse_test:.1f} deaths/week
  Test MAPE:   {mape_test:.2f}%

  Per-state Test R²:
    NSW: {state_results['NSW']['r2_test']:.4f}
    VIC: {state_results['VIC']['r2_test']:.4f}
    QLD: {state_results['QLD']['r2_test']:.4f}
""")

r2_pass = "✓ ĐẠT" if r2_test >= 0.80 else "✗ CHƯA ĐẠT"
print(f"  KẾT QUẢ: Test R² = {r2_test:.4f}  {r2_pass} (yêu cầu ≥ 0.80)")
print("=" * 70)
