"""
Hybrid GAM-DLNM Model — Per-State Training (v3)
=================================================
Train a SEPARATE model for each state (NSW, VIC, QLD)
to ensure R² > 0.80 for every state individually.

Architecture:
  SplineTransformer (non-linear basis) + RidgeCV (optimal regularization)
  This is equivalent to a penalized GAM but with superior regularization.

Components:
  - Linear trend + quadratic (extrapolates to test)
  - Fourier harmonics (annual + semi-annual + quarterly seasonality)
  - B-spline basis for weather (non-linear exposure-response)
  - DLNM: weighted-lag B-spline basis (distributed lag effects)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
print("=" * 70)
print("HYBRID GAM-DLNM v3 — PER-STATE TRAINING")
print("=" * 70)

df = pd.read_csv('weather_mortality_processed.csv')
df['Week_Start_Date'] = pd.to_datetime(df['Week_Start_Date'])

lag_cols = [c for c in df.columns if 'Lag' in c and '_Scaled' not in c]
df = df.dropna(subset=lag_cols).copy().reset_index(drop=True)

print(f"Dataset: {df.shape[0]} records × {df.shape[1]} columns")
print(f"States:  {list(df['State'].unique())}")

# ============================================================
# 2. BUILD FEATURES FOR ONE STATE
# ============================================================

def build_features(sdf):
    """
    Build feature matrix with B-spline basis for a single state.
    Uses SplineTransformer for non-linear weather features (≈ GAM smooth terms)
    and Fourier terms for seasonality (≈ GAM linear terms).
    """
    sdf = sdf.copy().reset_index(drop=True)
    n = len(sdf)
    week = sdf['Week'].values

    # ── 1) Linear trend (extrapolates!) ──
    t_norm = np.arange(n, dtype=float) / n
    linear_feats = np.column_stack([t_norm, t_norm ** 2])

    # ── 2) Fourier seasonality (6 terms) ──
    fourier = np.column_stack([
        np.sin(2 * np.pi * week / 52),
        np.cos(2 * np.pi * week / 52),
        np.sin(4 * np.pi * week / 52),
        np.cos(4 * np.pi * week / 52),
        np.sin(6 * np.pi * week / 52),
        np.cos(6 * np.pi * week / 52),
    ])

    # ── 3) DLNM: weighted-lag averages for each weather variable ──
    def wlag(v0, v1, v2):
        return (v0 * 1.0 + v1 * 0.5 + v2 * 0.25) / 1.75

    temp_wlag = wlag(sdf['Mean_Temp'].values, sdf['Mean_Temp_Lag1'].values,
                     sdf['Mean_Temp_Lag2'].values)
    maxtemp_wlag = wlag(sdf['Max_Temp'].values, sdf['Max_Temp_Lag1'].values,
                        sdf['Max_Temp_Lag2'].values)
    mintemp_wlag = wlag(sdf['Min_Temp'].values, sdf['Min_Temp_Lag1'].values,
                        sdf['Min_Temp_Lag2'].values)
    humid_wlag = wlag(sdf['Mean_Humidity_Max'].values,
                      sdf['Mean_Humidity_Max_Lag1'].values,
                      sdf['Mean_Humidity_Max_Lag2'].values)
    rain_wlag = wlag(sdf['Total_Rainfall'].values,
                     sdf['Total_Rainfall_Lag1'].values,
                     sdf['Total_Rainfall_Lag2'].values)
    heat_wlag = wlag(sdf['Heat_Days_Count'].values,
                     sdf['Heat_Days_Count_Lag1'].values,
                     sdf['Heat_Days_Count_Lag2'].values)
    sd_wlag = wlag(sdf['SD_Temp'].values,
                   sdf['SD_Temp_Lag1'].values,
                   sdf['SD_Temp_Lag2'].values)

    # ── 4) B-spline basis for weather (= GAM smooth terms) ──
    weather_vars = {
        'temp_wlag': temp_wlag,
        'maxtemp_wlag': maxtemp_wlag,
        'mintemp_wlag': mintemp_wlag,
        'humid_wlag': humid_wlag,
        'rain_wlag': rain_wlag,
        'heat_wlag': heat_wlag,
        'sd_wlag': sd_wlag,
        'mean_temp': sdf['Mean_Temp'].values,
        'min_temp': sdf['Min_Temp'].values,
        'humidity': sdf['Mean_Humidity_Max'].values,
        'solar': sdf['Mean_Solar_Radiation'].values,
        'temp_range': sdf['Max_Temp'].values - sdf['Min_Temp'].values,
    }

    spline_blocks = []
    spline_transformers = {}
    for vname, vals in weather_vars.items():
        n_knots = 6 if 'temp' in vname else 5
        sp = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False,
                               extrapolation='linear')
        basis = sp.fit_transform(vals.reshape(-1, 1))
        spline_blocks.append(basis)
        spline_transformers[vname] = sp

    # ── 5) Interactions: temp × season ──
    temp_sin = sdf['Mean_Temp'].values * np.sin(2 * np.pi * week / 52)
    temp_cos = sdf['Mean_Temp'].values * np.cos(2 * np.pi * week / 52)
    interactions = np.column_stack([temp_sin, temp_cos])

    # ── Assemble ──
    X = np.hstack([linear_feats, fourier, interactions] + spline_blocks)

    y = sdf['Deaths'].values
    dates = sdf['Week_Start_Date'].values

    return X, y, dates, spline_transformers

# ============================================================
# 3. TRAIN ONE STATE
# ============================================================

def train_state(state_name, sdf):
    """Train, tune, and evaluate for one state using Ridge + B-splines."""
    print(f"\n{'─' * 60}")
    print(f"  STATE: {state_name}  ({len(sdf)} weeks)")
    print(f"{'─' * 60}")

    X, y, dates, sp_trans = build_features(sdf)

    # Time-series split 80/20
    n = len(y)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  Features: {X.shape[1]}  |  Train: {split}  |  Test: {n - split}")

    # Scale features for Ridge
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_all_s = scaler.transform(X)

    # RidgeCV with wide alpha range — GCV scoring
    alphas = np.logspace(-2, 8, 200)
    model = RidgeCV(alphas=alphas, cv=None,  # GCV (Leave-One-Out)
                    scoring=None, store_cv_results=True)
    model.fit(X_train_s, y_train)

    print(f"  Best alpha: {model.alpha_:.2f}")

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)
    y_pred_all = model.predict(X_all_s)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    r2_all = r2_score(y, y_pred_all)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │ {state_name} RESULTS                          │")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │ Train R²  = {r2_train:.4f}                  │")
    print(f"  │ Test  R²  = {r2_test:.4f}  {'✓' if r2_test >= 0.8 else '✗'}               │")
    print(f"  │ Overall R²= {r2_all:.4f}                  │")
    print(f"  │ MAE       = {mae_test:.1f} deaths/wk          │")
    print(f"  │ RMSE      = {rmse_test:.1f} deaths/wk          │")
    print(f"  │ MAPE      = {mape_test:.2f}%                  │")
    print(f"  └─────────────────────────────────────┘")

    return {
        'state': state_name, 'model': model, 'scaler': scaler,
        'X': X, 'y': y, 'dates': dates, 'split': split,
        'y_pred_all': y_pred_all, 'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'y_train': y_train, 'y_test': y_test,
        'r2_train': r2_train, 'r2_test': r2_test, 'r2_all': r2_all,
        'mae_test': mae_test, 'rmse_test': rmse_test, 'mape_test': mape_test,
    }

# ============================================================
# 4. RUN PER-STATE TRAINING
# ============================================================
results = {}
for state in ['NSW', 'VIC', 'QLD']:
    sdf = df[df['State'] == state].copy()
    results[state] = train_state(state, sdf)

# ============================================================
# 5. OVERALL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("OVERALL SUMMARY — PER-STATE MODELS")
print("=" * 70)
print(f"\n{'State':<8} {'R²(train)':>10} {'R²(test)':>10} {'R²(all)':>10}"
      f" {'MAE':>8} {'RMSE':>8} {'MAPE%':>8}")
print("─" * 65)
all_pass = True
for state in ['NSW', 'VIC', 'QLD']:
    r = results[state]
    flag = "✓" if r['r2_test'] >= 0.80 else "✗"
    if r['r2_test'] < 0.80:
        all_pass = False
    print(f"{state:<8} {r['r2_train']:>10.4f} {r['r2_test']:>10.4f} {r['r2_all']:>10.4f}"
          f" {r['mae_test']:>8.1f} {r['rmse_test']:>8.1f} {r['mape_test']:>7.2f}%  {flag}")

print(f"\nTarget R² ≥ 0.80 for ALL states: {'✓ PASS' if all_pass else '✗ FAIL'}")

# ============================================================
# 6. TIME-SERIES CROSS-VALIDATION (per-state, expanding window)
# ============================================================
print(f"\n--- Time-Series Cross-Validation (4-fold, expanding window) ---")
for state in ['NSW', 'VIC', 'QLD']:
    r = results[state]
    X_st, y_st = r['X'], r['y']
    n_st = len(y_st)
    fold_size = n_st // 5
    r2_folds = []
    for fold in range(4):
        tr_end = fold_size * (fold + 2)
        te_end = min(tr_end + fold_size, n_st)
        if te_end <= tr_end:
            continue
        try:
            sc_cv = StandardScaler()
            X_tr_cv = sc_cv.fit_transform(X_st[:tr_end])
            X_te_cv = sc_cv.transform(X_st[tr_end:te_end])
            m_cv = RidgeCV(alphas=np.logspace(-2, 8, 100), cv=None)
            m_cv.fit(X_tr_cv, y_st[:tr_end])
            p_cv = m_cv.predict(X_te_cv)
            r2_folds.append(r2_score(y_st[tr_end:te_end], p_cv))
        except Exception:
            pass
    if r2_folds:
        print(f"  {state}: CV R² = {np.mean(r2_folds):.4f} ± {np.std(r2_folds):.4f}  "
              f"(folds: {[f'{v:.4f}' for v in r2_folds]})")

# ============================================================
# 7. VISUALIZATION
# ============================================================
print("\n--- Generating Per-State Plots ---")

from scipy.stats import pearsonr, spearmanr

colors = {'NSW': '#e74c3c', 'VIC': '#3498db', 'QLD': '#2ecc71'}
fig = plt.figure(figsize=(24, 42))
fig.suptitle('Hybrid GAM-DLNM v3 — Per-State Models (Australia 2015–2024)',
             fontsize=20, fontweight='bold', y=0.998)

# ── Row 1: Time-series per state ──
for idx, state in enumerate(['NSW', 'VIC', 'QLD']):
    r = results[state]
    ax_ts = fig.add_subplot(8, 3, idx + 1)
    dates_all = pd.to_datetime(r['dates'])
    ax_ts.plot(dates_all, r['y'], color=colors[state], alpha=0.35, lw=0.8, label='Actual')
    ax_ts.plot(dates_all, r['y_pred_all'], color=colors[state], lw=1.5, label='Predicted')
    ax_ts.fill_between(dates_all, r['y'], r['y_pred_all'], alpha=0.08, color=colors[state])
    split_date = dates_all[r['split']]
    ax_ts.axvline(split_date, color='gray', ls='--', alpha=0.7, label='Train|Test')
    pop = df[df['State'] == state]['Population'].iloc[0]
    ax_ts.set_title(f'{state} — Time Series (Pop: {pop:,.0f})', fontsize=13, fontweight='bold')
    ax_ts.set_ylabel('Deaths/week')
    ax_ts.legend(fontsize=7, loc='upper left')
    pr_all, _ = pearsonr(r['y'], r['y_pred_all'])
    info = (f"Train R²={r['r2_train']:.4f}\n"
            f"Test  R²={r['r2_test']:.4f}\n"
            f"MAE={r['mae_test']:.1f}\n"
            f"RMSE={r['rmse_test']:.1f}\n"
            f"MAPE={r['mape_test']:.1f}%\n"
            f"r={pr_all:.4f}")
    ax_ts.text(0.98, 0.02, info, transform=ax_ts.transAxes, fontsize=8,
               va='bottom', ha='right', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

# ── Row 2: Actual vs Predicted scatter ──
for idx, state in enumerate(['NSW', 'VIC', 'QLD']):
    r = results[state]
    ax_sc = fig.add_subplot(8, 3, idx + 4)
    # Train points (lighter)
    ax_sc.scatter(r['y_train'], r['y_pred_train'], s=18, alpha=0.3,
                  color=colors[state], edgecolors='none', label='Train')
    # Test points (darker)
    ax_sc.scatter(r['y_test'], r['y_pred_test'], s=30, alpha=0.7,
                  color=colors[state], edgecolors='k', linewidths=0.3, label='Test')
    mn, mx = r['y'].min() * 0.95, r['y'].max() * 1.05
    ax_sc.plot([mn, mx], [mn, mx], 'k--', lw=1, alpha=0.5, label='1:1 line')
    slope, intercept = np.polyfit(r['y_test'], r['y_pred_test'], 1)
    ax_sc.plot([mn, mx], [slope*mn+intercept, slope*mx+intercept],
               'r-', lw=1.5, alpha=0.7, label=f'Fit: y={slope:.2f}x+{intercept:.0f}')
    ax_sc.set_xlabel('Actual Deaths', fontsize=10)
    ax_sc.set_ylabel('Predicted Deaths', fontsize=10)
    ax_sc.set_title(f'{state} — Actual vs Predicted', fontsize=12, fontweight='bold')
    ax_sc.legend(fontsize=7, loc='upper left')
    pr, _ = pearsonr(r['y_test'], r['y_pred_test'])
    sr, _ = spearmanr(r['y_test'], r['y_pred_test'])
    ax_sc.text(0.97, 0.03, f"R²={r['r2_test']:.4f}\nPearson r={pr:.4f}\nSpearman ρ={sr:.4f}",
               transform=ax_sc.transAxes, fontsize=9, va='bottom', ha='right',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ── Row 3: Residual distribution ──
for idx, state in enumerate(['NSW', 'VIC', 'QLD']):
    r = results[state]
    ax_rd = fig.add_subplot(8, 3, idx + 7)
    resid = r['y_test'] - r['y_pred_test']
    mu_r, sig_r = np.mean(resid), np.std(resid)
    ax_rd.hist(resid, bins=25, color=colors[state], edgecolor='white', alpha=0.8, density=True)
    xr = np.linspace(resid.min(), resid.max(), 100)
    ax_rd.plot(xr, stats.norm.pdf(xr, mu_r, sig_r), 'r-', lw=2, label='Normal fit')
    ax_rd.axvline(0, color='k', ls='--', lw=1, alpha=0.5)
    ax_rd.axvline(mu_r, color='red', ls=':', lw=1.5, label=f'Mean={mu_r:.1f}')
    ax_rd.set_xlabel('Residual (Actual − Predicted)', fontsize=10)
    ax_rd.set_ylabel('Density', fontsize=10)
    ax_rd.set_title(f'{state} — Residual Distribution (Test)', fontsize=12, fontweight='bold')
    ax_rd.legend(fontsize=7)
    skew_r = stats.skew(resid)
    kurt_r = stats.kurtosis(resid)
    medae = np.median(np.abs(resid))
    max_err = np.max(np.abs(resid))
    ax_rd.text(0.97, 0.97,
               f"Mean={mu_r:.1f}\nStd={sig_r:.1f}\nSkew={skew_r:.3f}\nKurtosis={kurt_r:.3f}\n"
               f"Median AE={medae:.1f}\nMax |err|={max_err:.0f}",
               transform=ax_rd.transAxes, fontsize=8, va='top', ha='right',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ── Row 4: Residuals vs Predicted (heteroscedasticity check) ──
for idx, state in enumerate(['NSW', 'VIC', 'QLD']):
    r = results[state]
    ax_hc = fig.add_subplot(8, 3, idx + 10)
    resid_test = r['y_test'] - r['y_pred_test']
    sig_r = np.std(resid_test)
    ax_hc.scatter(r['y_pred_test'], resid_test, s=25, alpha=0.6,
                  color=colors[state], edgecolors='none')
    ax_hc.axhline(0, color='k', ls='--', lw=1)
    ax_hc.axhline(2*sig_r, color='red', ls=':', lw=1, alpha=0.6, label=f'+2σ ({2*sig_r:.0f})')
    ax_hc.axhline(-2*sig_r, color='red', ls=':', lw=1, alpha=0.6, label=f'−2σ ({-2*sig_r:.0f})')
    # LOWESS-like trend via moving average
    sort_idx = np.argsort(r['y_pred_test'])
    pred_sorted = r['y_pred_test'][sort_idx]
    resid_sorted = resid_test[sort_idx]
    win = max(len(resid_sorted) // 10, 3)
    resid_smooth = pd.Series(resid_sorted).rolling(win, center=True, min_periods=1).mean().values
    ax_hc.plot(pred_sorted, resid_smooth, 'r-', lw=2, alpha=0.7, label='Trend')
    ax_hc.set_xlabel('Predicted Deaths', fontsize=10)
    ax_hc.set_ylabel('Residual', fontsize=10)
    ax_hc.set_title(f'{state} — Residuals vs Predicted', fontsize=12, fontweight='bold')
    ax_hc.legend(fontsize=7)

# ── Row 5: Temperature-Mortality curve per state ──
for idx, state in enumerate(['NSW', 'VIC', 'QLD']):
    ax_tm = fig.add_subplot(8, 3, idx + 13)
    r = results[state]
    sdf = df[df['State'] == state]
    temp = sdf['Mean_Temp'].values
    deaths = sdf['Deaths'].values
    bins_cut = pd.cut(temp, bins=15)
    g = pd.DataFrame({'t': temp, 'd': deaths}).groupby(bins_cut).agg(['mean', 'std', 'count'])
    g.columns = ['t_mean', 't_std', 't_n', 'd_mean', 'd_std', 'd_n']
    g['d_se'] = g['d_std'] / np.sqrt(g['d_n'])
    ax_tm.errorbar(g['t_mean'], g['d_mean'], yerr=g['d_se'] * 1.96,
                   fmt='o-', color=colors[state], markersize=5, capsize=3, lw=1.5)
    ax_tm.set_xlabel('Mean Temperature (°C)', fontsize=10)
    ax_tm.set_ylabel('Deaths/week', fontsize=10)
    ax_tm.set_title(f'{state} — Temp-Mortality (±95% CI)', fontsize=12, fontweight='bold')

# ── Row 6: Residual ACF per state ──
for idx, state in enumerate(['NSW', 'VIC', 'QLD']):
    ax_acf = fig.add_subplot(8, 3, idx + 16)
    r = results[state]
    resid_all = r['y'] - r['y_pred_all']
    max_lags = 20
    acf_vals = [1.0]
    for lag in range(1, max_lags + 1):
        if lag < len(resid_all):
            acf_vals.append(np.corrcoef(resid_all[:-lag], resid_all[lag:])[0, 1])
    ax_acf.bar(range(len(acf_vals)), acf_vals, color=colors[state], alpha=0.7)
    ci = 1.96 / np.sqrt(len(resid_all))
    ax_acf.axhline(ci, color='gray', ls='--', lw=1, alpha=0.7, label=f'95% CI (±{ci:.3f})')
    ax_acf.axhline(-ci, color='gray', ls='--', lw=1, alpha=0.7)
    ax_acf.axhline(0, color='k', lw=0.5)
    # Durbin-Watson
    dw = np.sum(np.diff(resid_all)**2) / np.sum(resid_all**2)
    ax_acf.set_xlabel('Lag (weeks)', fontsize=10)
    ax_acf.set_ylabel('ACF', fontsize=10)
    ax_acf.set_title(f'{state} — Residual ACF (DW={dw:.3f})', fontsize=12, fontweight='bold')
    ax_acf.legend(fontsize=7)

# ── Row 7: Per-state performance comparison bar chart ──
ax_bar = fig.add_subplot(8, 3, (19, 20))
states_list = ['NSW', 'VIC', 'QLD']
x_pos = np.arange(len(states_list))
width = 0.18

r2_vals = [results[s]['r2_test'] for s in states_list]
mean_deaths = {s: np.mean(results[s]['y']) for s in states_list}
mae_norm = [results[s]['mae_test'] / mean_deaths[s] * 100 for s in states_list]
mape_vals = [results[s]['mape_test'] for s in states_list]
r2_train_vals = [results[s]['r2_train'] for s in states_list]

bars1 = ax_bar.bar(x_pos - 1.5*width, [v*100 for v in r2_train_vals], width,
                   color=[colors[s] for s in states_list], alpha=0.5, label='R²(train) ×100')
bars2 = ax_bar.bar(x_pos - 0.5*width, [v*100 for v in r2_vals], width,
                   color=[colors[s] for s in states_list], alpha=0.9, label='R²(test) ×100')
bars3 = ax_bar.bar(x_pos + 0.5*width, mae_norm, width,
                   color=[colors[s] for s in states_list], alpha=0.5, hatch='//', label='MAE/Mean (%)')
bars4 = ax_bar.bar(x_pos + 1.5*width, mape_vals, width,
                   color=[colors[s] for s in states_list], alpha=0.5, hatch='..', label='MAPE (%)')

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8)

ax_bar.axhline(80, color='red', ls='--', lw=1.5, alpha=0.6, label='R²=0.80 target')
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(states_list, fontsize=12, fontweight='bold')
ax_bar.set_ylabel('Value (%)', fontsize=11)
ax_bar.set_title('Per-State Performance Comparison', fontsize=13, fontweight='bold')
ax_bar.legend(fontsize=8, loc='upper right')

# ── Row 7 (right): Model Summary Table ──
ax_tbl = fig.add_subplot(8, 3, 21)
ax_tbl.axis('off')

summary_text = (
    f"{'═' * 48}\n"
    f"  HYBRID GAM-DLNM v3 — MODEL REPORT\n"
    f"  Per-State Independent Training\n"
    f"{'═' * 48}\n\n"
    f"  Architecture\n"
    f"  {'─' * 44}\n"
    f"  SplineTransformer + RidgeCV (penalized GAM)\n"
    f"  Fourier (annual + semi-annual + quarterly)\n"
    f"  DLNM weighted-lag B-spline basis\n"
    f"  Features: {results['NSW']['X'].shape[1]} per state\n\n"
    f"  {'State':<6} {'R²(tr)':>8} {'R²(te)':>8} {'MAE':>7} {'RMSE':>7} {'MAPE%':>7}\n"
    f"  {'─' * 44}\n"
)
for s in states_list:
    r = results[s]
    flag = "✓" if r['r2_test'] >= 0.80 else "✗"
    summary_text += (f"  {s:<6} {r['r2_train']:>8.4f} {r['r2_test']:>8.4f} "
                     f"{r['mae_test']:>7.1f} {r['rmse_test']:>7.1f} {r['mape_test']:>6.2f}% {flag}\n")
all_pass_flag = all(results[s]['r2_test'] >= 0.80 for s in states_list)
summary_text += (
    f"\n  Target R²≥0.80: {'✓ PASS' if all_pass_flag else '✗ FAIL'}\n"
    f"{'═' * 48}"
)
ax_tbl.text(0.02, 0.98, summary_text, transform=ax_tbl.transAxes, fontsize=9.5,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.95))

# ── Row 8: Q-Q plots per state ──
for idx, state in enumerate(['NSW', 'VIC', 'QLD']):
    r = results[state]
    ax_qq = fig.add_subplot(8, 3, idx + 22)
    resid_test = r['y_test'] - r['y_pred_test']
    sorted_resid = np.sort(resid_test)
    n_pts = len(sorted_resid)
    theoretical_q = stats.norm.ppf((np.arange(1, n_pts + 1) - 0.5) / n_pts)
    ax_qq.scatter(theoretical_q, sorted_resid, s=20, alpha=0.7,
                  color=colors[state], edgecolors='none')
    # Reference line through Q1-Q3
    q1_t, q3_t = stats.norm.ppf(0.25), stats.norm.ppf(0.75)
    q1_r, q3_r = np.percentile(sorted_resid, 25), np.percentile(sorted_resid, 75)
    slope_qq = (q3_r - q1_r) / (q3_t - q1_t)
    intercept_qq = q1_r - slope_qq * q1_t
    x_line = np.array([theoretical_q.min(), theoretical_q.max()])
    ax_qq.plot(x_line, slope_qq * x_line + intercept_qq, 'r-', lw=1.5, alpha=0.8)
    ax_qq.set_xlabel('Theoretical Quantiles', fontsize=10)
    ax_qq.set_ylabel('Sample Quantiles', fontsize=10)
    ax_qq.set_title(f'{state} — Q-Q Plot (Test Residuals)', fontsize=12, fontweight='bold')
    ax_qq.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.995])
plt.savefig('hybrid_gam_dlnm_v3_perstate_results.png', dpi=150, bbox_inches='tight')
print("  Saved: hybrid_gam_dlnm_v3_perstate_results.png")

# ============================================================
# 8. FINAL VERDICT
# ============================================================
print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)
for state in ['NSW', 'VIC', 'QLD']:
    r = results[state]
    flag = "✓ ĐẠT" if r['r2_test'] >= 0.80 else "✗ CHƯA ĐẠT"
    print(f"  {state}: Test R² = {r['r2_test']:.4f}  {flag}")
all_pass = all(results[s]['r2_test'] >= 0.80 for s in ['NSW', 'VIC', 'QLD'])
print(f"\n  Tất cả bang R² ≥ 0.80: {'✓ ĐẠT' if all_pass else '✗ CHƯA ĐẠT'}")
print("=" * 70)
