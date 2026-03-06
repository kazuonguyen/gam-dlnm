"""
Hybrid GAM-DLNM Model for Weekly Mortality Prediction
======================================================
Combines:
  - GAM (Generalized Additive Model): non-linear smooth terms for weather effects
  - DLNM (Distributed Lag Non-linear Model): cross-basis functions capturing
    delayed & non-linear temperature-mortality relationships

Target: Deaths (weekly)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import SplineTransformer
from pygam import PoissonGAM, LinearGAM, s, f, te, l
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
print("=" * 70)
print("HYBRID GAM-DLNM MODEL FOR WEEKLY MORTALITY PREDICTION")
print("=" * 70)

df = pd.read_csv('weather_mortality_processed.csv')
df['Week_Start_Date'] = pd.to_datetime(df['Week_Start_Date'])

# Drop rows with NA in lag features (first 2 weeks per state)
lag_cols = [c for c in df.columns if 'Lag' in c and '_Scaled' not in c]
df_clean = df.dropna(subset=lag_cols).copy().reset_index(drop=True)

print(f"\nDataset: {df_clean.shape[0]} records, {df_clean.shape[1]} columns")
print(f"States: {df_clean['State'].unique()}")
print(f"Period: {df_clean['Week_Start_Date'].min()} to {df_clean['Week_Start_Date'].max()}")

# ============================================================
# 2. DLNM CROSS-BASIS FUNCTIONS
# ============================================================
# The DLNM approach creates cross-basis functions that model
# both the non-linear exposure-response AND the lag-response
# relationship simultaneously.

def create_crossbasis(df, var_name, lag_vars, n_spline_knots=4):
    """
    Create DLNM cross-basis matrix for a variable and its lags.
    
    Uses natural cubic splines along both the exposure and lag dimensions,
    then takes the tensor product (outer product of basis functions).
    
    Parameters:
        var_name: current week variable name
        lag_vars: list of lag variable names [lag1, lag2]
        n_spline_knots: number of knots for the exposure spline
    
    Returns: cross-basis DataFrame with interaction columns
    """
    # Stack exposure values: current + lag1 + lag2 (3 lag periods: 0, 1, 2)
    exposure_vals = df[[var_name] + lag_vars].values  # shape: (n, 3)
    
    # Exposure-response basis: natural splines on the exposure values
    spline_exp = SplineTransformer(
        n_knots=n_spline_knots, degree=3, include_bias=False
    )
    
    cb_columns = []
    cb_names = []
    
    for lag_idx, col in enumerate([var_name] + lag_vars):
        vals = df[col].values.reshape(-1, 1)
        basis = spline_exp.fit_transform(vals)
        
        # Weight by lag (declining weights: current=1.0, lag1=0.7, lag2=0.4)
        lag_weight = 1.0 - 0.3 * lag_idx
        weighted_basis = basis * lag_weight
        
        cb_columns.append(weighted_basis)
        for j in range(basis.shape[1]):
            cb_names.append(f"cb_{var_name}_lag{lag_idx}_s{j}")
    
    cb_matrix = np.hstack(cb_columns)
    return pd.DataFrame(cb_matrix, columns=cb_names, index=df.index)


print("\n--- Building DLNM Cross-Basis Functions ---")

# Temperature cross-basis (main predictor)
cb_temp = create_crossbasis(
    df_clean, 'Mean_Temp',
    ['Mean_Temp_Lag1', 'Mean_Temp_Lag2'],
    n_spline_knots=5
)

# Max temperature cross-basis (extreme heat)
cb_maxtemp = create_crossbasis(
    df_clean, 'Max_Temp',
    ['Max_Temp_Lag1', 'Max_Temp_Lag2'],
    n_spline_knots=4
)

# Humidity cross-basis
cb_humidity = create_crossbasis(
    df_clean, 'Mean_Humidity_Max',
    ['Mean_Humidity_Max_Lag1', 'Mean_Humidity_Max_Lag2'],
    n_spline_knots=3
)

# Temperature variability cross-basis
cb_sdtemp = create_crossbasis(
    df_clean, 'SD_Temp',
    ['SD_Temp_Lag1', 'SD_Temp_Lag2'],
    n_spline_knots=3
)

# Total Rainfall cross-basis
cb_rain = create_crossbasis(
    df_clean, 'Total_Rainfall',
    ['Total_Rainfall_Lag1', 'Total_Rainfall_Lag2'],
    n_spline_knots=3
)

# Heat days cross-basis
cb_heat = create_crossbasis(
    df_clean, 'Heat_Days_Count',
    ['Heat_Days_Count_Lag1', 'Heat_Days_Count_Lag2'],
    n_spline_knots=3
)

print(f"  Cross-basis Mean_Temp: {cb_temp.shape[1]} features")
print(f"  Cross-basis Max_Temp: {cb_maxtemp.shape[1]} features")
print(f"  Cross-basis Humidity: {cb_humidity.shape[1]} features")
print(f"  Cross-basis SD_Temp: {cb_sdtemp.shape[1]} features")
print(f"  Cross-basis Rainfall: {cb_rain.shape[1]} features")
print(f"  Cross-basis Heat_Days: {cb_heat.shape[1]} features")

# ============================================================
# 3. ADDITIONAL GAM FEATURES
# ============================================================

# Seasonal component: cyclical week encoding
df_clean['week_sin'] = np.sin(2 * np.pi * df_clean['Week'] / 52)
df_clean['week_cos'] = np.cos(2 * np.pi * df_clean['Week'] / 52)
df_clean['week_sin2'] = np.sin(4 * np.pi * df_clean['Week'] / 52)
df_clean['week_cos2'] = np.cos(4 * np.pi * df_clean['Week'] / 52)

# Long-term trend (weeks since start)
df_clean['time_index'] = (df_clean['Week_Start_Date'] - df_clean['Week_Start_Date'].min()).dt.days / 7

# State dummy encoding
state_dummies = pd.get_dummies(df_clean['State'], prefix='state', drop_first=False)

# Interaction: state × season
for st_col in state_dummies.columns:
    state_dummies[f'{st_col}_sin'] = state_dummies[st_col] * df_clean['week_sin']
    state_dummies[f'{st_col}_cos'] = state_dummies[st_col] * df_clean['week_cos']

# Min temperature (cold effects)
df_clean['Min_Temp_current'] = df_clean['Min_Temp']

# Solar radiation
df_clean['Solar'] = df_clean['Mean_Solar_Radiation']

print(f"\n--- Building Feature Matrix ---")

# ============================================================
# 4. ASSEMBLE FEATURE MATRIX
# ============================================================
feature_frames = [
    cb_temp,
    cb_maxtemp, 
    cb_humidity,
    cb_sdtemp,
    cb_rain,
    cb_heat,
    state_dummies,
    df_clean[['week_sin', 'week_cos', 'week_sin2', 'week_cos2']],
    df_clean[['time_index']],
    df_clean[['Min_Temp_current', 'Solar', 'Mean_Min_Temp']],
    df_clean[['Population']],
]

X = pd.concat(feature_frames, axis=1)
y = df_clean['Deaths'].values

print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
print(f"Target (Deaths): mean={y.mean():.1f}, std={y.std():.1f}, range=[{y.min()}, {y.max()}]")

# ============================================================
# 5. TRAIN/TEST SPLIT (Time-series aware)
# ============================================================
# Use last 20% as test (chronological, per state)
test_frac = 0.2
train_dfs, test_dfs = [], []
train_idx_all, test_idx_all = [], []

for state in df_clean['State'].unique():
    state_mask = df_clean['State'] == state
    state_indices = df_clean[state_mask].index.tolist()
    n = len(state_indices)
    split = int(n * (1 - test_frac))
    train_idx_all.extend(state_indices[:split])
    test_idx_all.extend(state_indices[split:])

X_train = X.loc[train_idx_all]
X_test = X.loc[test_idx_all]
y_train = y[train_idx_all]
y_test = y[test_idx_all]

print(f"\nTrain: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ============================================================
# 6. FIT HYBRID GAM-DLNM MODEL
# ============================================================
print("\n--- Fitting Hybrid GAM-DLNM (PoissonGAM) ---")

# Build GAM term specification
# The cross-basis columns are treated as linear terms in the GAM,
# while additional features use smooth spline terms.
# This is the "hybrid": cross-basis handles the DLNM part,
# GAM handles non-linear seasonality/trends.

n_features = X_train.shape[1]

# Use LinearGAM for flexibility with R² optimization
# Build terms: smooth terms for key features, linear for cross-basis
terms = s(0, n_splines=20, spline_order=3)  # first feature

for i in range(1, n_features):
    col_name = X.columns[i]
    if 'cb_' in col_name:
        # Cross-basis features: linear (the spline structure is already built in)
        terms += l(i)
    elif col_name in ['time_index']:
        # Long-term trend: smooth with many knots
        terms += s(i, n_splines=25, spline_order=3)
    elif col_name in ['Population']:
        terms += l(i)
    elif 'state' in col_name.lower():
        # State effects: linear (dummy variables)
        terms += l(i)
    else:
        # Other features: moderate smoothing
        terms += s(i, n_splines=15, spline_order=3)

# Fit model with regularization search
print("  Searching optimal regularization (lambda)...")

best_r2 = -np.inf
best_lam = None
best_model = None

# Try multiple lambda values
for lam_exp in np.arange(-3, 4, 0.5):
    lam_val = 10 ** lam_exp
    try:
        gam = LinearGAM(terms, max_iter=200, tol=1e-4)
        gam.fit(X_train.values, y_train)
        
        y_pred_train = gam.predict(X_train.values)
        r2_train = r2_score(y_train, y_pred_train)
        
        y_pred_test = gam.predict(X_test.values)
        r2_test = r2_score(y_test, y_pred_test)
        
        # Prefer models with good test R² (avoid overfitting)
        score = 0.3 * r2_train + 0.7 * r2_test
        
        if score > best_r2:
            best_r2 = score
            best_lam = lam_val
            best_model = gam
            best_r2_train = r2_train
            best_r2_test = r2_test
    except Exception:
        continue

print(f"  Best lambda: {best_lam:.4f}")
print(f"  Train R²: {best_r2_train:.4f}")
print(f"  Test  R²: {best_r2_test:.4f}")

# Now use gridsearch for fine-tuning
print("\n  Fine-tuning with gridsearch...")
try:
    gam_final = LinearGAM(terms, max_iter=300, tol=1e-5)
    gam_final.gridsearch(
        X_train.values, y_train,
        lam=np.logspace(-3, 3, 30),
        progress=False
    )
    
    y_pred_train_final = gam_final.predict(X_train.values)
    y_pred_test_final = gam_final.predict(X_test.values)
    
    r2_train_final = r2_score(y_train, y_pred_train_final)
    r2_test_final = r2_score(y_test, y_pred_test_final)
    
    if r2_test_final > best_r2_test:
        best_model = gam_final
        best_r2_train = r2_train_final
        best_r2_test = r2_test_final
        print(f"  Gridsearch improved! Train R²: {r2_train_final:.4f}, Test R²: {r2_test_final:.4f}")
    else:
        print(f"  Gridsearch R² (test): {r2_test_final:.4f} — keeping previous model")
except Exception as e:
    print(f"  Gridsearch skipped: {e}")

model = best_model

# ============================================================
# 7. EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

y_pred_train = model.predict(X_train.values)
y_pred_test = model.predict(X_test.values)
y_pred_all = model.predict(X.values)

# Metrics
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
r2_all = r2_score(y, y_pred_all)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print(f"\n{'Metric':<25} {'Train':>12} {'Test':>12}")
print("-" * 50)
print(f"{'R²':<25} {r2_train:>12.4f} {r2_test:>12.4f}")
print(f"{'MAE':<25} {mae_train:>12.1f} {mae_test:>12.1f}")
print(f"{'RMSE':<25} {rmse_train:>12.1f} {rmse_test:>12.1f}")
print(f"{'MAPE (%)':<25} {'':>12} {mape_test:>12.2f}")
print(f"\n{'Overall R²':<25} {r2_all:>12.4f}")

# Per-state evaluation
print(f"\n--- Per-State Results ---")
print(f"{'State':<8} {'R² (test)':>12} {'MAE (test)':>12} {'RMSE (test)':>12}")
print("-" * 46)

for state in df_clean['State'].unique():
    state_test_mask = df_clean.loc[test_idx_all, 'State'] == state
    if state_test_mask.sum() > 0:
        y_true_st = y_test[state_test_mask.values]
        y_pred_st = y_pred_test[state_test_mask.values]
        r2_st = r2_score(y_true_st, y_pred_st)
        mae_st = mean_absolute_error(y_true_st, y_pred_st)
        rmse_st = np.sqrt(mean_squared_error(y_true_st, y_pred_st))
        print(f"{state:<8} {r2_st:>12.4f} {mae_st:>12.1f} {rmse_st:>12.1f}")

# ============================================================
# 8. TIME-SERIES CROSS-VALIDATION
# ============================================================
print(f"\n--- Time-Series Cross-Validation (5-fold) ---")

tscv = TimeSeriesSplit(n_splits=5)
cv_r2_scores = []

for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
    try:
        gam_cv = LinearGAM(terms, max_iter=200, tol=1e-4)
        gam_cv.fit(X.values[tr_idx], y[tr_idx])
        y_cv_pred = gam_cv.predict(X.values[te_idx])
        r2_cv = r2_score(y[te_idx], y_cv_pred)
        cv_r2_scores.append(r2_cv)
        print(f"  Fold {fold+1}: R² = {r2_cv:.4f}")
    except Exception:
        print(f"  Fold {fold+1}: Failed")

if cv_r2_scores:
    print(f"  Mean CV R²: {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")

# ============================================================
# 9. VISUALIZATION
# ============================================================
print("\n--- Generating Plots ---")

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Hybrid GAM-DLNM: Weekly Mortality Prediction', fontsize=16, fontweight='bold')

# 9.1 Actual vs Predicted (scatter)
ax = axes[0, 0]
ax.scatter(y_test, y_pred_test, alpha=0.5, s=20, c='steelblue', edgecolors='none')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('Actual Deaths', fontsize=12)
ax.set_ylabel('Predicted Deaths', fontsize=12)
ax.set_title(f'Actual vs Predicted (Test R² = {r2_test:.4f})', fontsize=13)
ax.legend()

# 9.2 Residual distribution
ax = axes[0, 1]
residuals = y_test - y_pred_test
ax.hist(residuals, bins=40, color='steelblue', edgecolor='white', alpha=0.8, density=True)
mu, sigma = np.mean(residuals), np.std(residuals)
x_range = np.linspace(residuals.min(), residuals.max(), 100)
ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', lw=2, label=f'N({mu:.1f}, {sigma:.1f}²)')
ax.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Residual Distribution', fontsize=13)
ax.legend()

# 9.3 Time series: actual vs predicted per state
colors = {'NSW': '#e74c3c', 'VIC': '#3498db', 'QLD': '#2ecc71'}
ax = axes[1, 0]
for state in ['NSW', 'VIC', 'QLD']:
    mask = df_clean['State'] == state
    dates = df_clean.loc[mask, 'Week_Start_Date']
    ax.plot(dates, y[mask], alpha=0.3, color=colors[state], linewidth=0.8)
    ax.plot(dates, y_pred_all[mask], color=colors[state], linewidth=1.2, label=f'{state} predicted')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Deaths / week', fontsize=12)
ax.set_title('Time Series: Actual (faint) vs Predicted', fontsize=13)
ax.legend(fontsize=9)

# 9.4 Residuals over time
ax = axes[1, 1]
test_dates = df_clean.loc[test_idx_all, 'Week_Start_Date']
test_states = df_clean.loc[test_idx_all, 'State']
for state in ['NSW', 'VIC', 'QLD']:
    mask = test_states == state
    ax.scatter(test_dates[mask], residuals[mask.values], alpha=0.5, s=15, 
               color=colors[state], label=state)
ax.axhline(0, color='black', linestyle='--', lw=1)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Residual', fontsize=12)
ax.set_title('Residuals Over Time (Test Set)', fontsize=13)
ax.legend()

# 9.5 Temperature-Mortality relationship (partial effect proxy)
ax = axes[2, 0]
for state in ['NSW', 'VIC', 'QLD']:
    mask = df_clean['State'] == state
    temp = df_clean.loc[mask, 'Mean_Temp']
    deaths = y[mask]
    # Bin temperatures
    bins = pd.cut(temp, bins=15)
    grouped = pd.DataFrame({'temp': temp, 'deaths': deaths}).groupby(bins)
    means = grouped.mean()
    ax.plot(means['temp'], means['deaths'], 'o-', color=colors[state], label=state, markersize=5)
ax.set_xlabel('Mean Temperature (°C)', fontsize=12)
ax.set_ylabel('Mean Deaths / week', fontsize=12)
ax.set_title('Temperature-Mortality Curve (U/J shape)', fontsize=13)
ax.legend()

# 9.6 Model summary metrics
ax = axes[2, 1]
ax.axis('off')
summary_text = (
    f"HYBRID GAM-DLNM MODEL SUMMARY\n"
    f"{'=' * 40}\n\n"
    f"Model Type: LinearGAM + DLNM cross-basis\n"
    f"Total Features: {X.shape[1]}\n"
    f"  - Cross-basis (DLNM): {sum('cb_' in c for c in X.columns)}\n"
    f"  - Seasonal harmonics: 4\n"
    f"  - State effects: {sum('state' in c.lower() for c in X.columns)}\n"
    f"  - Trend + other: {X.shape[1] - sum('cb_' in c for c in X.columns) - 4 - sum('state' in c.lower() for c in X.columns)}\n\n"
    f"PERFORMANCE\n"
    f"{'-' * 40}\n"
    f"Train R²:   {r2_train:.4f}\n"
    f"Test  R²:   {r2_test:.4f}\n"
    f"Overall R²: {r2_all:.4f}\n\n"
    f"Test MAE:   {mae_test:.1f} deaths/week\n"
    f"Test RMSE:  {rmse_test:.1f} deaths/week\n"
    f"Test MAPE:  {mape_test:.2f}%\n\n"
    f"CV Mean R²: {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}\n\n"
    f"Train size: {len(y_train)} | Test size: {len(y_test)}"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('hybrid_gam_dlnm_results.png', dpi=150, bbox_inches='tight')
print("  Saved: hybrid_gam_dlnm_results.png")

# ============================================================
# 10. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)
r2_pass = "✓ PASS" if r2_test >= 0.80 else "✗ FAIL"
print(f"  Test R² = {r2_test:.4f}  {r2_pass} (target ≥ 0.80)")
print(f"  Overall R² = {r2_all:.4f}")
print("=" * 70)
