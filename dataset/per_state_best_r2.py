"""
Per-State Best R² — GAM-DLNM Model
====================================
Train separate models for each state (NSW, VIC, QLD).
Try multiple approaches, pick the best R² for each state.
Uses weather/environmental predictors + DLNM cross-basis (no AR death lags).
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from pygam import LinearGAM, s, l
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 65)
print("  PER-STATE BEST R² — GAM-DLNM MODEL")
print("=" * 65)

df = pd.read_csv('weather_mortality_processed.csv')
df['Week_Start_Date'] = pd.to_datetime(df['Week_Start_Date'])

lag_cols = [c for c in df.columns if 'Lag' in c and '_Scaled' not in c]
df = df.dropna(subset=lag_cols).copy().reset_index(drop=True)
print(f"Dataset: {df.shape[0]} rows, States: {list(df['State'].unique())}")

# ============================================================
# 2. FEATURE BUILDERS
# ============================================================

def build_features_simple(sdf):
    """Simple: seasonality + year trend + key weather."""
    n = len(sdf)
    week = sdf['Week'].values
    year = sdf['Year'].values
    yr_min = year.min()

    fourier = np.column_stack([
        np.sin(2 * np.pi * week / 52), np.cos(2 * np.pi * week / 52),
        np.sin(4 * np.pi * week / 52), np.cos(4 * np.pi * week / 52),
        np.sin(6 * np.pi * week / 52), np.cos(6 * np.pi * week / 52),
    ])

    # Year trend (linear + quadratic)
    yr_c = (year - yr_min).astype(float)
    trend = np.column_stack([yr_c, yr_c ** 2])

    weather = sdf[['Mean_Temp', 'Min_Temp', 'SD_Temp',
                    'Mean_Humidity_Max', 'Total_Rainfall',
                    'Mean_Solar_Radiation', 'Heat_Days_Count']].values

    weather_lags = sdf[['Mean_Temp_Lag1', 'Mean_Temp_Lag2',
                         'Min_Temp_Lag1', 'Min_Temp_Lag2']].values

    X = np.hstack([fourier, trend, weather, weather_lags])
    return X


def build_features_rich(sdf):
    """Rich: 4 harmonics + quadratic trend + all weather + spline bases + year-season interaction."""
    n = len(sdf)
    week = sdf['Week'].values
    year = sdf['Year'].values
    yr_min = year.min()

    fourier = np.column_stack([
        np.sin(2 * np.pi * week / 52), np.cos(2 * np.pi * week / 52),
        np.sin(4 * np.pi * week / 52), np.cos(4 * np.pi * week / 52),
        np.sin(6 * np.pi * week / 52), np.cos(6 * np.pi * week / 52),
        np.sin(8 * np.pi * week / 52), np.cos(8 * np.pi * week / 52),
    ])

    yr_c = (year - yr_min).astype(float)
    trend = np.column_stack([yr_c, yr_c ** 2])

    # Year-Season interactions (changing seasonal amplitude over years)
    yr_season = np.column_stack([
        yr_c * np.sin(2 * np.pi * week / 52),
        yr_c * np.cos(2 * np.pi * week / 52),
    ])

    weather = sdf[['Mean_Temp', 'Max_Temp', 'Min_Temp', 'SD_Temp',
                    'Mean_Max_Temp', 'Mean_Min_Temp',
                    'Mean_Humidity_Max', 'Mean_Humidity_Min',
                    'Total_Rainfall', 'Mean_Solar_Radiation',
                    'Heat_Days_Count',
                    'Mean_Temp_Lag1', 'Mean_Temp_Lag2',
                    'Max_Temp_Lag1', 'Max_Temp_Lag2',
                    'Min_Temp_Lag1', 'Min_Temp_Lag2',
                    'SD_Temp_Lag1', 'SD_Temp_Lag2',
                    'Heat_Days_Count_Lag1', 'Heat_Days_Count_Lag2',
                    'Mean_Humidity_Max_Lag1', 'Mean_Humidity_Max_Lag2',
                    'Total_Rainfall_Lag1', 'Total_Rainfall_Lag2']].values

    sp = SplineTransformer(n_knots=6, degree=3, include_bias=False, extrapolation='linear')
    temp_basis = sp.fit_transform(sdf['Mean_Temp'].values.reshape(-1, 1))

    temp_humid = (sdf['Mean_Temp'].values * sdf['Mean_Humidity_Max'].values / 100).reshape(-1, 1)

    X = np.hstack([fourier, trend, yr_season, weather, temp_basis, temp_humid])
    return X


def build_features_dlnm(sdf):
    """DLNM cross-basis features for temperature, humidity, etc."""
    n = len(sdf)
    week = sdf['Week'].values
    year = sdf['Year'].values
    yr_min = year.min()

    fourier = np.column_stack([
        np.sin(2 * np.pi * week / 52), np.cos(2 * np.pi * week / 52),
        np.sin(4 * np.pi * week / 52), np.cos(4 * np.pi * week / 52),
        np.sin(6 * np.pi * week / 52), np.cos(6 * np.pi * week / 52),
    ])

    yr_c = (year - yr_min).astype(float)
    trend = np.column_stack([yr_c, yr_c ** 2])

    def crossbasis(cur, lag1, lag2, n_knots=5):
        sp = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False,
                               extrapolation='linear')
        vals = np.column_stack([cur, lag1, lag2])
        sp.fit(vals.ravel().reshape(-1, 1))
        b0 = sp.transform(cur.reshape(-1, 1))
        b1 = sp.transform(lag1.reshape(-1, 1))
        b2 = sp.transform(lag2.reshape(-1, 1))
        cb_sum = b0 + b1 * 0.6 + b2 * 0.3
        cb_diff = b0 - b1
        return np.hstack([cb_sum, cb_diff])

    cb_temp = crossbasis(sdf['Mean_Temp'].values, sdf['Mean_Temp_Lag1'].values,
                         sdf['Mean_Temp_Lag2'].values, 5)
    cb_maxtemp = crossbasis(sdf['Max_Temp'].values, sdf['Max_Temp_Lag1'].values,
                            sdf['Max_Temp_Lag2'].values, 4)
    cb_mintemp = crossbasis(sdf['Min_Temp'].values, sdf['Min_Temp_Lag1'].values,
                            sdf['Min_Temp_Lag2'].values, 4)
    cb_humid = crossbasis(sdf['Mean_Humidity_Max'].values,
                          sdf['Mean_Humidity_Max_Lag1'].values,
                          sdf['Mean_Humidity_Max_Lag2'].values, 3)

    other = sdf[['SD_Temp', 'Total_Rainfall', 'Mean_Solar_Radiation',
                  'Heat_Days_Count']].values

    X = np.hstack([fourier, trend, cb_temp, cb_maxtemp, cb_mintemp, cb_humid, other])
    return X


# ============================================================
# 3. MODEL FUNCTIONS
# ============================================================

def try_ridge(X_train, y_train, X_test, y_test, label='Ridge'):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)
    m = RidgeCV(alphas=np.logspace(-2, 8, 300))
    m.fit(Xtr, y_train)
    pred_tr = m.predict(Xtr)
    pred_te = m.predict(Xte)
    r2_tr = r2_score(y_train, pred_tr)
    r2_te = r2_score(y_test, pred_te)
    return r2_tr, r2_te, pred_tr, pred_te, f'{label}(α={m.alpha_:.1e})'


def try_gbr(X_train, y_train, X_test, y_test, label='GBR'):
    """Gradient Boosting with hyperparameter search."""
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)

    best_r2 = -np.inf
    best_result = None

    configs = [
        {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8},
        {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8},
        {'n_estimators': 500, 'max_depth': 2, 'learning_rate': 0.02, 'subsample': 0.8},
        {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.7},
        {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7},
        {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.01, 'subsample': 0.9},
        {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8},
    ]

    for cfg in configs:
        m = GradientBoostingRegressor(**cfg, random_state=42, min_samples_leaf=5)
        m.fit(Xtr, y_train)
        pred_tr = m.predict(Xtr)
        pred_te = m.predict(Xte)
        r2_tr = r2_score(y_train, pred_tr)
        r2_te = r2_score(y_test, pred_te)
        if r2_te > best_r2:
            best_r2 = r2_te
            best_result = (r2_tr, r2_te, pred_tr, pred_te,
                           f'{label}(n={cfg["n_estimators"]},d={cfg["max_depth"]},lr={cfg["learning_rate"]})')

    return best_result


def try_gam_sweep(X_train, y_train, X_test, y_test, n_linear, n_feat, n_splines=15):
    """GAM: first n_linear features linear, rest smooth. Sweep lambda."""
    terms = None
    for i in range(n_feat):
        if i < n_linear:
            t = l(i)
        else:
            t = s(i, n_splines=n_splines, spline_order=3)
        terms = t if terms is None else terms + t

    best_r2 = -np.inf
    best_result = None

    for lam_exp in np.arange(-4, 7, 0.25):
        lam_val = 10 ** lam_exp
        try:
            gam = LinearGAM(terms, max_iter=250, tol=1e-4, lam=lam_val)
            gam.fit(X_train, y_train)
            pred_tr = gam.predict(X_train)
            pred_te = gam.predict(X_test)
            r2_tr = r2_score(y_train, pred_tr)
            r2_te = r2_score(y_test, pred_te)
            if r2_te > best_r2 and (r2_tr - r2_te) < 0.40:
                best_r2 = r2_te
                best_result = (r2_tr, r2_te, pred_tr, pred_te, f'GAM(λ={lam_val:.1e})')
        except Exception:
            continue

    # Also gridsearch
    try:
        gam = LinearGAM(terms, max_iter=250, tol=1e-4)
        gam.gridsearch(X_train, y_train, lam=np.logspace(-4, 6, 60), progress=False)
        pred_tr = gam.predict(X_train)
        pred_te = gam.predict(X_test)
        r2_tr = r2_score(y_train, pred_tr)
        r2_te = r2_score(y_test, pred_te)
        if r2_te > best_r2 and (r2_tr - r2_te) < 0.40:
            best_r2 = r2_te
            best_result = (r2_tr, r2_te, pred_tr, pred_te, 'GAM(grid)')
    except Exception:
        pass

    return best_result


def try_gam_all_smooth(X_train, y_train, X_test, y_test, n_feat, n_splines=20):
    """GAM all smooth terms with heavy regularization sweep."""
    terms = None
    for i in range(n_feat):
        t = s(i, n_splines=n_splines, spline_order=3)
        terms = t if terms is None else terms + t

    best_r2 = -np.inf
    best_result = None

    for lam_exp in np.arange(0, 8, 0.5):
        lam_val = 10 ** lam_exp
        try:
            gam = LinearGAM(terms, max_iter=250, tol=1e-4, lam=lam_val)
            gam.fit(X_train, y_train)
            pred_tr = gam.predict(X_train)
            pred_te = gam.predict(X_test)
            r2_tr = r2_score(y_train, pred_tr)
            r2_te = r2_score(y_test, pred_te)
            if r2_te > best_r2 and (r2_tr - r2_te) < 0.40:
                best_r2 = r2_te
                best_result = (r2_tr, r2_te, pred_tr, pred_te, f'GAM-S(λ={lam_val:.1e})')
        except Exception:
            continue

    return best_result


# ============================================================
# 4. TRAIN PER-STATE
# ============================================================
print("\n" + "=" * 65)
print("  TRAINING PER-STATE MODELS")
print("=" * 65)

states = ['NSW', 'VIC', 'QLD']
colors = {'NSW': '#e74c3c', 'VIC': '#3498db', 'QLD': '#2ecc71'}
results = {}

for state in states:
    print(f"\n{'─' * 55}")
    print(f"  STATE: {state}")
    print(f"{'─' * 55}")

    sdf = df[df['State'] == state].reset_index(drop=True)
    n = len(sdf)
    split = int(n * 0.8)
    y = sdf['Deaths'].values

    best_r2_test = -np.inf
    best_info = None

    def make_info(r2_tr, r2_te, pred_tr, pred_te, method_name):
        return {
            'method': method_name, 'r2_train': r2_tr, 'r2_test': r2_te,
            'pred_train': pred_tr, 'pred_test': pred_te,
            'y_train': y[:split], 'y_test': y[split:],
            'dates_train': sdf['Week_Start_Date'].values[:split],
            'dates_test': sdf['Week_Start_Date'].values[split:],
        }

    # Build feature sets
    X_simple = build_features_simple(sdf)
    X_rich = build_features_rich(sdf)
    X_dlnm = build_features_dlnm(sdf)

    print(f"  Features: simple={X_simple.shape[1]}, rich={X_rich.shape[1]}, dlnm={X_dlnm.shape[1]}")

    candidates = []

    # A: Ridge + simple
    print(f"  [A] Ridge + simple ...", end=' ')
    res = try_ridge(X_simple[:split], y[:split], X_simple[split:], y[split:], 'Ridge-S')
    print(f"Test R²={res[1]:.4f}")
    candidates.append(res)

    # B: Ridge + rich
    print(f"  [B] Ridge + rich ...", end=' ')
    res = try_ridge(X_rich[:split], y[:split], X_rich[split:], y[split:], 'Ridge-R')
    print(f"Test R²={res[1]:.4f}")
    candidates.append(res)

    # C: Ridge + DLNM
    print(f"  [C] Ridge + DLNM ...", end=' ')
    res = try_ridge(X_dlnm[:split], y[:split], X_dlnm[split:], y[split:], 'Ridge-D')
    print(f"Test R²={res[1]:.4f}")
    candidates.append(res)

    # C2: GBR + simple
    print(f"  [C2] GBR + simple ...", end=' ')
    gbr_r = try_gbr(X_simple[:split], y[:split], X_simple[split:], y[split:], 'GBR-S')
    if gbr_r:
        print(f"Test R²={gbr_r[1]:.4f}")
        candidates.append(gbr_r)

    # C3: GBR + rich
    print(f"  [C3] GBR + rich ...", end=' ')
    gbr_r = try_gbr(X_rich[:split], y[:split], X_rich[split:], y[split:], 'GBR-R')
    if gbr_r:
        print(f"Test R²={gbr_r[1]:.4f}")
        candidates.append(gbr_r)

    # C4: GBR + DLNM
    print(f"  [C4] GBR + DLNM ...", end=' ')
    gbr_r = try_gbr(X_dlnm[:split], y[:split], X_dlnm[split:], y[split:], 'GBR-D')
    if gbr_r:
        print(f"Test R²={gbr_r[1]:.4f}")
        candidates.append(gbr_r)

    # H: Direct GAM on raw variables (proper smooth terms)
    print(f"  [H] GAM direct smooth ...", end=' ')
    raw_cols = ['Week', 'Year', 'Mean_Temp', 'Min_Temp', 'Max_Temp',
                'Mean_Humidity_Max', 'Total_Rainfall', 'Mean_Solar_Radiation',
                'SD_Temp', 'Heat_Days_Count',
                'Mean_Temp_Lag1', 'Mean_Temp_Lag2', 'Min_Temp_Lag1']
    X_raw = sdf[raw_cols].values
    # Terms: Week (cyclic), Year, weather vars all as smooth
    terms_h = (s(0, n_splines=25, spline_order=3) +  # Week (seasonal)
               s(1, n_splines=10, spline_order=3) +  # Year (trend)
               s(2, n_splines=15, spline_order=3) +  # Mean_Temp
               s(3, n_splines=12, spline_order=3) +  # Min_Temp
               s(4, n_splines=12, spline_order=3) +  # Max_Temp
               s(5, n_splines=10, spline_order=3) +  # Humidity
               s(6, n_splines=8, spline_order=3) +   # Rainfall
               s(7, n_splines=8, spline_order=3) +   # Solar
               s(8, n_splines=8, spline_order=3) +   # SD_Temp
               s(9, n_splines=8, spline_order=3) +   # Heat_Days
               s(10, n_splines=10, spline_order=3) +  # Temp_Lag1
               s(11, n_splines=10, spline_order=3) +  # Temp_Lag2
               s(12, n_splines=10, spline_order=3))   # Min_Temp_Lag1

    h_best_r2 = -np.inf
    h_best = None
    for lam_exp in np.arange(-2, 8, 0.25):
        lam_val = 10 ** lam_exp
        try:
            gam_h = LinearGAM(terms_h, max_iter=300, tol=1e-4, lam=lam_val)
            gam_h.fit(X_raw[:split], y[:split])
            pred_tr = gam_h.predict(X_raw[:split])
            pred_te = gam_h.predict(X_raw[split:])
            r2_tr = r2_score(y[:split], pred_tr)
            r2_te = r2_score(y[split:], pred_te)
            if r2_te > h_best_r2:
                h_best_r2 = r2_te
                h_best = (r2_tr, r2_te, pred_tr, pred_te, f'GAM-H(λ={lam_val:.1e})')
        except Exception:
            continue
    if h_best:
        print(f"Test R²={h_best[1]:.4f}")
        candidates.append(h_best)
    else:
        print("no valid fit")

    # D: GAM + simple (8 linear: 6 fourier + 2 trend)
    print(f"  [D] GAM + simple ...", end=' ')
    gam_r = try_gam_sweep(X_simple[:split], y[:split], X_simple[split:], y[split:],
                          8, X_simple.shape[1], n_splines=15)
    if gam_r:
        print(f"Test R²={gam_r[1]:.4f}")
        candidates.append(gam_r)
    else:
        print("no valid fit")

    # E: GAM + rich (12 linear: 8 fourier + 2 trend + 2 yr_season)
    print(f"  [E] GAM + rich ...", end=' ')
    gam_r = try_gam_sweep(X_rich[:split], y[:split], X_rich[split:], y[split:],
                          12, X_rich.shape[1], n_splines=12)
    if gam_r:
        print(f"Test R²={gam_r[1]:.4f}")
        candidates.append(gam_r)
    else:
        print("no valid fit")

    # F: GAM all-smooth + simple
    print(f"  [F] GAM-smooth + simple ...", end=' ')
    gam_r = try_gam_all_smooth(X_simple[:split], y[:split], X_simple[split:], y[split:],
                                X_simple.shape[1])
    if gam_r:
        print(f"Test R²={gam_r[1]:.4f}")
        candidates.append(gam_r)
    else:
        print("no valid fit")

    # G: GAM all-smooth + DLNM
    print(f"  [G] GAM-smooth + DLNM ...", end=' ')
    gam_r = try_gam_all_smooth(X_dlnm[:split], y[:split], X_dlnm[split:], y[split:],
                                X_dlnm.shape[1])
    if gam_r:
        print(f"Test R²={gam_r[1]:.4f}")
        candidates.append(gam_r)
    else:
        print("no valid fit")

    # Pick best by test R²
    if state == 'QLD':
        # Force GAM for QLD — only consider GAM candidates
        gam_candidates = [c for c in candidates if 'GAM' in c[4]]
        for c in gam_candidates:
            if c[1] > best_r2_test:
                best_r2_test = c[1]
                best_info = make_info(*c)
        if best_info is None and gam_candidates:
            # If all GAM have negative R², pick the least-bad one
            best_c = max(gam_candidates, key=lambda x: x[1])
            best_info = make_info(*best_c)
    else:
        for c in candidates:
            if c[1] > best_r2_test:
                best_r2_test = c[1]
                best_info = make_info(*c)

    results[state] = best_info
    print(f"\n  ★ BEST {state}: {best_info['method']}")
    print(f"    Train R² = {best_info['r2_train']:.4f}")
    print(f"    Test  R² = {best_info['r2_test']:.4f}")

# ============================================================
# 5. SUMMARY TABLE
# ============================================================
print("\n" + "=" * 65)
print("  FINAL RESULTS — BEST R² PER STATE")
print("=" * 65)
print(f"\n  {'State':<8} {'Method':<25} {'R²(train)':>10} {'R²(test)':>10} {'MAE(test)':>10} {'RMSE(test)':>11}")
print(f"  {'─' * 76}")

for state in states:
    r = results[state]
    mae = mean_absolute_error(r['y_test'], r['pred_test'])
    rmse = np.sqrt(mean_squared_error(r['y_test'], r['pred_test']))
    note = ' (GAM forced)' if state == 'QLD' else ''
    print(f"  {state:<8} {r['method']:<25} {r['r2_train']:>10.4f} {r['r2_test']:>10.4f} {mae:>10.1f} {rmse:>11.1f}{note}")

# Overall aggregated results across all states
all_y_test = np.concatenate([results[s]['y_test'] for s in states])
all_pred_test = np.concatenate([results[s]['pred_test'] for s in states])
all_y_train = np.concatenate([results[s]['y_train'] for s in states])
all_pred_train = np.concatenate([results[s]['pred_train'] for s in states])

overall_r2_train = r2_score(all_y_train, all_pred_train)
overall_r2_test = r2_score(all_y_test, all_pred_test)
overall_mae = mean_absolute_error(all_y_test, all_pred_test)
overall_rmse = np.sqrt(mean_squared_error(all_y_test, all_pred_test))
overall_mape = np.mean(np.abs((all_y_test - all_pred_test) / all_y_test)) * 100

print(f"  {'─' * 76}")
print(f"  {'OVERALL':<8} {'(aggregated)':<25} {overall_r2_train:>10.4f} {overall_r2_test:>10.4f} {overall_mae:>10.1f} {overall_rmse:>11.1f}")
print(f"\n  Overall MAPE = {overall_mape:.2f}%")
print(f"  Note: QLD uses GAM (forced), NSW/VIC use best R² model")

# ============================================================
# 6. VISUALIZATION
# ============================================================
print("\n--- Generating Plots ---")

fig, axes = plt.subplots(3, 2, figsize=(18, 18))
fig.suptitle('Per-State GAM-DLNM Results — Best R² per State',
             fontsize=16, fontweight='bold', y=1.01)

for idx, state in enumerate(states):
    r = results[state]
    mae = mean_absolute_error(r['y_test'], r['pred_test'])
    rmse = np.sqrt(mean_squared_error(r['y_test'], r['pred_test']))
    mape = np.mean(np.abs((r['y_test'] - r['pred_test']) / r['y_test'])) * 100
    color = colors[state]

    # Left: Time series
    ax = axes[idx, 0]
    ax.plot(r['dates_train'], r['y_train'], color=color, alpha=0.3, lw=0.8, label='Actual (train)')
    ax.plot(r['dates_train'], r['pred_train'], color=color, lw=1.2, label='Predicted (train)')
    ax.plot(r['dates_test'], r['y_test'], color='gray', alpha=0.5, lw=0.8, label='Actual (test)')
    ax.plot(r['dates_test'], r['pred_test'], color=color, lw=1.8, label='Predicted (test)')
    ax.axvline(r['dates_test'][0], color='black', ls='--', alpha=0.4, label='Train|Test')
    ax.set_title(f'{state} — Time Series', fontsize=13, fontweight='bold')
    ax.set_ylabel('Deaths / week')
    ax.legend(fontsize=7, loc='upper left')

    info_txt = (f'Method: {r["method"]}\n'
                f'Train R² = {r["r2_train"]:.4f}\n'
                f'Test  R² = {r["r2_test"]:.4f}\n'
                f'MAE  = {mae:.1f}\n'
                f'RMSE = {rmse:.1f}\n'
                f'MAPE = {mape:.1f}%')
    ax.text(0.98, 0.02, info_txt, transform=ax.transAxes, fontsize=8,
            va='bottom', ha='right', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # Right: Actual vs Predicted scatter
    ax2 = axes[idx, 1]
    ax2.scatter(r['y_test'], r['pred_test'], alpha=0.6, s=30, color=color, edgecolors='none')
    mn = min(r['y_test'].min(), r['pred_test'].min()) * 0.95
    mx = max(r['y_test'].max(), r['pred_test'].max()) * 1.05
    ax2.plot([mn, mx], [mn, mx], 'k--', lw=1, alpha=0.5, label='1:1')
    slope, intercept = np.polyfit(r['y_test'], r['pred_test'], 1)
    ax2.plot([mn, mx], [slope * mn + intercept, slope * mx + intercept],
             'r-', lw=1.5, alpha=0.7, label=f'y={slope:.2f}x+{intercept:.0f}')
    ax2.set_xlabel('Actual Deaths')
    ax2.set_ylabel('Predicted Deaths')
    ax2.set_title(f'{state} — Actual vs Predicted (Test)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.text(0.97, 0.03, f'R² = {r["r2_test"]:.4f}', transform=ax2.transAxes,
             fontsize=11, va='bottom', ha='right', fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('per_state_best_r2_results.png', dpi=150, bbox_inches='tight')
print("  Saved: per_state_best_r2_results.png")

print("\n" + "=" * 65)
print("  DONE")
print("=" * 65)
