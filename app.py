#!/usr/bin/env python
# coding: utf-8
"""
Fantasy Football AI — Predictive Modeling and Squad Optimization
================================================================

Purpose
-------
Streamlit app that:
1) Trains several regression models on historical FPL-style data.
2) Predicts points for the next gameweek.
3) Builds a 15-player squad + valid Starting XI via ILP (PuLP) or a robust greedy fallback.

How it works
------------
- Data split is automatic: all GW < max(GW) are training; GW == max(GW) is the
  "next gameweek" to predict/optimize for.
- Multiple models are fitted on training data. We keep in-sample metrics to compare them.
- You pick which model’s predictions to use for squad selection.
- We try an ILP optimizer first (with realistic FPL-like constraints). If PuLP is
  unavailable or infeasible, we build a valid squad with a greedy heuristic.

Expected input file
-------------------
- CSV named `merged_gw.csv` in the folder data in the working directory.
- Must contain:
  * `GW` (int) — gameweek.
  * Features listed in `feature_cols` (see below).
  * Target `total_points` (for training only).
  * Metadata columns: `name`, `team`, `position`, `value` (tenths of a million).
- Optional minute proxy: `expected_minutes`, `exp_minutes`, `proj_minutes`,
  `minutes_gw`, or `minutes`. (Used to filter likely starters.)

Running locally
---------------
    streamlit run app.py

Main dependencies
-----------------
streamlit, pandas, numpy, scikit-learn, plotly, xgboost, pulp
"""

# ====================================================
# Imports
# ====================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor  # optional, but enabled here

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
)
from sklearn.model_selection import GroupKFold

# Try to import PuLP for ILP squad optimization. If unavailable, we fall back to greedy.
try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False

# ====================================================
# Streamlit Page Setup
# ====================================================
st.set_page_config(page_title="Fantasy Football AI", layout="wide")
st.title("Fantasy Football AI — Predictive Modeling and Optimization")

# Convenience button to re-run the app (useful while iterating on CSV or code)
if st.button("🔄 Refresh Data"):
    st.rerun()

# ====================================================
# Utilities
# ====================================================
def format_millions_tenth(value_tenths: float) -> str:
    """
    Convert FPL-style 'tenths of a million' value (e.g., 85) to readable text (e.g., £8.5M)
    """
    return f"£{(value_tenths / 10.0):.1f}M"


def select_minutes_column(df_like: pd.DataFrame) -> pd.Series:
    """
    Heuristically pick a minutes/expected-minutes column for eligibility checks.
    If none exists, return zeros to avoid over-optimistic XI assumptions.

    Preference order:
      expected_minutes > exp_minutes > proj_minutes > minutes_gw > minutes
    """
    for col in ["expected_minutes", "exp_minutes", "proj_minutes", "minutes_gw", "minutes"]:
        if col in df_like.columns:
            return df_like[col]
    return pd.Series(0, index=df_like.index)  # safe fallback


# ====================================================
# Data Loading
# ====================================================
@st.cache_data(ttl=60)
def load_and_process_data():
    """
    Load CSV and create train/test split:
      - Train: rows with GW < max(GW)
      - Test : rows with GW == max(GW)  (treated as "next gameweek")

    Returns
    -------
    X_train : pd.DataFrame
    y_train : pd.Series
    X_test  : pd.DataFrame
    train_df, test_df : pd.DataFrame (full rows for diagnostics/metadata)
    feature_cols : list[str]
    target_col   : str
    latest_gw    : int
    """
    # Robust read: skip malformed lines instead of crashing
    df = pd.read_csv("data/merged_gw.csv", on_bad_lines="skip").copy()

    # Ensure GW is numeric and clean
    df["GW"] = pd.to_numeric(df["GW"], errors="coerce")
    df = df.dropna(subset=["GW"])
    df["GW"] = df["GW"].astype(int)

    latest_gw = int(df["GW"].max())
    train_df = df[df["GW"] < latest_gw].copy()
    test_df  = df[df["GW"] == latest_gw].copy()

    # Core features for supervised learning
    feature_cols = [
        "minutes", "goals_scored", "assists", "clean_sheets", "bps", "influence",
        "creativity", "threat", "ict_index", "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded"
    ]
    target_col = "total_points"

    # Drop rows missing features/targets to avoid model errors.
    train_df = train_df.dropna(subset=feature_cols + [target_col]).copy()
    test_df  = test_df.dropna(subset=feature_cols).copy()

    # Matrices for scikit-learn
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test  = test_df[feature_cols]

    return X_train, y_train, X_test, train_df, test_df, feature_cols, target_col, latest_gw


# ====================================================
# Metrics
# ====================================================
# Order and formatting for metric tables
METRIC_COLS = ["Model","MAE","RMSE","R²","ExplainedVar","MedAE","Spearman","TopKOverlap"]
METRIC_FORMATS = {
    "MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}",
    "ExplainedVar": "{:.3f}", "MedAE": "{:.3f}", "Spearman": "{:.3f}",
    "TopKOverlap": "{:.2%}",
}

def _spearman_like(y_true, y_pred) -> float:
    """
    Spearman-like rank correlation using pandas ranks (avoids SciPy dependency).
    Returns NaN if either side is constant.
    """
    a = pd.Series(y_true).rank(method="average")
    b = pd.Series(y_pred).rank(method="average")
    if a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _align_arrays(a, b):
    """
    Align two arrays to the same min length and flatten to 1D.
    Defensive helper when pipeline lengths can differ slightly.
    """
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    n = min(len(a), len(b))
    return a[:n], b[:n]


def top_k_overlap(y_true, y_pred, k=20):
    """
    Fractional overlap between true vs predicted top-K indices.
    Focuses on ranking quality (useful for "pick the best players" tasks).
    """
    y_true, y_pred = _align_arrays(y_true, y_pred)
    n = len(y_true)
    if n == 0:
        return np.nan
    k = min(k, n)
    true_top_idx = np.argsort(y_true)[-k:]
    pred_top_idx = np.argsort(y_pred)[-k:]
    return len(set(true_top_idx).intersection(set(pred_top_idx))) / k


def _metrics_block(y_true, y_pred, k_for_overlap=20):
    """
    Compute a broad set of regression + ranking metrics on aligned arrays.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R²": r2_score(y_true, y_pred),
        "ExplainedVar": explained_variance_score(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "Spearman": _spearman_like(y_true, y_pred),
        "TopKOverlap": top_k_overlap(y_true, y_pred, k=k_for_overlap),
    }


# ====================================================
# Train & Predict
# ====================================================
@st.cache_data(ttl=60)
def train_and_predict(X_train, y_train, X_test, test_df, k_for_overlap=20):
    """
    Fit a suite of models on training data, compute in-sample metrics,
    and attach each model's predictions to the *test* (next GW) frame.

    Also builds a simple tree-ensemble (RF + XGB + GBR) when all three are present.

    Returns
    -------
    test_df_with_preds : pd.DataFrame
        Original test_df plus one column per model (e.g., 'pred_rf', 'pred_xgb', ...)
    performance_df     : pd.DataFrame
        In-sample metrics table (sorted by RMSE).
    y_train            : pd.Series
        Ground-truth training target (convenience pass-through).
    train_predictions  : dict[str, np.ndarray]
        In-sample predictions per model (for diagnostics).
    pred_key_map       : dict[str, str]
        Mapping: model display name -> column name in test_df (used by UI selectbox).
    models_dict        : dict[str, sklearn.BaseEstimator]
        Fitted estimator per model name (for feature importances, etc.).
    """
    model_specs = [
        ("Linear Regression", "lr",  LinearRegression()),
        ("Random Forest",     "rf",  RandomForestRegressor(random_state=42, n_estimators=200)),
        ("XGBoost",           "xgb", XGBRegressor(random_state=42, n_estimators=300, max_depth=6, verbosity=0)),
        ("Gradient Boosting", "gbr", GradientBoostingRegressor(random_state=42)),
        ("KNN (k=10)",        "knn", KNeighborsRegressor(n_neighbors=10)),
    ]

    test_df = test_df.copy()
    rows = []                 # For metrics table
    train_predictions = {}    # In-sample yhat per model
    pred_key_map = {}         # Model name -> prediction column in test_df
    models_dict = {}          # Model name -> fitted estimator

    for name, key, model in model_specs:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)

        train_predictions[name] = y_pred_train
        models_dict[name] = model

        # Add next-GW predictions as a new column
        colname = f"pred_{key}"
        test_df[colname] = model.predict(X_test)
        pred_key_map[name] = colname

        # In-sample metrics (fast sanity check + comparison)
        metrics = _metrics_block(y_train, y_pred_train, k_for_overlap=k_for_overlap)
        metrics["Model"] = name
        rows.append(metrics)

    # Base performance table, sorted by RMSE
    performance = pd.DataFrame(rows)[METRIC_COLS].sort_values("RMSE").reset_index(drop=True)

    # Tree-ensemble (average predictions) for robustness
    if all(k in pred_key_map for k in ["Random Forest", "XGBoost", "Gradient Boosting"]):
        ens_name = "Ensemble (RF+XGB+GBR)"
        test_df["pred_ens"] = (
            test_df[pred_key_map["Random Forest"]]
          + test_df[pred_key_map["XGBoost"]]
          + test_df[pred_key_map["Gradient Boosting"]]
        ) / 3.0
        pred_key_map[ens_name] = "pred_ens"

        y_pred_train_ens = (
            train_predictions["Random Forest"]
          + train_predictions["XGBoost"]
          + train_predictions["Gradient Boosting"]
        ) / 3.0
        train_predictions[ens_name] = y_pred_train_ens

        ens_metrics = _metrics_block(y_train, y_pred_train_ens, k_for_overlap=k_for_overlap)
        ens_metrics["Model"] = ens_name

        performance = (
            pd.concat([performance, pd.DataFrame([ens_metrics])[METRIC_COLS]], ignore_index=True)
              .sort_values("RMSE")
              .reset_index(drop=True)
        )

    return test_df, performance, y_train, train_predictions, pred_key_map, models_dict


# ====================================================
# CV by GW
# ====================================================
@st.cache_data(ttl=60)
def evaluate_cv(train_df, feature_cols, target_col, _models_dict, n_splits=5, k_for_overlap=20):
    """
    GroupKFold cross-validation using GW as the grouping variable.
    - Preserves temporal grouping: no GW is split across folds.

    Returns per-model overall metrics and per-position breakdowns (aggregated over folds).
    """
    df = train_df.dropna(subset=feature_cols + [target_col]).copy()
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df["GW"].values
    pos = df.get("position", pd.Series(["UNK"]*len(df))).values  # optional

    uniq_groups = np.unique(groups)
    n_splits = min(max(2, n_splits), len(uniq_groups))
    gkf = GroupKFold(n_splits=n_splits)

    def _fresh(name):
        # Create a fresh, unfitted estimator for each fold to avoid leakage
        if name == "Linear Regression": return LinearRegression()
        if name == "Random Forest":     return RandomForestRegressor(random_state=42, n_estimators=200)
        if name == "XGBoost":           return XGBRegressor(random_state=42, n_estimators=300, max_depth=6, verbosity=0)
        if name == "Gradient Boosting": return GradientBoostingRegressor(random_state=42)
        if name.startswith("KNN"):      return KNeighborsRegressor(n_neighbors=10)
        return None

    overall_rows, perpos_rows = [], []
    model_names = list(_models_dict.keys())

    for name in model_names:
        all_true, all_pred = [], []
        perpos_true, perpos_pred = {}, {}

        if name.startswith("Ensemble"):
            # Rebuild the tree models inside each fold and average predictions
            base_names = ["Random Forest", "XGBoost", "Gradient Boosting"]
            for tr_idx, va_idx in gkf.split(X, y, groups):
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]
                pos_va = pos[va_idx]

                preds = []
                for bn in base_names:
                    m_ = _fresh(bn); m_.fit(X_tr, y_tr)
                    preds.append(m_.predict(X_va))
                yhat = np.mean(np.column_stack(preds), axis=1)

                all_true.append(y_va); all_pred.append(yhat)
                for p in np.unique(pos_va):
                    mask = (pos_va == p)
                    perpos_true.setdefault(p, []).append(y_va[mask])
                    perpos_pred.setdefault(p, []).append(yhat[mask])
        else:
            # Standard per-model CV
            for tr_idx, va_idx in gkf.split(X, y, groups):
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]
                pos_va = pos[va_idx]

                m = _fresh(name); m.fit(X_tr, y_tr)
                yhat = m.predict(X_va)

                all_true.append(y_va); all_pred.append(yhat)
                for p in np.unique(pos_va):
                    mask = (pos_va == p)
                    perpos_true.setdefault(p, []).append(y_va[mask])
                    perpos_pred.setdefault(p, []).append(yhat[mask])

        # Aggregate across folds
        y_all = np.concatenate(all_true); yhat_all = np.concatenate(all_pred)
        met = _metrics_block(y_all, yhat_all, k_for_overlap=k_for_overlap)
        met["Model"] = name
        overall_rows.append(met)

        # Per-position breakdown
        for p in sorted(perpos_true.keys()):
            yt = np.concatenate(perpos_true[p])
            yp = np.concatenate(perpos_pred[p])
            mpos = _metrics_block(yt, yp, k_for_overlap=k_for_overlap)
            mpos["Model"] = name
            mpos["Position"] = p
            perpos_rows.append(mpos)

    overall_df = pd.DataFrame(overall_rows)[METRIC_COLS].sort_values("RMSE").reset_index(drop=True)
    by_pos_df  = pd.DataFrame(perpos_rows)[["Model","Position"] + METRIC_COLS[1:]].copy()
    return overall_df, by_pos_df


def normalize_for_radar(perf_df):
    """
    Normalize metrics to [0,1] so different scales are comparable on a radar plot.
    Lower-better metrics (MAE, RMSE, MedAE) are inverted.
    """
    df = perf_df.copy()
    cols_hi = ["R²","ExplainedVar","Spearman","TopKOverlap"]
    cols_lo = ["MAE","RMSE","MedAE"]
    for c in cols_lo:
        m, M = df[c].min(), df[c].max()
        df[c] = 1 - (df[c] - m) / (M - m + 1e-9)
    for c in cols_hi:
        m, M = df[c].min(), df[c].max()
        df[c] = (df[c] - m) / (M - m + 1e-9)
    return df


# ====================================================
# Squad Construction (Greedy + ILP)
# ====================================================
# Global FPL-like constraints
SQUAD_SIZE       = 15
POSITION_QUOTAS  = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_XI_SIZE = 11
START_BOUNDS     = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}

def greedy_squad_and_xi(
    df: pd.DataFrame,
    pred_col: str,
    budget_tenths: int,
    quotas: dict,
    min_start_minutes: int,
    max_from_team: int | None
):
    """
    Greedy fallback that ALWAYS returns a usable squad (assuming data coverage):
      1) Sort by predicted points descending
      2) Add players while respecting budget, position quotas, per-club cap.
      3) Build a valid XI (minutes threshold + position bounds).
      4) Bench: outfield lowest to highest predicted, then GK last.
    """
    remaining = budget_tenths
    squad = {p: [] for p in quotas}
    team_counts = {}

    # Highest predicted first
    df_sorted = df.sort_values(pred_col, ascending=False)

    for _, row in df_sorted.iterrows():
        pos, val, team = row.get("position"), row.get("value"), row.get("team")
        # Skip incomplete rows to avoid runtime errors
        if pd.isna(pos) or pd.isna(val) or pd.isna(team):
            continue
        # Respect quotas + budget
        if len(squad[pos]) < quotas[pos] and remaining - val >= 0:
            # Per-club cap (if enabled)
            if max_from_team is not None and team_counts.get(team, 0) >= max_from_team:
                continue
            squad[pos].append(row)
            team_counts[team] = team_counts.get(team, 0) + 1
            remaining -= val
        # Stop early if all quotas are satisfied
        if all(len(squad[p]) == quotas[p] for p in quotas):
            break

    # Flatten to a DataFrame; bail out if we somehow couldn't fill quotas
    squad_df = pd.DataFrame([p for lst in squad.values() for p in lst]).copy()
    if squad_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None

    # Minutes-based eligibility for XI
    squad_df["exp_min"] = select_minutes_column(squad_df)
    eligible = squad_df.copy()
    eligible["eligible_start"] = (eligible["exp_min"] >= min_start_minutes).astype(int)

    # Build XI respecting position minima/maxima
    starters = []

    # Exactly 1 GK
    gks = eligible[(eligible["position"] == "GK") & (eligible["eligible_start"] == 1)].sort_values(pred_col, ascending=False)
    if len(gks) > 0:
        starters.append(gks.index[0])

    # Satisfy minimums for DEF/MID/FWD
    for pos, (lo, _) in START_BOUNDS.items():
        if pos == "GK":
            continue
        needed = lo - sum(eligible.loc[i, "position"] == pos for i in starters)
        if needed > 0:
            pool = eligible[
                (eligible["position"] == pos)
                & (eligible["eligible_start"] == 1)
                & (~eligible.index.isin(starters))
            ].sort_values(pred_col, ascending=False)
            starters += list(pool.index[:needed])

    # Fill remaining XI slots greedily within max bounds
    remaining_slots = STARTING_XI_SIZE - len(starters)
    pos_count = {p: sum(eligible.loc[i, "position"] == p for i in starters) for p in POSITION_QUOTAS}
    pool_any = eligible[(eligible["eligible_start"] == 1) & (~eligible.index.isin(starters))].sort_values(pred_col, ascending=False)
    for i in pool_any.index:
        if remaining_slots == 0:
            break
        pos = eligible.loc[i, "position"]
        if pos_count[pos] < START_BOUNDS[pos][1]:
            starters.append(i); pos_count[pos] += 1; remaining_slots -= 1

    # If still short of minutes, relax minutes
    if len(starters) < STARTING_XI_SIZE:
        pool_relax = eligible[~eligible.index.isin(starters)].sort_values(pred_col, ascending=False)
        starters += list(pool_relax.index[:STARTING_XI_SIZE - len(starters)])

    starters = starters[:STARTING_XI_SIZE]
    xi_df = squad_df.loc[starters].copy()

    # Captain/Vice: top 2 predicted within XI
    xi_sorted = xi_df.sort_values(pred_col, ascending=False)
    captain = xi_sorted.iloc[0]["name"] if len(xi_sorted) > 0 else None
    vice    = xi_sorted.iloc[1]["name"] if len(xi_sorted) > 1 else None

    # Bench ordering: outfield ascending by predicted points, GK(s) last
    bench_df = squad_df.drop(index=xi_df.index, errors="ignore").copy()
    if not bench_df.empty:
        bench_gk  = bench_df[bench_df["position"] == "GK"]
        bench_out = bench_df[bench_df["position"] != "GK"].sort_values(pred_col, ascending=True)
        bench_df  = pd.concat([bench_out, bench_gk], axis=0)

    return squad_df.reset_index(drop=True), xi_df.reset_index(drop=True), bench_df.reset_index(drop=True), captain, vice


def optimize_full(
    df: pd.DataFrame,
    pred_col: str,
    budget_tenths: int,
    quotas: dict,
    min_start_minutes: int,
    max_from_team: int | None,
    min_spend_pct: float,
    lambda_spend: float
):
    """
    ILP squad builder (requires PuLP):
      - Picks exactly 15 under budget, matching POSITION_QUOTAS.
      - Builds a valid XI (position bounds + minutes eligibility).
      - Enforces per-club cap (if set).
      - Selects exactly one Captain & one Vice (both must be starters).
      - Softly encourages spending via lambda_spend and a hard min_spend_pct floor.

    Returns squad_df, xi_df, bench_df, captain_name, vice_name on success; else None.
    """
    if not PULP_AVAILABLE:
        return None

    # Keep only rows with essential fields
    pool = df.dropna(subset=["position", "value", pred_col, "team"]).copy().reset_index(drop=True)

    # Precompute minutes/eligibility
    pool["exp_min"] = select_minutes_column(pool)
    pool["eligible_start"] = (pool["exp_min"] >= min_start_minutes).astype(int)
    pool["__idx"] = pool.index  # internal ID for bench derivation
    idx = pool.index

    # Decision variables per player i.
    x = pulp.LpVariable.dicts("x", idx, 0, 1, cat="Binary")  # picked in 15-man squad
    s = pulp.LpVariable.dicts("s", idx, 0, 1, cat="Binary")  # starting XI
    c = pulp.LpVariable.dicts("c", idx, 0, 1, cat="Binary")  # captain
    v = pulp.LpVariable.dicts("v", idx, 0, 1, cat="Binary")  # vice-captain

    prob = pulp.LpProblem("FPL_XI_Bench_Captain", pulp.LpMaximize)

    # Normalize values for objective stability (budget encouragement term)
    vmax = float(pool["value"].max()) if len(pool) and float(pool["value"].max()) > 0 else 1.0
    value_norm = pool["value"] / vmax

    # Objective: predicted points (squad) + extra for Captain (doubling proxy) + small for Vice
    # + budget encouragement
    prob += (
        pulp.lpSum(x[i] * pool.loc[i, pred_col] for i in idx) +
        pulp.lpSum(c[i] * pool.loc[i, pred_col] for i in idx) +
        0.01 * pulp.lpSum(v[i] * pool.loc[i, pred_col] for i in idx) +
        lambda_spend * pulp.lpSum(x[i] * value_norm.iloc[i] for i in idx)
    )

    # Squad size and spend envelope (with minimum spend)
    prob += pulp.lpSum(x[i] for i in idx) == SQUAD_SIZE
    spend_expr = pulp.lpSum(x[i] * pool.loc[i, "value"] for i in idx)
    prob += spend_expr <= budget_tenths
    prob += spend_expr >= (min_spend_pct * budget_tenths)

    # Exact position quotas for the 15
    for pos, req in POSITION_QUOTAS.items():
        prob += pulp.lpSum(x[i] for i in idx if pool.loc[i, "position"] == pos) == req

    # XI size + membership linkage
    prob += pulp.lpSum(s[i] for i in idx) == STARTING_XI_SIZE
    for i in idx:
        prob += s[i] <= x[i]  # can only start if selected in 15

    # XI position rules
    prob += pulp.lpSum(s[i] for i in idx if pool.loc[i, "position"] == "GK") == 1
    for pos, (lo, hi) in START_BOUNDS.items():
        if pos == "GK":
            continue
        prob += pulp.lpSum(s[i] for i in idx if pool.loc[i, "position"] == pos) >= lo
        prob += pulp.lpSum(s[i] for i in idx if pool.loc[i, "position"] == pos) <= hi

    # Minutes eligibility for starters
    for i in idx:
        prob += s[i] <= pool.loc[i, "eligible_start"]

    # Per-club cap (if enabled)
    if max_from_team is not None:
        for club in pool["team"].dropna().unique():
            prob += pulp.lpSum(x[i] for i in idx if pool.loc[i, "team"] == club) <= max_from_team

    # Exactly one Captain and one Vice; both must be starters; cannot be the same player
    prob += pulp.lpSum(c[i] for i in idx) == 1
    prob += pulp.lpSum(v[i] for i in idx) == 1
    for i in idx:
        prob += c[i] <= s[i]
        prob += v[i] <= s[i]
        prob += c[i] + v[i] <= 1

    # Solve quietly with CBC
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None

    # Extract decisions back to DataFrames
    chosen = [i for i in idx if pulp.value(x[i]) > 0.5]
    xi     = [i for i in idx if pulp.value(s[i]) > 0.5]
    cap    = [i for i in idx if pulp.value(c[i]) > 0.5]
    vice   = [i for i in idx if pulp.value(v[i]) > 0.5]

    cap_name  = pool.loc[cap[0], "name"] if len(cap) == 1 else None
    vice_name = pool.loc[vice[0], "name"] if len(vice) == 1 else None

    squad_df = pool.loc[chosen].copy()
    xi_df    = pool.loc[xi].copy()
    bench_df = squad_df[~squad_df["__idx"].isin(xi_df["__idx"])].copy()

    # Cleanup helper columns
    for _df in (squad_df, xi_df, bench_df):
        _df.reset_index(drop=True, inplace=True)
        _df.drop(columns=["__idx"], inplace=True, errors="ignore")

    return squad_df, xi_df, bench_df, cap_name, vice_name


# ====================================================
# Pipeline Execution
# ====================================================
# Load all data and compute train/test split + required columns
X_train, y_train, X_test, train_df, test_df, feature_cols, target_col, latest_gw = load_and_process_data()

# Sidebar controls that influence training metrics (Top-K overlap K)
st.sidebar.markdown("### Controls")
k_for_overlap = st.sidebar.slider("Top-K overlap (training)", 5, 50, 20, 5)

# Fit models and attach next-GW predictions to test_df
test_df, performance_df, y_train_true, train_predictions, pred_key_map, models_dict = train_and_predict(
    X_train, y_train, X_test, test_df, k_for_overlap=k_for_overlap
)

# ====================================================
# Sidebar: Squad Controls
# ====================================================
# Which model’s predictions to use for squad selection
model = st.sidebar.selectbox("Choose Prediction Model", list(pred_key_map.keys()))
pred_col = pred_key_map[model]

# Budget (in £M) — internally converted to tenths to match data convention
budget_millions = st.sidebar.slider("Budget (£M)", 70.0, 200.0, 100.0, 0.5)
budget_tenths   = int(round(budget_millions * 10))

# Standard FPL rule: max three per Premier League team
use_team_cap = st.sidebar.checkbox("Limit max players per club (FPL rule)", value=True)
max_per_team  = 3 if use_team_cap else None

# Limit candidate pool to positions of interest
positions_available = ["GK", "DEF", "MID", "FWD"]
pos_filter = st.sidebar.multiselect("Include positions in candidate pool", positions_available, default=positions_available)

# Expected minutes threshold for starting XI eligibility
min_start_minutes = st.sidebar.slider("Min expected minutes for starters", 0, 120, 60, 5)

# Spending behavior (ILP only)
min_spend_pct = st.sidebar.slider("Min spend (% of budget)", 0, 100, 95, 1) / 100.0
lambda_spend  = st.sidebar.slider("Spend encouragement (soft)", 0.0, 1.0, 0.05, 0.01)

# Choose optimizer vs greedy fallback
use_optimizer = st.sidebar.checkbox("Use optimizer (recommended)", value=True)

# Optional diagnostics
st.sidebar.markdown("### Advanced Diagnostics")
run_cv = st.sidebar.checkbox("Run GroupKFold CV (by GW)", value=False)
cv_splits = st.sidebar.slider("CV splits", 3, 8, 5, 1)
show_feature_importance = st.sidebar.checkbox("Show feature importances / coefficients", value=True)

# Prepare the candidate pool (post-position filter)
pool_df = test_df.copy()
if pos_filter:
    pool_df = pool_df[pool_df["position"].isin(pos_filter)]

# ====================================================
# Player Debug
# ====================================================
st.markdown("### Player Debug")
debug_name = st.text_input("Enter player name to inspect", "")
if debug_name.strip():
    dbg = pool_df.copy()
    dbg["predicted_points"] = dbg[pred_col]
    dbg["exp_min"] = select_minutes_column(dbg)
    # Quick substring search by name.
    dbg_rows = dbg[dbg["name"].str.contains(debug_name, case=False, na=False)]
    if not dbg_rows.empty:
        required_cols = ["position", "value", pred_col, "team"]
        dbg_rows = dbg_rows.copy()
        # Flags to show why a player might be dropped by optimizer/filters
        dbg_rows["dropped_by_optimizer"] = dbg_rows[required_cols].isna().any(axis=1)
        dbg_rows["passes_position_filter"] = dbg_rows["position"].isin(pos_filter)
        dbg_rows["eligible_starter"] = dbg_rows["exp_min"] >= min_start_minutes
        # Pretty cost.
        dbg_rows["cost (M)"] = dbg_rows["value"].apply(format_millions_tenth)
        dbg_rows = dbg_rows.drop(columns=["value"])
        st.dataframe(
            dbg_rows[["name","team","position","cost (M)","predicted_points","exp_min",
                      "passes_position_filter","eligible_starter","dropped_by_optimizer"]].reset_index(drop=True),
            use_container_width=True, hide_index=True
        )
    else:
        st.warning(f"No match for '{debug_name}' in current pool after filters (positions: {', '.join(pos_filter)}).")

# ====================================================
# Build Squad + XI + Bench
# ====================================================
st.markdown(f"### Selected Model: {model}")
st.caption(f"Predicting for Next GW (GW {latest_gw}) based on training over all previous weeks.")

optimizer_used = False
optimizer_infeasible = False

# Try ILP (if enabled and PuLP present). Otherwise, use greedy
if use_optimizer and PULP_AVAILABLE:
    result = optimize_full(
        pool_df, pred_col, budget_tenths, POSITION_QUOTAS,
        min_start_minutes, max_per_team,
        min_spend_pct=min_spend_pct, lambda_spend=lambda_spend
    )
    if result is None:
        optimizer_infeasible = True
        squad_df, xi_df, bench_df, captain, vice = greedy_squad_and_xi(
            pool_df, pred_col, budget_tenths, POSITION_QUOTAS, min_start_minutes, max_per_team
        )
    else:
        optimizer_used = True
        squad_df, xi_df, bench_df, captain, vice = result
else:
    if use_optimizer and not PULP_AVAILABLE:
        st.warning("PuLP not detected — using greedy fallback instead.")
    squad_df, xi_df, bench_df, captain, vice = greedy_squad_and_xi(
        pool_df, pred_col, budget_tenths, POSITION_QUOTAS, min_start_minutes, max_per_team
    )

# Helpful warning when ILP couldn't find a solution (tight constraints, missing data, etc.).
if optimizer_infeasible:
    st.warning("Optimizer infeasible — used greedy fallback.")

# ====================================================
# Display: Squad / XI / Bench / Totals
# ====================================================
if not isinstance(squad_df, pd.DataFrame) or squad_df.empty:
    st.warning("No feasible squad found with the current constraints.")
else:
    def _decorate_cost(df_in: pd.DataFrame, pred_col_name: str) -> pd.DataFrame:
        """
        Keep only relevant columns and convert value to '£x.xM' for readability
        """
        out = df_in[["name","team","position","value", pred_col_name]].copy()
        out.rename(columns={pred_col_name: "predicted_points"}, inplace=True)
        out["cost (M)"] = out["value"].apply(format_millions_tenth)
        out = out.drop(columns=["value"])
        return out

    # Full 15-man squad.
    st.markdown("#### Full Squad (15)")
    squad_display = _decorate_cost(squad_df, pred_col).sort_values(["position", "team", "name"]).reset_index(drop=True)
    st.dataframe(squad_display, use_container_width=True, hide_index=True)

    # Starting XI + Captain/Vice.
    if isinstance(xi_df, pd.DataFrame) and not xi_df.empty:
        st.markdown("#### Starting XI")
        xi_display = _decorate_cost(xi_df, pred_col).sort_values(["position", "team", "name"]).reset_index(drop=True)
        st.dataframe(xi_display, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Captain:** {captain if captain else '—'}")
        with col2:
            st.markdown(f"**Vice-Captain:** {vice if vice else '—'}")
    else:
        st.warning("Could not build a valid Starting XI under the current constraints.")

    # Bench ordering explanation shown in section header.
    if isinstance(bench_df, pd.DataFrame) and not bench_df.empty:
        st.markdown("#### Bench (Outfield by low points to GK last)")
        bench_display = _decorate_cost(bench_df, pred_col).reset_index(drop=True)
        st.dataframe(bench_display, use_container_width=True, hide_index=True)

    # Headline totals (cost & projected XI points with captain doubled)
    total_cost = (squad_df["value"] / 10.0).sum()   # in millions
    xi_points  = 0.0
    if isinstance(xi_df, pd.DataFrame) and not xi_df.empty:
        xi_points = xi_df[pred_col].sum()
        # Captain already included once; add again to simulate doubling
        if captain and (xi_df["name"] == captain).any():
            xi_points += float(xi_df.loc[xi_df["name"] == captain, pred_col].iloc[0])

    st.markdown(f"**Total Squad Cost:** {format_millions_tenth(10*total_cost)}")
    st.markdown(f"**Projected XI Points (captain doubled):** {xi_points:.2f}")

# ===========================
# Diagnostics & Comparisons
# ===========================
st.markdown("## Model Diagnostics & Comparisons")

TAB_TITLES = [
    "Performance Tables & Radar",
    "Cross-Validation (by GW)",
    "Per-Position Metrics",
    "Residuals & Calibration",
    "Feature Importance / Coefficients",
    "Next GW: Top Picks"
]
(tab_perf, tab_cv, tab_pos, tab_err, tab_feat, tab_next) = st.tabs(TAB_TITLES)

with tab_perf:
    # 1) In-sample metrics table for fast comparison
    st.markdown("### In-Sample Performance (Training Data)")
    st.dataframe(
        performance_df.style.format(METRIC_FORMATS),
        use_container_width=True, hide_index=True
    )

    # 2) Radar chart (metrics normalized to [0,1]) for visual comparison
    st.markdown("### Radar Chart (Normalized Metrics)")
    radar_df = normalize_for_radar(performance_df)
    metrics_for_radar = ["MAE","RMSE","R²","ExplainedVar","MedAE","Spearman","TopKOverlap"]

    radar_fig = go.Figure()
    for _, row in radar_df.iterrows():
        radar_fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics_for_radar],
            theta=metrics_for_radar,
            fill='toself',
            name=row["Model"]
        ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=True,
        height=550,
        title="Normalized (higher is better across all axes)"
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    # 3) Average predicted points per model for the next GW
    st.markdown(f"### Predicted Points for Next Gameweek (GW {latest_gw})")
    avg_preds = {m: test_df[pred_key_map[m]].mean() for m in pred_key_map}
    avg_pred_df = pd.DataFrame(list(avg_preds.items()), columns=["Model", "Predicted Points"])
    fig1 = px.bar(avg_pred_df, x="Model", y="Predicted Points", color="Model",
                  title=f"Predicted Points — GW {latest_gw}")
    st.plotly_chart(fig1, use_container_width=True)

    # 4) Training scatter (Actual vs Predicted) across all models
    st.markdown("### Actual vs Predicted Points (Training Data)")
    lengths = [len(np.asarray(y_train_true))] + [len(np.asarray(v)) for v in train_predictions.values()]
    min_n = min(lengths)
    actual_trim = np.asarray(y_train_true)[:min_n]
    scatter_df = pd.DataFrame({"Actual": actual_trim})
    for m, yhat in train_predictions.items():
        scatter_df[m] = np.asarray(yhat)[:min_n]

    scatter_fig = go.Figure()
    for m in [c for c in scatter_df.columns if c != "Actual"]:
        scatter_fig.add_trace(go.Scatter(x=scatter_df["Actual"], y=scatter_df[m], mode="markers", name=m))
    # Ideal diagonal line
    scatter_fig.add_trace(go.Scatter(
        x=[scatter_df["Actual"].min(), scatter_df["Actual"].max()],
        y=[scatter_df["Actual"].min(), scatter_df["Actual"].max()],
        mode="lines", name="Ideal Line", line=dict(dash="dash")
    ))
    scatter_fig.update_layout(
        xaxis_title="Actual Points",
        yaxis_title="Predicted Points",
        height=450, width=700,
        title="Training: Actual vs Predicted"
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    # 5) Side-by-side bar chart of all metrics
    st.markdown("### Model Error Comparison")
    melted_df = performance_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig2 = px.bar(melted_df, x="Model", y="Score", color="Metric", barmode="group",
                  title="Model Comparison: MAE, RMSE, R², EV, MedAE, Spearman, TopKOverlap")
    st.plotly_chart(fig2, use_container_width=True)

with tab_cv:
    # Slower: cross-validated performance by GW groups
    if run_cv:
        st.info("Running GroupKFold CV by GW…")
        cv_overall_df, cv_by_pos_df = evaluate_cv(
            train_df, feature_cols, target_col, models_dict,
            n_splits=cv_splits, k_for_overlap=k_for_overlap
        )

        st.markdown("### CV Overall (Grouped by GW)")
        st.dataframe(
            cv_overall_df.style.format(METRIC_FORMATS),
            use_container_width=True, hide_index=True
        )

        st.markdown("### CV Metric: RMSE by Model")
        fig_rmse = px.bar(cv_overall_df, x="Model", y="RMSE", color="Model", title="CV RMSE (lower is better)")
        st.plotly_chart(fig_rmse, use_container_width=True)

        st.markdown("### CV R² by Model")
        fig_r2 = px.bar(cv_overall_df, x="Model", y="R²", color="Model", title="CV R² (higher is better)")
        st.plotly_chart(fig_r2, use_container_width=True)
    else:
        st.warning("Enable 'Run GroupKFold CV (by GW)' in the sidebar to compute cross-validated metrics.")

with tab_pos:
    # In-sample per-position metrics to see where models excel/struggle
    st.markdown("### In-Sample Per-Position Metrics")
    lengths = [len(np.asarray(y_train_true))] + [len(np.asarray(v)) for v in train_predictions.values()]
    min_n = min(lengths)
    y_true_trim = np.asarray(y_train_true)[:min_n]
    pos_series = train_df.get("position", pd.Series(["UNK"]*len(train_df)))
    pos_trim = np.asarray(pos_series)[:min_n]

    perpos_rows = []
    for m, yhat in train_predictions.items():
        yhat_trim = np.asarray(yhat)[:min_n]
        for p in sorted(pd.unique(pos_trim)):
            mask = (pos_trim == p)
            if mask.sum() == 0:
                continue
            met = _metrics_block(y_true_trim[mask], yhat_trim[mask], k_for_overlap=k_for_overlap)
            perpos_rows.append({"Model": m, "Position": p, **met})
    perpos_df = pd.DataFrame(perpos_rows)

    fig_pos_mae = px.bar(perpos_df, x="Position", y="MAE", color="Model", barmode="group",
                         title="Per-Position MAE (Training)")
    st.plotly_chart(fig_pos_mae, use_container_width=True)

    fig_pos_rmse = px.bar(perpos_df, x="Position", y="RMSE", color="Model", barmode="group",
                          title="Per-Position RMSE (Training)")
    st.plotly_chart(fig_pos_rmse, use_container_width=True)

    st.markdown("#### Per-Position Full Table (Training)")
    st.dataframe(
        perpos_df.sort_values(["Position","RMSE"]).reset_index(drop=True).style.format(METRIC_FORMATS),
        use_container_width=True, hide_index=True
    )

with tab_err:
    # Residual diagnostics (bias/variance patterns) + calibration
    st.markdown("### Residual Analysis (Training)")
    yhat_sel = train_predictions[model]
    y_true_aligned, yhat_aligned = _align_arrays(y_train_true, yhat_sel)
    residuals = y_true_aligned - yhat_aligned

    # Histogram of residuals
    fig_hist = px.histogram(x=residuals, nbins=40, title=f"Residuals Histogram — {model}")
    fig_hist.update_layout(xaxis_title="Residual (Actual − Predicted)", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Calibration: bin by predicted score and compare mean predicted vs mean actual
    st.markdown("### Calibration (Binned Actual vs Predicted)")
    q = min(10, max(3, int(len(yhat_aligned)//100)))  # target ~100 samples/bin
    q = max(q, 3)
    bins = pd.qcut(pd.Series(yhat_aligned), q=q, duplicates="drop")
    cal_df = pd.DataFrame({"pred": yhat_aligned, "act": y_true_aligned, "bin": bins}).groupby("bin").agg(
        mean_pred=("pred","mean"), mean_act=("act","mean"), n=("act","size")
    ).reset_index(drop=True)

    cal_fig = go.Figure()
    cal_fig.add_trace(go.Scatter(x=cal_df["mean_pred"], y=cal_df["mean_act"],
                                 mode="markers+lines", name="Binned means"))
    both_min = float(min(cal_df["mean_pred"].min(), cal_df["mean_act"].min()))
    both_max = float(max(cal_df["mean_pred"].max(), cal_df["mean_act"].max()))
    cal_fig.add_trace(go.Scatter(x=[both_min, both_max], y=[both_min, both_max],
                                 mode="lines", name="Ideal", line=dict(dash="dash")))
    cal_fig.update_layout(xaxis_title="Mean Predicted", yaxis_title="Mean Actual",
                          title=f"Calibration Plot — {model}")
    st.plotly_chart(cal_fig, use_container_width=True)

with tab_feat:
    # Feature importances (trees) and coefficients (linear) for interpretability
    if show_feature_importance:
        st.markdown("### Feature Importances / Coefficients")
        for mname in ["Random Forest","XGBoost","Gradient Boosting"]:
            if mname in models_dict:
                m = models_dict[mname]
                try:
                    importances = getattr(m, "feature_importances_", None)
                    if importances is not None:
                        imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}) \
                                   .sort_values("Importance", ascending=False)
                        fig_imp = px.bar(imp_df, x="Feature", y="Importance", title=f"{mname} — Feature Importance")
                        st.plotly_chart(fig_imp, use_container_width=True)
                        st.dataframe(imp_df.reset_index(drop=True), use_container_width=True, hide_index=True)
                except Exception:
                    # Not all models expose importances
                    pass
        if "Linear Regression" in models_dict:
            m = models_dict["Linear Regression"]
            try:
                coefs = pd.DataFrame({
                    "Feature": feature_cols,
                    "Coefficient": m.coef_,
                    "AbsCoef": np.abs(m.coef_)
                }).sort_values("AbsCoef", ascending=False)
                fig_coef = px.bar(coefs, x="Feature", y="Coefficient", title="Linear Regression — Coefficients")
                st.plotly_chart(fig_coef, use_container_width=True)
                st.dataframe(coefs.drop(columns=["AbsCoef"]).reset_index(drop=True), use_container_width=True, hide_index=True)
            except Exception:
                pass
    else:
        st.info("Enable 'Show feature importances / coefficients' in the sidebar to view model explanations.")

with tab_next:
    # Top 20 players for the next GW by the selected model
    st.markdown(f"### Next GW — Top 20 by {model} (GW {latest_gw})")
    top20 = pool_df.sort_values(pred_col, ascending=False).head(20)
    tbl = top20[["name","team","position","value",pred_col]].copy()
    tbl.rename(columns={pred_col:"predicted_points"}, inplace=True)
    tbl["cost (M)"] = tbl["value"].apply(format_millions_tenth)
    tbl = tbl.drop(columns=["value"]).reset_index(drop=True)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

# ====================================================
# END
# ====================================================
