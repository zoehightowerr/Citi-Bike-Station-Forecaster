import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.dummy import DummyRegressor
import numpy as np

def load_final_dataset(path: str) -> pd.DataFrame:
    """
    Load the final aggregated dataset from a parquet file.
    """
    df = pd.read_parquet(path)
    return df

def build_feature_matrix(df: pd.DataFrame):
    """
    Split the DataFrame into features and target for model training.
    """
    y = df['hourly_count'].copy()
    X = df.drop(columns=['hourly_count', 'datetime', 'date', 'year'])
    return X, y

def train_lightgbm(X: pd.DataFrame, y: pd.Series):
    """
    Train a LightGBM regressor, evaluate on a hold-out test set, and compute various performance metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMRegressor()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=0) # disables logging
        ]
    )
    y_pred = model.predict(X_test)
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    y_base = baseline.predict(X_test)

    tscv = TimeSeriesSplit(n_splits=5, test_size=168*4)
    cv_maes = -cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
    y_pred_all = model.predict(X)

    errors     = y - y_pred_all
    abs_errors = errors.abs()
    mean_err     = errors.mean()
    mean_abs_err = abs_errors.mean()
    rmse_all     = np.sqrt((errors**2).mean())

    metrics = {
        "test_mae": mean_absolute_error(y_test, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "test_r2": r2_score(y_test, y_pred),
        "baseline_mae": mean_absolute_error(y_test, y_base),
        "baseline_rmse": np.sqrt(mean_squared_error(y_test, y_base)),
        "baseline_r2": r2_score(y_test, y_base),
        "cv_mae_mean": cv_maes.mean(),
        "cv_mae_std": cv_maes.std(),
        "fullset_mae": mean_abs_err,
        "fullset_rmse": rmse_all,
        "fullset_bias": mean_err,
    }
    return model, X_train, X_test, y_train, y_test, metrics


def save_model(model: lgb.LGBMRegressor, path: str) -> None:
    joblib.dump(model, path)

def compute_hourly_shares_from_wide(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each station's share of rides per weekday/hour.
    Returns long dataframe: ['station_name', 'dow', 'hour', 'share']
    """
    df = df_wide.copy()
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    result = []
    for day in weekday_names:
        for hour in range(24):
            col = f"{day}_{hour:02d}"
            total_this_hour = df[col].sum()
            # To avoid division by zero
            if total_this_hour == 0:
                shares = np.zeros(len(df))
            else:
                shares = df[col] / total_this_hour
            for idx, station_name in enumerate(df['start_station_name']):
                result.append({
                    'station_name': station_name,
                    'dow': day.lower(),
                    'hour': hour,
                    'share': shares.iloc[idx]
                })
    shares_long = pd.DataFrame(result)
    return shares_long


def allocate_by_hourly_share(
    df_totals: pd.DataFrame,       
    y_tot_pred: np.ndarray,
    shares: pd.DataFrame           
) -> pd.DataFrame:
    """
    Distribute predicted total rides to individual stations based on hourly share percentages.
    """
    d = df_totals.copy()
    d['dow'] = d['datetime'].dt.day_name().str.lower()
    d['hour'] = d['datetime'].dt.hour
    d['y_tot_pred'] = y_tot_pred
    stations = shares[['station_name']].drop_duplicates()
    d = (
        d.assign(key=1)
        .merge(stations.assign(key=1), on='key')
        .drop(columns='key')
    )
    # Merge on all three: station, dow, hour
    d = d.merge(shares, on=['station_name','dow','hour'], how='left')
    d['station_pred'] = d['y_tot_pred'] * d['share']
    return d[['station_name','datetime','station_pred']]


def train_pipeline():
    """
    Execute the end-to-end training pipeline:
      1. Load data and station totals.
      2. Compute hourly shares.
      3. Train LightGBM model and print metrics.
      4. Allocate station-level forecasts.
      5. Save model and shares to disk.
    """
    total_path = "/Users/zoehightower/Downloads/Citi-Bike-Station-Forecaster/data/final_aggregated.parquet"
    model_path = "/Users/zoehightower/Downloads/Citi-Bike-Station-Forecaster/model/bike_forecasting_model.pkl"
    station_path = "/Users/zoehightower/Downloads/Citi-Bike-Station-Forecaster/data/aggregated/station_weekday_hourly_totals.parquet"

    df_totals = load_final_dataset(total_path)
    df_station = pd.read_parquet(station_path)
    shares = compute_hourly_shares_from_wide(df_station)
    X, y = build_feature_matrix(df_totals)
    model, X_train, X_test, y_train, y_test, train_metrics = train_lightgbm(X, y)
    
    print("------ Model Train Metrics ------")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("---------------------------------")

    test_feats   = df_totals.loc[X_test.index, ['datetime']].reset_index(drop=True)
    y_tot_pred   = model.predict(X_test)
    station_preds = allocate_by_hourly_share(test_feats, y_tot_pred, shares)
    save_model(model, model_path)
    shares.to_parquet(model_path.replace(".pkl","_shares.parquet"))
    return model, X_train, X_test, y_train, y_test, train_metrics, df_totals, station_preds
