"""
Feature Engineering Script: Global Calculations
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
import calendar
from datetime import date, timedelta
from typing import Dict

# Feature engineering pipeline to prepare X-df and y-series for model


def nth_weekday(year, month, weekday, n):
    first_dow, days_in_month = calendar.monthrange(year, month)
    day = 1 + ((weekday - first_dow) % 7)
    return datetime.date(year, month, day + (n-1)*7)

def last_weekday(year, month, weekday):
    first_dow, days_in_month = calendar.monthrange(year, month)
    last_dow = (first_dow + days_in_month - 1) % 7
    offset = (last_dow - weekday) % 7
    return datetime.date(year, month, days_in_month - offset)

def easter_date(year):
    # Anonymous Gregorian algorithm
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day   = ((h + l - 7*m + 114) % 31) + 1
    return datetime.date(year, month, day)

def get_holiday(d: datetime.date) -> str | None:
    """Return the holiday name for date d, or None if not a holiday."""
    y, m, day = d.year, d.month, d.day

    # Fixed‐date
    fixed = {
        (1,  1): "New Year's Day",
        (6, 19): "Juneteenth",
        (7,  4): "Independence Day",
        (11,11): "Veterans Day",
        (12,25): "Christmas Day",
    }
    if (m, day) in fixed:
        return fixed[(m, day)]

    # Inauguration Day every 4 yrs starting 2017 on Jan 20
    if m == 1 and day == 20 and (y - 2017) % 4 == 0 and y >= 2017:
        return "Inauguration Day"

    # Weekday‐based
    if d == nth_weekday(y, 1, 0, 3):
        return "Martin Luther King Jr. Day"
    if d == nth_weekday(y, 2, 0, 3):
        return "Washington's Birthday"
    if d == last_weekday(y, 5, 0):
        return "Memorial Day"
    if d == nth_weekday(y, 9, 0, 1):
        return "Labor Day"
    if d == nth_weekday(y,10,0,2):
        return "Columbus Day"
    if d == nth_weekday(y,11,3,4):
        return "Thanksgiving Day"

    # Movable feast
    if d == easter_date(y):
        return "Easter"

    # Fun extra
    if m == 10 and day == 31:
        return "Halloween"

    return None


def load_hourly_data(path: str):
    """
    Loads your base hourly counts.
    - Reads the Parquet file at `path` (columns: ['date', 'hour', 'total'])
    - Renames 'total' to 'hourly_count'
    - Adds 'datetime' column for merging with weather
    - Returns a DataFrame with ['datetime', 'hour', 'hourly_count']
    """
    df = pd.read_parquet(path)
    # Rename total → hourly_count (from your data screenshot)
    if 'total' in df.columns:
        df = df.rename(columns={'total': 'hourly_count'})
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    df['hour'] = df['hour'].astype(int)
    df['hourly_count'] = df['hourly_count'].astype(int)
    return df[['datetime', 'hour', 'hourly_count']]


def add_basic_time_features(df):
    """
    Extract and add:
    - df['dow']   : day of week (0=Mon…6=Sun)
    - df['month'] : month number (1–12)
    - df['t']     : trend index (e.g. days since start)
    - Returns df.
    """

    df['dow'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year

    start = df['datetime'].min()
    df['day_index'] = (df['datetime'] - start).dt.days

    return df 

def add_cyclic_transforms(df):
    """
    Add cyclic sine/cosine transforms for:
    - hour (period=24) → 'hour_sin','hour_cos'
    - dow  (period=7)  → 'dow_sin','dow_cos'
    - month(period=12) → 'month_sin','month_cos'
    Returns df.
    """
    # hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # day of week
    df['dow_sin']  = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['dow'] / 7)

    # month
    df['month_sin'] = np.sin(2 * np.pi * (df['month']-1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month']-1) / 12)

    return df

def add_weekend_flag(df):
    """
    Add binary flag:
    - df['is_weekend']: 1 if Saturday or Sunday, else 0.
    - Returns df.
    """

    df['is_weekend'] = df['dow'].isin({5, 6}).astype(int)
    return df

def near(d):
        return any(abs((d - hd).days) <= 1 for hd in holiday_dates)


def get_holidays_for_year(year: int) -> Dict[date, str]:
    """
    Returns a dict mapping every holiday date in `year` to its holiday name,
    based on your get_holiday() function.
    """
    hols: Dict[date, str] = {}
    # iterate through every day of the year
    d = date(year, 1, 1)
    end = date(year + 1, 1, 1)
    while d < end:
        name = get_holiday(d)
        if name:
            hols[d] = name
        d += timedelta(days=1)
    return hols

def add_holiday_flags(df):
    df = df.copy()

    # 1) Make sure we have a datetime64 column
    df['datetime'] = pd.to_datetime(df['datetime'])
    # 2) Pull out the year for the holiday lookup
    years = df['datetime'].dt.year.unique()
    
    # 3) Build holiday map
    holiday_map: Dict[date, str] = {}
    for y in years:
        holiday_map.update(get_holidays_for_year(int(y)))
    
    # 4) Create a pure date column for mapping
    df['date'] = df['datetime'].dt.date

    # 5) Map date → holiday name (or NaN)
    df['holiday'] = df['date'].map(holiday_map)

    # 6) Flags
    df['is_holiday']       = df['holiday'].notnull().astype(int)
    flagged_dates = set(holiday_map.keys())
    df['is_near_holiday']  = df['date'].apply(lambda d: any(abs((d - hd).days) <= 1 for hd in flagged_dates)).astype(int)

    df['holiday_code'] = df['holiday'].astype('category').cat.codes
    df = df.drop(columns=['holiday'])
    return df



def add_weather_features(df, weather):
    """
    Merge in weather data by 'datetime'.
    - weather: DataFrame with ['datetime','temperature_2m','precipitation']
    """
    df = df.merge(weather, on='datetime', how='left')
    df['temperature_in_celsius'] = df['temperature_2m']
    df['precipitation_in_mm'] = df['precipitation']
    return df


def add_trend_features(df):
    """
    Compute rolling/trend features:
    - Returns df.
    """
    # 1) daily total
    df['daily_total'] = df.groupby('date')['hourly_count'].transform('sum')

    # 2) build daily‐level table
    daily = (
        df[['date', 'day_index', 'daily_total']]
        .drop_duplicates(subset=['date'])
        .sort_values('day_index')
        .copy()
    )

    # 3) guard against zeros (or negatives) before log:
    #    replace with a small positive constant
    daily['daily_total'] = daily['daily_total'].clip(lower=1e-3)

    # 4) fit log‐linear model
    X = daily[['day_index']].values
    y_log = np.log(daily['daily_total'].values).reshape(-1, 1)

    lr = LinearRegression().fit(X, y_log)
    # intercept_ is ln(a), coef_[0] is b

    # 5) predict in log space and exponentiate
    log_pred = lr.predict(X).flatten()
    daily['trend'] = np.exp(log_pred)

    # 6) map back to hourly rows & clean up
    df['trend'] = df['date'].map(daily.set_index('date')['trend'])
    df.drop(columns=['daily_total'], inplace=True)
    return df

def add_work_rush_flag(df):
    """
    Add binary flag:
    - df['work_rush']: 1 if (Mon–Fri AND (7 ≤ hour < 10 OR 16 ≤ hour < 18)), else 0.
    """
    # weekday is dow 0–4
    is_weekday = df['dow'].between(0, 4)
    # morning rush: 7–9 (inclusive of 7, exclusive of 10)
    morning = df['hour'].between(7, 9)
    # evening rush: 16–17 (4–6 PM → 16 inclusive, 18 exclusive)
    evening = df['hour'].between(16, 17)
    df['work_rush'] = (is_weekday & (morning | evening)).astype(int)
    return df


def save_feature_dataset(df, path: str):
    """
    Save the fully-featurized DataFrame to a Parquet file at `path`.
    """
    df.to_parquet(path, index=False)

def load_weather_data(path: str):
    """
    Loads weather data, renames columns, and parses datetime.
    - Expects columns: ['time', 'temperature__', 'precipitation', ...]
    - Returns DataFrame with columns ['datetime', 'temperature_2m', 'precipitation']
    """
    weather = pd.read_parquet(path)
    weather = weather.rename(columns={
        'time': 'datetime',
        'temperature__': 'temperature_2m', 
    })
    weather['datetime'] = pd.to_datetime(weather['datetime'])
    return weather[['datetime', 'temperature_2m', 'precipitation']]



if __name__ == "__main__":
    # 1) Load hourly usage data (and fix columns)
    df = load_hourly_data("/Users/zoehightower/Downloads/Citi-Bike-Station-Forecaster/data/aggregated/hourly_all.parquet")

    # 2) Load and fix weather data
    weather = load_weather_data("/Users/zoehightower/Downloads/Citi-Bike-Station-Forecaster/data/raw/weather/hourly_all.parquet")

    # 3) Time-based features
    df = add_basic_time_features(df)
    df = add_weekend_flag(df)
    df = add_work_rush_flag(df)
    df = add_cyclic_transforms(df)

    # 4) Holidays & near-holidays
    df = add_holiday_flags(df)

    # 5) Merge weather
    df = add_weather_features(df, weather)

    # 6) Trend fit
    df = add_trend_features(df)

    # 7) Save result
    save_feature_dataset(df, "/Users/zoehightower/Downloads/Citi-Bike-Station-Forecaster/data/final_aggregated.parquet")
    print("✅ Feature dataset saved to /Users/zoehightower/Downloads/Citi-Bike-Station-Forecaster/data/final_aggregated.parquet")


    