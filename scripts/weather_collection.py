import os
import requests
import pandas as pd
from datetime import datetime, timedelta

LATITUDE      = 40.7128
LONGITUDE     = -74.0060
TIMEZONE      = "America/New_York"
OVERALL_START = datetime(2013, 6, 1)
OVERALL_END   = datetime(2025, 5, 31)
BASE_URL      = "https://archive-api.open-meteo.com/v1/archive"


def fetch_hourly(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly weather data for the specified date range from the Open-Meteo archive API.
    """
    params = {
        "latitude":   LATITUDE,
        "longitude":  LONGITUDE,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     "temperature_2m,precipitation,weathercode",
        "timezone":   TIMEZONE
    }
    resp = requests.get(BASE_URL, params=params)
    resp.raise_for_status()
    return pd.DataFrame(resp.json().get("hourly", {}))


def last_day_of_month(year: int, month: int) -> int:
    """
    Compute the last calendar day for a given month.
    """
    next_month = month % 12 + 1
    next_year  = year + (month // 12)
    first_next = datetime(next_year, next_month, 1)
    return (first_next - timedelta(days=1)).day


def main():
    """
    Loop through each month in the overall date range, fetch hourly weather data,
    concatenate monthly DataFrames, and write the full dataset to a Parquet file.
    """
    all_dfs = []
    current = OVERALL_START.replace(day=1)

    while current <= OVERALL_END:
        year, month = current.year, current.month

        # determine this month’s span
        start_dt = current
        last_day = last_day_of_month(year, month)
        end_dt   = datetime(year, month, last_day)
        if end_dt > OVERALL_END:
            end_dt = OVERALL_END

        start_str = start_dt.strftime("%Y-%m-%d")
        end_str   = end_dt.strftime("%Y-%m-%d")
        print(f"→ Fetching {year}-{month:02d} ({start_str} → {end_str})")

        # fetch and collect
        df = fetch_hourly(start_str, end_str)
        all_dfs.append(df)

        # advance to next month
        if month == 12:
            current = datetime(year + 1, 1, 1)
        else:
            current = datetime(year, month + 1, 1)

    # concatenate everything and write one Parquet
    print("→ Concatenating all months into one DataFrame...")
    full = pd.concat(all_dfs, ignore_index=True)

    out_path = "/Users/zoehightower/Downloads/Citi-Bike-Station-Forecaster/data/raw/weather/hourly_all.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    full.to_parquet(out_path, index=False)
    print(f"✅ Written {len(full)} rows to {out_path}!")


if __name__ == "__main__":
    main()

