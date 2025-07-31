import os
import sys
import shutil
import pandas as pd
from glob import glob
from typing import List
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DateType


# Column renaming map for all Citi Bike schemas
STARTTIME_KEYS    = ["started_at", "starttime", "Start Time"]
STATION_ID_KEYS   = ["start_station_id", "start station id", "Start Station ID"]
STATION_NAME_KEYS = ["start_station_name", "start station name", "Start Station Name"]
LAT_KEYS          = ["start_lat", "start station latitude", "Start Station Latitude"]
LNG_KEYS          = ["start_lng", "start station longitude", "Start Station Longitude"]


def load_citibike_data(filepath: str) -> pd.DataFrame:
    """
    Load a raw Citi Bike CSV file, standardize column names, and return a DataFrame
    """
    print(f"  ⏳ Loading raw CSV: {os.path.basename(filepath)}")
    df = pd.read_csv(
        filepath,
        dtype=str,
        na_values=["\\N", ""],
    )
    col_map = {}
    for logical, keys in [
        ("started_at",         STARTTIME_KEYS),
        ("start_station_id",   STATION_ID_KEYS),
        ("start_station_name", STATION_NAME_KEYS),
        ("start_lat",          LAT_KEYS),
        ("start_lng",          LNG_KEYS),
    ]:
        found = next((k for k in keys if k in df.columns), None)
        if not found:
            raise Exception(f"Missing required column for {logical}")
        col_map[found] = logical
    df = df.rename(columns=col_map)
    return df[list(col_map.values())]


def filter_and_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and format the raw DataFrame by extracting datetime components and filtering invalid rows.
    """
    df['datetime'] = pd.to_datetime(df['started_at'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df['date']    = df['datetime'].dt.date
    df['hour']    = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.day_name()
    df['year']    = df['datetime'].dt.year
    return df.drop(columns=['started_at'])


def aggregate_hourly_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a cleaned DataFrame to compute hourly usage counts per station.
    """
    return (
        df
        .groupby(
            ['date', 'hour',
             'start_station_id', 'start_station_name',
             'start_lat', 'start_lng',
             'weekday', 'year'],
            as_index=False
        )
        .size()
        .rename(columns={'size': 'hourly_count'})
    )


def spark_aggregations(tmp_hourly_all_dir: str, tmp_hourly_2025_dir: str, out_base: str):
    """
    Perform Spark-based aggregations:
      1. Compute hourly totals across all stations.
      2. Compute per-station weekday-hour pivoted totals for 2025 data.
    """
    import logging
    logging.getLogger("pyspark").setLevel(logging.ERROR)
    logging.getLogger("py4j").setLevel(logging.ERROR)
    
    spark = SparkSession.builder \
        .appName("CitiBikeAggregations") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.local.dir", "/tmp/spark-temp") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    # Read ALL stations data for hourly totals
    df_all = spark.read.parquet(tmp_hourly_all_dir)
    
    # Hourly all totals with date, hour, and total (includes ALL stations)
    hourly = (
        df_all.groupBy('date', 'hour')
              .agg(F.sum('hourly_count').alias('total'))
              .orderBy('date', 'hour')
    )
    hourly.coalesce(1).write.mode('overwrite') \
           .parquet(os.path.join(out_base, 'hourly_all.parquet'))

    df_2025 = spark.read.parquet(tmp_hourly_2025_dir)

    # Station totals with every hour of every weekday as separate columns (2025 stations only)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Add weekday_hour combination column for pivoting
    df_with_weekday_hour = df_2025.withColumn(
        'weekday_hour', 
        F.concat(F.col('weekday'), F.lit('_'), F.format_string('%02d', F.col('hour')))
    )
    
    # Create all possible weekday_hour combinations
    weekday_hour_combinations = [f'{day}_{hour:02d}' for day in weekday_order for hour in range(24)]
    
    station_base = (
        df_2025.groupBy('start_station_id', 'start_station_name', 'start_lat', 'start_lng')
               .agg(F.sum('hourly_count').alias('total'))
               .dropDuplicates(['start_station_id'])
    )
    
    station_weekday_hourly = (
        df_with_weekday_hour.groupBy('start_station_id')
                           .pivot('weekday_hour', weekday_hour_combinations)
                           .sum('hourly_count')
                           .na.fill(0)
    )
    
    station_totals = (
        station_base.join(station_weekday_hourly, 'start_station_id')
                   .orderBy('start_station_id')
    )
    station_totals.coalesce(1).write.mode('overwrite') \
                   .parquet(os.path.join(out_base, 'station_weekday_hourly_totals.parquet'))

    spark.stop()
    print("💾 Spark aggregations complete!")


def main(raw_base=None, out_base=None):
    if raw_base is None:
        raw_base = "/Users/zoehightower/Downloads/Citi-Bike-Station-Forecaster/data/raw/station/"
    if out_base is None:
        out_base = "/Users/zoehightower/Downloads/Citi-Bike-Station-Forecaster/data/aggregated/"
    print(f"🔍 Searching for CSV files in: {raw_base}")
    files = sorted(glob(os.path.join(raw_base, "**", "*.csv"), recursive=True))
    print(f"📁 Found {len(files)} CSV files under {raw_base}\n")
    if not files:
        print("❗ No CSV files found."); return

    # build station whitelist from 2025 data
    station_ids_2025 = set()
    for fp in files:
        if os.path.basename(fp).startswith('2025'):
            try:
                df20 = load_citibike_data(fp)
                # Only add stations that have valid lat/lng
                df20_clean = df20.dropna(subset=['start_lat', 'start_lng'])
                df20_clean = df20_clean[
                    (df20_clean['start_lat'] != '') & 
                    (df20_clean['start_lng'] != '')
                ]
                station_ids_2025.update(df20_clean['start_station_id'])
            except Exception as e:
                print(f"    ⚠️ Skipping 2025 whitelist file {os.path.basename(fp)}: {e}")
    print(f"✅ Whitelisted {len(station_ids_2025)} station IDs from 2025 (with valid coordinates)")

    # Create separate temp directories for all stations vs 2025-only
    tmp_hourly_all_dir = os.path.join(out_base, "tmp_hourly_all")
    tmp_hourly_2025_dir = os.path.join(out_base, "tmp_hourly_2025")
    os.makedirs(tmp_hourly_all_dir, exist_ok=True)
    os.makedirs(tmp_hourly_2025_dir, exist_ok=True)

    total_rows_all, total_rows_2025, success = 0, 0, 0
    for i, fp in enumerate(files, start=1):
        print(f"—— [{i}/{len(files)}] Processing {os.path.basename(fp)}")
        try:
            raw   = load_citibike_data(fp)
            clean = filter_and_format(raw)
            if clean.empty:
                print("    ⚠️ No valid rows after cleaning, skipping."); continue

            hourly_df_all = aggregate_hourly_usage(clean)
            
            # Save ALL stations data (for hourly totals)
            if not hourly_df_all.empty:
                out_fp_all = os.path.join(tmp_hourly_all_dir, f"hourly_all_{i:04d}.parquet")
                hourly_df_all.to_parquet(out_fp_all, index=False)
                total_rows_all += len(hourly_df_all)
                print(f"    ➤ Wrote ALL stations: {out_fp_all} ({len(hourly_df_all)} rows)")
            
            # Filter to 2025 whitelisted stations (for station analysis)
            hourly_df_2025 = hourly_df_all[
                hourly_df_all['start_station_id'].isin(station_ids_2025) &
                hourly_df_all['start_lat'].notna() & 
                hourly_df_all['start_lng'].notna() &
                (hourly_df_all['start_lat'] != '') & 
                (hourly_df_all['start_lng'] != '')
            ]
            
            if not hourly_df_2025.empty:
                out_fp_2025 = os.path.join(tmp_hourly_2025_dir, f"hourly_2025_{i:04d}.parquet")
                hourly_df_2025.to_parquet(out_fp_2025, index=False)
                total_rows_2025 += len(hourly_df_2025)
                print(f"    ➤ Wrote 2025 stations: {out_fp_2025} ({len(hourly_df_2025)} rows)")
            
            if len(hourly_df_2025) < len(hourly_df_all):
                print(f"    📊 Kept {len(hourly_df_2025)}/{len(hourly_df_all)} rows for 2025 station analysis")
            
            success += 1
            print()
        except Exception as e:
            print(f"    ⚠️ Skipping file: {e}\n")

    print(f"📈 Processed {success}/{len(files)} files")
    print(f"   • {total_rows_all} total hourly rows (all stations)")
    print(f"   • {total_rows_2025} total hourly rows (2025 stations only)")
    
    if total_rows_all == 0:
        print("❗ No data processed successfully—check your CSVs and column names.")
        return

    print("⏳ Running Spark aggregations…")
    spark_aggregations(tmp_hourly_all_dir, tmp_hourly_2025_dir, out_base)

    print(f"🧹 Removing temporary files")
    shutil.rmtree(tmp_hourly_all_dir)
    shutil.rmtree(tmp_hourly_2025_dir)
    print(f"\n✅ All aggregated Parquets written to: {out_base}")


if __name__ == "__main__":
    main()