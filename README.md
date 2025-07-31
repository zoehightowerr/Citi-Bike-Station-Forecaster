# CitiBike Station Predictor 🚴‍♂️

**Hourly demand forecasts for 1,000+ Citi Bike stations in NYC**  
A two-step machine-learning system combining 12+ years of bike usage history with live weather data.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org) [![LightGBM](https://img.shields.io/badge/LightGBM-Latest-brightgreen.svg)](https://lightgbm.readthedocs.io) [![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.0+-orange.svg)](https://spark.apache.org) [![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-red.svg)](https://scikit-learn.org)

---

## 🚀 Overview
CitiBike Station Predictor delivers precise hourly ridership forecasts for 1,000+ Citi Bike stations across NYC through a sophisticated two-stage machine learning architecture. The system combines 12+ years of historical ridership data with real-time weather intelligence to enable data-driven operational planning and resource optimization.

**Core Innovation**: Rather than training individual models per station, CitiBike Station Predictor uses a global demand predictor with intelligent station-level allocation, achieving superior accuracy while maintaining computational efficiency.

---

## 📊 Validity

| Metric         | Result      | Improvement vs. “always average” |
| -------------- | ----------- | ------------------------------- |
| **Mean Absolute Error** (MAE) | 339 rides     | 83% better                    |
| **Root MAE** (RMSE)   | 597 rides     | 78% better                    |
| **R² Score**    | 0.95 (95%)  | Near-perfect fit               |
| **CV MAE**      | 655 ± 180   | Consistent across folds        |

---

## 🏗️ Technical Approach

### 1. Data Aggregation Pipeline (`data_aggregation.py`)
- **Multi-format ingestion**: Processes heterogeneous Citi Bike CSV schemas (2013-2025)
- **Apache Spark processing**: Scalable hourly aggregation across 100M+ trip records
- **Station filtering**: 2025 active station whitelist with coordinate validation
- **Output**: Structured Parquet files for downstream processing

### 2. Weather Intelligence (`weather_collection.py`)
- **Historical coverage**: 12+ years of NYC weather data via Open-Meteo Archive API
- **Temporal alignment**: Hourly temperature and precipitation matching ridership data
- **Geographic precision**: NYC-specific coordinates (40.7128°N, 74.0060°W)

### 3. Feature Engineering Pipeline (`feature_engineering.py`)
#### Temporal Intelligence
- **Cyclical encoding** for natural periodicity:
  ```python
  hour_sin = sin(2π × hour / 24)
  dow_sin = sin(2π × day_of_week / 7)
  month_sin = sin(2π × (month-1) / 12)
  ```
#### Calendar Intelligence
- **Holiday detection**: 15+ US federal holidays with proximity effects
- **Rush hour recognition**: Weekday peaks (7-10 AM, 4-6 PM)
- **Weekend behavior**: Distinct temporal patterns

#### Trend Analysis
- **Log-linear fitting**: Exponential growth trend via regression
  ```python
  trend = exp(α + β × day_index)
  ```

### 4. Exploratory Data Analysis (`eda.ipynb`)
- **Temporal patterns**: Day-of-week and time-of-day usage distributions
- **Weather correlation**: Temperature and precipitation impact analysis
- **Holiday effects**: Federal holiday usage reduction quantification
- **Trend validation**: Daily trend vs. actual ridership comparison
- **Calendar insights**: Monthly and seasonal usage patterns
- **Rush hour analysis**: Work vs. non-work hour demand profiling

### 5. Demand Forecasting (`model_training.py`)
#### Stage 1: Global Prediction
- **LightGBM Regressor** with early stopping and cross-validation
- **Time series splits**: Prevents data leakage with temporal validation
- **Feature importance**: Automated selection of predictive variables

#### Stage 2: Station Allocation
- **Historical usage patterns**: Weekday-hour specific allocation weights
- **Proportional distribution**: 
  ```python
  station_forecast = total_forecast × historical_share[station][dow][hour]
  ```

### 6. Model Assessment (`assess_model.ipynb`)
- **Prediction accuracy**: Actual vs. predicted scatter plots and correlation analysis
- **Baseline comparison**: Performance against naive forecasting methods
- **Residual analysis**: Error distribution and systematic bias detection
- **Diagnostic plots**: Model validation through comprehensive visualization

---

### EDA Findings

- **Temporal patterns**: Clear weekday rush hour peaks at 8 AM and 6 PM.
- **Seasonal trends**: Summer demand peaks in June–August.
- **Weather sensitivity**: Usage drops during rain, and rides increase with temperature.
- **Holiday impact**: Lower usage on federal holidays.
- **Work vs. leisure**: Different demand profiles for work hours vs. non-work hours.

---

## 🛠️ Getting Started

1. **Install**  
   ```bash
   pip install pandas numpy scikit-learn lightgbm pyspark joblib requests
   ```
2. **Run the pipeline**  
   ```bash
   # Fetch weather data
   python weather_collection.py

   # Process bike trip files
   python data_aggregation.py

   # Build features
   python feature_engineering.py

   # Explore data
   jupyter notebook eda.ipynb

   # Train model
   python model_training.py

   # Check performance
   jupyter notebook assess_model.ipynb
   ```

---

## 📁 Project Layout

```
BikeFlow-Predictor/
├── data/
│   ├── raw/                   # CSVs and weather archives- CSCs are not uploaded, download citibike data to replicate code
│   ├── aggregated/            # Hourly Parquet files
│   └── final_features.parquet # Ready-to-use training data
├── model/
│   ├── global_model.pkl       # LightGBM model
│   └── station_weights.parquet# Station share tables
├── notebooks/
│   ├── eda.ipynb
│   └── assess_model.ipynb
└── src/
    ├── data_aggregation.py
    ├── weather_collection.py
    ├── feature_engineering.py
    └── model_training.py
```

---

## 🎯 Why It Matters

- **Balance staffing** by station and hour  
- **Move bikes** before demand spikes  
- **Plan maintenance** when docks are least used  
- **Drive strategic growth** with data-driven site planning

---

## 🔮 Next Steps

- Try deep-learning (LSTM/Transformer) for trends  
- Add live event and transit-delay feeds  
- Build a real-time API and dashboard  
- Set up automated retraining and monitoring

---
