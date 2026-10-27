# 🚲 Bike Sharing Demand Prediction

An end-to-end Data Science and Machine Learning project for forecasting
hourly bike rental demand using historical usage patterns, calendar
information, weather conditions, and domain-driven features.

## 🎯 Business Objective

Develop a predictive model that accurately forecasts hourly bike rental
demand based on historical usage patterns and external factors such as
weather, season, time, holidays, and working days.

Accurate forecasts can help operators optimize bike availability, reduce
waiting time, improve operational planning, support fleet allocation,
and enable future data-driven pricing and marketing strategies.

> **Important:** The dataset does not contain pricing or revenue
> variables, so pricing optimization is a potential downstream
> application rather than a directly demonstrated capability.

## 🧩 Machine Learning Problem

-   **Type:** Supervised learning --- regression
-   **Target:** `cnt` --- total hourly bike rentals
-   **Granularity:** Hourly
-   **Period:** 2011--2012

## 📊 Dataset

`Dataset.csv` contains **17,379 rows and 17 columns**.

Main variables include:

`dteday`, `yr`, `mnth`, `hr`, `weekday`, `holiday`, `workingday`,
`season`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`, `casual`,
`registered`, and `cnt`.

### Data quality

-   0 duplicate rows
-   48 initial `?` values
-   Numeric columns converted to appropriate numeric types
-   Dates converted to datetime
-   `springer` standardized to `spring`
-   Missing values handled with appropriate imputation
-   Range and outlier checks performed

### Target leakage prevention

The project verified that:

`cnt = casual + registered`

There were **0 inconsistencies among complete rows**.

Therefore, `casual` and `registered` were excluded from the ML
predictors because they directly compose the target and would cause
target leakage.

## 📈 Exploratory Data Analysis

Key findings:

-   Demand is right-skewed.
-   Approximately **04:00** is one of the lowest-demand periods.
-   **17:00** has the highest average hourly demand at approximately
    **461 rentals**.
-   **18:00** averages approximately **426 rentals**.
-   **08:00** averages approximately **359 rentals**.
-   Fall has the highest average hourly demand (\~236).
-   Spring has the lowest (\~111).
-   Clear weather has the highest average demand (\~205).
-   Heavy rain has the lowest observed average (\~74), but only 3
    heavy-rain observations exist.
-   Working-day average demand is approximately 193 versus 181 on
    non-working days.

### Year-over-year growth

  Year     Total Rentals   Average Hourly Demand
  ------ --------------- -----------------------
  2011         1,243,103                 \~143.8
  2012         2,049,576                 \~234.7

Total rentals increased by approximately **64.9%**.

## 🔬 Correlation Analysis

Correlation with `cnt`, excluding leakage variables:

  Feature         Correlation
  ------------- -------------
  `temp`               +0.405
  `atemp`              +0.401
  `hr`                 +0.394
  `hum`                -0.323
  `windspeed`          +0.093

Temperature and hour have moderate positive relationships with demand,
while humidity has a moderate negative relationship.

## ⏱️ Time-Series Analysis

The project analyzed:

-   Hourly trends
-   Daily trends
-   Weekly patterns
-   Monthly trends
-   Monthly seasonality by year
-   Hourly seasonality
-   Yearly totals
-   Rolling averages
-   Seasonal demand patterns

Recurring peaks occur around **08:00 and 17:00**.

There are **76 calendar days with fewer than 24 observations**, so daily
aggregation must account for incomplete days.

## 🛠️ Feature Engineering

### Time features

-   `year`
-   `day`
-   `dayofyear`
-   `weekofyear`
-   `time_of_day`

### Demand-pattern features

-   `is_rush_hour`
-   `is_weekend`
-   `is_peak_hour`
-   `high_demand_hour`

### Interaction and nonlinear features

-   `temp_humidity_interaction`
-   `temp_squared`
-   `hum_squared`
-   `season_weather_interaction`

### Domain-driven features

-   `is_commute_period`
-   `morning_commute`
-   `evening_commute`
-   `business_hours`
-   `low_demand_period`
-   `favorable_weather`
-   `adverse_weather`
-   `comfortable_temperature`
-   `high_humidity`
-   `weather_comfort_score`
-   `workingday_commute`
-   `weekend_daytime`

## 🔢 Encoding & Scaling

The project evaluated:

-   Label Encoding
-   Ordinal Encoding
-   One-Hot Encoding
-   Binary Encoding
-   Frequency Encoding
-   Target Encoding

One-Hot Encoding was preferred for nominal categorical variables.

Scaling techniques evaluated included StandardScaler, MinMaxScaler,
RobustScaler, MaxAbsScaler, and Normalizer.

> Preprocessors must be fitted only on training data in production
> pipelines.

## 🤖 Algorithms Evaluated

### Linear / robust regression

1.  Linear Regression
2.  Ridge
3.  Lasso
4.  Elastic Net
5.  Huber
6.  Bayesian Ridge
7.  SGD Regression
8.  Passive Aggressive
9.  Polynomial Regression

### Instance / kernel methods

10. KNN Regression
11. SVR

### Tree-based models

12. Decision Tree
13. Random Forest
14. Extra Trees
15. Gradient Boosting
16. HistGradientBoosting
17. AdaBoost

### Neural network

18. MLP Regressor

### Optional boosting libraries

19. XGBoost
20. LightGBM
21. CatBoost

Optional models depend on library availability.

## 🧪 Train/Test Strategy

The data was sorted chronologically and split approximately:

-   **80% training**
-   **20% testing**

A random split was avoided because future observations must not
influence training when forecasting time-dependent demand.

Excluded predictors:

-   `cnt`
-   `casual`
-   `registered`
-   `instant`
-   raw date/datetime fields where replaced by engineered features

## 🔄 Cross-Validation

**TimeSeriesSplit** was used to preserve temporal ordering during
validation.

## ⚙️ Hyperparameter Tuning

The initial broad RandomizedSearchCV approach was computationally
expensive across all models.

The final strategy used:

-   Successive Halving Random Search
-   TimeSeriesSplit with 3 folds
-   Controlled candidate counts
-   Parallel search
-   Model-level parallelism control
-   Failure handling for incompatible configurations

This reduced unnecessary computation while still benchmarking the
complete model set.

## 📐 Model Evaluation

Metrics included:

-   MAE
-   MSE
-   RMSE
-   R²
-   MAPE
-   Explained Variance
-   Adjusted R²
-   RMSE gap
-   R² gap
-   Training and prediction time

RMSE and MAE are especially useful because they express prediction error
in bike-rental units.

MAPE is treated as supplementary because it can be unstable for very low
demand values.

## 🏆 Model Selection

Models were compared using:

-   Test RMSE
-   Test MAE
-   Test R²
-   MAPE
-   Train/test performance gaps
-   Generalization behavior
-   Computational cost

The preferred production model should provide strong test performance
with a small generalization gap rather than simply maximizing training
performance.

## 🚀 Deployment

The selected model was serialized and integrated into a Streamlit
application.

### Architecture

``` text
Historical Dataset
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Best Model
       ↓
Model Serialization
       ↓
Streamlit Application
       ↓
User Input
       ↓
Prediction
       ↓
Dashboard / Monitoring
```

### Streamlit dashboard

The application includes:

-   Executive Dashboard
-   Demand Prediction
-   Demand Analysis
-   Advanced Analytics
-   Model Monitoring
-   Model Information

Users can select date, hour, holiday, working day, season, weather,
temperature, feels-like temperature, humidity, and windspeed.

The application automatically derives engineered features before
prediction.

## ☁️ Cloud Deployment

The project was prepared for Streamlit Cloud deployment.

A deployment issue occurred because the serialized model required
XGBoost while the cloud environment did not have the package installed:

``` text
ModuleNotFoundError: No module named 'xgboost'
```

A scikit-learn version mismatch was also identified.

### Resolution

The deployment environment should reproduce the training environment as
closely as possible by:

-   Adding required ML libraries to `requirements.txt`
-   Pinning compatible versions
-   Using a compatible Python version
-   Redeploying after dependency installation

## 📁 Recommended Repository Structure

``` text
bike-sharing-demand-prediction/
│
├── README.md
├── Dataset.csv
├── requirements.txt
│
├── notebooks/
│   └── bike_sharing_demand_prediction.ipynb
│
├── models/
│   ├── bike_demand_model.pkl
│   └── model_metadata.json
│
├── app/
│   ├── app.py
│   ├── dashboard_data.csv
│   └── prediction_log.csv
│
├── reports/
│   └── model_comparison.csv
│
├── visualizations/
│   ├── target_distribution.png
│   ├── hourly_demand.png
│   ├── monthly_demand.png
│   ├── seasonal_demand.png
│   ├── weather_demand.png
│   ├── yearly_demand.png
│   └── correlation_heatmap.png
│
└── .gitignore
```

## 🧰 Technology Stack

  Area                Technology
  ------------------- -------------------------------
  Language            Python
  Data Processing     Pandas, NumPy
  Visualization       Matplotlib, Seaborn, Plotly
  Statistics          Statsmodels
  Machine Learning    Scikit-learn
  Gradient Boosting   XGBoost / LightGBM / CatBoost
  Notebook            Jupyter
  Deployment          Streamlit
  Version Control     Git / GitHub
  Serialization       Pickle

## 🔐 Best Practices

-   Chronological train/test split
-   Time-aware cross-validation
-   Target leakage prevention
-   Training-only preprocessing
-   Multiple evaluation metrics
-   Train/test generalization analysis
-   Hyperparameter tuning
-   Model serialization
-   Reproducible dependencies
-   Prediction logging
-   Model monitoring

## ⚠️ Limitations

-   Historical data covers only 2011--2012.
-   Demand behavior can change over time.
-   Some days contain fewer than 24 hourly observations.
-   Pricing/revenue data is unavailable.
-   Geographic station-level information is not available in the
    analyzed dataset.
-   Extreme-weather categories have limited observations.
-   Serialized models require compatible library versions.

## 🔮 Future Enhancements

-   Advanced XGBoost, LightGBM, and CatBoost optimization
-   SARIMA / Prophet / LSTM forecasting comparison
-   Station-level and geospatial demand forecasting when location data
    is available
-   Real-time weather integration
-   Real-time bike availability integration
-   Event and traffic data integration
-   Data drift and prediction drift monitoring
-   Automated retraining
-   REST API deployment
-   Cloud-based ML pipeline

## 💼 Business Impact

A production-ready solution can help operators:

-   Anticipate peak demand
-   Improve bike allocation
-   Reduce shortages during commuter peaks
-   Reduce idle capacity
-   Improve staffing and maintenance planning
-   Support targeted marketing
-   Provide a foundation for future dynamic pricing

## 🏁 Conclusion

This project demonstrates a complete end-to-end machine learning
workflow for bike rental demand forecasting.

The analysis shows that demand is strongly influenced by:

**Time of day + season + weather + temperature + working-day patterns +
historical demand trends**

The project progresses from raw-data analysis through feature
engineering, broad model benchmarking, time-aware validation,
hyperparameter tuning, evaluation, model selection, Streamlit
deployment, and monitoring.

## ⭐ Project Highlights

-   **17,379 hourly observations**
-   **17 original variables**
-   **0 duplicate records**
-   **0 target consistency violations**
-   **21 regression algorithms evaluated where libraries were
    available**
-   **Chronological 80/20 train/test strategy**
-   **TimeSeriesSplit cross-validation**
-   **Successive-halving hyperparameter optimization**
-   **Multiple regression metrics**
-   **Target leakage prevention**
-   **Domain-driven feature engineering**
-   **Interactive Streamlit dashboard**
-   **Cloud deployment preparation**
-   **Model monitoring architecture**

## 📄 License

This project is intended for educational, portfolio, and demonstration
purposes. Add an appropriate open-source license before public
distribution if required.
