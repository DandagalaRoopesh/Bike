
import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bike Demand Intelligence",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.2);
        text-align: center;
    }

    .dashboard-title {
        font-size: 2.4rem;
        font-weight: 700;
    }

    .dashboard-subtitle {
        font-size: 1rem;
        opacity: 0.7;
        margin-bottom: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "bike_demand_model.pkl"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "model_metadata.json"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "dashboard_data.csv"
)

LOG_PATH = os.path.join(
    BASE_DIR,
    "prediction_log.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

with open(
    MODEL_PATH,
    "rb"
) as file:

    model = pickle.load(file)


# ============================================================
# LOAD METADATA
# ============================================================

with open(
    METADATA_PATH,
    "r"
) as file:

    metadata = json.load(file)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

df["dteday"] = pd.to_datetime(
    df["dteday"],
    errors="coerce"
)


# ============================================================
# DATETIME
# ============================================================

if "datetime" not in df.columns:

    df["datetime"] = (
        df["dteday"]
        + pd.to_timedelta(
            df["hr"],
            unit="h"
        )
    )

else:

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )


df = df.sort_values(
    "datetime"
).reset_index(
    drop=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">'
    '🚲 Bike Demand Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Machine Learning powered bike rental demand '
    'prediction and analytics dashboard'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚲 Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Executive Dashboard",
        "Demand Prediction",
        "Demand Analysis",
        "Advanced Analytics",
        "Model Monitoring",
        "Model Information"
    ]
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.header(
        "📊 Executive Dashboard"
    )

    total_rentals = int(
        df["cnt"].sum()
    )

    average_demand = (
        df["cnt"].mean()
    )

    peak_hour = (
        df.groupby("hr")["cnt"]
        .mean()
        .idxmax()
    )

    peak_demand = (
        df.groupby("hr")["cnt"]
        .mean()
        .max()
    )

    best_season = (
        df.groupby("season")["cnt"]
        .mean()
        .idxmax()
    )


    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Rentals",
        f"{total_rentals:,}"
    )

    col2.metric(
        "Average Hourly Demand",
        f"{average_demand:.0f}"
    )

    col3.metric(
        "Peak Hour",
        f"{int(peak_hour):02d}:00"
    )

    col4.metric(
        "Peak Avg Demand",
        f"{peak_demand:.0f}"
    )

    col5.metric(
        "Top Season",
        str(best_season).title()
    )


    st.divider()


    # --------------------------------------------------------
    # YEARLY DEMAND
    # --------------------------------------------------------

    yearly = (
        df.groupby("yr")["cnt"]
        .sum()
        .reset_index()
    )

    yearly["Year"] = (
        yearly["yr"]
        .map({
            0: "2011",
            1: "2012"
        })
        .fillna(
            yearly["yr"].astype(str)
        )
    )


    fig = px.bar(
        yearly,
        x="Year",
        y="cnt",
        title="Total Bike Rentals by Year",
        labels={
            "cnt": "Total Rentals"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # HOURLY DEMAND
    # --------------------------------------------------------

    hourly = (
        df.groupby("hr")["cnt"]
        .mean()
        .reset_index()
    )

    fig_hour = px.line(
        hourly,
        x="hr",
        y="cnt",
        markers=True,
        title="Average Demand by Hour",
        labels={
            "hr": "Hour",
            "cnt": "Average Rentals"
        }
    )

    fig_hour.update_xaxes(
        dtick=1
    )

    col1.plotly_chart(
        fig_hour,
        use_container_width=True
    )


    # --------------------------------------------------------
    # SEASONAL DEMAND
    # --------------------------------------------------------

    seasonal = (
        df.groupby("season")["cnt"]
        .mean()
        .reset_index()
    )

    fig_season = px.bar(
        seasonal,
        x="season",
        y="cnt",
        title="Average Demand by Season",
        labels={
            "cnt": "Average Rentals",
            "season": "Season"
        }
    )

    col2.plotly_chart(
        fig_season,
        use_container_width=True
    )


# ============================================================
# DEMAND PREDICTION
# ============================================================

elif page == "Demand Prediction":

    st.header(
        "🔮 Bike Demand Prediction"
    )

    st.write(
        "Enter operational and weather conditions "
        "to predict hourly bike rental demand."
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        selected_date = st.date_input(
            "Date",
            value=pd.Timestamp(
                "2012-08-15"
            )
        )

        hour = st.slider(
            "Hour",
            0,
            23,
            17
        )

        holiday = st.selectbox(
            "Holiday",
            ["No", "Yes"]
        )

        workingday = st.selectbox(
            "Working Day",
            ["No", "Yes"]
        )


    with col2:

        season = st.selectbox(
            "Season",
            [
                "spring",
                "summer",
                "fall",
                "winter"
            ]
        )

        weather = st.selectbox(
            "Weather Situation",
            [1, 2, 3, 4]
        )

        temp = st.slider(
            "Temperature",
            0.0,
            1.0,
            0.50,
            0.01
        )

        atemp = st.slider(
            "Feels-like Temperature",
            0.0,
            1.0,
            0.50,
            0.01
        )


    with col3:

        humidity = st.slider(
            "Humidity",
            0.0,
            1.0,
            0.50,
            0.01
        )

        windspeed = st.slider(
            "Windspeed",
            0.0,
            1.0,
            0.20,
            0.01
        )


    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    weekday = selected_date.weekday()

    year = selected_date.year

    month = selected_date.month

    day = selected_date.day

    dayofyear = selected_date.timetuple().tm_yday

    weekofyear = selected_date.isocalendar().week


    # Dataset originally uses 0/1 for year.
    # 2011 = 0, 2012 = 1.
    yr = (
        0
        if year <= 2011
        else 1
    )


    # --------------------------------------------------------
    # ENGINEERED FEATURES
    # --------------------------------------------------------

    is_rush_hour = int(
        hour in [
            7, 8, 9,
            16, 17, 18, 19
        ]
    )

    is_weekend = int(
        weekday in [0, 6]
    )

    is_peak_hour = int(
        hour in [8, 17, 18]
    )

    temp_humidity_interaction = (
        temp * (1 - humidity)
    )

    temp_squared = temp ** 2

    hum_squared = humidity ** 2

    high_windspeed = int(
        windspeed > 0.20
    )

    is_commute_period = int(
        hour in [
            7, 8, 9,
            16, 17, 18, 19
        ]
    )

    morning_commute = int(
        hour in [7, 8, 9]
    )

    evening_commute = int(
        hour in [
            16, 17, 18, 19
        ]
    )

    business_hours = int(
        9 <= hour <= 17
    )

    low_demand_period = int(
        hour in [
            0, 1, 2, 3, 4, 5
        ]
    )

    favorable_weather = int(
        weather == 1
    )

    adverse_weather = int(
        weather in [3, 4]
    )

    comfortable_temperature = int(
        0.4 <= temp <= 0.7
    )

    high_humidity = int(
        humidity >= 0.70
    )

    weather_comfort_score = (
        temp * (1 - humidity)
    )

    workingday_commute = int(
        workingday == "Yes"
        and hour in [
            7, 8, 9,
            16, 17, 18, 19
        ]
    )

    weekend_daytime = int(
        weekday in [0, 6]
        and 9 <= hour <= 18
    )

    high_demand_hour = int(
        hour in [8, 17, 18]
    )


    # --------------------------------------------------------
    # TIME OF DAY
    # --------------------------------------------------------

    if hour <= 5:
        time_of_day = "Night"

    elif hour <= 11:
        time_of_day = "Morning"

    elif hour <= 16:
        time_of_day = "Afternoon"

    elif hour <= 20:
        time_of_day = "Evening"

    else:
        time_of_day = "Late Night"


    season_weather_interaction = (
        f"{season}_{weather}"
    )


    # --------------------------------------------------------
    # INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "yr": [yr],
        "mnth": [month],
        "hr": [hour],
        "weekday": [weekday],
        "holiday": [holiday],
        "workingday": [workingday],
        "season": [season],
        "weathersit": [weather],

        "temp": [temp],
        "atemp": [atemp],
        "hum": [humidity],
        "windspeed": [windspeed],

        "year": [year],
        "day": [day],
        "dayofyear": [dayofyear],
        "weekofyear": [weekofyear],

        "time_of_day": [time_of_day],

        "is_rush_hour": [
            is_rush_hour
        ],

        "is_weekend": [
            is_weekend
        ],

        "is_peak_hour": [
            is_peak_hour
        ],

        "temp_humidity_interaction": [
            temp_humidity_interaction
        ],

        "temp_squared": [
            temp_squared
        ],

        "hum_squared": [
            hum_squared
        ],

        "high_windspeed": [
            high_windspeed
        ],

        "is_commute_period": [
            is_commute_period
        ],

        "morning_commute": [
            morning_commute
        ],

        "evening_commute": [
            evening_commute
        ],

        "business_hours": [
            business_hours
        ],

        "low_demand_period": [
            low_demand_period
        ],

        "favorable_weather": [
            favorable_weather
        ],

        "adverse_weather": [
            adverse_weather
        ],

        "comfortable_temperature": [
            comfortable_temperature
        ],

        "high_humidity": [
            high_humidity
        ],

        "weather_comfort_score": [
            weather_comfort_score
        ],

        "workingday_commute": [
            workingday_commute
        ],

        "weekend_daytime": [
            weekend_daytime
        ],

        "high_demand_hour": [
            high_demand_hour
        ],

        "season_weather_interaction": [
            season_weather_interaction
        ]
    })


    # --------------------------------------------------------
    # ALIGN FEATURES
    # --------------------------------------------------------

    expected_features = (
        metadata["feature_columns"]
    )


    for column in expected_features:

        if column not in input_data.columns:

            input_data[column] = np.nan


    input_data = input_data[
        expected_features
    ]


    st.divider()


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🚀 Generate Prediction",
        type="primary",
        use_container_width=True
    ):

        prediction = model.predict(
            input_data
        )[0]

        prediction = max(
            0,
            prediction
        )

        prediction = round(
            prediction
        )


        # Demand classification

        if prediction < 100:

            demand_category = "Low Demand"

        elif prediction < 300:

            demand_category = "Moderate Demand"

        elif prediction < 500:

            demand_category = "High Demand"

        else:

            demand_category = "Very High Demand"


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader(
            "Prediction Result"
        )


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Predicted Rentals",
            f"{prediction:,}"
        )

        col2.metric(
            "Demand Category",
            demand_category
        )

        col3.metric(
            "Prediction Hour",
            f"{hour:02d}:00"
        )


        # ----------------------------------------------------
        # GAUGE
        # ----------------------------------------------------

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={
                    "text":
                    "Predicted Hourly Demand"
                },
                gauge={
                    "axis": {
                        "range": [
                            0,
                            max(
                                1000,
                                prediction + 100
                            )
                        ]
                    }
                }
            )
        )


        st.plotly_chart(
            gauge,
            use_container_width=True
        )


        # ----------------------------------------------------
        # SAVE PREDICTION
        # ----------------------------------------------------

        new_record = pd.DataFrame({

            "prediction_timestamp": [
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ],

            "date": [
                str(selected_date)
            ],

            "hour": [
                hour
            ],

            "predicted_demand": [
                prediction
            ],

            "actual_demand": [
                np.nan
            ]
        })


        if os.path.exists(
            LOG_PATH
        ):

            existing = pd.read_csv(
                LOG_PATH
            )

            updated = pd.concat(
                [
                    existing,
                    new_record
                ],
                ignore_index=True
            )

        else:

            updated = new_record


        updated.to_csv(
            LOG_PATH,
            index=False
        )


# ============================================================
# DEMAND ANALYSIS
# ============================================================

elif page == "Demand Analysis":

    st.header(
        "📈 Demand Analysis"
    )


    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Hourly",
            "Monthly",
            "Seasonal",
            "Weather"
        ]
    )


    # --------------------------------------------------------
    # HOURLY
    # --------------------------------------------------------

    with tab1:

        hourly = (
            df.groupby("hr")["cnt"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            hourly,
            x="hr",
            y="cnt",
            markers=True,
            title="Average Bike Demand by Hour"
        )

        fig.update_xaxes(
            dtick=1
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # Hour x workingday

        hourly_working = (
            df.groupby(
                ["hr", "workingday"]
            )["cnt"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            hourly_working,
            x="hr",
            y="cnt",
            color="workingday",
            markers=True,
            title="Hourly Demand: Working Day vs Non-Working Day"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    with tab2:

        monthly = (
            df.groupby("mnth")["cnt"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            monthly,
            x="mnth",
            y="cnt",
            title="Average Demand by Month"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        monthly_year = (
            df.groupby(
                ["yr", "mnth"]
            )["cnt"]
            .mean()
            .reset_index()
        )

        monthly_year["Year"] = (
            monthly_year["yr"]
            .map({
                0: "2011",
                1: "2012"
            })
        )


        fig = px.line(
            monthly_year,
            x="mnth",
            y="cnt",
            color="Year",
            markers=True,
            title="Monthly Demand by Year"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # SEASONAL
    # --------------------------------------------------------

    with tab3:

        seasonal = (
            df.groupby("season")["cnt"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            seasonal,
            x="season",
            y="cnt",
            title="Average Demand by Season"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        season_hour = (
            df.groupby(
                ["season", "hr"]
            )["cnt"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            season_hour,
            x="hr",
            y="cnt",
            color="season",
            title="Hourly Demand by Season"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    with tab4:

        weather = (
            df.groupby("weathersit")["cnt"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            weather,
            x="weathersit",
            y="cnt",
            title="Demand by Weather Situation"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        weather_hour = (
            df.groupby(
                ["weathersit", "hr"]
            )["cnt"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            weather_hour,
            x="hr",
            y="cnt",
            color="weathersit",
            title="Hourly Demand by Weather"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# ADVANCED ANALYTICS
# ============================================================

elif page == "Advanced Analytics":

    st.header(
        "🔬 Advanced Analytics"
    )


    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Weather Relationships",
            "Correlation",
            "Heatmaps",
            "User Analysis"
        ]
    )


    # --------------------------------------------------------
    # WEATHER RELATIONSHIPS
    # --------------------------------------------------------

    with tab1:

        col1, col2 = st.columns(2)


        fig = px.scatter(
            df,
            x="temp",
            y="cnt",
            opacity=0.35,
            trendline="ols",
            title="Temperature vs Demand"
        )

        col1.plotly_chart(
            fig,
            use_container_width=True
        )


        fig = px.scatter(
            df,
            x="hum",
            y="cnt",
            opacity=0.35,
            trendline="ols",
            title="Humidity vs Demand"
        )

        col2.plotly_chart(
            fig,
            use_container_width=True
        )


        col1, col2 = st.columns(2)


        fig = px.scatter(
            df,
            x="windspeed",
            y="cnt",
            opacity=0.35,
            trendline="ols",
            title="Windspeed vs Demand"
        )

        col1.plotly_chart(
            fig,
            use_container_width=True
        )


        fig = px.scatter(
            df,
            x="atemp",
            y="cnt",
            opacity=0.35,
            trendline="ols",
            title="Feels-like Temperature vs Demand"
        )

        col2.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    with tab2:

        correlation_columns = [
            "yr",
            "mnth",
            "hr",
            "weekday",
            "temp",
            "atemp",
            "hum",
            "windspeed",
            "casual",
            "registered",
            "cnt"
        ]

        correlation = (
            df[
                correlation_columns
            ]
            .corr()
        )


        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Matrix"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # HEATMAPS
    # --------------------------------------------------------

    with tab3:

        hour_working = (
            df.groupby(
                ["hr", "workingday"]
            )["cnt"]
            .mean()
            .unstack()
        )


        fig = px.imshow(
            hour_working,
            text_auto=".0f",
            aspect="auto",
            title="Hour × Working Day Demand"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        hour_season = (
            df.groupby(
                ["hr", "season"]
            )["cnt"]
            .mean()
            .unstack()
        )


        fig = px.imshow(
            hour_season,
            text_auto=".0f",
            aspect="auto",
            title="Hour × Season Demand"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        month_season = (
            df.groupby(
                ["mnth", "season"]
            )["cnt"]
            .mean()
            .unstack()
        )


        fig = px.imshow(
            month_season,
            text_auto=".0f",
            aspect="auto",
            title="Month × Season Demand"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # USER ANALYSIS
    # --------------------------------------------------------

    with tab4:

        user_data = pd.DataFrame({

            "User Type": [
                "Registered",
                "Casual"
            ],

            "Total Rentals": [
                df["registered"].sum(),
                df["casual"].sum()
            ]
        })


        fig = px.pie(
            user_data,
            names="User Type",
            values="Total Rentals",
            title="Registered vs Casual Rentals"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        user_hour = (
            df.groupby("hr")[
                [
                    "registered",
                    "casual"
                ]
            ]
            .mean()
            .reset_index()
        )


        fig = px.line(
            user_hour,
            x="hr",
            y=[
                "registered",
                "casual"
            ],
            markers=True,
            title="Registered vs Casual Demand by Hour"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# MODEL MONITORING
# ============================================================

elif page == "Model Monitoring":

    st.header(
        "🧠 Model Performance Monitoring"
    )


    if not os.path.exists(
        LOG_PATH
    ):

        st.warning(
            "No prediction history available."
        )

    else:

        log_df = pd.read_csv(
            LOG_PATH
        )


        total_predictions = len(
            log_df
        )


        actual_available = (
            log_df[
                "actual_demand"
            ].notna()
        )


        col1, col2 = st.columns(2)


        col1.metric(
            "Total Predictions",
            total_predictions
        )

        col2.metric(
            "Actual Values Available",
            int(actual_available.sum())
        )


        if actual_available.sum() > 0:

            monitoring = log_df[
                actual_available
            ].copy()


            actual = monitoring[
                "actual_demand"
            ]

            predicted = monitoring[
                "predicted_demand"
            ]


            mae = np.mean(
                np.abs(
                    actual - predicted
                )
            )


            mse = np.mean(
                (actual - predicted) ** 2
            )


            rmse = np.sqrt(
                mse
            )


            ss_total = np.sum(
                (actual - actual.mean()) ** 2
            )

            ss_residual = np.sum(
                (actual - predicted) ** 2
            )


            if ss_total != 0:

                r2 = (
                    1 -
                    ss_residual /
                    ss_total
                )

            else:

                r2 = np.nan


            col1, col2, col3, col4 = st.columns(4)


            col1.metric(
                "MAE",
                f"{mae:.2f}"
            )

            col2.metric(
                "MSE",
                f"{mse:.2f}"
            )

            col3.metric(
                "RMSE",
                f"{rmse:.2f}"
            )

            col4.metric(
                "R²",
                f"{r2:.3f}"
            )


            comparison = monitoring[
                [
                    "predicted_demand",
                    "actual_demand"
                ]
            ]


            fig = px.line(
                comparison,
                y=[
                    "predicted_demand",
                    "actual_demand"
                ],
                title="Predicted vs Actual Demand"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        else:

            st.info(
                "Actual demand values are not available. "
                "Once actual values are entered into the "
                "prediction log, model monitoring metrics "
                "will appear here."
            )


        st.subheader(
            "Prediction History"
        )


        st.dataframe(
            log_df,
            use_container_width=True,
            height=400
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.header(
        "⚙️ Model Information"
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Model",
        metadata["model_name"]
    )

    col2.metric(
        "Model Version",
        metadata["model_version"]
    )

    col3.metric(
        "Features",
        metadata["feature_count"]
    )


    st.divider()


    st.subheader(
        "Deployment Details"
    )


    deployment_info = pd.DataFrame({

        "Property": [
            "Model Name",
            "Model Type",
            "Model Version",
            "Deployment Date",
            "Target Variable",
            "Training Records",
            "Testing Records",
            "Feature Count"
        ],

        "Value": [
            metadata["model_name"],
            metadata["model_type"],
            metadata["model_version"],
            metadata["deployment_date"],
            metadata["target"],
            metadata["training_records"],
            metadata["testing_records"],
            metadata["feature_count"]
        ]
    })


    st.table(
        deployment_info
    )


    st.subheader(
        "Model Features"
    )


    st.dataframe(
        pd.DataFrame({
            "Feature": metadata[
                "feature_columns"
            ]
        }),
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Bike Demand Intelligence | "
    "Machine Learning Deployment & Monitoring"
)
