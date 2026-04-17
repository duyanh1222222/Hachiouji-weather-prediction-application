import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime
import copy

# Set up page config
st.set_page_config(page_title="Weather Prediction AI", page_icon="🌤️")


# Load the model and data once
@st.cache_resource
def load_package():
    return pickle.load(open("weather_AI_package.pkl", 'rb'))


package = load_package()


# --- Prediction Logic (Your Script) ---
def predict_multitask_recursive(target_date, data, model):
    # Convert streamlit date object to pandas datetime if necessary
    target_date = pd.to_datetime(target_date)
    last_row = data.iloc[-1]
    current_date = last_row["年月日"]

    temp_lags = [last_row["平均気温(℃)"]] + [last_row[f"平均気温_{i}"] for i in range(1, 5)]
    maxT_lags = [last_row["最高気温(℃)"]] + [last_row[f"最高気温_{i}"] for i in range(1, 5)]
    minT_lags = [last_row["最低気温(℃)"]] + [last_row[f"最低気温_{i}"] for i in range(1, 5)]

    new_data = copy.deepcopy(data)
    data_columns = new_data.drop(["年月日", "target_temp", "target_maxtemp", "target_mintemp"], axis=1)

    while current_date < target_date:
        current_date += datetime.timedelta(days=1)
        m_sin = np.sin(2 * np.pi * current_date.month / 12)
        m_cos = np.cos(2 * np.pi * current_date.month / 12)

        x_input = pd.DataFrame([[
            temp_lags[0], maxT_lags[0], minT_lags[0], temp_lags[1], maxT_lags[1], minT_lags[1],
            temp_lags[2], maxT_lags[2], minT_lags[2], temp_lags[3], maxT_lags[3], minT_lags[3],
            temp_lags[4], maxT_lags[4], minT_lags[4],
            current_date.year, current_date.month, m_sin, m_cos, current_date.day
        ]], columns=data_columns.columns)

        prediction = model.predict(x_input)[0]
        pred_temp, pred_maxT, pred_minT = prediction[0], prediction[1], prediction[2]

        temp_lags = [pred_temp] + temp_lags[:-1]
        maxT_lags = [pred_maxT] + maxT_lags[:-1]
        minT_lags = [pred_minT] + minT_lags[:-1]

    return pred_temp, pred_maxT, pred_minT


# --- UI Layout ---
st.title("🌡️ Weather Forecasting Tool")
st.write("Select a date to predict the average, maximum, and minimum temperatures.")

# Input: Date Picker
# Defaulting to 2026-04-15 as per your original script
selected_date = st.date_input("Target Date", value=datetime.date(2026, 4, 15))

if st.button("Predict Weather"):
    with st.spinner('Calculating recursive prediction...'):
        p_temp, p_maxT, p_minT = predict_multitask_recursive(
            selected_date, package['data_frame'], package['model']
        )

        st.divider()
        st.subheader(f"Results for {selected_date}")

        # Displaying results in a clean metric row
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Temp", f"{p_temp:.1f} ℃")
        col2.metric("Max Temp", f"{p_maxT:.1f} ℃")
        col3.metric("Min Temp", f"{p_minT:.1f} ℃")