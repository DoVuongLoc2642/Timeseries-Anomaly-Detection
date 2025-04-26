import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from adtk.detector import SeasonalAD
from adtk.data import validate_series
import pandas as pd

class ARIMAAnomalyDetector:
    def __init__(self, order=(2, 0, 2), threshold_sigma=3.0):
        self.order = order
        self.threshold_sigma = threshold_sigma
        self.model_fit = None
        self.threshold = None

    def fit_predict(self, series: pd.Series) -> pd.Series:
        # validate series
        s = validate_series(series)

        self.model_fit = ARIMA(s, order=self.order).fit()
        pred = self.model_fit.predict(start=s.index[0], end=s.index[-1])

        # calculate squared errors
        errors = (s - pred) ** 2
        # Z-score threshold: mean + n * std
        self.threshold = errors.mean() + self.threshold_sigma * errors.std()

        anomalies = errors > self.threshold
        return anomalies.astype(bool)

# ADTK: SeasonalAD: Seasonal Anomaly
# Detects if value is abnormal relative to its seasonality.
class ADTKAnomalyDetector:
    def __init__(self, period=288):
        self.period = period
        self.model = SeasonalAD(freq=self.period)

    def fit_predict(self, series: pd.Series) -> pd.Series:
        s = validate_series(series)
        return self.model.fit_detect(s)


def plot_anomalies(df, anomalies, model):
    model_name = model
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df.index, df['value'], label='Actual')
    ax.scatter(df[anomalies].index, df[anomalies]['value'], color='red', label='Anomaly')
    ax.set_title(f"{model_name} Anomaly Detection")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def plot_anomalous_windows(df, anomalies, window_size=288):
    values = df['value'].values
    timestamps = df.index
    num_windows = len(values) // window_size
    windows = [
        values[i * window_size:(i + 1) * window_size]
        for i in range(num_windows)
    ]
    window_times = [
        timestamps[i * window_size] for i in range(num_windows)
    ]
    anomaly_windows = []

    # save anomalies window index
    for i in range(num_windows):
        start = i * window_size
        end = (i + 1) * window_size
        if np.any(anomalies[start:end]):
            anomaly_windows.append(i)

    st.subheader("Anomalous Windows")
    if not anomaly_windows:
        st.info("No anomalous windows detected.")
    else:
        for i in anomaly_windows:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(windows[i], label='Actual Values')
            # Mark anomaly points
            for j in range(window_size):
                if anomalies[i * window_size + j]: # flag point j
                    ax.scatter(j, windows[i][j], color='red', zorder=5)
            ax.set_title(f"Window starting at {window_times[i]}")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)




def main():
    st.set_page_config(layout="wide")
    st.title("Time Series Anomaly Detection with Seasonal Sliding Window")

    upload_file = st.file_uploader("Upload your time series CSV file", type=["csv"])

    if upload_file is not None:
        df = pd.read_csv(upload_file, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)

        # select model to use
        model = st.selectbox("Select Anomaly Detection Model", ["ARIMA", "ADTK"])
        if model == "ADTK":
            st.subheader("ADTK Anomaly Detection")
            detector = ADTKAnomalyDetector(period=288)
            anomalies = detector.fit_predict(df['value'])
            df['anomaly'] = anomalies.fillna(False)
            # Plot
            plot_anomalies(df, df['anomaly'], model)
            plot_anomalous_windows(df, df['anomaly'], window_size=288)

        else:
            st.subheader("ARIMA Anomaly Detection")
            arima_detector = ARIMAAnomalyDetector(order=(2, 0, 2), threshold_sigma=3.0)
            anomalies = arima_detector.fit_predict(df['value'])
            df['anomaly_arima'] = anomalies

            # Plot
            plot_anomalies(df, df['anomaly_arima'], model)
            plot_anomalous_windows(df, df['anomaly_arima'], window_size=288)



if __name__ == "__main__":
    main()