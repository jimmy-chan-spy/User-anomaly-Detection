# User Behaviour Anomaly Detection System with Streamlit UI

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

# Streamlit UI setup
st.title("User Behaviour Anomaly Detection System")
st.write("Upload user activity data to detect anomalous behavior.")

# Upload CSV file
data_file = st.file_uploader("Upload CSV", type=["csv"])

if data_file is not None:
    df = pd.read_csv(data_file)
    st.write("### Preview of Uploaded Data:")
    st.dataframe(df.head())

    # Feature selection (Assume columns: login_time, downloads)
    if 'login_time' in df.columns and 'downloads' in df.columns:
        X = df[['login_time', 'downloads']]

        # Isolation Forest Model
        model = IsolationForest(contamination=0.2, random_state=42)
        df['anomaly'] = model.fit_predict(X)

        anomalies = df[df['anomaly'] == -1]
        st.write("### Anomalies Detected:")
        st.dataframe(anomalies)

        # Visualization
        st.write("### Anomaly Visualization:")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='login_time', y='downloads', hue='anomaly', palette='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("The CSV must contain 'login_time' and 'downloads' columns.")
else:
    st.info("Awaiting CSV file upload.")
