# User Behaviour Anomaly Detection System using Streamlit UI

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns


st.title("User Behaviour Anomaly Detection System")
st.write("Upload user activity data to detect anomalous behavior.")


data_file = st.file_uploader("Upload CSV", type=["csv"])  # Upload CSV file

if data_file is not None:
    df = pd.read_csv(data_file)
    st.write("### Preview of Uploaded Data:")
    st.dataframe(df.head())

    
    if 'login_time' in df.columns and 'downloads' in df.columns:  # Feature selection 
        X = df[['login_time', 'downloads']]

        
        model = IsolationForest(contamination=0.2, random_state=42)   # Isolation Forest Model
        df['anomaly'] = model.fit_predict(X)

        anomalies = df[df['anomaly'] == -1]
        st.write("### Anomalies Detected:")
        st.dataframe(anomalies)

      
        st.write("### Anomaly Visualization:")    # Visualization
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='login_time', y='downloads', hue='anomaly', palette='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("The CSV must contain 'login_time' and 'downloads' columns.")
else:
    st.info("Awaiting CSV file upload.")
