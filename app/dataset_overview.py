import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


@st.cache_data
def load_data():
    df = pd.read_csv('../datasets/clean_heart_disease.csv')
    return df

df = load_data()

def imbalance_data():
    cmap = 'Blues' 
    df["heartdisease"].value_counts()
    df["heartdisease_label"] = df["heartdisease"].map({0: "Normal", 1: "Heart Disease"})

    # Create count plot using Plotly Express with 'Blues' color scale
    fig = px.histogram(df, x="heartdisease_label", color="heartdisease_label", 
                   color_discrete_sequence=["#A9C7E1", "#1F77B4"], # Custom blue shades
                   title="Imbalance Data", 
                   labels={"heartdisease_label": "Diagnose"})

    # Update layout for axis titles and style
    fig.update_layout(
        xaxis_title="Diagnose",
        yaxis_title="Count",
        template="plotly_white",  # White background to match Seaborn style
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

def data_overview():
    st.subheader("📈 Data Overview")
    df = load_data()
    st.write(df)
    st.metric("Total Samples", df.shape[0])
    st.metric("Total Features", df.shape[1])

    st.subheader("📊 Descriptive Statistics for Continuous Data")
    num_cols = df.loc[:,df.nunique()>10]
    st.write(num_cols.describe().T)

    st.subheader("⚖️ Imbalance Data")
    imbalance_data()

