import streamlit as st


# Custom libraries
from summary import about
from dataset_overview import data_overview
from ml_models import models

# Project libraries
import utils.viz as viz
import utils.classification_model as cm

st.markdown(
    """
    <style>
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
        }
        .stMarkdown, .stPlotlyChart, .stDataFrame {
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🫀 Heart Disease Prediction Using Machine Learning")

tab1, tab2, tab3 = st.tabs(["📌 Summary", "📋 Dataset Overview", "🤖 Machine Learning Models"])

with tab1:
    about()

with tab2:
    data_overview()

with tab3:
    models()
