import streamlit as st


def about():
    st.write("""
    <div style="text-align: left;">
             
    ### Heart Disease Prediction: why is it important?
             
    Cardiovascular diseases (CVDs) are the **number 1 cause of death globally**, taking an estimated 17.9 million lives each year, 
    which accounts for 31% of all deaths worldwide. Four out of 5 CVD deaths are due to heart attacks and strokes, 
    and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs 
    and this dataset contains 11 features that can be used to predict a possible heart disease.

    People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors 
    such as hypertension, diabetes, hyperlipidaemia, or already established disease) need early detection and management 
    wherein a machine learning model can be of great help. 
   
    ### About the Dataset

    This dataset was created by combining different datasets already available independently but not combined before. 
    In this dataset, 5 heart datasets are combined over 11 common features, which makes it the largest heart disease dataset available 
    so far for research purposes. The five datasets used for its curation are:

    - **Cleveland**: 303 observations
    - **Hungarian**: 294 observations
    - **Switzerland**: 123 observations
    - **Long Beach VA**: 200 observations
    - **Stalog (Heart) Data Set**: 270 observations

    You can access the dataset on Kaggle using the following link:

    [Heart Failure Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
    </div>
    """,  unsafe_allow_html=True)
