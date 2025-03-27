import shap
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report

# File system libraries
import os
import sys

# Add the root directory to sys.path 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Project libraries
import utils.classification_model as cm

@st.cache_data
def load_data():
    df = pd.read_csv('../datasets/clean_heart_disease.csv')
    return df

df = load_data()

# Load the Random Forest model
filename = '../models/random_forest_model.pkl'
pipeline_random = cm.load_model_from_pickle(filename)

# Load the SVM model
filename = '../models/svm_model.pkl'
pipeline_svm = cm.load_model_from_pickle(filename)

# Load best model
filename = '../models/best_model.pkl'
pipelinebest = cm.load_model_from_pickle(filename)

# Load SHAP values
filename = '../models/shap_values.pkl'
shap_values = cm.load_model_from_pickle(filename)

# Split data for training and testing
X = df.drop("heartdisease", axis=1)
y = df["heartdisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Apply SMOTE
smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

@st.cache_data
def train_best_model(X_resampled, y_resampled):
    """Train the best model and cache the result."""
    pipelinebest.fit(X_resampled, y_resampled)
    return pipelinebest

@st.cache_data
def make_predictions(_model, X_test):
    """Make predictions using the trained model and cache the results."""
    return _model.predict(X_test)


# Plot ROC curve and AUC using Plotly
def plot_roc_curve(y_test, y_pred_rf, y_pred_svm):
    color = '#ff0051'
    color2 = '#008bfb'
    # Compute ROC curve for model 1
    fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred_svm)
    auc1 = auc(fpr1, tpr1) # Compute AUC for model 1

    # Compute ROC curve for model 2
    fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_rf)
    auc2 = auc(fpr2, tpr2)  # Compute AUC for model 2

    fig = go.Figure()

    # ROC curve line 1
    fig.add_trace(go.Scatter(x=fpr1, y=tpr1, mode='lines', name=f'ROC curve for SVM model (AUC = {auc1:.2f})',
                             line=dict(color=color, width=2)))
    
    # ROC curve line 1
    fig.add_trace(go.Scatter(x=fpr2, y=tpr2, mode='lines', name=f'ROC curve for Random Forest model (AUC = {auc2:.2f})',
                             line=dict(color=color2, width=2)))

    # No skill line (diagonal)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='No Skill',
                             line=dict(color='gray', width=1, dash='dash')))

    # Update layout
    fig.update_layout(
        title='Interactive ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True,
        template='plotly_white'
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

# Function to plot confusion matrix with Plotly
def plot_confusion_matrix(y_test, y_pred):
    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a DataFrame for the confusion matrix
    cm_df = pd.DataFrame(cm, index=["Normal", "Heart Disease"], columns=["Predicted Normal", "Predicted Heart Disease"])
    
    # Plot using Plotly
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", 
                    title="Confusion Matrix", labels=dict(x="Predicted", y="Actual"))
    
    # Update layout for better visual appearance
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", 
                      xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Normal", "Heart Disease"]),
                      yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Normal", "Heart Disease"]))
    
    # Display the Plotly chart
    st.plotly_chart(fig)

def shap_bar_plot(shap_values, df=None):
    """
    Create a Plotly bar plot showing the mean absolute SHAP values for each feature.
    
    :param shap_values: SHAP values calculated from the model.
    :param df: The original dataframe used for model training/prediction (optional).
    """
    # If shap_values is a numpy array, compute mean absolute SHAP values
    if isinstance(shap_values, np.ndarray):
        shap_values_abs = np.abs(shap_values)
        feature_importance = np.mean(shap_values_abs, axis=0)
    else:
        # If shap_values is already a DataFrame
        shap_values_abs = np.abs(shap_values.values)
        feature_importance = np.mean(shap_values_abs, axis=0)

    # If df (dataframe) is provided, use its columns as feature names
    if df is not None:
        feature_names = df.columns
    else:
        # If no dataframe is provided, generate generic labels
        feature_names = [f"Feature {i+1}" for i in range(shap_values_abs.shape[1])]

    # Create a DataFrame to hold feature names and their importance
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Create a Plotly bar chart
    fig = go.Figure([go.Bar(x=feature_importance_df['Feature'], y=feature_importance_df['Importance'], 
                           marker_color='royalblue')])
    
    fig.update_layout(
        title="Mean Absolute SHAP Values for Each Feature",
        xaxis_title="Feature",
        yaxis_title="Mean Absolute SHAP Value",
        showlegend=False
    )
    
    # Display plot in Streamlit
    st.plotly_chart(fig)

def shap_beeswarm_plot(shap_values, df=None):
    """
    Create a Plotly beeswarm plot for SHAP values, showing feature impact on the model.
    
    :param shap_values: SHAP values calculated from the model.
    :param df: The original dataframe used for model training/prediction (optional).
    """
    # If shap_values is a numpy array, it might not contain feature names
    if isinstance(shap_values, np.ndarray):
        shap_values_abs = shap_values  # If numpy, we only have SHAP values
    else:
        shap_values_abs = shap_values.values  # If pandas, use .values for the numpy array of SHAP values
    
    # If df (dataframe) is provided, use its columns as feature names
    if df is not None:
        feature_names = df.columns  # Use column names from the dataframe
    else:
        # If no dataframe is provided, create generic feature names
        feature_names = [f"Feature {i+1}" for i in range(shap_values_abs.shape[1])]
    
    # Create the beeswarm plot using Plotly
    data = []
    for i, feature in enumerate(feature_names):
        data.append(go.Scatter(
            x=shap_values_abs[:, i],  # SHAP values for each feature
            y=[feature] * len(shap_values_abs),  # Repeat the feature name for all samples
            mode='markers',
            marker=dict(
                color=shap_values_abs[:, i],  # Use SHAP value for coloring
                colorscale='Blues',
                size=8
            ),
            name=feature,
            opacity=0.7
        ))
    
    # Create a Plotly figure
    fig = go.Figure(data)
    fig.update_layout(
        title="SHAP Beeswarm Plot for Heart Disease",
        xaxis_title="SHAP Value",
        yaxis_title="Feature",
        showlegend=False,
        hovermode='closest'
    )
    
    # Display plot in Streamlit
    st.plotly_chart(fig)


def models():

    st.subheader("🧠 Machine Learning Models")
    chart_type = st.radio("Select Model", ["Random Forest vs. Super Vector Machine", "Random Forest"])

    if chart_type == "Random Forest vs. Super Vector Machine":
        st.subheader("📈 ROC Curve")
        # Predict rf
        pipeline_random.fit(X_resampled, y_resampled)
        y_pred_rf = pipeline_random.predict(X_test)
        # Predict svm
        pipeline_svm.fit(X_resampled, y_resampled)
        y_pred_svm = pipeline_svm.predict(X_test)
        # Plot ROC curve and AUC
        plot_roc_curve(y_test, y_pred_rf, y_pred_svm)
    else:
        st.subheader("📈 Best Model")

        st.subheader('Classification Report')
        # Cache model training
        best_model = train_best_model(X_resampled, y_resampled)
        # Cache prediction
        y_pred_best = best_model.predict(X_test)

        report = classification_report(y_test, y_pred_best, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader('Confusion Matrix')
        plot_confusion_matrix(y_test, y_pred_best)

        st.subheader("SHAP Values: How Features Impact the Model's Predictions")
        # Display SHAP bar plot
        shap_bar_plot(shap_values, X)

        # Display SHAP beeswarm plot
        shap_beeswarm_plot(shap_values, X)
