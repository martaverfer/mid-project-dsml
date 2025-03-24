# Heart Failure Prediction

## Overview
This dataset contains medical records of patients who are at risk of heart failure. It includes various clinical and demographic features that can be used to predict the likelihood of heart failure and, more specifically, whether a patient will experience a **death event** due to heart failure.

## Objectives
- The goal of this project is to use machine learning models to predict the probability of a death event based on the patient's medical records and relevant features. In other words, the model aims to classify whether a patient is at risk of dying from heart failure (a binary classification problem). 
- The investigation aims to identify the model that performs best for the dataset, considering factors like model interpretability, scalability, and generalization to new data.
- The ultimate goal is to enhance early detection, personalized treatment, and resource optimization in healthcare.

## Project Workflow
1. **Load and Explore Data**:
   - Load the dataset and check for missing values, basic statistics, and distributions of the features.
2. **Exploratory Data Analysis**:
   - Descriptive Statistics, distributions and correlations.
3. **Data Preprocessing**:
   - Handle missing data, scale numerical features, and address any class imbalance.
3. **Train-Test Split**:
   - Split the data into training and testing sets (typically 80/20 or 70/30 split).
4. **Model Training**:
   - Train various classification models (Logistic Regression, Random Forest, Gradient Boosting, etc.) and compare their performance.
5. **Model Validation**:
   - Evaluate the models using appropriate metrics (accuracy, precision, recall, F1-score, ROC-AUC).
6. **Model Tuning**:
   - Fine-tune hyperparameters using GridSearchCV or RandomizedSearchCV for better model performance.
7. **Interpretability**:
   - If needed, interpret model predictions using feature importance or tools like SHAP or LIME.

## Analysis and Results

## Conclusion

## Additional content
- [Presentation](./docs/heart_failure_prediction.pdf)
- [Dataset Details](./datasets/dataset.md)
- [Developer setup](./docs/setup.md)
