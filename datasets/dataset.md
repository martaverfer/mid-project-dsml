# Heart Failure Prediction Dataset

## Overview
This dataset contains medical records of patients who are at high cardiovascular risk. It includes various clinical and demographic features that can be used to predict the likelihood of heart disease. The goal of this project is to use machine learning models to predict the probability of a heart disease based on the patient's health information.

## Dataset Source
- **Kaggle Dataset:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

## Dataset Details

### Features:
1. **Age:** age of the patient [years]
2. **Sex:** sex of the patient [0: Male, 1: Female]
3. **ChestPainType:** chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
4. **RestingBP:** resting blood pressure [mm Hg]
5. **Cholesterol:** serum cholesterol [mm/dl]
6. **FastingBS:** fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
7. **RestingECG:** resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
8. **MaxHR:** maximum heart rate achieved [Numeric value between 60 and 202]
9. **ExerciseAngina:** exercise-induced angina [Y: Yes, N: No]
10. **Oldpeak:** oldpeak = ST [Numeric value measured in depression]
11. **ST_Slope:** the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
12. **HeartDisease:** output class [1: heart disease, 0: Normal]


### Target Variable:
- **HeartDisease:** This is a binary classification target variable indicating whether the patient experience a heart disease (1 = heart disease, 0 = normal).

### Data Size:
- **Number of rows:** 918 rows
- **Number of columns:** 12 columns (including the target variable)

## Source
This dataset was created by combining different datasets already available independently but not combined before. In this dataset, 5 heart datasets are combined over 11 common features which makes it the largest heart disease dataset available so far for research purposes. The five datasets used for its curation are:

- Cleveland: 303 observations
- Hungarian: 294 observations
- Switzerland: 123 observations
- Long Beach VA: 200 observations
- Stalog (Heart) Data Set: 270 observations

## Data Preprocessing
- **Missing Values**: It’s important to check and handle any missing data (if present) by imputation or removal.
- **Feature Scaling**: Since this dataset contains numeric features with varying ranges, scaling might be needed for some machine learning algorithms.
- **Encoding Categorical Data**: Features like **FastingBS** and **HeartDisease** are binary, so they are already encoded (0 or 1). Others like **ChestPainType**, **RestingECG**, **ExerciseAngina** and **ST_Slope** need to be encoded.
- **Class Imbalance**: If there is a significant imbalance between the classes (1 = deheart disease, 0 = normal), techniques such as resampling (SMOTE or undersampling) or adjusting class weights may be needed.
