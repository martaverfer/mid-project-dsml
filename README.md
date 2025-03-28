# Heart Failure Prediction

## Overview
This dataset contains medical records of patients who are at risk of heart disease. It includes various clinical and demographic features that can be used to predict the likelihood of heart disease.

## Objectives
- The goal of this project is to use machine learning models to predict the probability of a disease event based on the patient's medical records and relevant features. In other words, the model aims to classify whether a patient is at risk of suffering from heart disease (a binary classification problem). 

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
   - Train various classification models (Logistic Regression, Decision Tree, Random Forest, XGBOOST and SVM) and compare their performance.
5. **Model Validation**:
   - Evaluate the models using appropriate metrics (accuracy, precision, recall, F1-score, ROC-AUC).
6. **Model Tuning**:
   - Fine-tune hyperparameters using GridSearchCV or RandomizedSearchCV for better model performance.
7. **Interpretability**:
   - If needed, interpret model predictions using feature importance or tools like SHAP or LIME.

## Analysis and Results

In this project, we aimed to predict the likelihood of heart disease using machine learning models trained on patient medical records. After exploring and preprocessing the data, we trained various classification models and evaluated their performance. After testing several models we can conclude:

- Compensation of imbalance data and standardization of data can impact to the performance of models like SVM.

- SVM and Random Forest are the models that performed the best when compared with others.
 
- Random Forest is the model that has performed the best with a 0.91 of accuracy after hyperparameter tuning.

- The presence of st_slope_Flat, chestpaintype_ASY and oldpeak, and the absence of st_slope_Up are indicators of heart disease (high impact on the model).

## Conclusion

This project demonstrates the effectiveness of machine learning in early detection of heart disease, which could aid in personalized treatment plans and resource allocation in healthcare. While Random Forest performed the best in this case, SHAP could offer deeper insights into patient risk factors. Future work could focus on integrating more advanced algorithms or exploring larger datasets for improved prediction capabilities and model generalization.

Overall, this approach has the potential to enhance early detection, reduce healthcare costs, and improve patient outcomes by identifying at-risk individuals more effectively. 

## Additional content
- [Presentation](https://docs.google.com/presentation/d/11iYIil5NDXSlMP2Q6uov4dAgefgMNFpu184u43WceIE/edit?usp=sharing)
- [Dataset Details](./datasets/dataset.md)
- [Developer setup](./docs/setup.md)
