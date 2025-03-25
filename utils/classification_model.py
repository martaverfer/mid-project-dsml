# 📚 Basic libraries
import pandas as pd
import numpy as np 
import os

# 🤖 Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix, precision_recall_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def classification_metrics(X, y, df, random_state, model, standardize=False):
    """
    Returns a dataframe with the metrics of the model you have passed as a parameter for different test sizes.
    """
    test_sizes = [0.1, 0.2, 0.3, 0.4]
    
    # List to store results for each test size
    results = []

    for i in test_sizes:
        print(f"{i*100}% test size")
        print("====================================")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=random_state)
        
        print(f'100% of our data: {len(df)}.')
        print(f'{(1-i)*100}% for training data: {len(X_train)}.')
        print(f'{i*100}% for test data: {len(X_test)}.')
        print("====================================")
        print()

        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            pass

        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, predictions) 
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Store the results for this test size
        results.append({
            'test_size': f'{i*100}%',
            'accuracy': round(accuracy, 3),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3)
        })

    pd.set_option('display.float_format', '{:.3f}'.format)

    # Return the final dataframe containing all results
    return pd.DataFrame(results)

def test_train_r_analysis(model, X_train, X_test, y_train, y_test):
    """
    Calculate `accuracy` for training and testing sets
    """
    # Predict on the training and testing sets
    prediction_train = model.predict(X_train)
    prediction_test = model.predict(X_test)
        
    # Calculate R² scores for training and testing
    train_accuracy= accuracy_score(y_train, prediction_train)
    test_accuracy= accuracy_score(y_test, prediction_test)

    print("Accuracy train: ", round(train_accuracy, 3))
    print("Accuracy test: ", round(test_accuracy, 3))


def model_evaluation(models, preprocessor, smote, X_train, y_train):
    """
    Evaluate models using cross-validation and the preprocessing pipeline
    """
    folds = [3, 5, 8, 10]  # Number of cross-validation folds

    for fold in folds:
        print(f"\n📈 Evaluating with {fold}-fold cross-validation:")
        print()
        for name, model in models.items():
            # Create a pipeline that includes both preprocessing and the model
            pipeline = Pipeline([
                ('preprocessor', preprocessor),  # Apply preprocessing
                ('classifier', model)  # Add the classifier
            ])
            
            # Apply SMOTE manually inside the loop during training
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)  # Resample the training data

            # Perform cross-validation and evaluate performance
            scores = cross_val_score(pipeline, X_resampled, y_resampled, cv=fold, scoring='accuracy')  #fold cross-validation
            print(f"\t{name} Average Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

def reporting_dataframe(X_test, y_test, model):
    """
    Returns a dataframe with actual vs. predictions 
    Evaluates and compares the predicted values of a model with the true values of the dataset
    """

    # Make predictions
    predictions = model.predict(X_test)
    
    # Create a dataframe to compare
    eval_df = pd.DataFrame({"actual": y_test, "pred": predictions})
    eval_df["dif"] = abs(eval_df["actual"]-eval_df["pred"])
    eval_df.reset_index(drop=True, inplace=True)
    
    # Return the final dataframe containing actual vs predictions
    return eval_df

def save_dataframe_to_pickle(df, filename):
    """
    Saves the DataFrame to a pickle file if the file doesn't already exist.
    """
    if not os.path.exists(filename):  # Check if the file already exists
        df.to_pickle(filename)  # Save DataFrame as pickle file
        print(f"DataFrame saved as {filename}")
    else:
        print(f"{filename} already exists. File not overwritten.")
