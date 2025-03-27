# 📚 Basic libraries
import pandas as pd

# File system libraries
import os
import joblib

# 🤖 Machine Learning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

def classification_metrics(X, y, df, random_state, model, smote, preprocessor):
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

        pipeline = Pipeline([
                ('preprocessor', preprocessor),  # Apply preprocessing
                ('classifier', model)  # Add the classifier
            ])
            
        # Apply SMOTE manually inside the loop during training
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # Train the model
        pipeline.fit(X_resampled, y_resampled)
        
        # Make predictions
        predictions = pipeline.predict(X_test)
        
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

def save_dataframe_to_csv(df, filepath):
    """
    Saves the dataframe to a CSV file in the /data folder if the file doesn't already exist.
    """
    # Check if the file already exists
    if not os.path.exists(filepath):
        # If the file doesn't exist, save the dataframe
        df.to_csv(filepath, index=False)
        print(f"File saved as {filepath}")
    else:
        print(f"The file {filepath} already exists. Skipping save.")

def save_model_to_pickle(pipeline, filename):
    """
    Saves the classification model (including Pipeline) to a pickle file in the /models folder if the file doesn't already exist.
    """
    # Define the path to the models folder
    models_folder = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # Ensure the models folder exists
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    
    # Define the full path for the pickle file
    full_path = os.path.join(models_folder, filename)
    
    # Check if the file already exists
    if not os.path.exists(full_path):
        # Save the model (or pipeline) as a pickle file using joblib
        joblib.dump(pipeline, full_path)  # Save the model as a pickle file
        print(f"Model saved as {full_path}")
    else:
        print(f"{full_path} already exists. File not overwritten.")

def load_model_from_pickle(filename):
    """
    Load a classification model from a pickle file in the /models folder.
    """
    # Define the path to the models folder
    models_folder = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # Define the full path for the pickle file
    full_path = os.path.join(models_folder, filename)
    
    # Check if the file exists
    if os.path.exists(full_path):
        # Load the model (pipeline) from the pickle file
        pipeline = joblib.load(full_path)
        print(f"Model loaded from {full_path}")
        return pipeline
    else:
        print(f"{full_path} not found. Model not loaded.")
        return None
