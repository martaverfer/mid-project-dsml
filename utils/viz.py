# 📚 Basic libraries
import numpy as np
import shap
shap.initjs()

# 📊 Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

# 🤖 Machine Learning
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc

# Define the color palette
color = '#ff0051'
color2 = '#008bfb'

# Define the color map
cmap = 'Blues' 

def distribution_plot(df,nrows,ncols):
    """
    Plots the distribution of numerical features in the DataFrame.
    """
    # grid size
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 10))
    axes = axes.flatten()

    # Plot each numerical feature
    for i, ax in enumerate(axes):
        if i >= len(df.columns):
            ax.set_visible(False)  # hide unesed plots
            continue
        ax.hist(df.iloc[:, i], bins=30, color=color, edgecolor='black')
        ax.set_title(df.columns[i])

    plt.tight_layout()
    plt.show()

def outliers_distribution(df, nrows, ncols):
    """
    Plots the outliers of numerical features in the DataFrame.
    """
    # grid size
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(df.columns):
            ax.set_visible(False)
            continue
        ax.boxplot(df.iloc[:, i].dropna(), vert=False, patch_artist=True, 
                boxprops=dict(facecolor=color, color='black'), 
                medianprops=dict(color='yellow'), whiskerprops=dict(color='black'), 
                capprops=dict(color='black'), flierprops=dict(marker='o', color='red', markersize=5))
        ax.set_title(df.columns[i], fontsize=10)
        ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    plt.show()

def multicollinearity_heatmap(df):
    """
    Plots a heatmap to show multicollinearity between features.
    """
    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = True # optional, to hide repeat half of the matrix

    f, ax = plt.subplots(figsize=(25, 15))
    sns.set(font_scale=1.5) # increase font size

    ax = sns.heatmap(df, mask=mask, annot=True, annot_kws={"size": 12}, linewidths=.5, cmap=cmap, fmt=".2f", ax=ax) # round to 2 decimal places
    ax.set_title("Dealing with Multicollinearity", fontsize=20) # add title
    plt.xticks(rotation=45)
    plt.show()

def imbalance_data(df, column):
    """
    Plots the imbalance data.
    """
    sns.set_style("whitegrid")
    sns.countplot(data=df, x=column, palette=cmap)
    plt.xlabel("Diagnose")
    plt.title("Imbalance Data")

def confusion_matrix_plot(y_test, y_pred):
    """
    Plots the confusion matrix.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=cmap)  
    plt.grid(True)
    # Add a title
    plt.title("Confusion Matrix")
    plt.show()

def roc_curve_auc_plot(y_test1, predictions1, y_test2, predictions2, label1, label2):
    """
    Plots the ROC curve for each model.
    """
    # Compute ROC curve for model 1
    fpr1, tpr1, thresholds1 = roc_curve(y_test1, predictions1)
    auc1 = auc(fpr1, tpr1) # Compute AUC for model 1

    # Compute ROC curve for model 2
    fpr2, tpr2, thresholds2 = roc_curve(y_test2, predictions2)
    auc2 = auc(fpr2, tpr2)  # Compute AUC for model 2

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr1, tpr1, color=color, lw=2, label=f'{label1} (AUC = {auc1:.2f})')
    plt.plot(fpr2, tpr2, color=color2, lw=2, label=f'{label2} (AUC = {auc2:.2f})')
    # Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def max_depth_plot(model, X_train, y_train, X_test, y_test):
    """
    Plots the max depth of the model vs the accuracy.
    """
    max_depth = range(1, 30)
    test = []
    train = []

    for depth in max_depth:
        model = model.set_params(classifier__max_depth=depth)
        model.fit(X_train, y_train)
        test.append(model.score(X_test,y_test))
        train.append(model.score(X_train,y_train))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(max_depth), y=train, name='Training Accuracy', marker=dict(color=color)))
    fig.add_trace(go.Scatter(x=list(max_depth), y=test, name='Testing Accuracy', marker=dict(color=color2)))
    fig.update_layout(xaxis_title='Max Tree Depth', yaxis_title='Accuracy', title='Accuracy vs Max Tree Depth')
    fig.show()

def shap_bar_plot(shap_values):
    shap.plots.bar(shap_values, show=False)
    plt.title('Mean Absolute SHAP values for each feature in the model', fontsize=16)
    plt.show()

def shap_beeswarm_plot(shap_values):
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Beeswarm Plot for Heart Disease", fontsize=16)
    plt.show()  
