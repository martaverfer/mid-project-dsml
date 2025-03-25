# 📚 Basic libraries
import numpy as np

# 📊 Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# 🤖 Machine Learning
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, confusion_matrix

# Define the color palette
color = '#fcbf49'

# Define the color map
cmap = 'YlOrBr' 

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
    sns.countplot(data=df, x=column, palette="YlOrBr")
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

def roc_curve_plot(y_test, predictions):
    """
    Plots the ROC curve.
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, predictions)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='orange', lw=2, label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()