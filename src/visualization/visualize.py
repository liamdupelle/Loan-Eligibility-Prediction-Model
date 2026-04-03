# import all modules for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# import confusion matrix
from sklearn.metrics import confusion_matrix
# import numpy
import numpy as np
# import log_decorator for error handling
from ..logger.log_decorator import log_decorator

@log_decorator    
def plot_feature_importance(model, x):
    """
    Plot the importance of each feature to the model

    Args:
        model (model): trained Random Forest model
        x (DataFrame): input features
    """
    
    plt.figure(figsize=(8,6))
    sns.barplot(x=model.feature_importances_, 
                y=x.columns)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title("Feature Importance Chart", fontsize=16)
    plt.tight_layout()
    plt.show()
    
@log_decorator    
def plot_correlation_heatmap(df):
    """
    Plot the correlation heatmap for the numeric features

    Args:
        df (DataFrame): dataset
    """
    
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.show()
    
@log_decorator       
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False,title="Confusion Matrix"):
    """
    Plot the confusion matrix 

    Args:
        y_true (Series): actual loan eligibility
        y_pred (Series): predicted loan eligibility
        classes (list): distinct classes of loan eligibility (0 or 1)
        normalize (bool, optional): normalize the confusion matrix. Defaults to False.
        title (str, optional): title of the confusion matrix. Defaults to "Confusion Matrix".
    """
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm.astype('float') / cm.sum(axis=1) [:, np.newaxis]
         
    fig = plt.figure(figsize=(8,6))   
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes, fmt=".0f")
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    fig.savefig("confusion_matrix.png")
    