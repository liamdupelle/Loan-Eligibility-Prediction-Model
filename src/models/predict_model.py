# import metrics to calculate the accuracy of the model
from sklearn.metrics import accuracy_score, confusion_matrix
# import the log_decorator for error handling
from ..logger.log_decorator import log_decorator

@log_decorator
def evaluate_model(model, x_test_scaled, y_test):
    """
    Predict the target variable using scaled input features from the test set
    Calculate the performance of the model on the test set

     Args:
        model (model): Random Forest Classifier trained model
        x_test_scaled (DataFrame): Scaled input features of the test set
        y_test (Series): Target variable of the test set

    Returns:
        Series: Predicted values of the test set
        float: Accuracy of the model
        ndarray: Confusion matrix of test set
    """
    
    # raise error if the model is None
    if model is None:
        raise ValueError("Error in model training")
    
    # predict the loan eligibility using the input features of the test set    
    y_pred = model.predict(x_test_scaled)
    
    # calculate the accuracy score of the model
    test_accuracy = accuracy_score(y_pred, y_test)
    
    # calculate the confusion matrix
    test_confusion = confusion_matrix(y_test, y_pred)
    
    return y_pred, test_accuracy, test_confusion