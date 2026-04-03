# import all necessary modules
from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import create_dummy_vars
from src.models.train_model import model_training
from src.models.predict_model import evaluate_model
from src.logger.configure_logger import configure_logger
import warnings
warnings.filterwarnings("ignore")
from src.visualization.visualize import *
from src.logger.log_decorator import log_decorator

@log_decorator
def pipeline():
    
    # store the path for the raw dataset
    data_path = "data/raw/credit.csv"
    
    # configure the logger for error handling
    configure_logger()
    
    # load and preprocess the raw dataset
    df = load_and_preprocess_data(data_path)
    
    # plot correlation heatmap for numerical features
    plot_correlation_heatmap(df)
    
    # perform feature engineering
    x, y = create_dummy_vars(df)
    
    # split the dataset into training and test sets, and train the model
    model, x_test_scaled, y_test = model_training(x, y)
    
    # plot feature importance for the trained model
    plot_feature_importance(model, x)
           
    # evaluate the performance of the model using the test set 
    y_pred, test_accuracy, test_confusion = evaluate_model(model, x_test_scaled, y_test)
    
    # plot confusion matrix 
    plot_confusion_matrix(y_test, y_pred, model.classes_)
    
    # display the accuracy and confusion matrix of the model
    print(f"Accuracy : {test_accuracy}")
    print(f"Confusion Matrix : {test_confusion}")
    
if __name__ == "__main__":
    pipeline()
