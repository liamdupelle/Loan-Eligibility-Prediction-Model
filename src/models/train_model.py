# import random forest model
from sklearn.ensemble import RandomForestClassifier
# import module for splitting dataset
from sklearn.model_selection import train_test_split
# import the MinMaxScaler 
from sklearn.preprocessing import MinMaxScaler
# import pickle module
import pickle
# import the log_decorator for error handling
from ..logger.log_decorator import log_decorator

@log_decorator
def model_training(x,y):
    """
    Train the Random Forest model on the dataset

    Args:
        x (DataFrame): Input features
        y (Series): Target variable

    Returns:
        model: Trained Random Forest model,
        DataFrame: Scaled input features of the test set,
        Series: Target variable of the test set
    """
    
    # split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, stratify=y, random_state=42)

    # scale the data using the MinMaxScaler
    sc = MinMaxScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)
    
    # train the Random Forest model on the training set
    model = RandomForestClassifier(n_estimators=100,
                                   max_depth=2,
                                   max_features=15,
                                   random_state=42)
    RFmodel = model.fit(x_train_scaled, y_train)
    
    # save the training model in the "models" folder
    with open("models/RFmodel.pkl","wb") as file:
        pickle.dump(RFmodel,file)
        
    # save the scaler in the "models" folder
    with open("models/sc.pkl","wb") as file:
        pickle.dump(sc,file)
        
    return RFmodel, x_test_scaled, y_test