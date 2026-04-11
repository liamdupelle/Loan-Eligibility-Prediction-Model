# import the pandas module
import pandas as pd
# import the log_decorator for error handling
from ..logger.log_decorator import log_decorator

@log_decorator
def create_dummy_vars(df):
    """
    Create dummy variables for categorical features
    Split dataset into input features and target variable

    Args:
        df (DataFrame): Preprocessed dataset

    Returns:
        DataFrame: Input features,
        Series: Target variable
    """
    
    # raise error if the DataFrame is None
    if df is None:
        raise ValueError("DataFrame was not properly loaded")
    
    # raise error if required features are missing
    cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Approved']
    cols_missing = [col for col in cols if col not in df.columns]
    if cols_missing:
        raise ValueError(f"Required columns missing : {cols_missing}")
    
    # create dummy variables for all categorical features
    df = pd.get_dummies(df, columns=['Gender', 
                                     'Married', 
                                     'Dependents',
                                     'Education',
                                     'Self_Employed',
                                     'Property_Area'], dtype=int)
    
    # replace categorical values in "Loan_Approved" feature with numerical (0, 1)
    df['Loan_Approved'] = df['Loan_Approved'].replace({'Y':1, 'N':0})
    
    # separate input features from the target variable
    x = df.drop('Loan_Approved', axis=1) 

    # store the target variable in y
    y = df['Loan_Approved']
    
    return x, y