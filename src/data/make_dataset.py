# import the pandas module for loading the raw dataset
import pandas as pd
# import the log_decorator for error handling
from ..logger.log_decorator import log_decorator

@log_decorator
def load_and_preprocess_data(file_path):
    """
    Load the raw data file "credit.csv"
    Impute the missing values in all the features
    Drop the unnecessary identifier column "Loan_ID"
    Drop all duplicate rows

    Args:
        file_path (str): File path of the raw dataset

    Returns:
        DataFrame: Dataset with preprocessing performed
    """
    
    # raise error if the file path is empty
    if not file_path:
        raise ValueError(f"File path is empty : {file_path}")
    
    # load the raw data from "credit.csv" into a DataFrame    
    df = pd.read_csv(file_path)
    
    # impute all missing values in each feature
    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Education'].fillna(df['Education'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['ApplicantIncome'].fillna(df['ApplicantIncome'].median(), inplace=True)
    df['CoapplicantIncome'].fillna(df['CoapplicantIncome'].median(), inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    df['Property_Area'].fillna(df['Property_Area'].mode()[0], inplace=True)
    
    # drop the unnecessary 'Loan_ID' column from the DataFrame
    df = df.drop('Loan_ID', axis=1)
    
    # remove duplicate rows from the DataFrame
    df.drop_duplicates(inplace = True)
    
    return df