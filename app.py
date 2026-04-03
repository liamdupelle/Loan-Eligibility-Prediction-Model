# import all necessary modules
import pickle
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
from src.logger.configure_logger import configure_logger
from src.logger.log_decorator import log_decorator
import logging

# configure the logger
configure_logger()

# retrieve the logger object
logger = logging.getLogger(__name__)

# show the title and a description of the application
st.title("Credit Loan Eligibility Predictor")
st.write("""
This app predicts whether a loan applicant is eligible for a loan 
based on various personal and financial characteristics.
""")

# unpickle a file 
@log_decorator
def unpickle_file(file_path, message):
    
    logger.info("Unpickling " + message)

    # load the file
    file_pickle = open(file_path, "rb")
    file = pickle.load(file_pickle)
    file_pickle.close()
    
    logger.info(f"Successfully Unpickled {file_path}")
    
    return file

# load the pre-trained model
rf_model = unpickle_file("models/RFmodel.pkl", "Machine Learning Model")

# load the scaler used for training the model
sc = unpickle_file("models/sc.pkl", "Scaler")

# prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Loan Applicant Details")
    
    # gender of the applicant
    Gender = st.selectbox("Gender", options=["Male", "Female"])
    
    # marital status of applicant
    Married = st.selectbox("Marital Status", options=["Yes", "No"])
    
    # number of dependents
    Dependents = st.selectbox("Number of Dependents", 
                               options=["0", "1", "2", "3+"])
    
    # highest level of education
    Education = st.selectbox("Highest Education Level", 
                              options=["Graduate", "Not Graduate"])
    
    # self-employed (y/n)
    Self_Employed = st.selectbox("Self Employed", options=["Yes", "No"])
    
    # applicant's monthly income
    ApplicantIncome = st.number_input("Applicant Monthly Income", 
                                       min_value=0, 
                                       step=500)
    
    # coapplicant's monthly income
    CoapplicantIncome = st.number_input("Co-applicant Monthly Income", 
                                         min_value=0, 
                                         step=100)
    
    # loan amount in thousands
    LoanAmount = st.number_input("Loan Amount (x1000)", 
                                  min_value=0, 
                                  step=50)
    
    # loan amount term in months
    Loan_Amount_Term = st.selectbox("Loan Amount Term (Months)", 
                                    options=["360","300", "240", "180", "120", "60"])
    
    # credit history
    Credit_History = st.selectbox("Credit History", 
                                  options=["1", "0"])
    
    # current property location
    Property_Area = st.selectbox("Property Area", 
                                 options=["Urban", "Semiurban", "Rural"])
    
    # submit button
    submitted = st.form_submit_button("Predict Loan Eligibility")

# handle the dummy variables to pass to the model
if submitted & (rf_model is not None) & (sc is not None):
    
    Gender_Male = 0 if Gender == "Female" else 1
    Gender_Female = 1 if Gender == "Female" else 0

    Married_Yes = 1 if Married == "Yes" else 0
    Married_No = 1 if Married == "No" else 0

    # handle the number of dependents
    Dependents_0 = 1 if Dependents == "0" else 0
    Dependents_1 = 1 if Dependents == "1" else 0
    Dependents_2 = 1 if Dependents == "2" else 0
    Dependents_3 = 1 if Dependents == "3+" else 0

    Education_Graduate = 1 if Education == "Graduate" else 0
    Education_Not_Graduate = 1 if Education == "Not Graduate" else 0

    Self_Employed_Yes = 1 if Self_Employed == "Yes" else 0
    Self_Employed_No = 1 if Self_Employed == "No" else 0

    Property_Area_Rural = 1 if Property_Area == "Rural" else 0
    Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
    Property_Area_Urban = 1 if Property_Area == "Urban" else 0

    # convert Loan Amount Term and Credit History to integers
    Loan_Amount_Term = int(Loan_Amount_Term)
    Credit_History = int(Credit_History)

    # prepare the input for prediction, same order as training
    prediction_input = [[ApplicantIncome, CoapplicantIncome, LoanAmount,
        Loan_Amount_Term, Credit_History, Gender_Female, Gender_Male,
        Married_No, Married_Yes, Dependents_0, Dependents_1,
        Dependents_2, Dependents_3, Education_Graduate,
        Education_Not_Graduate, Self_Employed_No, Self_Employed_Yes,
        Property_Area_Rural, Property_Area_Semiurban, Property_Area_Urban
    ]]
    
    # scale the input features
    @log_decorator
    def transform_inputs(prediction_input):
        logger.info("Starting Transformation of Inputs")
        
        prediction_input = sc.transform(prediction_input)
        
        logger.info("Finished Transformations")
        return prediction_input
    
    prediction_input_scaled = transform_inputs(prediction_input)

    # perform prediction
    @log_decorator
    def predict_price(prediction_input):
        logger.info("Prediction Started")
        
        new_prediction = rf_model.predict(prediction_input)
        
        logger.info("Prediction Finished")
        return new_prediction
    
    new_prediction = predict_price(prediction_input_scaled)
    
    if new_prediction is not None:

        # display the prediction result
        st.subheader("Prediction Result:")
        if new_prediction[0] == 1:
            st.write("You are eligible for the loan!")
            logger.info("You are eligible for the loan!")
        else:
            st.write("Sorry, you are not eligible for the loan.")
            logger.info("Sorry, you are not eligible for the loan.")
        
    else:
        
        # prediction not able to run
        st.write("The Prediction was not able to run!")

st.write(
    """We used a machine learning (Random Forest) model to predict your eligibility, the features used in this prediction are ranked by relative
    importance below."""
)

# display the importance of each feature to the model
st.image("feature_importance.png")
