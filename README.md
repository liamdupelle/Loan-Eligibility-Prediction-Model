# Credit Loan Eligibility Predictor
This application predicts the eligibility of an applicant to receive a loan at a bank based on inputs derived from the German Credit Risk dataset. The model aims to help users assess loan eligibility by leveraging machine learning predictions. 

## Features
- User-friendly interface powered by Streamlit.
- Input form to enter details such as credit history, marriage status, monthly income, and other relevant features.
- Real-time prediction of loan eligibility based on the trained model.

## Dataset
The application includes features like:
- Gender
- Age
- Monthly income
- Loan amount
- Loan term
- And other factors affecting credit risk.

## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization.

## Model
A Random Forest Classifier model is used to predict loan eligibility. 

## Instructions 
- Enter the applicant's personal and financial characteristics into the Loan Applicant Details form
- Submit the form by pressing the "Predict Loan Eligibility" button
- The eligibility of the applicant will be displayed in the "Prediction Result:" section
- The feature importance chart and confusion matrix are displayed below the prediction result