from fastapi import FastAPI
from pydantic import BaseModel,Field
from fastapi.responses import FileResponse,JSONResponse
from typing import Annotated,Literal
import joblib
import os
import pandas as pd

app = FastAPI()

model = joblib.load('model.joblib')
encoders = joblib.load('encoders.joblib')
scaler = joblib.load('scaler.joblib')
MODEL_VERSION = '1.11.0'

class Input_Data(BaseModel):
    person_age:Annotated[int, Field(...,description = "Age of the Applicant")]
    person_gender:Annotated[Literal['female','male'],Field(...,description = "Gender of the Applicatn")]
    person_education:Annotated[Literal['Bachelor','Associate','Hight School','Master','Doctorate'],Field(...,description = "Education of the Applicatn")]
    person_income:int =Field(...,description = "How many year when he employed")
    person_home_ownership:Annotated[Literal['RENT', 'OWN', 'MORTGAGE'],Field(...,description = "Home ownership of the Applicant")]
    person_emp_exp:Annotated[int,Field(...,description = "Year Income in Dollar of the Applicatn")]
    loan_amnt:Annotated[int,Field(...,description = "Loan Amount in Dollar for applicant")]
    loan_intent:Annotated[Literal['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT','DEBTCONSOLIDATION'],Field(...,description = "Loan Purpose for applicant")]
    loan_int_rate:float =Field(...,description = "Loan interest ratio for applicant")
    cb_person_cred_hist_length:int =Field(...,description = "when he start his credict card")
    credit_score:int=Field(...,description = "Credit Score of the Applicant")
    previous_loan_defaults_on_file:Annotated[Literal['No', 'Yes'],Field(...,description = "Previous the applicant loan default or not")]

    def loan_percent_income(self) -> int:
        return (self.loan_amount / self.person_income) * 100

@app.get("/")
def read_index():
    return FileResponse("templates/index.html")

@app.get("/health")
def health_check():
    return {
        'status':'OK',
        'version':MODEL_VERSION,
        'model_loaded':model is not None
    }

@app.post('/predict')
def predict(data: Input_Data):
    # Encode categorical features
    encoded = {}
    encoded['person_gender'] = encoders['person_gender'].transform([data.person_gender])[0]
    encoded['person_education'] = encoders['person_education'].transform([data.person_education])[0]
    encoded['person_home_ownership'] = encoders['person_home_ownership'].transform([data.person_home_ownership])[0]
    encoded['loan_intent'] = encoders['loan_intent'].transform([data.loan_intent])[0]
    encoded['previous_loan_defaults_on_file'] = encoders['previous_loan_defaults_on_file'].transform([data.previous_loan_defaults_on_file])[0]

    encoded_df = pd.DataFrame([encoded])

    # Scale numeric features
    scaled = scaler.transform([[data.person_age, data.person_income,
                                data.person_emp_exp, data.loan_amnt,
                                data.loan_int_rate, data.credit_score]])
    scaled_df = pd.DataFrame(scaled, columns=['person_age','person_income','person_emp_exp',
                                              'loan_amnt','loan_int_rate','credit_score'])

    # Extra numeric features
    cred_hist_df = pd.DataFrame([[data.cb_person_cred_hist_length]], 
                                columns=['cb_person_cred_hist_length'])

    loan_percent_income_df = pd.DataFrame([[data.loan_amnt / data.person_income]], 
                                          columns=['loan_percent_income'])

    # Final feature vector (keep same column order as training)
    final_features = pd.concat(
        [encoded_df, scaled_df, cred_hist_df, loan_percent_income_df],
        axis=1
    )

    # Prediction
    prediction = model.predict(final_features)[0]
    result = int(prediction)

    return JSONResponse(
    status_code=200,content={"Prediction": "Approved" if result == 1 else "Not Approved"})
