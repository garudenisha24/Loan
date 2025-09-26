import streamlit as st 
import pandas as pd 
import numpy as np
import joblib

model = joblib.load("Logistic_model.pkl")

st.title("Titanic Survival Prediction")
# st.background_image("titenic.jpg")

# User Input Form

with st.form('Prediction_Form'):
   # pclass = st.selectbox('Passenger Class',['First Class','Second Class','Third Class'])
    age = st.slider('Age',0,100,25)
    
    
    sibsp = st.number_input('Number of Siblings/Spouses Aboard', 0, 10, 0)
    parch = st.number_input('Number of Parents/Children Aboard', 0, 10, 0)
    fare = st.number_input('Fare',0.0, 500.0, 50.0)
    sex = st.radio('Sex',['Male','Female'])
    submit = st.form_submit_button('Predict')
    
if submit:
    sex = 0 if sex == 'male' else 1
    input_data = pd.DataFrame([{age,sibsp,parch,fare,sex}])
    
    
    
                              # columns=['Pclass','Sex','Age','Sibsp','Parch','Fare'])
    
    # Make Prediction
    
    prediction = model.predict(input_data)
    result = 'Survived' if prediction[0] == 1 else "Did not Survived"
    st.success(f"Prediction: {result}")
 