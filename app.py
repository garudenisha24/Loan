import streamlit as st 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
train_data = pd.read_csv("Titanic_Train.csv")
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
import joblib
import re 

model = joblib.load("Logistic_model.pkl")
st.title("Titanic Logistic Regression Model")

# Prepare title encoder from training data
#def extract_title(name):
   # match = re.search(r',\s*([^.]*)\.', name)
    #if match:
     #   return match.group(1).strip()
    #match2 = re.search(r'\b(Mr|Mrs|Miss|Master|Dr|Rev|Col|Major|Mlle|Mme|Ms|Sir|Lady|Capt|Don|Jonkheer)\b', name)
    #if match2:
     #   return match2.group(1)
    #return "Mr"

#train_titles = train_data['Name'].apply(extract_title)
#title_le = LabelEncoder()
# title_le.fit(train_titles)
    
with st.form("titanic_form"):
     
    passengerid = st.number_input("Passenger I", 1,1000,1)
    
    pclass_options = {
        "First Class": 1,
        "Second Class": 2,
        "Third Class": 3
    }
    pclass_name = st.selectbox("Passenger Class", list(pclass_options.keys()))
    
    
   # Name = st.text_input("Name")

    sex = st.radio('Sex',["Male","Female"])
    age = st.slider("Age", 0,100,25)
    sibsp = st.number_input('Number of Siblings/Spouses Aboard', 0, 10, 0)
    parch = st.number_input('Number of Parents/Children Aboard', 0, 10, 0)
    ticket = st.text_input("Ticket")
    fare = st.number_input('Fare',0.0, 500.0, 50.0)
    #cabin = st.text_input("Cabin")
    
    embarked_le = LabelEncoder()
    embarked = st.selectbox("Embarked", ["C","Q","S"])
    embarked_le.fit(train_data['Embarked'])
    
    submit = st.form_submit_button("Predict")
    

if submit:
    pclass_num = pclass_options[pclass_name]
    #pclass = 1 if pclass == "First Class" else (2 if pclass == "Second Class" else 3)   
    sex_num = 0 if sex == 'Male' else 1
    embarked_num = embarked_le.transform([embarked])[0]
    #embarked = 0 if embarked == 'C' else (1 if embarked == 'Q' else 2)
    # Handle unseen titles
    # Extract title from Name input
    #title = extract_title(Name)

# Handle unseen titles
    #if Name not in title_le.classes_:
     #   st.warning(f"Title '{Name}' not recognized. Using default title 'Mr'.")
      #  title = "Mr"

# Encode the title
   #name_encoding = int(title_le.transform([Name])[0])
    
    
    
   
    
    input_data = pd.DataFrame([{
        'PassengerId': passengerid,
        'Pclass': pclass_num,
        #'Name' : name_encoding,
        'Sex': sex_num,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Ticket': ticket,
        'Fare': fare,
        #'Cabin': cabin,
        'Embarked': embarked_num
        }], dtype='object')
    
   # prediction = model.predict(input_data)
    #probability = model.predict_proba(input_data)[:, 1]
    
   # print("Prediction (0=Not Survived, 1=Survived):", prediction[0])
   # print("Survival Probability:", probability[0])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]

    st.subheader("Prediction Result")
    st.success(f"Prediction: {'Survived' if prediction[0] == 1 else 'Not Survived'}")
    st.info(f"Survival Probability: {probability[0]:.2f}")