import streamlit as st
import pandas as pd
import numpy as np
import joblib

model=joblib.load("model.pkl")
scaler=joblib.load("scaler.pkl")
encoder=joblib.load("encoder.pkl")

st.title("Carrer Sucess  Predictor 💼")
st.write("Fill your details below to predict your Salary:")

age=st.number_input("Age",min_value=18,max_value=80,value=25)
gender=st.selectbox("Gender",["Male","Female"])
education=st.selectbox("Education Level",["Bachelor's","Master's","PhD"])
experience=st.number_input("Years of Experience",min_value=0.0,max_value=50.0,value=2.0)
job_title=st.text_input("Job Title (e.g., Junior Engineer ,Senior Analyst")

def extract_level(title):
    title = title.lower()
    if "junior" in title:
        return "Junior"
    elif any(x in title for x in ["senior", "principal"]):
        return "Senior"
    elif any(x in title for x in ["lead", "manager", "director", "chief", "vp"]):
        return "Senior"
    else:
        return "Mid"

career_level = extract_level(job_title)

input_df=pd.DataFrame({
    "Gender":[gender],
    "Education Level":[education],
    "Carrer_Level":[career_level]
})

encoded_cat=encoder.transform(input_df)
encoded_cat_df=pd.DataFrame(encoded_cat,columns=encoder.get_feature_names_out())

num_data=pd.DataFrame({
    "Age":[age],
    "Years of Experience":[experience]
})
scaled_num=scaler.transform(num_data)
scaled_num_df=pd.DataFrame(scaled_num,columns=num_data.columns)

final_input=pd.concat([scaled_num_df,encoded_cat_df],axis=1)

if st.button("Predict Salary 💰"):
    prediction=model.predict(final_input)[0]
    st.success(f"Estimated Salary: ${prediction:,.2f}")