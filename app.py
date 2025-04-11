import streamlit as st
import pandas as pd
import numpy as np

from disease_classifier import predict_user_disease


if __name__ == "__main__":
    st.title('Symptom-Disease Matcher')
    st.subheader("Enter your symptoms to learn what disease/issue your body may be experiencing. Type in each symptom separared by commas.")
    user_input  = st.chat_input("Type your symptoms here")

    if user_input:
        st.write(f"You entered: {user_input}")
        symptoms = user_input.split(",");  
        cleaned_symptoms = []; 
        for symptom in symptoms:
            symptom = symptom.strip();
            cleaned_symptoms.append(symptom)
        print(cleaned_symptoms);
        st.write(predict_user_disease(cleaned_symptoms)); 
        