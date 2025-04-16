import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from disease_classifier import predict_user_disease

if __name__ == "__main__":
    st.title('Symptom-Disease Matcher')
    st.subheader("Enter your symptoms to learn what issue you may be experiencing.")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    training_data_path = os.path.join(BASE_DIR, "training_dataset.csv")
    df = pd.read_csv(training_data_path)

    symptom_list = df.columns[1:].tolist()
    symptoms = st.multiselect('Select Your Symptoms:', symptom_list)

    if st.button("Predict"):
        group_output, disease_output = predict_user_disease(symptoms)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(group_output, unsafe_allow_html=True)

        with col2:
            st.markdown(disease_output, unsafe_allow_html=True)
