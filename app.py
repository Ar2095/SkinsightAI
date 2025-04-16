import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import streamlit as st
from disease_classifier import predict_user_disease


if __name__ == "__main__":
    st.title('Symptom-Disease Matcher')
    st.subheader("Enter your symptoms to learn what disease/issue your body may be experiencing. Type in each symptom separared by commas.")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    training_data_path = os.path.join(BASE_DIR, "training_dataset.csv")
    df = pd.read_csv(training_data_path)
    X = df.columns[1:].tolist()
    y = df['disease']
    symptoms = st.multiselect('Select Your Symptoms: ', X)
    if (st.button("Predict")):
        st.write(predict_user_disease(symptoms)); 
        