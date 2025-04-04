import streamlit as st
import pandas as pd
import numpy as np



if __name__ == "__main__":
    st.title('Symptom-Disease Matcher')
    st.subheader("Enter your symptoms to learn what disease/issue your body may be experiencing.")
    st.chat_input("Type your symptoms here")