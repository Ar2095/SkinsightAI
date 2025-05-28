import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

st.title("Disease Classification Accuracy Test")

# Load local CSV
csv_path = "new_data.csv"
df = pd.read_csv(csv_path).dropna()
df = pd.get_dummies(df, columns=[col for col in df.columns if df[col].dtype == 'object' and col != 'Disease'])


# Show preview
st.write("📊 Dataset Preview:")
st.write(df.head())
st.write(df["Disease"].value_counts())


if "Disease" not in df.columns:
    st.error("❌ The dataset must contain a 'Disease' column.")
else:
    diseases = sorted(df["Disease"].unique())
    selected_disease = st.selectbox("🎯 Choose a target disease for binary classification:", diseases)

    # Binary label for selected disease
    df["Target"] = df["Disease"].apply(lambda x: 1 if x == selected_disease else 0)

    # Encode categorical columns
    encoders = {}
    for col in df.columns:
        if df[col].dtype == "object" and col not in ["Disease"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # Binary classification
    X = df.drop(columns=["Disease", "Target"])
    y_binary = df["Target"]
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    #model_binary = RandomForestClassifier(random_state=42)
    base_model =  XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    base_model.fit(Xb_train, yb_train)
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv="prefit")

    #gaussian_model = GaussianNB()
    #regression_model = LogisticRegression(max_iter=1000)



    calibrated_model.fit(Xb_train, yb_train)
    yb_pred = calibrated_model.predict(Xb_test)
    binary_accuracy = accuracy_score(yb_test, yb_pred)


    st.success(f"✅ Binary accuracy (detecting '{selected_disease}'): **{binary_accuracy * 100:.2f}%**")

    # Multiclass classification
    label_encoder = LabelEncoder() 

    y_multi = df["Disease"]
    y_multi_encoded = label_encoder.fit_transform(y_multi)

    Xm_train, Xm_test, ym_train, ym_test = train_test_split(X, y_multi_encoded, test_size=0.2, random_state=42)

    multi_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    multi_model.fit(Xm_train, ym_train)
    ym_pred = multi_model.predict(Xm_test)

    overall_accuracy = accuracy_score(ym_test, ym_pred)

    st.info(f"📈 Multiclass accuracy (exact match): **{overall_accuracy * 100:.2f}%**")

    # Top-5 accuracy calculation
    proba = multi_model.predict_proba(Xm_test)

    top5_preds = np.argsort(proba, axis=1)[:, -5:]  # Get indices of top 5 classes
    #class_labels = multi_model.classes_
    top5_labels = multi_model.classes_[top5_preds]

    # Check if true label is in top 5
    correct_top5 = [true_label in pred_row for true_label, pred_row in zip(ym_test, top5_labels)]
    top5_accuracy = np.mean(correct_top5)

    st.success(f"⭐ Top-5 accuracy (true disease in top 5 guesses): **{top5_accuracy * 100:.2f}%**")

    # Optional: Show top 5 predictions for a few test samples
    if st.checkbox("🔍 Show top-5 predictions for first 5 test cases"):
        test_df = Xm_test.copy()
        test_df["True Disease"] = ym_test.values
        for i in range(min(5, len(test_df))):
            row = test_df.iloc[i]
            top5 = top5_labels[i][::-1]  # Reverse to show best first
            st.write(f"🧪 **Test case {i+1}**")
            st.write(f"True disease: **{row['True Disease']}**")
            st.write("Top-5 predicted diseases (best to worst):")
            for rank, pred in enumerate(top5, start=1):
                st.write(f"{rank}. {pred}")

