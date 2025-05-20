import streamlit as st
import pandas as pd
<<<<<<< HEAD
=======
import numpy as np
>>>>>>> 353ba621 (skin classification app)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
<<<<<<< HEAD
import numpy as np

st.title("Disease Classification Accuracy Test")

# Load local CSV
csv_path = "new_data.csv"
df = pd.read_csv(csv_path).dropna()

# Show preview
st.write("📊 Dataset Preview:")
st.write(df.head())

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

    model_binary = RandomForestClassifier(random_state=42)
    model_binary.fit(Xb_train, yb_train)
    yb_pred = model_binary.predict(Xb_test)
    binary_accuracy = accuracy_score(yb_test, yb_pred)

    st.success(f"✅ Binary accuracy (detecting '{selected_disease}'): **{binary_accuracy * 100:.2f}%**")

    # Multiclass classification
    y_multi = df["Disease"]
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(X, y_multi, test_size=0.2, random_state=42)

    model_multi = RandomForestClassifier(random_state=42)
    model_multi.fit(Xm_train, ym_train)
    ym_pred = model_multi.predict(Xm_test)
    overall_accuracy = accuracy_score(ym_test, ym_pred)

    st.info(f"📈 Multiclass accuracy (exact match): **{overall_accuracy * 100:.2f}%**")

    # Top-5 accuracy calculation
    proba = model_multi.predict_proba(Xm_test)
    top5_preds = np.argsort(proba, axis=1)[:, -5:]  # Get indices of top 5 classes
    class_labels = model_multi.classes_
    top5_labels = class_labels[top5_preds]

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
=======

st.title("Disease Classification Accuracy Test")

# Load dataset
csv_path = "symbipredict.csv"
df = pd.read_csv(csv_path)

# Check for missing values
df = df.dropna()

# Show initial class distribution
st.subheader("📊 Original Class Distribution")
st.write(df["prognosis"].value_counts())

# Remove diseases with fewer than 5 instances
class_counts = df["prognosis"].value_counts()
valid_diseases = class_counts[class_counts >= 5].index
df = df[df["prognosis"].isin(valid_diseases)].copy()

# Display remaining diseases
st.subheader("✅ Remaining Diseases After Filtering (≥ 5 instances)")
st.write(df["prognosis"].value_counts())

# Encode the target
label_encoder = LabelEncoder()
df["Target"] = label_encoder.fit_transform(df["prognosis"])

# Binary classification setup
diseases = sorted(df["prognosis"].unique())
selected_disease = st.selectbox("🎯 Choose a target disease for binary classification:", diseases)

# Create binary labels
df["BinaryTarget"] = df["prognosis"].apply(lambda x: 1 if x == selected_disease else 0)

# Features and labels
X = df.drop(columns=["prognosis", "Target", "BinaryTarget"])
y_binary = df["BinaryTarget"]

# Split for binary classification
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

model_binary = RandomForestClassifier(random_state=42)
model_binary.fit(Xb_train, yb_train)
yb_pred = model_binary.predict(Xb_test)
binary_accuracy = accuracy_score(yb_test, yb_pred)

st.success(f"✅ Binary accuracy (detecting '{selected_disease}'): **{binary_accuracy * 100:.2f}%**")

# Multiclass classification
y_multi = df["Target"]
Xm_train, Xm_test, ym_train, ym_test = train_test_split(X, y_multi, test_size=0.2, random_state=42)

model_multi = RandomForestClassifier(random_state=42)
model_multi.fit(Xm_train, ym_train)
ym_pred = model_multi.predict(Xm_test)
overall_accuracy = accuracy_score(ym_test, ym_pred)

st.info(f"📈 Multiclass accuracy (exact match): **{overall_accuracy * 100:.2f}%**")

# Top-5 accuracy
proba = model_multi.predict_proba(Xm_test)
top5_preds = np.argsort(proba, axis=1)[:, -5:]
class_labels = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))
top5_labels = class_labels[top5_preds]

true_labels = label_encoder.inverse_transform(ym_test)
correct_top5 = [true_label in pred_row for true_label, pred_row in zip(true_labels, top5_labels)]
top5_accuracy = np.mean(correct_top5)

st.success(f"⭐ Top-5 accuracy (true disease in top 5 guesses): **{top5_accuracy * 100:.2f}%**")

# Optional: Show top-5 predictions for a few test samples
if st.checkbox("🔍 Show top-5 predictions for first 5 test cases"):
    test_df = Xm_test.copy()
    test_df["True Disease"] = true_labels
    for i in range(min(5, len(test_df))):
        row = test_df.iloc[i]
        top5 = top5_labels[i][::-1]
        st.write(f"🧪 **Test case {i+1}**")
        st.write(f"True disease: **{row['True Disease']}**")
        st.write("Top-5 predicted diseases (best to worst):")
        for rank, pred in enumerate(top5, start=1):
            st.write(f"{rank}. {pred}")
>>>>>>> 353ba621 (skin classification app)
