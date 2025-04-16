import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Define disease groups based on the provided list.
disease_groups = {
    "Neurological": [
        "Alzheimer's disease", "accident cerebrovascular", "aphasia", "confusion", "encephalopathy",
        "epilepsy", "migraine disorders", "neuropathy", "parkinson disease", "tonic-clonic epilepsy",
        "tonic-clonic seizures", "transient ischemic attack", "deglutition disorder", "incontinence",
        "delirium", "dementia"
    ],
    "Infectious": [
        "HIV", "Pneumocystis carinii pneumonia", "acquired immuno-deficiency syndrome", "bacteremia",
        "candidiasis", "cellulitis", "infection", "infection urinary tract", "influenza",
        "oral candidiasis", "pneumonia", "pneumonia aspiration", "sepsis (invertebrate)",
        "septicemia", "systemic infection", "upper respiratory infection", "osteomyelitis",
        "hiv infections"
    ],
    "Cardiovascular": [
        "cardiomyopathy", "coronary arteriosclerosis", "coronary heart disease", "deep vein thrombosis",
        "edema pulmonary", "effusion pericardial", "embolism pulmonary", "failure heart",
        "failure heart congestive", "hypertensive disease", "hypertension pulmonary", "ischemia",
        "mitral valve insufficiency", "myocardial infarction", "overload fluid", "paroxysmal dyspnea",
        "pericardial effusion body substance", "peripheral vascular disease", "stenosis aortic valve",
        "tachycardia sinus", "thrombus", "tricuspid valve insufficiency", "endocarditis"
    ],
    "Gastrointestinal": [
        "adhesion", "biliary calculus", "cholecystitis", "cholelithiasis", "cirrhosis", "colitis",
        "diverticulitis", "diverticulosis", "gastritis", "gastroenteritis", "gastroesophageal reflux disease",
        "hemorrhoids", "hepatitis", "hepatitis B", "hepatitis C", "hernia", "hernia hiatal",
        "ileus", "pancreatitis", "ulcer peptic"
    ],
    "Neoplasms": [
        "adenocarcinoma", "carcinoma", "carcinoma breast", "carcinoma colon", "carcinoma of lung",
        "carcinoma prostate", "fibroid tumor", "malignant neoplasm of breast", "malignant neoplasm of lung",
        "malignant neoplasm of prostate", "malignant neoplasms", "malignant tumor of colon", "melanoma",
        "neoplasm", "neoplasm metastasis", "primary carcinoma of the liver cells", "primary malignant neoplasm",
        "lymphoma"
    ],
    "Psychiatric": [
        "affect labile", "anxiety state", "bipolar disorder", "delusion", "dependence",
        "depression mental", "depressive disorder", "manic disorder", "paranoia", "personality disorder",
        "psychotic disorder", "schizophrenia", "suicide attempt"
    ],
    "Renal": [
        "chronic kidney failure", "failure kidney", "insufficiency renal", "kidney disease",
        "kidney failure acute", "pyelonephritis"
    ],
    "Musculoskeletal": [
        "arthritis", "degenerative polyarthritis", "osteoporosis"
    ],
    "Endocrine/Metabolic": [
        "diabetes", "hyperglycemia", "hypoglycemia", "obesity", "obesity morbid",
        "ketoacidosis diabetic", "hypercholesterolemia", "hyperlipidemia", "hypothyroidism",
        "hyperbilirubinemia", "gout"
    ],
    "Hematologic": [
        "anemia", "lymphatic diseases", "neutropenia", "pancytopenia", "sickle cell anemia", "thrombocytopaenia"
    ],
    "Respiratory": [
        "asthma", "bronchitis", "chronic obstructive airway disease", "emphysema pulmonary",
        "pneumothorax", "spasm bronchial", "respiratory failure"
    ],
    "Ophthalmologic": [
        "glaucoma"
    ],
    "Urological": [
        "benign prostatic hypertrophy"
    ],
    "Dermatological": [
        "exanthema"
    ],
    "Miscellaneous": [
        "chronic alcoholic intoxication", "decubitus ulcer", "dehydration"
    ]
}

# Create a reverse lookup: map each specific disease to its group.
disease_to_group = {}
for group, diseases in disease_groups.items():
    for disease in diseases:
        disease_to_group[disease] = group

# Load the training dataset (which contains one row per [disease, symptom, occurrence_count])
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "diseases.xlsx")
cleaned_data_path = os.path.join(BASE_DIR, "cleaned_data.csv")
training_data_path = os.path.join(BASE_DIR, "training_dataset.csv")
df = pd.read_csv(training_data_path)

# Features are the one-hot encoded symptoms (starting at column 2) and labels are diseases (first column)
X = df.iloc[:, 1:]
y = df['disease']

# Map the actual disease labels to their groups.
# If a disease isn't found in our mapping, label it as 'unknown'
y_grouped = y.map(disease_to_group).fillna('unknown')

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_grouped, test_size=0.2, random_state=101)

# Train a Multinomial Naïve Bayes model
clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)

# Get predictions and probability estimates
y_pred = clf_nb.predict(X_test)
y_pred_proba = clf_nb.predict_proba(X_test)
group_classes = clf_nb.classes_

correct_predictions = 0
total_predictions = len(y_test)

# Evaluate each test case
for i in range(len(y_test)):
    actual_group = y_test.iloc[i]
    predicted_group = y_pred[i]

    # Retrieve symptoms present in this test case (features with value 1)
    symptoms_present = X_test.iloc[i]
    active_symptoms = symptoms_present[symptoms_present == 1].index.tolist()

    # Determine the top 5 predicted groups with confidence scores
    prediction_probs = y_pred_proba[i]
    top_5_indices = np.argsort(prediction_probs)[::-1][:5]
    top_5_predictions = [(group_classes[idx], prediction_probs[idx] * 100) for idx in top_5_indices]

    # Output the test case details
    print(f"\nTest Case {i+1}:")
    print(f"Symptoms Present: {', '.join(active_symptoms)}")
    print(f"Actual Disease Group: {actual_group}")
    print(f"Predicted Disease Group: {predicted_group}")
    print("Top 5 Predictions:")
    for group, confidence in top_5_predictions:
        print(f"- {group}: {confidence:.2f}% confidence")

    if predicted_group == actual_group:
        correct_predictions += 1

# Calculate and print group-based accuracy
accuracy = correct_predictions / total_predictions
print(f"\nModel Accuracy (group-based): {accuracy:.2%}")

#Predictions based on user input 

def predict_user_disease(symptoms):
    user_input= pd.DataFrame([[0] * X.shape[1]], columns=X.columns)
    #one hot encodes if the symptom is present or not 
    for symptom in symptoms:
        if symptom in user_input.columns:
            user_input.at[0, symptom] = 1;
    # Make prediction
    predicted_group = clf_nb.predict(user_input)[0]
    prediction_probs = clf_nb.predict_proba(user_input)[0]
  
    top_5_indices = np.argsort(prediction_probs)[::-1][:5]
    top_5_predictions = [(group_classes[idx], prediction_probs[idx] * 100) for idx in top_5_indices]
    
    predictions = f"Predicted Disease Group: {predicted_group}\n"; 
    predictions += "\nTop 5 Predictions:\n"
    for group, confidence in top_5_predictions:
         predictions += (f"- {group}: {confidence:.2f}% confidence\n")
    return predictions; 
