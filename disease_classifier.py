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

# Map each disease to its group
disease_to_group = {}
for group, diseases in disease_groups.items():
    for disease in diseases:
        disease_to_group[disease] = group

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
training_data_path = os.path.join(BASE_DIR, "training_dataset.csv")
df = pd.read_csv(training_data_path)

# Prepare features and labels
X = df.iloc[:, 1:]
y = df['disease']
y_grouped = y.map(disease_to_group).fillna('unknown')

# Split data for group classifier
X_train, X_test, y_train, y_test = train_test_split(X, y_grouped, test_size=0.2, random_state=101)
clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)
group_classes = clf_nb.classes_

# Split data for individual disease classifier
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y, test_size=0.2, random_state=101)
clf_nb_disease = MultinomialNB()
clf_nb_disease.fit(X_train_d, y_train_d)
disease_classes = clf_nb_disease.classes_

# Evaluate on test set
y_pred = clf_nb.predict(X_test)
y_pred_proba = clf_nb.predict_proba(X_test)
correct_predictions = 0
total_predictions = len(y_test)

for i in range(len(y_test)):
    actual_group = y_test.iloc[i]
    actual_disease = y_test_d.iloc[i]
    predicted_group = y_pred[i]
    symptoms_present = X_test.iloc[i]
    active_symptoms = symptoms_present[symptoms_present == 1].index.tolist()

    prediction_probs = y_pred_proba[i]
    top_5_indices = np.argsort(prediction_probs)[::-1][:5]
    top_5_predictions = [(group_classes[idx], prediction_probs[idx] * 100) for idx in top_5_indices]

    disease_probs = clf_nb_disease.predict_proba(pd.DataFrame([symptoms_present], columns=X.columns))[0]
    top_5_disease_indices = np.argsort(disease_probs)[::-1][:5]
    top_5_diseases = [(disease_classes[idx], disease_probs[idx] * 100) for idx in top_5_disease_indices]

    print(f"\nTest Case {i+1}:")
    print(f"Symptoms Present: {', '.join(active_symptoms)}")
    print(f"Actual Disease Group: {actual_group}")
    print(f"Actual Disease: {actual_disease}")

    print("Top 5 Disease Group Predictions:")
    for group, confidence in top_5_predictions:
        print(f"- {group}: {confidence:.2f}% confidence")

    print("Top 5 Disease Predictions:")
    for disease, confidence in top_5_diseases:
        print(f"- {disease}: {confidence:.2f}% confidence")

    if predicted_group == actual_group:
        correct_predictions += 1

# Accuracy metrics
group_accuracy = correct_predictions / total_predictions
print(f"\nModel Accuracy (group-based): {group_accuracy:.2%}")

y_pred_disease = clf_nb_disease.predict(X_test_d)
disease_accuracy = accuracy_score(y_test_d, y_pred_disease)
print(f"Model Accuracy (individual disease-based): {disease_accuracy:.2%}")

# Predict diseases from user symptoms
def predict_user_disease(symptoms):
    user_input = pd.DataFrame([[0] * X.shape[1]], columns=X.columns)
    for symptom in symptoms:
        if symptom in user_input.columns:
            user_input.at[0, symptom] = 1

    predicted_group = clf_nb.predict(user_input)[0]
    prediction_probs = clf_nb.predict_proba(user_input)[0]
    top_5_indices = np.argsort(prediction_probs)[::-1][:5]
    top_5_groups = [(group_classes[idx], prediction_probs[idx] * 100) for idx in top_5_indices]

    disease_probs = clf_nb_disease.predict_proba(user_input)[0]
    top_5_disease_indices = np.argsort(disease_probs)[::-1][:5]
    top_5_diseases = [(disease_classes[idx], disease_probs[idx] * 100) for idx in top_5_disease_indices]

    group_results = f"### Predicted Disease Groups:\n"
    for group, confidence in top_5_groups:
        group_results += f"- {group}<br><sub>{confidence:.2f}% confidence</sub>\n"

    disease_results = f"### Predicted Individual Diseases:\n"
    for disease, confidence in top_5_diseases:
        disease_results += f"- {disease}<br><sub>{confidence:.2f}% confidence</sub>\n"

    return group_results, disease_results
