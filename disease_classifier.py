import csv
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Source

# Ensure correct working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "Desktop/AIApp/diseases.xlsx")
cleaned_data_path = os.path.join(BASE_DIR, "cleaned_data.csv")
training_data_path = os.path.join(BASE_DIR, "training_dataset.csv")
tree_output_path = os.path.join(BASE_DIR, "tree.dot")
tree_img_path = os.path.join(BASE_DIR, "tree.png")

# Read dataset
df = pd.read_excel(data_path, engine='openpyxl')
data = df.ffill()

def process_data(data):
    data_list = []
    data_name = data.replace('^', '_').split('_')
    for i, name in enumerate(data_name):
        if i % 2 == 1:
            data_list.append(name.strip())
    return data_list

disease_list = []
disease_symptom_dict = defaultdict(list)
disease_symptom_count = {}

for _, row in data.iterrows():
    disease = row['Disease'].strip()
    if disease:
        disease_list = process_data(disease)
        count = row['Count of Disease Occurrence']

    symptom = row['Symptom'].strip()
    if symptom:
        symptom_list = process_data(symptom)
        for d in disease_list:
            for s in symptom_list:
                disease_symptom_dict[d].append(s)
            disease_symptom_count[d] = count

# Save cleaned data
with open(cleaned_data_path, 'w', newline='') as f:
    writer = csv.writer(f)
    for key, val in disease_symptom_dict.items():
        for symptom in val:
            writer.writerow([key, symptom, disease_symptom_count[key]])

df = pd.read_csv(cleaned_data_path, names=['disease', 'symptom', 'occurence_count'])
df.replace(float('nan'), np.nan, inplace=True)
df.dropna(inplace=True)

# Encode categorical symptoms
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['symptom'])

onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

cols = df['symptom'].unique()
df_ohe = pd.DataFrame(onehot_encoded, columns=cols)

df_disease = df['disease']
df_concat = pd.concat([df_disease, df_ohe], axis=1).drop_duplicates()
df_concat = df_concat.groupby('disease').sum().reset_index()

df_concat.to_csv(training_data_path, index=False)

# Train Model
X = df_concat.iloc[:, 1:]  # Features
y = df_concat['disease']   # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

clf_dt = DecisionTreeClassifier().fit(X_train, y_train)

# Export Decision Tree
export_graphviz(clf_dt, out_file=tree_output_path, feature_names=X.columns)
graph = Source(export_graphviz(clf_dt, out_file=None, feature_names=X.columns))

# Save tree visualization
png_bytes = graph.pipe(format='png')
with open(tree_img_path, 'wb') as f:
    f.write(png_bytes)

# Evaluate Model
disease_pred = clf_dt.predict(X_test)
disease_real = y_test.values

for i in range(len(disease_real)):
    if disease_pred[i] != disease_real[i]:
        print(f'Predicted: {disease_pred[i]}, Actual: {disease_real[i]}')
