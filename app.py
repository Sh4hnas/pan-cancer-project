import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from flask import Flask, request, jsonify, render_template
import sqlite3
import os

file_path = "data/1-s2.0-S1535947625000179-mmc1.xlsx"
df = pd.read_excel(file_path)

# Data Preprocessing (Modify according to dataset structure)
df = df.dropna()  # Remove missing values if any
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target variable (Cancer stage/risk)

# Step 2: Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Save the trained model
with open("cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Step 3: Setup Flask App
app = Flask(__name__)

# Ensure database exists
db_path = "patients.db"
if not os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, risk INTEGER)''')
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_features = np.array(data['features']).reshape(1, -1)
    
    with open("cancer_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    prediction = loaded_model.predict(input_features)[0]
    
    # Store in database
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("INSERT INTO patients (name, age, risk) VALUES (?, ?, ?)", (data['name'], data['age'], int(prediction)))
    conn.commit()
    conn.close()
    
    return jsonify({'prediction': int(prediction)})

@app.route('/patients', methods=['GET'])
def get_patients():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM patients")
    patients = c.fetchall()
    conn.close()
    return jsonify(patients)

if __name__ == '__main__':
    app.run(debug=True)
