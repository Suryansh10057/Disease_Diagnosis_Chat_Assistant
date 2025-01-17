from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import requests

app = Flask(__name__)

# Load the model and datasets
rnd_forest = joblib.load("model.pkl")
discrp = pd.read_csv("content/symptom_Description.csv")
ektra7at = pd.read_csv("content/symptom_precaution.csv")
symptom_weights = pd.read_csv("content/Symptom-severity.csv")

# Hugging Face API setup
HUGGINGFACE_API_KEY = "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"
LLM_URL = "https://api-inference.huggingface.co/models/gpt2"

# Helper function to call Hugging Face LLM
def query_huggingface(payload):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    response = requests.post(LLM_URL, headers=headers, json=payload)
    return response.json()

# List of all possible symptoms (from the dataset)
# psymptoms = list(symptom_weights["Symptom"].unique())
lst = joblib.load("symptom_list.pkl");
symptoms=lst
# Function to predict disease and provide details
def predict_disease(model, symptoms):
    input_symptoms = [symptoms.get(f"symptom{i}", 0) for i in range(1, 18)]
    a = np.array(symptom_weights["Symptom"])
    b = np.array(symptom_weights["weight"])
    for j in range(len(input_symptoms)):
        for k in range(len(a)):
            if input_symptoms[j] == a[k]:
                input_symptoms[j] = b[k]
    psy = [input_symptoms]
    pred2 = model.predict(psy)
    disease = pred2[0]
    confidence_scores = model.predict_proba(psy)[0]
    return disease, confidence_scores

def explain_disease(disease):
    disp = discrp[discrp["Disease"] == disease].values[0][1]
    recomnd = ektra7at[ektra7at["Disease"] == disease]
    c = np.where(ektra7at["Disease"] == disease)[0][0]
    precautions = []
    for i in range(1, len(ektra7at.iloc[c])):
        precautions.append(ektra7at.iloc[c, i])
    return disp, precautions

# /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected_symptoms = data.get("symptoms", [])
    
    if not selected_symptoms:
        return jsonify({"error": "No symptoms selected"}), 400

    symptoms = {f"symptom{i + 1}": selected_symptoms[i] if i < len(selected_symptoms) else 0 for i in range(17)}
    
    disease, confidence_scores = predict_disease(rnd_forest, symptoms)
    
    # Fetch explanation of the disease
    description, precautions = explain_disease(disease)

    return jsonify({
        "disease": disease, 
        "confidence_scores": confidence_scores.tolist(),
        "description": description, 
        "precautions": precautions
    }), 200

# New LLM route
@app.route('/generate-llm-response', methods=['POST'])
def generate_llm_response():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    llm_response = query_huggingface({"inputs": prompt})
    return jsonify(llm_response), 200

if __name__ == '__main__':
    app.run(debug=True)

