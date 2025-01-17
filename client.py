from flask import Flask, request, jsonify, render_template
import requests
import re

app = Flask(__name__)

REST_API_URL = "http://127.0.0.1:5000"

import joblib
lst = joblib.load("symptom_list.pkl");
psymptoms=lst
######### Helper function to extract symptoms from user input
# def extract_symptoms(user_input):
#     extracted = [symptom for symptom in psymptoms if symptom in user_input.lower()]
#     return extracted


## Helper function to extract symptoms from user input
def extract_symptoms(user_input):
    # Normalize input: replace spaces and hyphens with underscores
    normalized_input = re.sub(r"[\s\-]+", "_", user_input.strip().lower())
    
    # Split by common delimiters (comma, space, etc.)
    input_symptoms = re.split(r",|\s+", normalized_input)
    
    # Match symptoms in the preloaded symptom list
    return [symptom for symptom in input_symptoms if symptom in psymptoms]


@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/process-chat', methods=['POST'])
def process_chat():
    user_input = request.form.get('user_input')
    
    if not user_input:
        return render_template('chat.html', error="Please enter some symptoms.")
    
    # Extract symptoms from the user input
    symptoms = extract_symptoms(user_input)
    
    if symptoms:
        # Send the extracted symptoms to the /predict endpoint for disease prediction
        response = requests.post(f"{REST_API_URL}/predict", json={"symptoms": symptoms})
        
        if response.status_code == 200:
            result = response.json()
            return render_template('chat.html', user_input=user_input, result=result)
        else:
            return render_template('chat.html', error="Error fetching prediction from REST API.")
    else:
        # If no symptoms are found, treat user input as a prompt for LLM response
        llm_prompt = user_input
        response = requests.post(f"{REST_API_URL}/generate-llm-response", json={"prompt": llm_prompt})
        
        if response.status_code == 200:
            llm_output = response.json()
            return render_template('chat.html', user_input=user_input, llm_output=llm_output)
        else:
            return render_template('chat.html', error="Error generating response from LLM.")

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)


