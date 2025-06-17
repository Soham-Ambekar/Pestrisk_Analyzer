from flask import Flask, render_template, request
import modules.pesticide as pesticide_module
import modules.skin_cancer as skin_cancer_module
import modules.chatbot as chatbot_module
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Chatbot route
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    response = None
    error = None
    if request.method == 'POST':
        user_query = request.form['query']
        try:
            response = chatbot_module.get_gemini_response(user_query)
        except Exception as e:
            error = str(e)
            print(f"Chatbot Error: {e}")
    return render_template('chatbot.html', response=response, error=error)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json()
        user_query = data.get('message')
        if not user_query:
            return jsonify({'error': 'Missing "message" in request'}), 400

        response = chatbot_module.get_gemini_response(user_query)
        return jsonify({'response': response})
    except Exception as e:
        print(f"API Chatbot Error: {e}")
        return jsonify({'error': str(e)}), 500



# Pesticide Prediction route
# Pesticide Prediction route
# Load the saved label encoder and models
with open("S:/FinalYearProject_PestiRisk/FinalYearProject/modules/label_encoder.pkl", 'rb') as f:
    label_encoder = pickle.load(f)


with open('S:/FinalYearProject_PestiRisk/FinalYearProject/modules/model_disease.pkl', 'rb') as f:
    model_disease = pickle.load(f)

with open('S:/FinalYearProject_PestiRisk/FinalYearProject/modules/model_symptoms.pkl', 'rb') as f:
    model_symptoms = pickle.load(f)

@app.route('/pesticide-prediction', methods=['GET', 'POST'])
def pesticide_prediction():
    chronic_disease = None
    symptoms = None

    if request.method == 'POST':
        input_data = request.form.get('input_data')

        if input_data:
            try:
                # Encode the pesticide name
                pesticide_encoded = label_encoder.transform([input_data])

                # Predict disease and symptoms
                predicted_disease = model_disease.predict([pesticide_encoded])
                predicted_symptoms = model_symptoms.predict([pesticide_encoded])

                # Decode the predicted outputs
                chronic_disease = predicted_disease[0]
                symptoms = predicted_symptoms[0]
            except Exception as e:
                chronic_disease = "Error in prediction"
                symptoms = str(e)

    return render_template('pesticide.html', chronic_disease=chronic_disease, symptoms=symptoms)



#Skin Cancer Detection route

@app.route('/skin-cancer-detection', methods=['GET', 'POST'])
def skin_cancer_detection():
    prediction = None
    if request.method == 'POST':
        image = request.files.get('image')
        if image:
            prediction = skin_cancer_module.predict(image)  # Call the actual prediction function
        else:
            prediction = "No image uploaded!"
    return render_template('skin_cancer.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)