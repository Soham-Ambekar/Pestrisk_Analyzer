import pickle
from sklearn.preprocessing import LabelEncoder

# Load the serialized models
model_path_disease = "S:/FinalYearProject_PestiRisk/FinalYearProject/modules/model_disease.pkl"
model_path_symptoms = "S:/FinalYearProject_PestiRisk/FinalYearProject/modules/model_symptoms.pkl"
encoder_path = "S:/FinalYearProject_PestiRisk/FinalYearProject/modules/label_encoder.pkl"
with open(model_path_disease, 'rb') as file:
    model_disease = pickle.load(file)

with open(model_path_symptoms, 'rb') as file:
    model_symptoms = pickle.load(file)

# Load the label encoder (if applicable)
with open(encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)

def predict(pesticide_name):
    # Encode the pesticide name to match the model input
    try:
        pesticide_encoded = label_encoder.transform([pesticide_name])
    except ValueError:
        return "Error: Pesticide name not recognized. Please use a valid name."

    # Get predictions
    chronic_disease_prediction = model_disease.predict([pesticide_encoded])
    symptoms_prediction = model_symptoms.predict([pesticide_encoded])

    return chronic_disease_prediction[0], symptoms_prediction[0]
