from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

# Load the trained model
model = load_model("S:/FinalYearProject_PestiRisk/FinalYearProject/modules/skin_cancer_model.h5")

def predict(uploaded_file):
    try:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)  # Ensure temp directory exists

        temp_path = os.path.join(temp_dir, uploaded_file.filename)
        uploaded_file.save(temp_path)  # Save uploaded file to temp directory

        # ✅ Correct way to load the image
        image = load_img(temp_path, target_size=(150, 150))  # Use the file path

        image_array = img_to_array(image) / 255.0  # Normalize the image
        image_array = image_array.reshape((1, 150, 150, 3))

        # Make prediction
        prediction = model.predict(image_array)

        # ✅ Ensure the file is removed after processing
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return "Cancer Detected" if prediction[0][0] > 0.7 else "No Cancer Detected"
    
    except Exception as e:
        return f"Error: {str(e)}"
