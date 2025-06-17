import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

try:
    # Load the dataset
    dataset_path = 'C:/Users/Shravani/OneDrive/Desktop/FinalYearProject/modules/pesticides dataset.csv'  # Adjust path if necessary
    data = pd.read_csv(dataset_path)
    print("Dataset loaded successfully!")

    # Check if 'Pesticides Name' column exists
    if 'Pesticides Name' not in data.columns:
        raise ValueError("The dataset does not contain a 'Pesticides Name' column.")

    # Extract the "Pesticides Name" column
    pesticide_names = data['Pesticides Name'].unique()  # Get unique pesticide names
    print(f"Found {len(pesticide_names)} unique pesticide names.")

    # Create and fit the LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(pesticide_names)

    # Ensure the 'modules' directory exists
    os.makedirs('modules', exist_ok=True)

    # Save the LabelEncoder to a file
    encoder_path = 'modules/label_encoder.pkl'
    with open(encoder_path, 'wb') as file:
        pickle.dump(encoder, file)

    print(f"Label encoder created and saved as {encoder_path}.")

except Exception as e:
    print(f"Error: {e}")
