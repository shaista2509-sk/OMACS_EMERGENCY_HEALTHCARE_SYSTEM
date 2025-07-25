import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from groupnet_model import GroupNet


def load_model(model_path='ml_models/models/groupnet.pth'):
    """Load the trained GroupNet model"""
    checkpoint = torch.load(model_path, map_location='cpu')

    model = GroupNet(
        input_size=checkpoint['input_size'],
        num_diseases=checkpoint['num_diseases']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def preprocess_patient_data(patient_data, label_encoders, feature_names):
    """Preprocess patient data for model input"""
    # Ensure data is in correct order
    processed_data = []

    for feature in feature_names:
        if feature in ['Gender', 'Blood Type', 'Test Results', 'Medication', 'Admission Type']:
            # Apply label encoder
            le = label_encoders[feature]
            try:
                encoded_value = le.transform([str(patient_data[feature])])[0]
            except ValueError:
                # Handle unknown categories
                encoded_value = 0
            processed_data.append(encoded_value)
        else:
            # Numerical feature
            processed_data.append(float(patient_data[feature]))

    return np.array(processed_data, dtype=np.float32)


def predict_diseases(model, patient_data, disease_names, threshold=0.5):
    """Make disease predictions for a patient"""
    with torch.no_grad():
        input_tensor = torch.FloatTensor(patient_data).unsqueeze(0)
        predictions = model(input_tensor)
        probabilities = predictions.cpu().numpy()[0]

        # Apply threshold
        predicted_diseases = []
        for i, prob in enumerate(probabilities):
            if prob > threshold:
                predicted_diseases.append({
                    'disease': disease_names[i],
                    'probability': float(prob)
                })

        return predicted_diseases, probabilities
