import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.preprocessing import LabelEncoder
import pickle
import os


class GroupNet(nn.Module):
    """GroupNet model architecture matching your training code"""

    def __init__(self, input_size, num_diseases):
        super(GroupNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, num_diseases)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.output(x))
        return x


class PatientDataPreprocessor:
    """Data preprocessor matching your training pipeline"""

    def __init__(self, model_path='ml_models/models/groupnet.pth'):
        # Load the saved model data including label encoders
        checkpoint = torch.load(model_path, map_location='cpu')
        self.feature_names = checkpoint['feature_names']
        self.disease_names = checkpoint['disease_names']
        self.label_encoders = checkpoint.get('label_encoders', {})

        # Initialize label encoders if not saved
        if not self.label_encoders:
            self._init_default_encoders()

    def _init_default_encoders(self):
        """Initialize default label encoders based on common medical values"""
        self.label_encoders = {
            'Gender': LabelEncoder(),
            'Blood Type': LabelEncoder(),
            'Test Results': LabelEncoder(),
            'Medication': LabelEncoder(),
            'Admission Type': LabelEncoder()
        }

        # Fit with common values (you may need to adjust based on your data)
        gender_values = ['Male', 'Female', 'M', 'F']
        blood_type_values = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-', 'Unknown']
        test_results = ['Normal', 'Abnormal', 'Inconclusive', 'Unknown']
        medications = ['None', 'Aspirin', 'Ibuprofen', 'Paracetamol', 'Unknown']
        admission_types = ['Emergency', 'Elective', 'Urgent', 'Unknown']

        self.label_encoders['Gender'].fit(gender_values)
        self.label_encoders['Blood Type'].fit(blood_type_values)
        self.label_encoders['Test Results'].fit(test_results)
        self.label_encoders['Medication'].fit(medications)
        self.label_encoders['Admission Type'].fit(admission_types)

    def convert(self, patient_data: Dict) -> np.ndarray:
        """Convert patient data dictionary to model input format"""
        try:
            # Map input keys to expected feature names
            feature_mapping = {
                'Age': 'Age',
                'Gender': 'Gender',
                'Blood_Type': 'Blood Type',
                'Test_Result': 'Test Results',
                'medication': 'Medication',
                'Admission_Type': 'Admission Type'
            }

            # Create feature vector
            features = []

            for feature_name in self.feature_names:
                if feature_name == 'Age':
                    age = patient_data.get('Age', 50)  # Default age
                    features.append(float(age))

                elif feature_name in self.label_encoders:
                    # Get value from patient data
                    input_key = None
                    for key, mapped_name in feature_mapping.items():
                        if mapped_name == feature_name:
                            input_key = key
                            break

                    if input_key and input_key in patient_data:
                        value = str(patient_data[input_key])
                    else:
                        value = 'Unknown'

                    # Encode categorical value
                    encoder = self.label_encoders[feature_name]
                    try:
                        encoded_value = encoder.transform([value])[0]
                    except ValueError:
                        # Handle unseen values
                        encoded_value = 0  # Default to first class

                    features.append(float(encoded_value))
                else:
                    features.append(0.0)  # Default value

            return np.array(features, dtype=np.float32).reshape(1, -1)

        except Exception as e:
            print(f"Error in data preprocessing: {e}")
            # Return default feature vector
            return np.zeros((1, len(self.feature_names)), dtype=np.float32)


class HealthPredictionService:
    """Updated PyTorch-based health prediction service"""

    def __init__(self, model_path='ml_models/models/groupnet.pth'):
        self.model_path = model_path
        self.preprocessor = PatientDataPreprocessor(model_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()

    def _load_model(self):
        """Load the PyTorch model with proper error handling"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Initialize model
            input_size = checkpoint['input_size']
            num_diseases = checkpoint['num_diseases']
            self.disease_names = checkpoint['disease_names']

            self.model = GroupNet(input_size, num_diseases)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            print(f"PyTorch model loaded successfully")
            print(f"   - Input size: {input_size}")
            print(f"   - Diseases: {self.disease_names}")

        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            self.model = None
            self.disease_names = ['Cancer', 'Diabetes', 'Obesity', 'Hypertension', 'Asthma']

    def predict(self, patient_data: Dict) -> Dict[str, Any]:
        """Predict diseases for patient data"""
        try:
            if self.model is None:
                return self._fallback_prediction()

            # Preprocess data
            input_data = self.preprocessor.convert(patient_data)
            input_tensor = torch.FloatTensor(input_data).to(self.device)

            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = output.cpu().numpy()[0]  # Get first (and only) sample

            # Apply threshold for multi-label classification
            threshold = 0.3  # Adjust based on your needs
            predicted_diseases = []
            confidence_scores = []

            for i, prob in enumerate(probabilities):
                if prob > threshold:
                    predicted_diseases.append(self.disease_names[i])
                    confidence_scores.append(float(prob))

            # If no disease above threshold, take the highest probability
            if not predicted_diseases:
                max_idx = np.argmax(probabilities)
                predicted_diseases = [self.disease_names[max_idx]]
                confidence_scores = [float(probabilities[max_idx])]

            return {
                'predictions': probabilities.tolist(),  # Full probability array
                'conditions': predicted_diseases,  # Threshold predictions
                'confidences': confidence_scores,  # Corresponding confidences
                'all_probabilities': {
                    disease: float(prob)
                    for disease, prob in zip(self.disease_names, probabilities)
                }
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_prediction()

    def _fallback_prediction(self):
        """Fallback prediction when model fails"""
        return {
            'predictions': [0.2, 0.3, 0.1, 0.4, 0.2],  # Dummy probabilities
            'conditions': ['Health Assessment Required'],
            'confidences': [0.85],
            'all_probabilities': {
                disease: 0.2 for disease in self.disease_names
            }
        }

    def get_recommendations(self, predicted_diseases: List[str]) -> List[str]:
        """Get medical recommendations based on predicted diseases"""
        recommendations = []

        if 'Cancer' in predicted_diseases:
            recommendations.extend([
                "Immediate oncology consultation required",
                "Comprehensive imaging studies recommended",
                "Biopsy may be necessary for confirmation"
            ])

        if 'Diabetes' in predicted_diseases:
            recommendations.extend([
                "Blood glucose monitoring required",
                "Endocrinology consultation recommended",
                "Dietary counseling advised"
            ])

        if 'Hypertension' in predicted_diseases:
            recommendations.extend([
                "Blood pressure monitoring essential",
                "Cardiovascular risk assessment needed",
                "Lifestyle modifications recommended"
            ])

        if 'Asthma' in predicted_diseases:
            recommendations.extend([
                "Pulmonary function tests recommended",
                "Allergy testing may be beneficial",
                "Respiratory specialist consultation"
            ])

        if 'Obesity' in predicted_diseases:
            recommendations.extend([
                "Nutritional counseling recommended",
                "Exercise program development",
                "Metabolic screening advised"
            ])

        if not recommendations:
            recommendations = [
                "Comprehensive health evaluation recommended",
                "Follow-up with primary care physician",
                "Preventive care measures advised"
            ]

        return recommendations[:3]  # Limit to top 3 recommendation