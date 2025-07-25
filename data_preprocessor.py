import numpy as np
from app.models import Patient



class PatientDataPreprocessor:
    """Converts SQLAlchemy Patient objects to ML model input format"""

    def __init__(self):
        self.feature_order = [
            'Age', 'Gender_Female', 'Gender_Male',
            'Blood Type_A+', 'Blood Type_B+', 'Blood Type_O+',
            'Medical Condition_Hypertension', 'Medical Condition_Diabetes'
        ]

    def convert_patient(self, patient: Patient) -> np.ndarray:
        """Convert SQLAlchemy Patient to feature vector"""
        features = []

        # Age (normalized 0-1 with max age 100)
        features.append(patient.Age / 100)

        # Gender one-hot encoding
        features.extend([1 if patient.Gender == 'Female' else 0,
                         1 if patient.Gender == 'Male' else 0])

        # Blood Type encoding
        blood_type = patient.Blood_Type.replace(' ', '_')  # Handle spaces
        for bt in ['A+', 'B+', 'O+']:
            features.append(1 if blood_type == bt else 0)

        # Medical Conditions
        features.append(1 if 'Hypertension' in patient.Medical_Condition else 0)
        features.append(1 if 'Diabetes' in patient.Medical_Condition else 0)

        return np.array(features).astype(np.float32)
