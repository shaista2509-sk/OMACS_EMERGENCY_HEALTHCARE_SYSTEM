import torch
import onnxruntime
import numpy as np
import pandas as pd
from model_utils import load_model, preprocess_patient_data, predict_diseases


def test_pytorch_model():
    """Test the PyTorch model"""
    print("=== Testing PyTorch Model ===")

    # Load model
    model, checkpoint = load_model()

    # Sample patient data (replace with real data)
    sample_patient = {
        'Age': 45,
        'Gender': 'Male',
        'Blood Type': 'A+',
        'Test Results': 'Normal',
        'Medication': 'Aspirin',
        'Admission Type': 'Emergency'
    }

    # Preprocess
    processed_data = preprocess_patient_data(
        sample_patient,
        checkpoint['label_encoders'],
        checkpoint['feature_names']
    )

    # Predict
    predictions, probabilities = predict_diseases(
        model,
        processed_data,
        checkpoint['disease_names']
    )

    print(f"Patient: {sample_patient}")
    print(f"Predicted diseases: {predictions}")
    print(f"All probabilities: {dict(zip(checkpoint['disease_names'], probabilities))}")


def test_onnx_model():
    """Test the ONNX model"""
    print("\n=== Testing ONNX Model ===")

    try:
        # Load ONNX model
        session = onnxruntime.InferenceSession(r'C:\Users\shast\PycharmProjects\emergency_healthcare_system\ml_models\models\groupnet.onnx')

        # Load PyTorch model info for comparison
        checkpoint = torch.load('ml_models/models/groupnet.pth', map_location='cpu')

        # Sample patient data
        sample_patient = {
            'Age': 45,
            'Gender': 'Male',
            'Blood Type': 'A+',
            'Test Results': 'Normal',
            'Medication': 'Aspirin',
            'Admission Type': 'Emergency'
        }

        # Preprocess
        processed_data = preprocess_patient_data(
            sample_patient,
            checkpoint['label_encoders'],
            checkpoint['feature_names']
        )

        # Run ONNX inference
        input_data = processed_data.reshape(1, -1).astype(np.float32)
        result = session.run(None, {'patient_features': input_data})
        probabilities = result[0][0]

        # Apply threshold
        predicted_diseases = []
        for i, prob in enumerate(probabilities):
            if prob > 0.5:
                predicted_diseases.append({
                    'disease': checkpoint['disease_names'][i],
                    'probability': float(prob)
                })

        print(f"Patient: {sample_patient}")
        print(f"Predicted diseases: {predicted_diseases}")
        print(f"All probabilities: {dict(zip(checkpoint['disease_names'], probabilities))}")

    except Exception as e:
        print(f"ONNX test failed: {e}")


def compare_models():
    """Compare PyTorch and ONNX model outputs"""
    print("\n=== Comparing PyTorch vs ONNX ===")

    # Load PyTorch config (not ONNX file!)
    checkpoint = torch.load(r'C:\Users\shast\PycharmProjects\emergency_healthcare_system\ml_models\models\groupnet.pth',
                          map_location='cpu')

    for i in range(3):
        print(f"\nTest {i + 1}:")
        # Generate random patient data
        test_input = np.random.randn(checkpoint['input_size']).astype(np.float32)

        # PyTorch prediction
        model, _ = load_model()
        with torch.no_grad():
            pytorch_pred = model(torch.FloatTensor(test_input).unsqueeze(0)).cpu().numpy()[0]

        # ONNX prediction
        session = onnxruntime.InferenceSession(r'C:\Users\shast\PycharmProjects\emergency_healthcare_system\ml_models\models\groupnet.onnx')
        onnx_pred = session.run(None, {'patient_features': test_input.reshape(1, -1)})[0][0]

        # Compare
        diff = np.abs(pytorch_pred - onnx_pred).max()
        print(f"Max difference: {diff:.8f}")
        print(f"PyTorch output: {pytorch_pred}")
        print(f"ONNX output: {onnx_pred}")



if __name__ == "__main__":
    test_pytorch_model()
    test_onnx_model()
    compare_models()
