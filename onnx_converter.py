import os
import torch
import onnx
import onnxruntime
import numpy as np
from groupnet_model import GroupNet

# Define correct paths

PYTORCH_PATH =  r'C:\Users\shast\PycharmProjects\emergency_healthcare_system\ml_models\models\groupnet.pth'
ONNX_PATH = r'C:\Users\shast\PycharmProjects\emergency_healthcare_system\ml_models\models\groupnet.onnx'



def convert_to_onnx():
    """Convert PyTorch model to ONNX format"""
    print("Loading PyTorch model...")

    # Load the saved model
    checkpoint = torch.load(PYTORCH_PATH, map_location='cpu')

    model = GroupNet(
        input_size=checkpoint['input_size'],
        num_diseases=checkpoint['num_diseases']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(1, checkpoint['input_size'])

    print("Exporting to ONNX...")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=['patient_features'],
        output_names=['disease_predictions'],
        opset_version=17,
        dynamic_axes={
            'patient_features': {0: 'batch_size'},
            'disease_predictions': {0: 'batch_size'}
        }
    )

    print("ONNX model saved successfully!")

    # Verify the ONNX model
    verify_onnx_model()

def verify_onnx_model():
    """Verify ONNX model works correctly"""
    print("Verifying ONNX model...")

    try:
        # Load ONNX model
        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")

        # Test with ONNX Runtime
        session = onnxruntime.InferenceSession(ONNX_PATH)

        # Get input shape
        input_shape = session.get_inputs()[0].shape
        print(f"✓ Input shape: {input_shape}")

        # Create test input
        test_input = np.random.randn(1, input_shape[1]).astype(np.float32)

        # Run inference
        result = session.run(None, {'patient_features': test_input})
        print(f"✓ Output shape: {result[0].shape}")
        print(f"✓ Sample predictions: {result[0][0]}")

        return True

    except Exception as e:
        print(f"✗ ONNX verification failed: {e}")
        return False

if __name__ == "__main__":
    convert_to_onnx()
