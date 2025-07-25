import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\shast\Downloads\archive (1)\healthcare_dataset.csv")

# Feature engineering
relevant_features = ['Age', 'Gender', 'Blood Type', 'Test Results', 'Medication', 'Admission Type']
top_diseases = ['Cancer', 'Diabetes', 'Obesity', 'Hypertension', 'Asthma']

# Create multi-label targets
for disease in top_diseases:
    df[f'has_{disease}'] = (df['Medical Condition'] == disease).astype(int)

X = df[relevant_features].copy()
y = df[[f'has_{disease}' for disease in top_diseases]]

# Preprocessing
categorical_cols = ['Gender', 'Blood Type', 'Test Results', 'Medication', 'Admission Type']
for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

print(f"Using {len(relevant_features)} relevant features instead of all columns")
print(f"Features: {relevant_features}")


# Neural network architecture
class GroupNet(nn.Module):
    def __init__(self, input_size, num_diseases):
        super(GroupNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Reduced since fewer features
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


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization
model = GroupNet(input_size=len(relevant_features), num_diseases=len(top_diseases))
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.FloatTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.FloatTensor(y_test.values)

# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Calculate accuracy every 10 epochs
    if epoch % 10 == 0 or epoch == 99:
        model.eval()
        with torch.no_grad():
            preds = model(X_test_tensor)
            preds_binary = (preds > 0.5).int().cpu().numpy()
            y_true = y_test_tensor.cpu().numpy()
            # Hamming accuracy: fraction of correct labels
            acc = (preds_binary == y_true).mean()
            print(f"Epoch {epoch}: Train Loss: {loss.item():.4f} | Multi-label Hamming Accuracy: {acc:.3f}")


# Save the trained model (add this after the training loop)
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': len(relevant_features),
    'num_diseases': len(top_diseases),
    'feature_names': relevant_features,
    'disease_names': top_diseases,
    'label_encoders': {col: le for col in categorical_cols}
}, r'C:\Users\shast\PycharmProjects\emergency_healthcare_system\ml_models\models\groupnet.pth')

print("Model saved successfully!")
