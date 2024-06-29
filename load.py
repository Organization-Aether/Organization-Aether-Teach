import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved model
model_path = 'best_random_forest_model.pkl'
loaded_model = joblib.load(model_path)

# Load the dataset to get the features for prediction (replace with your test dataset path if different)
file_path = 'diabetes.csv'
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Predict the outcomes on the test set using the loaded model
y_pred_loaded = loaded_model.predict(X_test_scaled)

# Evaluate the loaded model
from sklearn.metrics import accuracy_score, classification_report

accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
report_loaded = classification_report(y_test, y_pred_loaded)

print("Loaded Model Accuracy:", accuracy_loaded)
print("Loaded Model Classification Report:\n", report_loaded)
