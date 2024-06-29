import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the dataset
file_path = 'diabetes.csv'
data = pd.read_csv(file_path)

# Handle missing values
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Separate features and target
X = data_imputed.drop('Outcome', axis=1)
y = data_imputed['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the hyperparameter grid
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Initialize RandomizedSearchCV with 5-fold cross-validation
random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist,
                                   n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42, scoring='accuracy')

# Start timing
start_time = time.time()

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train_scaled, y_train)

# End timing
end_time = time.time()
execution_time = end_time - start_time
logger.info(f"RandomizedSearchCV completed in {execution_time:.2f} seconds")

# Get the best parameters
best_params = random_search.best_params_

# Train the model with the best parameters
best_rf_classifier = RandomForestClassifier(**best_params, random_state=42)
best_rf_classifier.fit(X_train_scaled, y_train)

# Cross-validate the model
cv_scores = cross_val_score(best_rf_classifier, X_train_scaled, y_train, cv=5)
logger.info(f"Cross-validation scores: {cv_scores}")
logger.info(f"Mean cross-validation score: {cv_scores.mean():.4f}")

# Predict the outcomes on the test set
y_pred_best = best_rf_classifier.predict(X_test_scaled)

# Evaluate the tuned model
accuracy_best = accuracy_score(y_test, y_pred_best)
report_best = classification_report(y_test, y_pred_best)

# Print the results
logger.info("Best Parameters: %s", best_params)
logger.info("Accuracy: %.4f", accuracy_best)
logger.info("Classification Report:\n%s", report_best)

# Print feature importances
feature_importances = best_rf_classifier.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
logger.info("Feature Importances:\n%s", importance_df.to_string(index=False))

# Save the model
joblib.dump(best_rf_classifier, 'best_random_forest_model.pkl')
logger.info("Model saved to best_random_forest_model.pkl")
