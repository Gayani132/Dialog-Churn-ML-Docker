
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Generate sample customer data
np.random.seed(42)

n = 2000

usage = np.random.uniform(1, 30, n)
tenure = np.random.randint(1, 60, n)
topup = np.random.uniform(100, 3000, n)
complaints = np.random.randint(0, 6, n)

# Generate demonstration churn labels
score = (
    -0.08 * usage
    -0.04 * tenure
    -0.0005 * topup
    +0.9 * complaints
    +2
)

probability = 1 / (1 + np.exp(-score))

churn = (
    np.random.random(n) < probability
).astype(int)

# Prepare features
X = np.column_stack([
    usage, tenure, topup, complaints
])

y = churn

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create ML pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save trained model
joblib.dump(model, "churn_model.joblib")

print("Model saved successfully!")
