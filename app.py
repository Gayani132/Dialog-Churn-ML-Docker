
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Create Flask application
app = Flask(__name__)

# Load trained model once
model = joblib.load("churn_model.joblib")


# Home page
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Dialog Churn Prediction API is running"
    })


# Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON"}), 400

    required = [
        "monthly_usage_gb",
        "tenure_months",
        "monthly_topup",
        "complaints_last_3_months"
    ]

    if not all(key in data for key in required):
        return jsonify({"error": "Missing customer data"}), 400

    try:
        features = np.array([[
            float(data[key]) for key in required
        ]])

        if not np.isfinite(features).all():
            raise ValueError()

    except (TypeError, ValueError):
        return jsonify({"error": "Invalid numeric values"}), 400

    prediction = int(model.predict(features)[0])

    probability = float(
        model.predict_proba(features)[0][1]
    )

    result = "Churn" if prediction == 1 else "Not Churn"

    return jsonify({
        "prediction": result,
        "churn_probability": round(probability, 4)
    })


# Start Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
