import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        float_features = [float(x) for x in request.form.values()]
        features = np.array([float_features])  # Convert to NumPy array

        # Make prediction
        prediction = model.predict(features)

        # Convert prediction to readable format
        prediction_text = "Malignant" if prediction[0] == 1 else "Benign"

        # Show the result on a separate page
        return render_template("result.html", prediction_text=f"The predicted diagnosis is: {prediction_text}")

    except Exception as e:
        return render_template("result.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
