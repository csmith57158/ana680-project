from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('svm_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract feature values from the form
    feature_values = [request.form.get(feature) for feature in ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                                                                'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                                                                'pH', 'sulphates', 'alcohol']]
    # Convert feature values to float and create a numpy array for prediction
    features_array = np.array([float(value) for value in feature_values if value]).reshape(1, -1)
    # Predict the quality using your model
    prediction = model.predict(features_array)
    # Render the result
    return render_template('index.html', prediction_text=f'Predicted Quality: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
