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
    feature_values = [request.form.get(feature) for feature in ['Sex', 'Length', 'Diameter', 'Height',
                                                                'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']]
    # Convert feature values to float, reshape
    features_array = np.array([float(value) for value in feature_values if value]).reshape(1, -1)
    # Predict the Rings using your model
    prediction = model.predict(features_array)
    # Render the result
    return render_template('index.html', prediction_text=f'Predicted Rings: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)