from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')  # Load the model
label_encoder = joblib.load('label_encoder.pkl')  # Load the label encoder

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extracting form data
            data = request.form
            features = [float(data.get('sepal_length')), float(data.get('sepal_width')),
                        float(data.get('petal_length')), float(data.get('petal_width'))]
            # Predict the species
            prediction = model.predict([features])
            # Inverse transform the prediction to get the species name
            predicted_species = label_encoder.inverse_transform(prediction)
            # Render the prediction result
            return render_template('index.html', prediction_text=f'Predicted Iris Species: {predicted_species[0]}')
        except Exception as e:
            # Handle error
            return jsonify({'error': str(e)})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
