from flask import Flask, jsonify,request
from flask_cors import CORS
import joblib
import numpy as np



app = Flask(__name__)
CORS(app)
try:
    loaded_model = joblib.load('./branch_model_updated.pkl')
except Exception as e:
    print(f"Error loading the model: {e}")
    loaded_model = None

@app.route('/')
def hello_world():
    return jsonify({'message': 'Hello prasanth'})

@app.route('/predict',methods=['POST'])
def pedict():
    
    if loaded_model is None:
        return jsonify({'error': 'Model not loaded'})
    input_data = request.get_json()

    input_data = np.array([[
        input_data['rank'],
        input_data['gender'],
        input_data['caste'],
        input_data['math_cgpa'],
        input_data['phy_cgpa'],
        input_data['chem_cgpa'],
        input_data['total_cgpa']
    ]])

    try:
        probaba = loaded_model.predict_proba(input_data)

        classes = loaded_model.classes_

        predictions_dict = {class_name: float(prob) for class_name, prob in zip(classes, probaba[0])}

        return jsonify({'predictions': predictions_dict})

    except Exception as e:
        return jsonify({'error': f"Prediction error: {e}"})
if __name__ == '__main__':
    app.run(debug=True)