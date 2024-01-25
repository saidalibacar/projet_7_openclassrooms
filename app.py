from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS

# Load the pre-trained model
model = joblib.load('lightgbm_model.joblib')

# Assuming 'X_test_app.csv' is in the same directory as your Flask app
X_test_app = pd.read_csv('X_test_app.csv')

# Endpoint for predictions
@app.route('/predict_id', methods=['GET', 'POST'])
def predict_id():
    try:
        # Ensure that the input data has the same features as the model expects
        # Assuming you want to exclude the 'ID' column from the input data
        features_to_use = [col for col in X_test_app.columns if col != 'ID']

        if request.method == 'GET':
            # Handle GET request
            # Get custom threshold from query parameters or use default
            threshold = float(request.args.get('threshold', 0.065))
            id = request.args.get("id")
        elif request.method == 'POST':
            # Handle POST request
            data = request.json
            threshold = float(data.get('threshold', 0.065))
            id = data.get('id')

        X_test_app['ID'] = X_test_app.index + 1 

        # Perform prediction using the pre-trained model
        predictions_prob = model.predict(X_test_app[X_test_app["ID"] == int(id)][features_to_use])

        # Convert probabilities to binary predictions based on the threshold
        predictions_binary = (predictions_prob > threshold).astype(int)

        response_data = {
            'True Prediction' : float(predictions_prob[0]),
            'Binary Prediction' : int(predictions_binary[0]),
            'Class' : int((predictions_prob > threshold))
        }
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, port=8000)


    # ne pas effectuer toutes les predictions en meme tps mais par id 






