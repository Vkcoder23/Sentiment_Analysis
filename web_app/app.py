from flask import Flask, request, jsonify, send_from_directory
import pickle
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained model

pkl = os.path.join(os.path.dirname(__file__), '..', 'model', 'sentiment_model.pkl')
with open(pkl, 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text') if data else None
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Log the received text
    logging.debug(f"Received text: {text}")

    try:
        # Predict the sentiment
        prediction = model.predict([text])

        # Log the prediction
        logging.debug(f"Prediction: {prediction[0]}")

        return jsonify({'sentiment': prediction[0]})
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)
