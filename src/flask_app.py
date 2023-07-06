from flask import Flask, request, jsonify
import joblib

# Load the vectorizer and model
vectorizer = joblib.load('models/vectorizer.joblib')
model = joblib.load('models/model.joblib')

# Create a Flask app
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.json
    sentence = data['sentence']

    sentence_features = vectorizer.transform([sentence])
    prediction = model.predict(sentence_features.toarray())

    return jsonify({'prediction': prediction.tolist()})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='localhost', port=3000, debug=True, threaded=True)
