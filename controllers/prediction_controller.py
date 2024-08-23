from flask import Flask, request, jsonify
from flask_cors import cross_origin
class PredictionController:
    def __init__(self, prediction_service):
        self.app = Flask(__name__)
        self.prediction_service = prediction_service

    def setup_routes(self):
        @self.app.route('/predict', methods=['POST'])
        @cross_origin() 
        def predict():
            data = request.json
            text = data['text']
            prediction = self.prediction_service.predict(text)
            return jsonify(prediction)

    def run(self):
        self.app.run(debug=True)