from flask import Flask, request, jsonify, send_file
from flask_cors import cross_origin
from data_management.data_handler import get_db, Prediction

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

        @self.app.route('/predictions', methods=['GET'])
        @cross_origin()
        def get_predictions():
            db = next(get_db())
            predictions = db.query(Prediction).all()
            return jsonify([{
                'id': p.id,
                'text': p.text,
                'prediction': p.prediction,
                'confidence': p.confidence,
                'timestamp': p.timestamp.isoformat()
            } for p in predictions])

        @self.app.route('/explanation/<filename>', methods=['GET'])
        @cross_origin()
        def get_explanation(filename):
            return send_file(f"explanation_results/{filename}")

    def run(self):
        self.app.run(debug=True)