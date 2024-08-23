import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch.nn.functional as F
from prediction.prediction_service import PredictionService
from explanation.visualizer import Visualizer
from controllers.prediction_controller import PredictionController
from flask import Flask, request, jsonify 
from flask_cors import CORS
from preprocessor.preprocessor import TextPreprocessor
from model.model import AlbertTextClassifier
from transformers import AlbertTokenizer

app = Flask(__name__)
CORS(app) 

def main():
    # Initialize the model
    model = AlbertTextClassifier(num_labels=2)
    best_model_path = "best-checkpoint.ckpt"
    
    # Load the best model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    preprocessor = TextPreprocessor(tokenizer)
    
    # Set up PredictionService
    prediction_service = PredictionService(model, preprocessor)
    
    # Set up and run API
    api_handler = PredictionController(prediction_service)
    api_handler.setup_routes()
    api_handler.run()

if __name__ == "__main__":
    main()