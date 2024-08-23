import os
from flask import send_file
import torch
import torch.nn.functional as F
import uuid
from pathlib import Path
from explanation.visualizer import Visualizer 
# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Define the output folder relative to the project root
HTML_OUTPUT_FOLDER = PROJECT_ROOT /"explanation_results"

class PredictionService:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, text):
        self.model.eval()
        inputs = self.preprocessor.preprocess(text)
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            print(predicted_class)
            confidence = probs[0][predicted_class].item()

            
        # Get attributions
        attributions, delta, tokens = self.model.attribute_text(text, predicted_class)
        text_id = uuid.uuid4()
        random_id_str = str(text_id)
        # Generate and save HTML
        html_file_name = f"{random_id_str}.html"
        html_file_path = os.path.join(HTML_OUTPUT_FOLDER, html_file_name)
        # Visualize attributions
        Visualizer.visualize_attributions(self.model, self.preprocessor.tokenizer, attributions, tokens, delta, text, predicted_class, html_file_path)
        send_file(html_file_path, as_attachment=True)

        return {
            'text': text,
            'prediction': 'AI-generated' if predicted_class == 1 else 'Human-written',
            'confidence': confidence
        }
    


    