import mimetypes
import os
import uuid
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from flask import send_file
import torch
import torch.nn.functional as F
from explanation.visualizer import Visualizer
from data_management.data_handler import Prediction, get_db
from config import config
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from PIL import Image
import io
import numpy as np

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Define the output folder relative to the project root
HTML_OUTPUT_FOLDER = PROJECT_ROOT / "explanation_results"
IMAGE_OUTPUT_FOLDER = PROJECT_ROOT / "explanation_images"

class PredictionService:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.b2_client = self._get_b2_client()
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")

    def _get_b2_client(self):
        return boto3.client(
            service_name='s3',
            endpoint_url=config.B2_ENDPOINT_URL,
            aws_access_key_id=config.B2_KEY_ID,
            aws_secret_access_key=config.B2_APPLICATION_KEY
        )

    def _html_to_image(self, html_path, image_path):
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.chrome_options)
        driver.get(f"file://{html_path}")
        driver.set_window_size(1920, 1080)  # Set a fixed size for consistency
        png = driver.get_screenshot_as_png()
        driver.quit()

        # Use PIL to crop and save the image
        img = Image.open(io.BytesIO(png))
        img_array = np.array(img)

        # Find the bounding box of non-white pixels
        non_white_pixels = np.where(img_array[:, :, :3] != [255, 255, 255])
        top, left = np.min(non_white_pixels[0]), np.min(non_white_pixels[1])
        bottom, right = np.max(non_white_pixels[0]), np.max(non_white_pixels[1])

        # Add a small padding (e.g., 10 pixels) around the content
        padding = 10
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(img.height, bottom + padding)
        right = min(img.width, right + padding)

        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(image_path)

    def _upload_to_b2(self, file_path, object_name):
        try:
            # content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
            # self.b2_client.upload_file(file_path, config.B2_BUCKET_NAME, object_name)
            mimetype, _ = mimetypes.guess_type(file_path)
            if mimetype is None:
                raise Exception("Failed to guess mimetype")

            self.b2_client.upload_file(
                Filename=file_path,
                Bucket=config.B2_BUCKET_NAME,
                Key=object_name,
                ExtraArgs={
                    "ContentType": mimetype
                }
            ) 
        except ClientError as e:
            print(f"Error uploading to B2: {e}")
            return None
        return f"{config.B2_ENDPOINT_URL}/{config.B2_BUCKET_NAME}/{object_name}"
    
   

    def predict(self, text):
        self.model.eval()
        inputs = self.preprocessor.preprocess(text)
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()

        # Get attributions
        attributions, delta, tokens = self.model.attribute_text(text, predicted_class)
        text_id = uuid.uuid4()
        random_id_str = str(text_id)

        # Generate and save HTML
        html_file_name = f"{random_id_str}.html"
        html_file_path = os.path.join(HTML_OUTPUT_FOLDER, html_file_name)
        Visualizer.visualize_attributions(self.model, self.preprocessor.tokenizer, attributions, tokens, delta, text, predicted_class, html_file_path)

        # Convert HTML to image
        image_file_name = f"{random_id_str}.png"
        image_file_path = os.path.join(IMAGE_OUTPUT_FOLDER, image_file_name)
        self._html_to_image(html_file_path, image_file_path)

        # Upload image to B2
        b2_object_name = f"explanations/{image_file_name}"
        b2_image_url = self._upload_to_b2(image_file_path, b2_object_name)

        # Store prediction in the database
        db = next(get_db())
        prediction_result = 'AI-generated' if predicted_class == 1 else 'Human-written'
        new_prediction = Prediction(text=text, prediction=prediction_result, confidence=confidence, explanation_url=b2_image_url)
        db.add(new_prediction)
        db.commit()

        return {
            'text': text,
            'prediction': prediction_result,
            'confidence': confidence,
            'explanation_url': b2_image_url
        }