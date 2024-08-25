import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    DATABASE_URI = os.getenv('DATABASE_URI')
    #DEBUG = os.getenv('DEBUG', 'False') == 'True'
    
    # Backblaze B2 configuration
    B2_ENDPOINT_URL = os.getenv('B2_ENDPOINT_URL')
    B2_KEY_ID = os.getenv('B2_KEY_ID')
    B2_APPLICATION_KEY = os.getenv('B2_APPLICATION_KEY')
    B2_BUCKET_NAME = os.getenv('B2_BUCKET_NAME')

config = Config()