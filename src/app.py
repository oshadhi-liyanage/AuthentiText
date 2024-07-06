from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
import shap
import os
import nltk
import string

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import joblib
from helper import load_dataset
from model import TransformerModel, SoftMaxLit
import textstat
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords


app = Flask(__name__)

# Load the Logistic Regression model
lr_model = joblib.load('logistic_regression_model.pkl')

# Initialize the model and other components
device = torch.device("cpu")
DEV = False

# Load dataset and models
df = load_dataset('../dataset/training.json', test=True)

checkpoints = [
    'lightning_logs/version_80/checkpoints/model=albert--dev=False--epoch=297-step=33674--val_loss=0.36.ckpt',
    'lightning_logs/version_81/checkpoints/model=electra--dev=False--epoch=298-step=33787--val_loss=0.39.ckpt',
    'lightning_logs/version_82/checkpoints/model=deberta--dev=False--epoch=299-step=33900--val_loss=0.37.ckpt',
    'lightning_logs/version_83/checkpoints/model=roberta--dev=False--epoch=297-step=33674--val_loss=0.35.ckpt',
    'lightning_logs/version_84/checkpoints/model=xlnet--dev=False--epoch=64-step=7345--val_loss=0.37.ckpt'
]

pretrained_datasets_x = [
    f"pretrained--dev={DEV}--model=albert.pt",
    f"pretrained--dev={DEV}--model=electra.pt",
    f"pretrained--dev={DEV}--model=deberta.pt",
    f"pretrained--dev={DEV}--model=roberta.pt",
    f"pretrained--dev={DEV}--model=xlnet.pt"
]

model_y_arr = []
for model_name, pretrained_dataset_x, ckpt in zip(list(TransformerModel.MODELS.keys()), pretrained_datasets_x, checkpoints):
    n_inputs = TransformerModel.MODELS[model_name]['dim']
    model = SoftMaxLit(n_inputs, 2).load_from_checkpoint(n_inputs=n_inputs, n_outputs=2, checkpoint_path=ckpt)
    x = torch.load(pretrained_dataset_x).to(device)
    y_hat = model(x)

    del x
    torch.cuda.empty_cache()
    y_first = y_hat

    model_y_arr.append(y_first)

lr_dataset_x = torch.cat(model_y_arr, dim=1).detach()

def extract_features(text):
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    
    word_count = len(words)
    sentence_count = len(sentences)
    lexical_diversity = len(set(words)) / word_count if word_count != 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count != 0 else 0
    not_stopword_ratio = len([word for word in words if word.lower() not in stop_words]) / word_count if word_count != 0 else 0
    punctuation_count = len([char for char in text if char in string.punctuation])
    avg_word_length = np.mean([len(word) for word in words]) if word_count != 0 else 0
    
    features = {
        'WordCount': word_count,
        'LexicalDiversity': lexical_diversity,
        'NotStopwordRatio': not_stopword_ratio,
        'PunctuationCount': punctuation_count,
        'AvgWordLength': avg_word_length,
        'AvgSentenceLength': avg_sentence_length,
        'ReadingEase': textstat.flesch_reading_ease(text),
    }
    return features

# Define route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text_features = extract_features(data['text'])
    text_features_array = np.array(list(text_features.values())).reshape(1, -1)  # Reshape to 2D array

    # Since lr_dataset_x is the output of models for the entire dataset, we should take the specific instance we are predicting for.
    # We should ensure lr_dataset_x corresponds to the text being analyzed.
    # For simplicity, this example assumes lr_dataset_x contains the correct feature for the input text.
    # In a real-world scenario, you would need to ensure lr_dataset_x is generated for the input text.
    
    # Assuming lr_dataset_x contains only one row corresponding to the input text
    # if lr_dataset_x.shape[0] != 1:
    #     raise ValueError("lr_dataset_x should contain features for a single instance.")

    combined_features = np.hstack([ text_features_array])

    prediction = lr_model.predict(combined_features)[0]

    # Calculate SHAP values
    explainer = shap.Explainer(lr_model, combined_features)
    shap_values = explainer(combined_features)
    
    # Generate SHAP force plot
    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values.values[0], combined_features, feature_names=list(text_features.keys()))
    plt.savefig('shap_force_plot.png')
    plt.close()

    return jsonify({
        'prediction': int(prediction)
        # 'explanation': 'SHAP force plot saved as shap_force_plot.png'
    })

# Define route to serve the SHAP force plot
@app.route('/shap_force_plot', methods=['GET'])
def get_shap_plot():
    return send_file('shap_force_plot.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)

def extract_features(text):
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    
    word_count = len(words)
    sentence_count = len(sentences)
    lexical_diversity = len(set(words)) / word_count if word_count != 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count != 0 else 0
    not_stopword_ratio = len([word for word in words if word.lower() not in stop_words]) / word_count if word_count != 0 else 0
    punctuation_count = len([char for char in text if char in string.punctuation])
    avg_word_length = np.mean([len(word) for word in words]) if word_count != 0 else 0
    
    features = {
        'WordCount': word_count,
        'LexicalDiversity': lexical_diversity,
        'NotStopwordRatio': not_stopword_ratio,
        'PunctuationCount': punctuation_count,
        'AvgWordLength': avg_word_length,
        'AvgSentenceLength': avg_sentence_length,
        'ReadingEase': textstat.flesch_reading_ease(text),
    }
    return features
