import torch
from transformers import AutoTokenizer, AutoModel
from model import SoftMaxLit  # Make sure this is the correct import for your model class

# Function to load models
def load_model(checkpoint_path, n_inputs, n_outputs, device):
    model = SoftMaxLit(n_inputs, n_outputs)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

# Function to get predictions from transformer models
def get_weak_learner_predictions(text, models, tokenizers, device):
    embeddings = []
    for model, tokenizer in zip(models, tokenizers):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            embeddings.append(outputs.last_hidden_state[:, 0, :])
    return torch.cat(embeddings, dim=1)

# Function to predict with the meta-learner
def predict_with_meta_learner(features, meta_learner):
    with torch.no_grad():
        probabilities = meta_learner(features)
        predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model specifications
n_inputs = {'albert': 768, 'electra': 256, 'roberta': 768, 'xlnet': 768}
n_outputs = 2  # Assuming binary classification
total_input_features = sum(n_inputs.values())

# Paths to model checkpoints
weak_learner_paths =  [
    'lightning_logs/version_0/checkpoints/model=albert--dev=False--epoch=89-step=10170--val_loss=0.35.ckpt',
    'lightning_logs/version_1/checkpoints/model=electra--dev=False--epoch=297-step=33674--val_loss=0.38.ckpt',
    'lightning_logs/version_2/checkpoints/model=roberta--dev=False--epoch=289-step=32770--val_loss=0.36.ckpt',
    'lightning_logs/version_3/checkpoints/model=xlnet--dev=False--epoch=30-step=3503--val_loss=0.37.ckpt'
]
meta_learner_path = 'lightning_logs/version_4/checkpoints/model=lr--dev=False.ckpt'

# Load models and tokenizers
tokenizers = {
    'albert': AutoTokenizer.from_pretrained('albert-base-v2'),
    'electra': AutoTokenizer.from_pretrained('google/electra-small-discriminator'),
    'roberta': AutoTokenizer.from_pretrained('roberta-base'),
    'xlnet': AutoTokenizer.from_pretrained('xlnet-base-cased')
}
models = [AutoModel.from_pretrained(name) for name in ['albert-base-v2', 'google/electra-small-discriminator', 'roberta-base', 'xlnet-base-cased']]

# Load meta-learner
meta_learner = load_model(meta_learner_path, 2560, n_outputs, device)

# Example text for classification
text = "This complete script handles the entire workflow from data preprocessing through prediction, tailored to work with an ensemble of transformer models and a custom logistic regression model for final output."

# Get predictions from weak learners
features = get_weak_learner_predictions(text, models, [tokenizers[name] for name in n_inputs.keys()], device)

# Get final prediction from meta-learner
final_prediction, probabilities = predict_with_meta_learner(features, meta_learner)
print(f"Final Prediction: {final_prediction.item()}, Probabilities: {probabilities.numpy()}")