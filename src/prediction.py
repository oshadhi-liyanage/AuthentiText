import torch
from transformers import AutoTokenizer
from model import TransformerModel, SoftMaxLit

# Function to load the models and prepare them for inference
def load_models(model_names, checkpoints, lr_checkpoint_path, device):
    models = []
    for model_name, ckpt in zip(model_names, checkpoints):
        n_inputs = TransformerModel.MODELS[model_name]['dim']
        model = SoftMaxLit(n_inputs, 2).load_from_checkpoint(n_inputs=n_inputs, n_outputs=2, checkpoint_path=ckpt).to(device)
        model.eval()  # Set the model to evaluation mode
        models.append(model)
    # Load logistic regression model
    # Assuming each transformer model outputs a feature vector of size `dim`
    total_input_features = sum([TransformerModel.MODELS[m]['dim'] for m in model_names])
    lr_model = SoftMaxLit(2560, 2).load_from_checkpoint(checkpoint_path=lr_checkpoint_path, n_inputs=2560, n_outputs=2, device=device)

#   lr_model = SoftMaxLit(sum([TransformerModel.MODELS[m]['dim'] for m in model_names]), 2).load_from_checkpoint(n_inputs=sum([TransformerModel.MODELS[m]['dim'] for m in model_names]), n_outputs=2, checkpoint_path=lr_checkpoint_path).to(device)
    lr_model.eval()
    return models, lr_model

# Function to get predictions from a single text input
def get_prediction(text, models, lr_model, device):
    tokenized_inputs = [model.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device) for model in models]
    
    # Get embeddings from each model
    embeddings = [model.pretrained(**inputs).last_hidden_state[:, 0, :].detach() for model, inputs in zip(models, tokenized_inputs)]
    
    # Concatenate all embeddings
    combined_embeddings = torch.cat(embeddings, dim=1)
    
    # Predict with logistic regression model
    logits = lr_model(combined_embeddings)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class, probabilities[0].tolist()

# Assuming model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_names = ['albert', 'electra', 'roberta', 'xlnet']
checkpoints = [
    'lightning_logs/version_0/checkpoints/model=albert--dev=False--epoch=89-step=10170--val_loss=0.35.ckpt',
    'lightning_logs/version_1/checkpoints/model=electra--dev=False--epoch=297-step=33674--val_loss=0.38.ckpt',
    'lightning_logs/version_2/checkpoints/model=roberta--dev=False--epoch=289-step=32770--val_loss=0.36.ckpt',
    'lightning_logs/version_3/checkpoints/model=xlnet--dev=False--epoch=30-step=3503--val_loss=0.37.ckpt'
]
lr_checkpoint_path = 'lightning_logs/version_4/checkpoints/model=lr--dev=False.ckpt'
models, lr_model = load_models(model_names, checkpoints, lr_checkpoint_path, device)

# Example usage
text = "Your example text here"
prediction, probabilities = get_prediction(text, models, lr_model, device)
print("Predicted class:", prediction)
print("Probabilities:", probabilities)
