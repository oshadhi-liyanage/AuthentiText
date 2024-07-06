import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from model import SoftMaxLit, TransformerModel

# Function to load transformer models and their corresponding tokenizers
def load_model(model_name, checkpoint_path, device):
    # Retrieve model information from the MODELS dictionary
    model_info = TransformerModel.MODELS[model_name]
    
    # Load tokenizer and model using pre-trained settings
    tokenizer = model_info['tokenizer'].from_pretrained(model_info['name'])
    
    # Load model configuration dynamically if available from transformers
    config = AutoConfig.from_pretrained(model_info['name'])
    model = model_info['pretrained'](config)
    
    # Load the entire checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check for the presence of 'state_dict' in checkpoint (common with PyTorch Lightning)
    if 'state_dict' in checkpoint:
        # Filter out unnecessary keys if needed, and load the state dictionary into the model
        model_keys = set(model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_keys}
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        # Directly load the state dictionary assuming it's a plain PyTorch checkpoint
        model.load_state_dict(checkpoint)

    # Move the model to the specified device and set it to evaluation mode
    model.to(device).eval()

    return model, tokenizer

# Function to load the logistic regression model from a checkpoint
def load_soft_max_lit_from_checkpoint(checkpoint_path, input_dim, output_dim, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SoftMaxLit(input_dim, output_dim)
    
    # Check for the presence of 'state_dict'
    if 'state_dict' in checkpoint:
        state_dict = {k.partition('module.')[2]: v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)

    model.to(device).eval()
    return model

# Function to get prediction for a given text
def get_prediction(text, model_details, lr_checkpoint_path, device):
    embeddings = []
    for model, tokenizer in model_details:
        encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**encoded_input)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding)
    
    # Concatenate all embeddings to form a single feature vector
    concatenated_features = torch.cat(embeddings, dim=1)

    # Load logistic regression model and make prediction
    lr_model = load_soft_max_lit_from_checkpoint(lr_checkpoint_path, concatenated_features.shape[1], 2, device)
    with torch.no_grad():
        logits = lr_model(concatenated_features)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()
        predicted_probability = probabilities.max(dim=1).values.item()

    return predicted_class, predicted_probability

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_names = ['albert', 'electra', 'roberta', 'xlnet']
checkpoints = [
    'lightning_logs/version_0/checkpoints/model=albert--dev=False--epoch=89-step=10170--val_loss=0.35.ckpt',
    'lightning_logs/version_1/checkpoints/model=electra--dev=False--epoch=297-step=33674--val_loss=0.38.ckpt',
    'lightning_logs/version_2/checkpoints/model=roberta--dev=False--epoch=289-step=32770--val_loss=0.36.ckpt',
    'lightning_logs/version_3/checkpoints/model=xlnet--dev=False--epoch=30-step=3503--val_loss=0.37.ckpt'
]
lr_checkpoint_path = 'lightning_logs/version_4/checkpoints/model=lr--dev=False.ckpt'
model_details = [load_model(name, ckpt, device) for name, ckpt in zip(model_names, checkpoints)]

text = "Your sample text here"
predicted_class, predicted_probability = get_prediction(text, model_details, lr_checkpoint_path, device)
print(f"Predicted Class: {predicted_class}, Probability: {predicted_probability}")
