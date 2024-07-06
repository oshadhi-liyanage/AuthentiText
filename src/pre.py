import torch
from transformers import AutoTokenizer

def get_prediction(text, models, lr_model, device):
    tokenized_inputs = [model['tokenizer'](text, return_tensors="pt", padding=True, truncation=True).to(device) for model in models]

    # Process each model to get embeddings
    embeddings = []
    for model in models:
        model['model'].eval()
        with torch.no_grad():
            outputs = model['model'](input_ids=tokenized_inputs[model['index']]['input_ids'], 
                                     attention_mask=tokenized_inputs[model['index']]['attention_mask'])
            embeddings.append(outputs.last_hidden_state[:, 0, :])  # Assuming using the [CLS] token's embedding

    # Concatenate all embeddings
    concatenated_embeddings = torch.cat(embeddings, dim=1)

    # Get predictions from the logistic regression model
    lr_model.eval()
    with torch.no_grad():
        logits = lr_model(concatenated_embeddings)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()

    return predicted_class, probabilities.tolist()

# Example usage:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text = "Example text to classify."

# Load models and tokenizer (this part is conceptual, you should adjust based on your actual setup)
models = [
    {'model': model_albert, 'tokenizer': AutoTokenizer.from_pretrained('albert-base-v2'), 'index': 0},
    {'model': model_electra, 'tokenizer': AutoTokenizer.from_pretrained('google/electra-small-discriminator'), 'index': 1},
    {'model': model_roberta, 'tokenizer': AutoTokenizer.from_pretrained('roberta-base'), 'index': 2},
    {'model': model_xlnet, 'tokenizer': AutoTokenizer.from_pretrained('xlnet-base-cased'), 'index': 3}
]

# Assuming lr_model is already loaded
predicted_class, probabilities = get_prediction(text, models, lr_model, device)
print(f"Predicted Class: {predicted_class}, Probabilities: {probabilities}")
