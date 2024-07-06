import os
import joblib
from matplotlib import pyplot as plt
import torch
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from helper import load_dataset
from model import TransformerModel, Data, get_dataloaders, SoftMaxLit
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
DEV = False

# Print current working directory for debugging
print("Current Working Directory:", os.getcwd())

checkpoints = [
    'lightning_logs/version_80/checkpoints/model=albert--dev=False--epoch=297-step=33674--val_loss=0.36.ckpt',
    'lightning_logs/version_81/checkpoints/model=electra--dev=False--epoch=298-step=33787--val_loss=0.39.ckpt',
    'lightning_logs/version_82/checkpoints/model=deberta--dev=False--epoch=299-step=33900--val_loss=0.37.ckpt',
    'lightning_logs/version_83/checkpoints/model=roberta--dev=False--epoch=297-step=33674--val_loss=0.35.ckpt',
    'lightning_logs/version_84/checkpoints/model=xlnet--dev=False--epoch=64-step=7345--val_loss=0.37.ckpt'
]


device = torch.device("cpu")

# Use the correct relative path
df = load_dataset('../dataset/training.json', test=True)

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

# Combine with text features
df_features = df[['WordCount',  'LexicalDiversity', 'NotStopwordRatio', 'PunctuationCount', 'AvgWordLength', 'AvgSentenceLength', 'ReadingEase']]
# combined_features = np.hstack([lr_dataset_x, df_features.values])
combined_features = df_features

# Debugging: Print shapes and lengths to ensure consistency
print(f"Shape of df_features: {df_features.shape}")
print(f"Shape of lr_dataset_x: {lr_dataset_x.shape}")
print(f"Shape of combined_features: {combined_features.shape}")


X_train, X_test, y_train, y_test = train_test_split(combined_features, df['label'], test_size=0.2, random_state=42)

# Train meta learner
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(lr_model, 'logistic_regression_model.pkl')
# Calculate SHAP values
explainer = shap.Explainer(lr_model, X_train)
shap_values = explainer(X_test)

# Debugging: Print shapes of SHAP values and length of feature names
print(f"SHAP values shape: {shap_values.shape}")
print(f"Length of feature_names: {len(df_features.columns)}")

# # Feature names
# feature_names = list(df_features.columns) + [f'Model_{i}' for i in range(len(model_y_arr))]
# print(f"Feature names: {feature_names}")

# # Ensure feature names match SHAP values dimensions
# if len(feature_names) != shap_values.shape[1]:
#     print(f"Mismatch in dimensions: {len(feature_names)} vs {shap_values.shape[1]}")
#     # Adjust feature names or investigate the cause of mismatch
print(df_features.columns)
# Display SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=df_features.columns)

# Generate explanation for a single instance
def explain_prediction(instance, shap_values, feature_names):
    explanation = "The following factors contributed to the prediction:\n"
    for name, shap_value in zip(feature_names, shap_values):
        print(f"Name: {name}, Type of shap_value: {type(shap_value)}, shap_value: {shap_value}")  # Debugging line
        
        if isinstance(shap_value, shap.Explanation):
            shap_value = shap_value.values if isinstance(shap_value.values, float) else shap_value.values[0]
        elif isinstance(shap_value, np.ndarray):
            shap_value = shap_value.item()  # Convert numpy array to scalar
        
        explanation += f"Feature '{name}' with SHAP value {float(shap_value):.2f}\n"
    return explanation

# Select a single instance
single_instance = X_test[0]
single_shap_values = shap_values[0]

# Print the single instance and its SHAP values
print(df_features.columns)
print(single_instance)
print(single_shap_values.values)

# Generate explanation
explanation = explain_prediction(single_instance, single_shap_values, df_features.columns)
print(explanation)

# Display SHAP force plot for a single prediction
shap.force_plot(explainer.expected_value, single_shap_values.values, single_instance, feature_names=feature_names)
plt.savefig('shap_force_plot.png')