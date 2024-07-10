BATCH_SIZE = 128
NUM_EPOCH = 300

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from helper import load_dataset
from model import TransformerModel, Data, get_dataloaders, SoftMaxLit

import shap

DEV = False

df = load_dataset('../dataset/training.json', test=True)
checkpoints = []
for cur_model_name in list(TransformerModel.MODELS.keys()):
    cur_dataset_x = torch.load(f'pretrained--dev={DEV}--model={cur_model_name}.pt')
    cur_data = Data(df, x=cur_dataset_x)
    cur_dataloaders = get_dataloaders(cur_data, BATCH_SIZE)
    additional_feature_dim = df.shape[1] - 2  # -2 to exclude 'text' and 'label'
    combined_feature_dim = TransformerModel.MODELS[cur_model_name]['dim'] + additional_feature_dim
    cur_model = SoftMaxLit(combined_feature_dim, 2)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        filename=f'model={cur_model_name}--dev={DEV}' + '--{epoch}-{step}--{val_loss:.2f}'
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback], max_epochs=NUM_EPOCH)
    trainer.fit(model=cur_model, train_dataloaders=cur_dataloaders['train'], val_dataloaders=cur_dataloaders['val'])

    checkpoints.append(checkpoint_callback.best_model_path)
    best_model = cur_model.load_from_checkpoint(
        n_inputs=combined_feature_dim,
        n_outputs=2,
        checkpoint_path=checkpoint_callback.best_model_path
    )
    trainer.test(best_model, dataloaders=cur_dataloaders['test'])

    # Initialize explainer
    explainer = shap.DeepExplainer(best_model, cur_data.x)

    # Select a subset of the dataset for explanation
    sample_indices = np.random.choice(len(cur_data), size=5, replace=False)
    sample_data = cur_data.x[sample_indices]

    #    Compute SHAP values
    shap_values = explainer.shap_values(sample_data)
    #Verify dimensions
    print(f"Sample data shape: {sample_data.shape}")
    print(f"SHAP values shape: {np.array(shap_values).shape}")

# Check the first few elements to ensure correctness
    # print(f"First sample data: {sample_data[0]}")
    print(f"First SHAP values (output 0): {shap_values[0][0].shape}")
    print(f"First SHAP values (output 1): {shap_values[:, :, 1].shape}")

 # Generate combined feature names
    bert_feature_dim = TransformerModel.MODELS[cur_model_name]['dim']
    additional_features = list(df.columns[2:])
    feature_names = [f'BERT_feature_{i}' for i in range(bert_feature_dim)] + additional_features


# Generate combined feature names
    # combined_feature_dim = TransformerModel.MODELS[cur_model_name]['dim'] + len(df.columns[2:])
    # feature_names = [f'BERT_feature_{i}' for i in range(TransformerModel.MODELS[cur_model_name]['dim'])] + list(df.columns[2:])

# Ensure the lengths match
    assert sample_data.shape[1] == len(feature_names), "Mismatch between feature names and sample data dimensions"

# Select SHAP values for the first output (index 0)
shap_values_output_0 = shap_values[:, :, 1]  # Shape: (5, 775, 2)

# Extract the SHAP values for the first class and first sample
shap_values_output_0_single_sample = shap_values[:, :, 1]    # Correctly index to get (775,)

# Print the shapes of the SHAP values and the first sample's SHAP values to debug
print(f"SHAP values for output 0 shape: {shap_values_output_0_single_sample.shape}")
print(f"First sample data shape: {sample_data[0].shape}")
additional_feature_indices = list(range(bert_feature_dim, len(feature_names)))
shap_values_additional_features = shap_values_output_0_single_sample[:, additional_feature_indices]
additional_feature_names = [feature_names[i] for i in additional_feature_indices]

# Verify lengths
print(f"Length of SHAP values for first sample: {len(shap_values_output_0_single_sample)}")  # Should be 775
print(f"Length of feature names: {len(feature_names)}")  # Should be 775


# Visualize the first prediction's explanation for a single sample
shap.force_plot(explainer.expected_value[1], shap_values_additional_features[0], sample_data[0][additional_feature_indices].cpu().numpy(), feature_names=additional_feature_names)

    # Visualize feature importance for the additional features
shap.summary_plot(shap_values_additional_features, sample_data[:, additional_feature_indices].cpu().numpy(), feature_names=additional_feature_names)

del cur_dataset_x
del cur_data.x
torch.cuda.empty_cache()
