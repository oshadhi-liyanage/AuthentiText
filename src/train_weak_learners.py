import shap
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from helper import load_dataset
from model import TransformerModel, Data, get_dataloaders, SoftMaxLit

BATCH_SIZE = 128
NUM_EPOCH = 300
DEV = False

df = load_dataset('../dataset/training.json', test=True)
checkpoints = []

for cur_model_name in list(TransformerModel.MODELS.keys()):
    # Load the saved embeddings and features
    cur_dataset_x, cur_features = torch.load(f'pretrained--dev={DEV}--model={cur_model_name}.pt')
    cur_data = Data(df, x=cur_dataset_x, features=cur_features)
    cur_dataloaders = get_dataloaders(cur_data, BATCH_SIZE)
    cur_model = SoftMaxLit(TransformerModel.MODELS[cur_model_name]['dim'], 2)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        filename=f'model={cur_model_name}--dev={DEV}' + '--{epoch}-{step}--{val_loss:.2f}'
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback], max_epochs=NUM_EPOCH)
    trainer.fit(model=cur_model, train_dataloaders=cur_dataloaders['train'], val_dataloaders=cur_dataloaders['val'])

    checkpoints.append(checkpoint_callback.best_model_path)
    best_model = cur_model.load_from_checkpoint(n_inputs=TransformerModel.MODELS[cur_model_name]['dim'], n_outputs=2, checkpoint_path=checkpoint_callback.best_model_path)
    trainer.test(best_model, dataloaders=cur_dataloaders['test'])

    # SHAP Integration
    def model_predict(x):
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return best_model(x).numpy()

    print("Data matrix shape:", cur_data.features.shape)

    print("Data columns:", cur_data.features)


    # Explain model predictions using SHAP
    background = cur_data.x[:100].numpy()
    explainer = shap.Explainer(model_predict, background, max_evals=2 * TransformerModel.MODELS[cur_model_name]['dim'] + 1)
    shap_values = explainer(cur_data.x.numpy())
    print("SHAP values shape:", shap_values.shape)
    print("SHAP values:", shap_values)

    # Generate feature names
    feature_names = ["avg_word_length", "avg_sentence_length", "sentiment"]
    
    print("Feature names:", feature_names)

    # Plot the SHAP values for the first sample in the dataset
    shap.summary_plot(shap_values, cur_data.features, plot_type="bar", feature_names=feature_names)

    for i in range(5):
        shap.initjs()
        shap.force_plot(explainer.expected_value, shap_values[i], cur_data.features[i], feature_names=feature_names)

    del cur_dataset_x
    del cur_data.x
    torch.cuda.empty_cache()
