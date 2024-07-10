BATCH_SIZE = 128
NUM_EPOCH = 100

# Update the following checkpoints in the following order: albert, electra, roberta, xlnet
checkpoints = [
    'lightning_logs/version_212/checkpoints/model=albert--dev=False--epoch=7-step=8--val_loss=0.32.ckpt'
    # 'lightning_logs/version_10/checkpoints/model=electra--dev=False--epoch=299-step=33900--val_loss=0.39.ckpt',
    # 'lightning_logs/version_11/checkpoints/model=deberta--dev=False--epoch=298-step=33787--val_loss=0.37.ckpt'
    # 'lightning_logs/version_12/checkpoints/model=roberta--dev=False--epoch=298-step=33787--val_loss=0.36.ckpt',
    # 'lightning_logs/version_13/checkpoints/model=xlnet--dev=False--epoch=52-step=5989--val_loss=0.37.ckpt'
]

import numpy as np
import shap
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from helper import load_dataset
from model import TransformerModel, Data, get_dataloaders, SoftMaxLit

DEV = False
device = torch.device("cpu")
df = load_dataset('../dataset/training.json', test=True)

pretrained_datasets_x = [
    f"pretrained--dev={DEV}--model=albert.pt"
    # f"pretrained--dev={DEV}--model=electra.pt",
    # f"pretrained--dev={DEV}--model=deberta.pt",
    # f"pretrained--dev={DEV}--model=roberta.pt",
    # f"pretrained--dev={DEV}--model=xlnet.pt"
]
# print(df.shape)
model_y_arr = []
for model_name, pretrained_dataset_x, ckpt in zip(list(TransformerModel.MODELS.keys()), pretrained_datasets_x, checkpoints):
    additional_feature_dim = df.shape[1] - 2  # -2 to exclude 'text' and 'label'
    combined_feature_dim = TransformerModel.MODELS[model_name]['dim'] + additional_feature_dim
    n_inputs = TransformerModel.MODELS[model_name]['dim']
    model = SoftMaxLit(combined_feature_dim, 2).load_from_checkpoint(n_inputs=combined_feature_dim, n_outputs=2, checkpoint_path=ckpt)
    x = torch.load(pretrained_dataset_x).to(device)
    y_hat = model(x)
    print("y_hat")
    print(y_hat)

    # Free up memory
    del x
    torch.cuda.empty_cache()
    y_first = y_hat

    model_y_arr.append(y_first)
print("model_y_arr shape")
print(model_y_arr)
lr_dataset_x = torch.cat(model_y_arr, dim=1).detach()
print("lr_dataset_x shape")
print(lr_dataset_x.shape)

lr_dataset = Data(df, x=lr_dataset_x)
lr_dataloaders = get_dataloaders(lr_dataset, BATCH_SIZE)

lr_model = SoftMaxLit(lr_dataset_x.shape[1], 2)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor='val_loss',
    mode='min',
    filename=f'model=lr--dev={DEV}'
)

trainer = pl.Trainer(callbacks = [checkpoint_callback], max_epochs=NUM_EPOCH) # callbacks=[checkpoint_callback]
trainer.fit(model=lr_model, train_dataloaders=lr_dataloaders['train'], val_dataloaders=lr_dataloaders['val'])
best_lr_model = lr_model.load_from_checkpoint(n_inputs=lr_dataset_x.shape[1], n_outputs=2, checkpoint_path=checkpoint_callback.best_model_path)
trainer.test(best_lr_model, dataloaders=lr_dataloaders['test'])

def model_predict(data):
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():
        return best_lr_model(data_tensor).cpu().numpy()
    
# Compute SHAP values for the meta-learner using LinearExplainer
explainer_lr = shap.KernelExplainer(model_predict, lr_dataset_x[:4].cpu().numpy())
shap_values_lr = explainer_lr.shap_values(lr_dataset_x.cpu().numpy())
shap_values_output_0 = shap_values_lr[:, :, 1]

# Debugging SHAP values structure
print(f"Type of shap_values_lr: {type(shap_values_lr)}")
print(f"Shape of shap_values_lr: {np.array(shap_values_lr).shape}")
print(f"Shape of shap_values_output_0 : {np.array(shap_values_output_0).shape}")

# Visualize SHAP values for the meta-learner
shap.summary_plot(shap_values_output_0, lr_dataset_x.cpu().numpy())