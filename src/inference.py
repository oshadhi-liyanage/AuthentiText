lr_checkpoint_path = 'lightning_logs/version_19/checkpoints/model=lr--dev=False.ckpt'
# Update the following checkpoints in the following order: albert, electra, roberta, xlnet
checkpoints = [
    'lightning_logs/version_14/checkpoints/model=albert--dev=False--epoch=122-step=13899--val_loss=0.36.ckpt',
    'lightning_logs/version_15/checkpoints/model=electra--dev=False--epoch=299-step=33900--val_loss=0.39.ckpt',
    'lightning_logs/version_16/checkpoints/model=deberta--dev=False--epoch=298-step=33787--val_loss=0.38.ckpt',
    'lightning_logs/version_17/checkpoints/model=roberta--dev=False--epoch=291-step=32996--val_loss=0.36.ckpt',
    'lightning_logs/version_18/checkpoints/model=xlnet--dev=False--epoch=241-step=27346--val_loss=0.38.ckpt'
]

from sklearn.metrics import classification_report, confusion_matrix
import torch
from helper import load_dataset
from model import TransformerModel, SoftMaxLit

DEV = False
# device = torch.cuda.current_device()
device = torch.device("cpu")
df = load_dataset('../dataset/training.json', test=True)

validation_df = load_dataset('../dataset/test_data.json', test=True)
model_names = ['albert', 'electra','deberta', 'roberta', 'xlnet'] #albert: 128, electra: 64, roberta: 128, xlnet: 128

# print(validation_df.head())
model_y_arr = []
for model_name, ckpt in zip(model_names, checkpoints):
    n_inputs = TransformerModel.MODELS[model_name]['dim']
    model = SoftMaxLit(n_inputs, 2).load_from_checkpoint(n_inputs=n_inputs, n_outputs=2, checkpoint_path=ckpt)
    print(f'Loaded model {model_name} from checkpoint {ckpt}')
    x = TransformerModel(model_name).dataset(validation_df, DEV, save=False, delete=False).x.to(device)
    y_hat = model(x)

    # Free up memory
    del x
    torch.cuda.empty_cache()
    y_first = y_hat

    model_y_arr.append(y_first)
lr_dataset_x = torch.cat(model_y_arr, dim=1).detach()
x = lr_dataset_x.to(device)

lr_model = SoftMaxLit(lr_dataset_x.shape[1], 2).load_from_checkpoint(n_inputs=lr_dataset_x.shape[1], n_outputs=2, checkpoint_path=lr_checkpoint_path).to(device)

validation_out = lr_model(x)
validation_out = validation_out.detach()
out = torch.argmax(validation_out, dim=1)
# f = open('answer.json', 'w')
# f.write('')
# f.close()

# f = open('answer.json', 'a')
# for idx, label_out in enumerate(out.tolist()):
#     to_write = '{"id": ' + str(idx) + ', "label": ' + str(label_out) + '}\n'
#     f.write(to_write)
# f.close()
# Write predictions to JSON file
with open('answer.json', 'w') as f:
    for idx, label_out in enumerate(out.tolist()):
        to_write = '{"id": ' + str(idx) + ', "label": ' + str(label_out) + '}\n'
        f.write(to_write)

# Calculate confusion matrix
true_labels = validation_df['label'].tolist()  # Replace 'label' with the actual column name in your dataset
predicted_labels = out.tolist()
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)

# Generate classification report
report = classification_report(true_labels, predicted_labels, target_names=['Class 0', 'Class 1'], output_dict=True)
print("Classification Report:")
for key, value in report.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for sub_key, sub_value in value.items():
            print(f"  {sub_key}: {sub_value}")
    else:
        print(f"{key}: {value}")

# Extract and print additional metrics
accuracy = report['accuracy']
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

print(f"\nAccuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1_score}")