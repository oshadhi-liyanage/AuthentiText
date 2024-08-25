import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
import torchmetrics

class AlbertTextClassifier(pl.LightningModule):
    def __init__(self, num_labels):
        super(AlbertTextClassifier, self).__init__()
        self.model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=num_labels)
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, batch['labels'])
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.validation_step_outputs.append({'val_loss': loss, 'val_acc': acc})
        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = self.val_acc.compute()
        self.log('avg_val_loss', avg_loss)
        self.log('avg_val_acc', avg_acc)
        print(f'Validation Accuracy: {avg_acc:.4f}')
        self.val_acc.reset()
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)

    def attribute_text(self, text, label):
        self.model.eval()
        self.model.zero_grad()

        encoded = self.tokenizer.encode_plus(text, add_special_tokens=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        def forward_func(input_ids, attention_mask):
            return self.forward(input_ids, attention_mask)

        lig = LayerIntegratedGradients(forward_func, self.model.albert.embeddings)

        attributions, delta = lig.attribute(inputs=input_ids,
                                            additional_forward_args=(attention_mask,),
                                            target=label,
                                            return_convergence_delta=True)

        return attributions, delta, input_ids.squeeze().tolist()

def visualize_attributions(model, tokenizer, attributions, tokens, delta, text, label):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    token_words = tokenizer.convert_ids_to_tokens(tokens)
    token_words = [word.replace('▁', ' ').strip() for word in token_words]

    input_ids = torch.tensor([tokens]).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model(input_ids, attention_mask)

    pred_prob = torch.softmax(output, dim=1)
    pred_class = torch.argmax(pred_prob).item()

    vis_data_records = []
    vis_data_records.append(viz.VisualizationDataRecord(
                            attributions,
                            pred_prob[0, pred_class].item(),
                            pred_class,
                            label,
                            "label",
                            attributions.sum(),
                            token_words,
                            delta))

    viz.visualize_text(vis_data_records)