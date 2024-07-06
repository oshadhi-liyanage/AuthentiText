import math
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim, nn, utils
from torch.utils.data import Dataset, DataLoader
import nltk
from textblob import TextBlob
nltk.download('punkt')

import lightning.pytorch as pl
# import pytorch_lightning as plpl
# from lightning.pytorch.callbacks import ModelCheckpoint

from transformers import XLNetTokenizer, XLNetModel, AutoTokenizer, AlbertModel, AutoModel, ElectraModel, RobertaModel, AlbertTokenizer,DebertaModel,DebertaTokenizer

class SoftMaxLit(pl.LightningModule):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.softmax(self.linear(x))
        
    def training_step(self, batch, batch_idx):
        x, features, y = batch
        y_hat = self(x)  # Assuming the model uses x only, not the additional features directly
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, features, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        
    def test_step(self, batch, batch_idx):
        x, features, y = batch
        y = torch.argmax(y, dim=1)
        y_hat = torch.argmax(self(x), dim=1)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        self.log('test_acc', accuracy)


class Data(Dataset):
    def __init__(self, df, *, x=None, features=None, load_batch_size=None, tokenizer=None, pretrained=None):
        self.y, self.len = self._get_y_and_len_from_df(df)
        
        if x is not None and features is not None:
            self.x = x
            self.features = features
        else:
            self.x, self.features = self._get_x_and_features_from_df(df, load_batch_size, tokenizer, pretrained)
        
    def _get_x_and_features_from_df(self, df, load_batch_size, tokenizer, pretrained):
        docs = df['text'].tolist()
        inputs = tokenizer(docs, return_tensors="pt", padding=True, truncation=True)
        cls_arr = []
        features_arr = []
        
        for i, (x, y) in zip(tqdm(range(math.ceil(len(df) / load_batch_size))), self._get_x_y_from_df_with_batch(df, load_batch_size)):
            cls = pretrained(**{k: inputs[k][x:y] for k in list(inputs.keys())}).last_hidden_state[:, 0, :].detach()
            cls_arr.append(cls)
            features_arr.extend(self._extract_features(docs[x:y]))
        
        embeddings = torch.concat(cls_arr).type(torch.float32)
        features = np.array(features_arr)
        return embeddings, features
    
    def _extract_features(self, texts):
        features = []
        for text in texts:
            tokens = nltk.word_tokenize(text)
            avg_word_length = np.mean([len(word) for word in tokens])
            sentences = nltk.sent_tokenize(text)
            avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences])
            sentiment = TextBlob(text).sentiment.polarity
            features.append([avg_word_length, avg_sentence_length, sentiment])
        return features
    
    def _get_y_and_len_from_df(self, df):
        dim_0 = df['text'].shape[0]
        matrix = np.zeros((dim_0, 2))
        for i, y in enumerate(df['label'].tolist()):
            matrix[i][y] = 1
        return torch.from_numpy(matrix).type(torch.float32), dim_0

    def _get_x_y_from_df_with_batch(self, df, step_size):
        l = list(range(0, len(df), step_size))
        for ind, _ in enumerate(l):
            if l[ind] + step_size >= len(df):
                yield (l[ind], len(df))
            else:    
                yield (l[ind], l[ind + 1])

    def __getitem__(self, idx):
        return self.x[idx], self.features[idx], self.y[idx]
    
    def __len__(self):
        return self.len

    @staticmethod
    def concat(df, datasets):
        x = torch.cat([dataset.x for dataset in datasets], 1)
        features = np.concatenate([dataset.features for dataset in datasets], axis=1)
        return Data(df, x=x, features=features)

# MODELS
class TransformerModel():

    MODELS = {
        'albert': {'name': 'albert-base-v2', 'dim': 768, 'tokenizer': AlbertTokenizer, 'pretrained': AlbertModel}
        # 'electra': {'name': 'google/electra-small-discriminator', 'dim': 256, 'tokenizer': AutoTokenizer, 'pretrained': ElectraModel},
        # 'deberta': {'name': 'microsoft/deberta-base', 'dim': 768, 'tokenizer': DebertaTokenizer, 'pretrained': DebertaModel},
        # 'roberta': {'name': 'roberta-base', 'dim': 768, 'tokenizer': AutoTokenizer, 'pretrained': RobertaModel},
        # 'xlnet': {'name': 'xlnet-base-cased', 'dim': 768, 'tokenizer': XLNetTokenizer, 'pretrained': XLNetModel}, 
    }

    def __init__(self, model_tag):
        if model_tag not in list(self.MODELS.keys()):
            raise ValueError(f'Invalid model: {model_tag}. Valid models are: {" ".join(self.MODELS.keys())}')
        
        self.model_tag = model_tag
        self.dim = self.MODELS[model_tag]['dim']
        self.tokenizer = self.MODELS[model_tag]['tokenizer'].from_pretrained(self.MODELS[model_tag]['name'])
        self.pretrained = self.MODELS[model_tag]['pretrained'].from_pretrained(self.MODELS[model_tag]['name'])
        
    def dataset(self, df, dev, save=False, delete=False):
        # cur_df = df[:100] if dev else df
        dataset = Data(df, load_batch_size=30, tokenizer=self.tokenizer, pretrained=self.pretrained)

        if save:
            torch.save((dataset.x, dataset.features), f"pretrained--dev={dev}--model={self.model_tag}.pt")
        
        if delete:
            del dataset.x
            del dataset.features
            torch.cuda.empty_cache()

        return dataset

def get_dataloaders(dataset, batch_size):
    train_dataset, val_dataset, test_dataset = utils.data.random_split(dataset, (0.8, 0.1, 0.1))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
