import os
import sys

import torch
import pandas as pd
from torch.utils.data import Dataset, dataloader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from build_vocab import build_vocab, tokenizer


class IMDBDataset(Dataset):
    
    def __init__(self):
        data = pd.read_csv('/Users/chenbai/Projects/ml/imdb-sentiment/data/IMDB Dataset.csv')
        texts, labels = data['review'], data['sentiment']
        tokens = []
        texts_tokens = []
        for text in texts:
            text_tokens = tokenizer(text)
            texts_tokens.append(text_tokens)
            tokens.extend(text_tokens)
        self.vocab = build_vocab(tokens)
        
        # [[0, 1, 5, ...], [5, 2, 1, ...], ...]
        self.texts_indices = []
        for tokens in texts_tokens:
            self.texts_indices.append(self.vocab.lookup_indices(tokens))
        self.labels = [0 if label == 'negative' else 1 for label in labels]
        self.max_length = len(max(self.texts_indices, key=len))

        
    def __len__(self):
        assert len(self.texts_indices) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        # padding
        pad_idx = self.vocab.get_stoi()['[pad]']
        if len(self.texts_indices[idx]) < self.max_length:
           self.texts_indices[idx].extend([pad_idx for i in range(self.max_length - len(self.texts_indices[idx]))])
        return (torch.tensor(self.texts_indices[idx], dtype=torch.int64),
                torch.tensor(self.labels[idx], dtype=torch.int64))

def build_dataloader(dataset, batch_size):    
    return dataloader.DataLoader(dataset, batch_size)

if __name__ == '__main__':
    imdb_dataset = IMDBDataset()
    data_iter = build_dataloader(imdb_dataset, batch_size=10)
    texts, labels = next(iter(data_iter))
    print(texts[0].shape, labels[0].shape)
    texts, labels = next(iter(data_iter))
    print(texts[0].shape, labels[0].shape)