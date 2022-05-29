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
        
    def __len__(self):
        assert len(self.texts_indices) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.texts_indices[idx], self.labels[idx])

def build_dataloader(dataset, batch_size):
    def collate_fn(batch):
        # 遍历得到max length of text, 然后将len < max length的padding
        new_batch = []
        max_length = 0
        for i, (text, label) in enumerate(batch):
            if len(text) > max_length:
                max_length = len(text)
        
        for i, (text, label) in enumerate(batch):
            if len(text) < max_length:
                text.extend([0 for j in range(max_length - len(text))])
            new_batch.append((torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)))
        return new_batch
    
    return dataloader.DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)

if __name__ == '__main__':
    imdb_dataset = IMDBDataset()
    data_iter = build_dataloader(imdb_dataset, batch_size=10)
    text, label = next(iter(data_iter))[0]
    print(text, label)
    print(text.shape, label.shape)