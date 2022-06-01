import torch
from torch import nn

INPUT_SIZE = 147158 # vocab size
EMBEDDING_SIZE = 300
SEQUENCE_SIZE = 2752
BATCH_SIZE = 10

class FcNet(nn.Module):
    def __init__(self):
        super(FcNet, self).__init__()
        self.embedding = nn.Embedding(INPUT_SIZE, EMBEDDING_SIZE)
        self.linear = nn.Linear(EMBEDDING_SIZE * SEQUENCE_SIZE, 2)
        self.softmax = nn.LogSoftmax(dim=0)
    
    def forward(self, input):
        # inputs [b, s] => [b, s, EMBEDDING_SIZE]
        embedding = self.embedding(input)
        # [b, s, EMBEDDING_SIZE] => [b, s*EMBEDDING_SIZE]
        embedding = embedding.view(BATCH_SIZE, -1)
        # [b, s*EMBEDDING_SIZE] => [b, 2]
        out = self.linear(embedding)
        # [b, 2]
        return self.softmax(out)

if __name__ == '__main__':
    net = FcNet()
    X = torch.randint(0, INPUT_SIZE, (BATCH_SIZE, SEQUENCE_SIZE))
    y = torch.randint(0, 1, (BATCH_SIZE,))
    y_hat = net(X)
    print(y_hat.shape)
    print(y.shape)