import torch
from data_helper import data_reader

def create_emb_layer(weights_matrix, non_trainable=False):
    emb_layer = torch.nn.Embedding.from_pretrained(torch.tensor(weights_matrix, dtype=torch.float), padding_idx=0)
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer

class Model(torch.nn.Module):
    def __init__(self, weights_matrix,labels):
        super(Model, self).__init__()
        self.embedding = create_emb_layer(weights_matrix, False)
        self.rnn = torch.nn.LSTM(input_size=50, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(50, len(labels))

    def forward(self,sentence):
        token_idx = torch.tensor(data_reader.token_2_id(sentence))
        rnn_embedding = self.rnn(self.embedding(token_idx))
        logits = self.linear(rnn_embedding)
        return logits