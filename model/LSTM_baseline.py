import torch

def create_emb_layer(weights_matrix, non_trainable=False):
    emb_layer = torch.nn.Embedding.from_pretrained(torch.tensor(weights_matrix, dtype=torch.float), padding_idx=0)
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer

class LSTM_Model(torch.nn.Module):
    def __init__(self, weights_matrix,labels):
        super(LSTM_Model, self).__init__()
        self.embedding = create_emb_layer(weights_matrix, False)
        self.rnn = torch.nn.LSTM(input_size=50, hidden_size=100, num_layers=2, batch_first=False, bidirectional=True)
        self.linear = torch.nn.Linear(200, len(labels))

    def forward(self,sentence):
         # emb: length * batch * dim
        emb = self.embedding(sentence)
        rnn_embedding, _ = self.rnn(emb)
        logits = self.linear(rnn_embedding)
        return logits