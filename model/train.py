import torch
from tqdm import tqdm
import sys
sys.path.append('../')
from data_helper import data_reader
from model.LSTM_baseline import LSTM_Model

def train(model, opt, new_train_sample):
    ls = []
    for example in tqdm(new_train_sample):
        sid = 0
        token, label_list = example
        opt.zero_grad()
        logit = model(torch.tensor(token))
        label_vec = torch.zeros(len(token), 1, len(labels))
        for each_label in label_list:
            label_vec[sid, 0, each_label] = 1
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label_vec)
        loss.backward()
        opt.step()
        ls.append(loss.item())
        sid += 1
    return ls

if __name__ == '__main__':

    train_set, dev_set, emb, vocab, labels = data_reader.data_preprocesing('../data/BIO-formatted/conll2012.train.txt',
                                                                           '../data/BIO-formatted/conll2012.devel.txt',
                                                                           '../data/glove.6B.50d.txt', 20)
    save_file_path = 'model-lstm.th'
    model = LSTM_Model(emb, labels)
    train_samples_np, train_mask_np, train_labels_np, train_predicate_np = train_set
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(50):
        new_train_sample = zip(train_samples_np, train_labels_np)
        ls = train(model, opt, new_train_sample)
    torch.save({'model': model.state_dict()}, save_file_path)
