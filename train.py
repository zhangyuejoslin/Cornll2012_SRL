import torch
from tqdm import tqdm
from data_helper.data_reader import data_preprocesing
from model.LSTM_baseline import LSTM_Model
import random


def train(model, opt, new_train_sample, voacb_label):
    ls = []
    for example in tqdm(new_train_sample):
        # token: batch * length of sentence
        # label_list: batch * length
        token, label_list = example
         # token: length * batch
        token = zip(*token)

        opt.zero_grad()
        #logit: length * batch * dim
        token = torch.tensor(tuple(token)).cuda()
        logit = model(token)
        #label_vec: length * batch * label_length
        label_vec = torch.zeros(token.shape[0], token.shape[1], len(voacb_label)).cuda()

        for batch_num in range(token.shape[1]):
            for each_label in range(len(label_list[batch_num])):
                label_num = label_list[batch_num][each_label]
                label_vec[each_label, batch_num, label_num] = 1

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label_vec).cuda()
        loss.backward()
        opt.step()
        ls.append(loss.item())
    return ls

def find_list(indices, data):
    out = []
    for i in indices:
        out = out +[data[i][:20]]
    return out

def generate_batch(vocab_list, label_list, batch_size, shuffle=False):
    rows = len(vocab_list)
    indices = list(range(rows))
    number = 1
    if shuffle:
        random.seed(100)
        random.shuffle(indices)
    while True:
        batch_indices = indices[0:batch_size]
        indices = indices[batch_size:] + indices[:batch_size]
        temp_data_vocab = find_list(batch_indices, vocab_list)
        temp_data_label = find_list(batch_indices, label_list)
        batch_data = (temp_data_vocab, temp_data_label)
        yield batch_data
        if batch_size * number >= rows:
            break
        number += 1
       


if __name__ == '__main__':

    train_set, dev_set, emb, vocab, labels = data_preprocesing('data/BIO-formatted/conll2012.train.txt',
                                                                           'data/BIO-formatted/conll2012.devel.txt',
                                                                           'data/glove.6B.50d.txt', 20)
    save_file_path = 'model-lstm.th'
    model = LSTM_Model(emb, labels.stoi).cuda()
    train_samples_np, train_mask_np, train_labels_np, train_predicate_np = train_set
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(50):
        new_train_sample =  generate_batch(train_samples_np, train_labels_np, 50, False)
        ls = train(model, opt, new_train_sample,labels.stoi)
    torch.save({'model': model.state_dict()}, save_file_path)

 