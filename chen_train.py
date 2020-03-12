from sklearn import metrics
from statistics import mean
import torch
from tqdm import tqdm
from data_helper.data_reader import *
from model.SRL_model import SRL_Model
from config.global_config import CONFIG
import os
import random

#### load config
cfg = CONFIG()

def train(model, opt, new_train_sample, voacb_label):
    ls = []
    for example in tqdm(new_train_sample):
        # token: batch * length of sentence
        # label_list: batch * length
        token, mask, label_list = example
         # token: length * batch
        token = zip(*token)

        opt.zero_grad()
        #logit: length * batch * dim
        token = torch.tensor(tuple(token)).cuda()
        mask = torch.tensor(tuple(mask)).cuda()
        logit = model(token, mask)
        #label_vec: length * batch * label_length
        label_vec = torch.zeros(token.shape[1], token.shape[0], len(voacb_label)).cuda()

        for batch_num in range(token.shape[1]):
            for each_label in range(len(label_list[batch_num])):
                label_num = label_list[batch_num][each_label]
                # label_vec[each_label, batch_num, label_num] = 1
                label_vec[batch_num, each_label, label_num] = 1
                

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

def generate_batch(train_samples_np, train_mask_np, train_labels_np, batch_size, shuffle=False):
    rows = len(train_samples_np)
    indices = list(range(rows))
    number = 1
    if shuffle:
        random.seed(100)
        random.shuffle(indices)
    while True:
        batch_indices = indices[0:batch_size]
        indices = indices[batch_size:] + indices[:batch_size]
        temp_data_vocab = find_list(batch_indices, train_samples_np)
        temp_data_mask = find_list(batch_indices, train_mask_np)
        temp_data_label = find_list(batch_indices, train_labels_np)
        batch_data = (temp_data_vocab, temp_data_mask, temp_data_label)
        yield batch_data
        if batch_size * number >= rows:
            break
        number += 1

if __name__ == '__main__':

    train_set, dev_set, emb, vocab, labels = data_preprocesing('data/BIO-formatted/conll2012.train.txt',
                                                                'data/BIO-formatted/conll2012.devel.txt',
                                                                '/home/hlr/shared/data/glove6B/glove.6B.50d.txt', 20)
    if not os.path.exists(cfg.model_store_dir):
        os.makedirs(cfg.model_store_dir)
    model = SRL_Model(emb, labels.stoi, is_test=False).cuda()
    train_samples_np, train_mask_np, train_labels_np, train_predicate_np = train_set
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(50):
        # new_train_sample =  generate_batch(train_samples_np, train_labels_np, 50, False)
        # ls = train(model, opt, new_train_sample,labels.stoi)
        print(f'Starting epoch {epoch+1}') 
        new_train_sample =  generate_batch(train_samples_np, train_mask_np, train_labels_np, 50, False)
        ls = train(model, opt, new_train_sample,labels.stoi)
        print(f'Epoch {epoch+1} finished, avg loss: {mean(ls)}')
    torch.save({'model': model.state_dict()}, cfg.model_store_file)

