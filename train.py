from sklearn import metrics
from statistics import mean
import torch
import random
from tqdm import tqdm
from data_helper.data_reader import data_preprocesing
from data_helper.prediction_builder import save_predictions
from model.LSTM_baseline import LSTM_Model
import random
from model.SRL_model import SRL_Model
from model.Decoder import viterbi_decode, ilp_decoder
from config.global_config import CONFIG
import os
import numpy as np
import math

cfg = CONFIG()

def train(model, opt, new_train_sample, vocab_label):
    ls = []
    for example in tqdm(new_train_sample, total=(82510//cfg.batch_size), desc="Training"):
        # token: batch * length of sentence
        # label_list: batch * length
        token, mask, label_list, gold_predicate = example
        # token: length * batch
        token = zip(*token)

        opt.zero_grad()
        token = torch.tensor(tuple(token)).cuda(cfg.use_which_gpu)
        mask = torch.tensor(tuple(mask)).cuda(cfg.use_which_gpu)
        gold_predicate = torch.tensor(tuple(gold_predicate), dtype=torch.float32).cuda(cfg.use_which_gpu)
        #logit: length * batch * dim
        logit = model(token, gold_predicate)
        #label_vec: length * batch * label_length
        label_vec = torch.zeros(token.shape[0], token.shape[1], len(vocab_label)).cuda(cfg.use_which_gpu)

        for batch_num in range(token.shape[1]):
            for each_label in range(len(label_list[batch_num])):
                label_num = label_list[batch_num][each_label]
                label_vec[each_label, batch_num, label_num] = 1

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label_vec).cuda(cfg.use_which_gpu)
        loss.backward()
        opt.step()
        ls.append(loss.item())
    return ls

def eval_micro_f1_score(model, samples, masks, labels, gold_predicate, label_vocab):
    """
    model: A pytorch module
    samples: dataset samples (n * max_len)
    masks: dataset mask (n * max_len)
    labels: dataset labels (n * max_len)
    label_vocab: a torchtext vocab for labels
    """
    all_preds = torch.tensor([],dtype=torch.long).cuda(cfg.use_which_gpu)
    all_labels = torch.tensor([],dtype=torch.long).cuda(cfg.use_which_gpu)
    with torch.no_grad():
        for i in tqdm(range(samples.shape[0]), total=(len(samples)), desc="Validation"):
            # tokens: 1 * length of sentence
            # label_list: 1 * length
            tokens = torch.tensor(samples[i,:][masks[i]==1], dtype=torch.long).unsqueeze(0).cuda(cfg.use_which_gpu)
            label_list = torch.tensor(labels[i,:][masks[i]==1], dtype=torch.long).unsqueeze(0).cuda(cfg.use_which_gpu)
            cur_gold_predicate = torch.tensor(gold_predicate[i, :][masks[i]==1], dtype=torch.float32).unsqueeze(0).cuda(cfg.use_which_gpu)
            
            cur_gold_predicate = torch.tensor(cur_gold_predicate, dtype=torch.float32).cuda(cfg.use_which_gpu)
        

            # tokens: len * 1
            tokens = torch.t(tokens)
            #logit: length * 1 * labels
            logit: torch.Tensor = model(tokens, cur_gold_predicate)

            # argmax predictions
            # predictions: length * 1
            _, predictions = logit.max(dim=2)

            # predictions: length
            predictions.squeeze_(-1)
            # label_list: length
            label_list.squeeze_(0)

            all_preds = torch.cat((all_preds, predictions))
            all_labels = torch.cat((all_labels, label_list))
    
    return metrics.f1_score(y_true=all_labels.cpu(), y_pred=all_preds.cpu(), average='micro')

def eval_with_baseline(model, samples, masks, labels,gold_predicate, label_vocab):
    """
    model: A pytorch module
    samples: dataset samples (n * max_len)
    masks: dataset mask (n * max_len)
    labels: dataset labels (n * max_len)
    label_vocab: a torchtext vocab for labels
    """
    all_preds = torch.tensor([],dtype=torch.long).cuda(cfg.use_which_gpu)
    all_labels = torch.tensor([],dtype=torch.long).cuda(cfg.use_which_gpu)
    predict_labels = []
    label_dict = label_vocab.stoi
    new_label_dict = {value:key for key, value in label_dict.items()}
    with torch.no_grad():
        for i in tqdm(range(samples.shape[0]), total=(len(samples)), desc="Validation"):
            # tokens: 1 * length of sentence
            # label_list: 1 * length
            tokens = torch.tensor(samples[i,:][masks[i]==1], dtype=torch.long).unsqueeze(0).cuda(cfg.use_which_gpu)
            label_list = torch.tensor(labels[i,:][masks[i]==1], dtype=torch.long).unsqueeze(0).cuda(cfg.use_which_gpu)
            cur_gold_predicate = torch.tensor(gold_predicate[i, :][masks[i]==1], dtype=torch.float32).unsqueeze(0).cuda(cfg.use_which_gpu)
            
            cur_gold_predicate = torch.tensor(cur_gold_predicate, dtype=torch.float32).cuda(cfg.use_which_gpu)
            #tokens: len * 1
            tokens = torch.t(tokens)
            #logit: length * 1 * labels
            logit: torch.Tensor = model(tokens, cur_gold_predicate)

            # argmax predictions
            # predictions: length * 1
            _, predictions = logit.max(dim=2)

            # predictions: length
            predictions.squeeze_(-1)

            # label_list: length
            label_list.squeeze_(0)
            new_prediction = [new_label_dict[each_label_index] for each_label_index in predictions.cpu().numpy().tolist()]
            predict_labels.append(new_prediction)
    return predict_labels

def eval_with_ILP(model, samples, masks, labels, gold_predicate, label_vocab, transition_matrix):
    """
    model: A pytorch module
    samples: dataset samples (n * max_len)
    masks: dataset mask (n * max_len)
    gold_predicate: 0/1 gold predicate(n * max_len)
    labels: dataset labels (n * max_len)
    label_vocab: a torchtext vocab for labels
    """
    binary_matrix = torch.tensor([],dtype=torch.bool).cuda(cfg.use_which_gpu)
    all_preds = torch.tensor([],dtype=torch.long).cuda(cfg.use_which_gpu)
    all_labels = torch.tensor([],dtype=torch.long).cuda(cfg.use_which_gpu)
    prediction_labels = []
    predicts_list = []
    with torch.no_grad():
        for i in tqdm(range(samples.shape[0]), total=(len(samples)), desc="Validation"):
            tokens = torch.tensor(samples[i,:][masks[i]==1], dtype=torch.long).unsqueeze(0).cuda(cfg.use_which_gpu)
            label_list = torch.tensor(labels[i,:][masks[i]==1], dtype=torch.long).unsqueeze(0).cuda(cfg.use_which_gpu)
            cur_masks = torch.tensor(masks[i, :][masks[i]==1], dtype=torch.long).unsqueeze(0).cuda(cfg.use_which_gpu)
            cur_gold_predicate = torch.tensor(gold_predicate[i, :][masks[i]==1], dtype=torch.float32).unsqueeze(0).cuda(cfg.use_which_gpu)

            tokens = torch.tensor(tokens, dtype=torch.long).cuda(cfg.use_which_gpu)
            cur_masks = torch.tensor(cur_masks, dtype=torch.long).cuda(cfg.use_which_gpu)
            cur_gold_predicate = torch.tensor(cur_gold_predicate, dtype=torch.float32).cuda(cfg.use_which_gpu)
            # tokens: len * 1
            tokens = torch.t(tokens)

            #logit: length * 1 * labels
            logit: torch.Tensor = model(tokens, cur_masks, cur_gold_predicate)
            logit = logit.view(logit.shape[0],logit.shape[2])

            predictions, predicates_index = ilp_decoder(logit.cpu().numpy(), label_vocab)
            prediction_labels.append(predictions)
            predicts_list.append(predicates_index)
    
    return prediction_labels, predicts_list

def generate_gold_predicate_0_1_matrix(gold_predicate):
    res = []
    for i in range(gold_predicate.shape[0]):
        tmp = [0.0]*cfg.max_len
        if gold_predicate[i] < cfg.max_len:
            tmp[gold_predicate[i]] = 1.0
        else:
            pass
        res.append(tmp)
    res_np = np.array(res)
    return res_np

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

def generate_batch_new(train_samples_np, train_mask_np, train_labels_np, train_gold_predicate_np, batch_size, shuffle=False):
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
        temp_data_gold_predicate = find_list(batch_indices, train_gold_predicate_np)
        batch_data = (temp_data_vocab, temp_data_mask, temp_data_label, temp_data_gold_predicate)
        yield batch_data
        if batch_size * number >= rows:
            break
        number += 1

def eval_with_xentropy(model, samples, vocab_label):
    with torch.no_grad():
        ls = []
        for example in samples:
            # token: batch * length of sentence
            # label_list: batch * length
            token, mask, label_list, gold_predicate = example
            # token: length * batch
            token = zip(*token)

            #logit: length * batch * dim
            token = torch.tensor(tuple(token)).cuda(cfg.use_which_gpu)
            mask = torch.tensor(tuple(mask)).cuda(cfg.use_which_gpu)
            gold_predicate = torch.tensor(tuple(gold_predicate), dtype=torch.float32).cuda(cfg.use_which_gpu)
            logit = model(token, gold_predicate)
            #label_vec: length * batch * label_length
            label_vec = torch.zeros(token.shape[0], token.shape[1], len(vocab_label)).cuda(cfg.use_which_gpu)

            for batch_num in range(token.shape[1]):
                for each_label in range(len(label_list[batch_num])):
                    label_num = label_list[batch_num][each_label]
                    label_vec[each_label, batch_num, label_num] = 1
                    #label_vec[batch_num, each_label, label_num] = 1
                    

            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label_vec).cuda(cfg.use_which_gpu)
            ls.append(loss.item())

        return mean(ls)

if __name__ == '__main__':
    
    train_set, dev_set, test_set, emb, vocab, labels, transition_matrix = data_preprocesing(cfg.train_loc,
                                                               cfg.dev_loc,
                                                               cfg.test_loc,
                                                               cfg.glove_embedding_loc, 
                                                               cfg.max_len)
   
    save_file_path = 'model-lstm.th'
    model = LSTM_Model(emb, labels).cuda(cfg.use_which_gpu)
    train_samples_np, train_mask_np, train_labels_np, train_predicate_np = train_set
    dev_samples_np, dev_mask_np, dev_labels_np, dev_predicate_np, dev_token_list = dev_set
    test_samples_np, test_mask_np, test_labels_np, test_predicate_np, test_token_list = test_set

    train_predicate_np = generate_gold_predicate_0_1_matrix(train_predicate_np)
    dev_predicate_np = generate_gold_predicate_0_1_matrix(dev_predicate_np)
    test_predicate_01_np = generate_gold_predicate_0_1_matrix(test_predicate_np)

    opt = torch.optim.Adam(model.parameters())
    best_loss = math.inf
    # for epoch in range(cfg.epochs):
    #     print(f'Starting epoch {epoch+1}') 
    #     new_train_sample =  generate_batch_new(train_samples_np, train_mask_np, train_labels_np, train_predicate_np, cfg.batch_size, False)
    #     ls = train(model, opt, new_train_sample,labels)
    #     #validation
    #     validation_samples = generate_batch_new(dev_samples_np, dev_mask_np, dev_labels_np, dev_predicate_np, 32, shuffle=False)
    #     validation_loss = eval_with_xentropy(model, validation_samples, labels)
    #     if validation_loss < best_loss:
    #          torch.save({'model': model.state_dict()}, cfg.model_store_dir + '/' + cfg.model_store_file)

        # Validation
        # f1_score_dev = eval(model, dev_samples_np, dev_mask_np, dev_labels_np, labels)
        # print(f'Epoch {epoch+1} finished, validation F1: {f1_score_dev}')
    #torch.save({'model': model.state_dict()}, save_file_path)

    #load model to test
    checkpoint = torch.load(save_file_path)
    model.load_state_dict(checkpoint['model'])
    predict_label_list = eval_with_baseline(model, test_samples_np, test_mask_np,test_labels_np, test_predicate_01_np, labels)   
    save_predictions(test_token_list, test_predicate_np, predict_label_list, "prediction_for_baseline_test.txt")
    # f1_score = eval(model, test_samples_np, test_mask_np,test_labels_np, test_predicate_01_np, labels)
    # print(f1_score)
    

