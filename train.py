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

cfg = CONFIG()

def train(model, opt, new_train_sample, vocab_label):
    ls = []
    for example in tqdm(new_train_sample, total=(82510/50), desc="Training"):
        # token: batch * length of sentence
        # label_list: batch * length
        token, label_list = example
        # token: length * batch
        token = zip(*token)

        opt.zero_grad()
        token = torch.tensor(tuple(token)).cuda()
        #logit: length * batch * dim
        logit = model(token)
        #label_vec: length * batch * label_length
        label_vec = torch.zeros(token.shape[0], token.shape[1], len(vocab_label)).cuda()

        for batch_num in range(token.shape[1]):
            for each_label in range(len(label_list[batch_num])):
                label_num = label_list[batch_num][each_label]
                label_vec[each_label, batch_num, label_num] = 1

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label_vec).cuda()
        loss.backward()
        opt.step()
        ls.append(loss.item())
    return ls

def eval(model, samples, masks, labels, label_vocab):
    """
    model: A pytorch module
    samples: dataset samples (n * max_len)
    masks: dataset mask (n * max_len)
    labels: dataset labels (n * max_len)
    label_vocab: a torchtext vocab for labels
    """
    all_preds = torch.tensor([],dtype=torch.long).cuda()
    all_labels = torch.tensor([],dtype=torch.long).cuda()
    with torch.no_grad():
        for i in tqdm(range(samples.shape[0]), total=(len(samples)), desc="Validation"):
            # tokens: 1 * length of sentence
            # label_list: 1 * length
            tokens = torch.tensor(samples[i,:][masks[i]==1], dtype=torch.long).unsqueeze(0).cuda()
            label_list = torch.tensor(labels[i,:][masks[i]==1], dtype=torch.long).unsqueeze(0).cuda()

            # tokens: len * 1
            tokens = torch.t(tokens)
            #logit: length * 1 * labels
            logit: torch.Tensor = model(tokens)

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

if __name__ == '__main__':
    
    train_set, dev_set, test_set, emb, vocab, labels, transition_matrix = data_preprocesing(cfg.train_loc,
                                                               cfg.dev_loc,
                                                               cfg.test_loc,
                                                               cfg.glove_embedding_loc, 
                                                               cfg.max_len)
   

    num_train_set = train_set[0].shape[0]
    # num_dev_set = dev_set[0].shape[0]
    transition_matrix = transition_matrix.cuda(cfg.use_which_gpu)  ## [num_labels, num_labels]

    if not os.path.exists(cfg.model_store_dir):
        os.makedirs(cfg.model_store_dir)
    model = SRL_Model(emb, labels, is_test=False).train().cuda(cfg.use_which_gpu)
    train_samples_np, train_mask_np, train_labels_np, train_predicate_np = train_set
    dev_samples_np, dev_mask_np, dev_labels_np, dev_predicate_np, dev_token_list = dev_set
    test_samples_np, test_mask_np, test_labels_np, test_predicate_np, test_token_list = test_set

    train_predicate_np = generate_gold_predicate_0_1_matrix(train_predicate_np)
    dev_predicate_np = generate_gold_predicate_0_1_matrix(dev_predicate_np)
    test_predicate_np = generate_gold_predicate_0_1_matrix(test_predicate_np)

    opt = torch.optim.Adam(model.parameters())
    # for epoch in range(cfg.epochs):
    #     print(f'Starting epoch {epoch+1}') 
    #     new_train_sample =  generate_batch(train_samples_np, train_mask_np, train_labels_np, train_predicate_np, cfg.batch_size, False)
    #     ls = train(model, opt, new_train_sample, labels, num_train_set)
    #     # Validation
    #     # f1_score = eval(model, dev_samples_np, dev_mask_np, dev_labels_np, labels, transition_matrix)

    #     # print(f'Epoch {epoch+1} finished, validation F1: {f1_score}, avg loss: {mean(ls)}')
    #     print(f'Epoch {epoch+1} finished, avg loss: {mean(ls)}')
    # torch.save({'model': model.state_dict()}, cfg.model_store_dir + '/' + cfg.model_store_file)
    

    checkpoint = torch.load(cfg.model_store_dir + '/' + cfg.model_store_file)
    model.load_state_dict(checkpoint['model'])
    #f1_score = eval(model, test_samples_np, test_mask_np,test_labels_np, labels)
    # print(f1_score)

    ### test
   # f1_score_val = eval_with_ILP(model.eval(), dev_samples_np, dev_mask_np, dev_labels_np, dev_predicate_np, labels, transition_matrix)
    predict_label_list, preidcts_list = eval_with_ILP(model.eval(), test_samples_np, test_mask_np, test_labels_np, test_predicate_np, labels, transition_matrix)
    save_predictions(test_token_list, preidcts_list, predict_label_list, "prediction_for_joslin_test.txt")

    #print(f'Val F1: {f1_score_val}, Test F1: {f1_score_test}')
    #print(f'Test F1: {f1_score_test}')





    # train_set, dev_set, test_set, emb, vocab, labels, transition_matrix = data_preprocesing('data/BIO-formatted/conll2012.train.txt',
    #                                                                        'data/BIO-formatted/conll2012.devel.txt',
    #                                                                        'data/BIO-formatted/conll2012.test.txt',
    #                                                                        '/home/hlr/shared/data/glove6B/glove.6B.50d.txt', 20)
    # save_file_path = 'model-lstm.th'
    # model = LSTM_Model(emb, labels).cuda()
    # train_samples_np, train_mask_np, train_labels_np, train_predicate_np = train_set
    # dev_samples_np, dev_mask_np, dev_labels_np, dev_predicate_np = dev_set
    # test_samples_np, test_mask_np, test_labels_np, test_predicate_np = test_set
    # opt = torch.optim.Adam(model.parameters())
    # for epoch in range(50):
    #     print(f'Starting epoch {epoch+1}') 
    #     new_train_sample =  generate_batch(train_samples_np, train_labels_np, 50, False)
    #     ls = train(model, opt, new_train_sample,labels)

    #     # Validation
    #     f1_score_dev = eval(model, dev_samples_np, dev_mask_np, dev_labels_np, labels)
    #     print(f'Epoch {epoch+1} finished, validation F1: {f1_score_dev}')
    # torch.save({'model': model.state_dict()}, save_file_path)

    # load model to test
    # checkpoint = torch.load(save_file_path)
    # model.load_state_dict(checkpoint['model'])
    # f1_score = eval(model, test_samples_np, test_mask_np,test_labels_np, labels)
    # print(f1_score)
    

