from sklearn import metrics
from statistics import mean
import torch
from tqdm import tqdm
from data_helper.data_reader import *
from model.SRL_model import SRL_Model
from model.Decoder import viterbi_decode, ilp_decoder
from data_helper.prediction_builder import save_predictions
from config.global_config import CONFIG
import os
import random
import math

import wandb

#### load config
cfg = CONFIG()

def train(model, opt, new_train_sample, vocab_label, total_num_trian_sample):
    ls = []
    for example in tqdm(new_train_sample, total=(total_num_trian_sample//cfg.batch_size), desc="Training"):
        # token: batch * length of sentence
        # label_list: batch * length
        token, mask, label_list, gold_predicate = example
         # token: length * batch
        token = zip(*token)

        opt.zero_grad()
        #logit: length * batch * dim
        token = torch.tensor(tuple(token)).cuda(cfg.use_which_gpu)
        mask = torch.tensor(tuple(mask)).cuda(cfg.use_which_gpu)
        gold_predicate = torch.tensor(tuple(gold_predicate), dtype=torch.float32).cuda(cfg.use_which_gpu)
        logit = model(token, mask, gold_predicate)
        #label_vec: length * batch * label_length
        label_vec = torch.zeros(token.shape[0], token.shape[1], len(vocab_label)).cuda(cfg.use_which_gpu)

        for batch_num in range(token.shape[1]):
            for each_label in range(len(label_list[batch_num])):
                label_num = label_list[batch_num][each_label]
                label_vec[each_label, batch_num, label_num] = 1
                # label_vec[batch_num, each_label, label_num] = 1
                

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label_vec).cuda(cfg.use_which_gpu)
        loss.backward()
        opt.step()
        ls.append(loss.item())
    return ls

def call_viterbi(logit, transition_matrix, label_dict):
    new_label_dict = {value:key for key, value in label_dict.items()}
    viterbi_paths, viterbi_scores = viterbi_decode(logit.view(-1, logit.size()[-1]), transition_matrix, cfg.beam_search_top_k)
    predictions = viterbi_paths[0] ## choose top 1
    new_prediction = []
    for each_lablel_index in predictions:
        new_prediction.append(new_label_dict[each_lablel_index])
    try:
        return new_prediction, predictions.index(5)
    except ValueError:
        return new_prediction, 0

    # viterbi_paths, viterbi_scores = viterbi_decode(logit.view(-1, logit.size()[-1]), transition_matrix, cfg.beam_search_top_k)
    # predictions = viterbi_paths[0] ## choose top 1
    # return predictions

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
            logit = model(token, mask, gold_predicate)
            #label_vec: length * batch * label_length
            label_vec = torch.zeros(token.shape[0], token.shape[1], len(vocab_label)).cuda(cfg.use_which_gpu)

            for batch_num in range(token.shape[1]):
                for each_label in range(len(label_list[batch_num])):
                    label_num = label_list[batch_num][each_label]
                    label_vec[each_label, batch_num, label_num] = 1
                    # label_vec[batch_num, each_label, label_num] = 1
                    

            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label_vec).cuda(cfg.use_which_gpu)
            ls.append(loss.item())

        return mean(ls)

def eval_with_micro_F1(model, samples, masks, labels, gold_predicate, label_vocab, transition_matrix):
    """
    model: A pytorch module
    samples: dataset samples (n * max_len)
    masks: dataset mask (n * max_len)
    gold_predicate: 0/1 gold predicate(n * max_len)
    labels: dataset labels (n * max_len)
    label_vocab: a torchtext vocab for labels
    """
    all_preds = torch.tensor([],dtype=torch.long).cuda(cfg.use_which_gpu)
    all_labels = torch.tensor([],dtype=torch.long).cuda(cfg.use_which_gpu)
    prediction_labels = []
    predicts_list = []
    with torch.no_grad():
        # for i in tqdm(range(0, samples.shape[0], cfg.batch_size), total=(samples.shape[0]//cfg.batch_size), desc="Validation"):
        for i in tqdm(range(samples.shape[0]), total=(len(samples)), desc="Validation"):
        # for i in range(samples.shape[0]):
            # tokens: 1 * length of sentence
            # label_list: 1 * length

            # tokens = torch.tensor(samples[i: i+cfg.batch_size,:][masks[i: i+cfg.batch_size]==1], dtype=torch.long).unsqueeze(0).cuda()
            # label_list = torch.tensor(labels[i: i+cfg.batch_size,:][masks[i: i+cfg.batch_size]==1], dtype=torch.long).unsqueeze(0).cuda()
            # cur_masks = torch.tensor(masks[i: i+cfg.batch_size,:][masks[i: i+cfg.batch_size]==1], dtype=torch.long).unsqueeze(0).cuda()
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

            # argmax predictions
            # predictions: length * 1
            # _, predictions = logit.max(dim=2)
            # _, predictions_drew = logit.max(dim=2)
            predictions = call_viterbi(logit, transition_matrix)
            predictions = torch.from_numpy(np.array(predictions)).cuda(cfg.use_which_gpu)
            # print(predictions)
            # print(predictions_drew)
            # import sys
            # sys.exit()

            # predictions: length
            predictions.squeeze_()
            # label_list: length
            label_list = torch.tensor(label_list, dtype=torch.long).squeeze().cuda(cfg.use_which_gpu)
            try:
                all_preds = torch.cat((all_preds, predictions))
                all_labels = torch.cat((all_labels, label_list))
            except:
                pass


    
    return metrics.f1_score(y_true=all_labels.cpu(), y_pred=all_preds.cpu(), average='micro')

def eval_with_viterbi(model, samples, masks, labels, gold_predicate, label_vocab, transition_matrix):
    """
    model: A pytorch module
    samples: dataset samples (n * max_len)
    masks: dataset mask (n * max_len)
    gold_predicate: 0/1 gold predicate(n * max_len)
    labels: dataset labels (n * max_len)
    label_vocab: a torchtext vocab for labels
    """
    all_preds = torch.tensor([],dtype=torch.long).cuda(cfg.use_which_gpu)
    all_labels = torch.tensor([],dtype=torch.long).cuda(cfg.use_which_gpu)
    prediction_labels = []
    predicts_list = []
    with torch.no_grad():
        # for i in tqdm(range(0, samples.shape[0], cfg.batch_size), total=(samples.shape[0]//cfg.batch_size), desc="Validation"):
        for i in tqdm(range(samples.shape[0]), total=(len(samples)), desc="Validation"):
        # for i in range(samples.shape[0]):
            # tokens: 1 * length of sentence
            # label_list: 1 * length

            # tokens = torch.tensor(samples[i: i+cfg.batch_size,:][masks[i: i+cfg.batch_size]==1], dtype=torch.long).unsqueeze(0).cuda()
            # label_list = torch.tensor(labels[i: i+cfg.batch_size,:][masks[i: i+cfg.batch_size]==1], dtype=torch.long).unsqueeze(0).cuda()
            # cur_masks = torch.tensor(masks[i: i+cfg.batch_size,:][masks[i: i+cfg.batch_size]==1], dtype=torch.long).unsqueeze(0).cuda()
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

            # argmax predictions
            # predictions: length * 1
            # _, predictions = logit.max(dim=2)
            # _, predictions_drew = logit.max(dim=2)
            predictions, predicates_index = call_viterbi(logit, transition_matrix, label_vocab.stoi)
            prediction_labels.append(predictions)
            predicts_list.append(predicates_index)
            
    return prediction_labels, predicts_list



def find_list(indices, data):
    out = []
    for i in indices:
        out = out +[data[i][:cfg.max_len]]
    return out

def generate_batch(train_samples_np, train_mask_np, train_labels_np, train_gold_predicate_np, batch_size, shuffle=False):
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

if __name__ == '__main__':

    wandb.init()
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
    wandb.watch(model)
    train_samples_np, train_mask_np, train_labels_np, train_predicate_np = train_set
    dev_samples_np, dev_mask_np, dev_labels_np, dev_predicate_np, dev_token_list = dev_set
    test_samples_np, test_mask_np, test_labels_np, test_predicate_np, test_token_list = test_set
    
    train_predicate_np = generate_gold_predicate_0_1_matrix(train_predicate_np)
    dev_predicate_np = generate_gold_predicate_0_1_matrix(dev_predicate_np)
    test_predicate_01_np = generate_gold_predicate_0_1_matrix(test_predicate_np)


    opt = torch.optim.Adam(model.parameters())
    best_loss = math.inf
    stop_counter = 0
    for epoch in range(cfg.epochs):
        print(f'Starting epoch {epoch+1}') 
        new_train_sample =  generate_batch(train_samples_np, train_mask_np, train_labels_np, train_predicate_np, cfg.batch_size, False)
        ls = train(model, opt, new_train_sample, labels, num_train_set)
        # Validation
        validation_samples = generate_batch(dev_samples_np, dev_mask_np, dev_labels_np, dev_predicate_np, 32, shuffle=False)
        validation_loss = eval_with_xentropy(model, validation_samples, labels)
        # f1_score = eval(model, dev_samples_np, dev_mask_np, dev_labels_np, labels, transition_matrix)

        wandb.log({'Train Loss': mean(ls), 'Val Loss': validation_loss})
        # print(f'Epoch {epoch+1} finished, validation F1: {f1_score}, avg loss: {mean(ls)}')
        print(f'Epoch {epoch+1} finished, avg loss: {mean(ls)}')
        if validation_loss < best_loss:
            torch.save({'model': model.state_dict()}, cfg.model_store_dir + '/' + cfg.model_store_file)
            stop_counter = 0
        else:
            stop_counter = stop_counter + 1

        if stop_counter >= cfg.early_stopping:
            print(f'Performance hasn\'t improved for {stop_counter} epochs, stopping')
            break
    

    checkpoint = torch.load(cfg.model_store_dir + '/' + cfg.model_store_file)
    model.load_state_dict(checkpoint['model'])

    ### test
    # f1_score_val = eval_with_micro_F1(model.eval(), dev_samples_np, dev_mask_np, dev_labels_np, dev_predicate_np, labels, transition_matrix)
    print(f"Outputting test predictions with best weights to {cfg.predictions_file}")
    predict_label_list, preidcts_list = eval_with_viterbi(model.eval(), test_samples_np, test_mask_np, test_labels_np, test_predicate_01_np, labels, transition_matrix)
    save_predictions(test_token_list, test_predicate_np, predict_label_list, cfg.predictions_file)
    #print(f'Val F1: {f1_score_val}, Test F1: {f1_score_test}')
    #print(f'Test F1: {f1_score_test}')

