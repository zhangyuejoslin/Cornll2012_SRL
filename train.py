from sklearn import metrics
from statistics import mean
import torch
import random
from tqdm import tqdm
from data_helper.data_reader import data_preprocesing
from model.LSTM_baseline import LSTM_Model
import random

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

    train_set, dev_set, test_set, emb, vocab, labels = data_preprocesing('data/BIO-formatted/conll2012.train.txt',
                                                                           'data/BIO-formatted/conll2012.devel.txt',
                                                                           'data/BIO-formatted/conll2012.test.txt',
                                                                           '/home/hlr/shared/data/glove6B/glove.6B.50d.txt', 20)
    save_file_path = 'model-lstm.th'
    model = LSTM_Model(emb, labels).cuda()
    train_samples_np, train_mask_np, train_labels_np, train_predicate_np = train_set
    dev_samples_np, dev_mask_np, dev_labels_np, dev_predicate_np = dev_set
    test_samples_np, test_mask_np, test_labels_np, test_predicate_np = test_set
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(50):
        print(f'Starting epoch {epoch+1}') 
        new_train_sample =  generate_batch(train_samples_np, train_labels_np, 50, False)
        ls = train(model, opt, new_train_sample,labels)

        # Validation
        f1_score_dev = eval(model, dev_samples_np, dev_mask_np, dev_labels_np, labels)
        print(f'Epoch {epoch+1} finished, validation F1: {f1_score_dev}')
    torch.save({'model': model.state_dict()}, save_file_path)

    # load model to test
    # checkpoint = torch.load(save_file_path)
    # model.load_state_dict(checkpoint['model'])
    # f1_score = eval(model, test_samples_np, test_mask_np,test_labels_np, labels)
    # print(f1_score)

