from collections import Counter
import random
import numpy as np
from torchtext.vocab import Vocab
'''
sample example:
5 We respectfully invite you to watch a special edition of Across China . ||| O O O B-ARG0 O B-V B-ARG1 I-ARG1 I-ARG1 I-ARG1 I-ARG1 I-ARG1 O
where 5 is the index of predicate in this sample.
'''
def read_raw_sentences(file_path):
    word_label_pair_sentences = []
    label_counter = Counter()
    f = open(file_path, 'r')
    for line in f:
        line = line.strip().split('|||')
        words_tmp = line[0].strip().split()
        words = [word.lower() for word in words_tmp]
        labels = line[1].strip().split()
        for label in labels:
            label_counter[label] += 1
        if len(line) <= 1: ### if the sample only has words but not has labels
            labels = ['O' for i in words[1:]]
        predicate = int(words[0])
        word_label_pair_sentences.append([words[1:], labels, predicate]) ## x, y, index of predicate
    f.close()
    return word_label_pair_sentences, Vocab(label_counter)

# ## test case for read_raw_sentences: pass
# res, label_dict= read_raw_sentences('../data/BIO-formatted-sample/train.txt')
# print(res)
# print(label_dict)


### use glove embedding
def load_embeddings(file_path):
    embeddings_res = []
    token_counter = Counter()
    f = open(file_path, 'r', encoding='UTF-8')
    for line in f:
        line = line.strip().split()
        embeddings_res.append([float(emb) for emb in line[1:]])
        token_counter[line[0]] += 1
    embeddings_res.insert(0, [random.gauss(0, 0.01) for _ in range(len(embeddings_res[0]))])
    f.close()

    return embeddings_res, Vocab(token_counter)

### test case for read_raw_sentences: pass
# embeddings_res, word_to_id = load_embeddings('/home/hlr/shared/glove6B/glove.6B.50d.txt')
# print(len(embeddings_res), len(word_to_id))

def sequence_to_id(sequence, vocab):
    return [vocab.stoi[elem] for elem in sequence]
    
# def label_2_id(sequence, label_dict):
#     ids = []
#     for word in sequence:
#         ids.append(label_dict[word])
#     return ids

### final outputs are numpy formats
def data_preprocesing(train_file, dev_file, embed_file, max_len):
    train_pair, label_vocab = read_raw_sentences(train_file)
    # dev_pair, dev_label_set = read_raw_sentences(dev_file)
    # label_set = list(set(train_label_set + dev_label_set))
    dev_pair, _ = read_raw_sentences(dev_file)
    embed, token_vocab = load_embeddings(embed_file)

    # training data token to index
    train_samples = [sequence_to_id(sent[0], token_vocab) for sent in train_pair]
    train_mask = [len(sent[0]) for sent in train_pair]
    train_labels = [sequence_to_id(sent[1], label_vocab) for sent in train_pair]
    train_predicate = [sent[2] for sent in train_pair]

    # training data token to index
    dev_samples = [sequence_to_id(sent[0], token_vocab) for sent in dev_pair]
    dev_mask = [len(sent[0]) for sent in dev_pair]
    dev_labels = [sequence_to_id(sent[1], label_vocab) for sent in dev_pair]
    dev_predicate = [sent[2] for sent in dev_pair]

    ### mask the training sample length
    for i in range(len(train_samples)):
        if len(train_samples[i]) < max_len:
            # train_samples[i] = train_samples[i] + [0]*(max_len - len(train_samples[i]))
            train_samples[i] = train_samples[i] + [0]*(max_len - len(train_samples[i]))
            train_mask[i] = [1]*train_mask[i] + [0]*(max_len - train_mask[i])
            train_labels[i] = train_labels[i] + ['O']*(max_len - len(train_samples[i]))
        else:
            train_samples[i] = train_samples[i][0:max_len]
            train_mask[i] = [1]*max_len
            train_labels[i] = train_labels[i]

    ### mask the dev sample length
    for i in range(len(dev_samples)):
        if len(dev_samples[i]) < max_len:
            # dev_samples[i] = dev_samples[i] + [0]*(max_len - len(dev_samples[i]))
            dev_samples[i] = dev_samples[i] + [0]*(max_len - len(dev_samples[i]))
            dev_mask[i] = [1]*dev_mask[i] + [0]*(max_len - dev_mask[i])
            dev_labels[i] = dev_labels[i] + ['O']*(max_len - len(dev_samples[i]))
        else:
            dev_samples[i] = dev_samples[i][0:max_len]
            dev_mask[i] = [1]*max_len
            dev_labels[i] = dev_labels[i]
    
    ### transfer list to numpy
    train_samples_np = np.array(train_samples)
    train_mask_np = np.array(train_mask)
    train_labels_np = np.array(train_labels)
    train_predicate_np = np.array(train_predicate)
    dev_samples_np = np.array(dev_samples)
    dev_mask_np = np.array(dev_mask)
    dev_labels_np = np.array(dev_labels)
    dev_predicate_np = np.array(dev_predicate)
    emb_np = np.array(embed)

    # Convert dicts back into lists for compatibility
  
    return (train_samples_np, train_mask_np, train_labels_np, train_predicate_np),\
            (dev_samples_np, dev_mask_np, dev_labels_np, dev_predicate_np), emb_np, token_vocab, label_vocab

### test data_preprocesing
#### sample data
# train, dev, emb, vocab, labels = data_preprocesing('../data/BIO-formatted-sample/train.txt',
#                                     '../data/BIO-formatted-sample/dev.txt',
#                                     '/home/hlr/shared/data/glove6B/glove.6B.50d.txt',20)

#### full data
# train, dev, emb, vocab, labels = data_preprocesing('../data/BIO-formatted/conll2012.train.txt',
#                                     '../data/BIO-formatted/conll2012.devel.txt',
#                                     '/home/hlr/shared/data/glove6B/glove.6B.50d.txt',20)
# print('first train sample data:', train[0][0])
# print('first train sample mask:', train[1][0])
# print('first train sample label:', train[2][0])
# print('first train sample predicate position:', train[3][0])
# print('----------------------------------------------------')
# print('first dev sample data:', dev[0][0])
# print('first dev sample mask:', dev[1][0])
# print('first dev sample label:', dev[2][0])
# print('first dev sample predicate position:', dev[3][0])
# print('----------------------------------------------------')
# print('embeding shape:', emb.shape)
# print('vocab list length:', len(vocab))
# print('label_list:', labels)


    








