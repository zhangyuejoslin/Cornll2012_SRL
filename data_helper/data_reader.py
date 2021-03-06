from collections import Counter
import random
import numpy as np
import torch
from torchtext.vocab import Vocab, Vectors
'''
sample example:
5 We respectfully invite you to watch a special edition of Across China . ||| O O O B-ARG0 O B-V B-ARG1 I-ARG1 I-ARG1 I-ARG1 I-ARG1 I-ARG1 O
where 5 is the index of predicate in this sample.
'''
def read_raw_sentences(file_path, counter=None):
    word_label_pair_sentences = []
    label_counter = counter if counter is not None else Counter()
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
    return word_label_pair_sentences, label_counter

# ## test case for read_raw_sentences: pass
# res, label_dict= read_raw_sentences('../data/BIO-formatted-sample/train.txt')
# print(res)
# print(label_dict)


### use glove embedding
def load_embeddings(file_path):
    token_counter = Counter()
    f = open(file_path, 'r', encoding='UTF-8')
    for line in f:
        line = line.strip().split()
        token_counter[line[0]] += 1
    f.close()

    token_vocab = Vocab(token_counter)
    glove_vectors = Vectors(file_path)
    token_vocab.load_vectors(glove_vectors)

    return token_vocab.vectors, token_vocab

### test case for read_raw_sentences: pass
# embeddings_res, word_to_id = load_embeddings('/home/hlr/shared/glove6B/glove.6B.50d.txt')
# print(len(embeddings_res), len(word_to_id))

def sequence_to_id(sequence, vocab):
    return [vocab.stoi[elem] for elem in sequence]

### final outputs are numpy formats
def data_preprocesing(train_file, dev_file, test_file, embed_file, max_len):
    print('Reading training data...')
    train_pair, label_counter = read_raw_sentences(train_file)
    print(f'Loaded {len(train_pair)} examples from {train_file}')
    # dev_pair, dev_label_set = read_raw_sentences(dev_file)
    # label_set = list(set(train_label_set + dev_label_set))
    print('Reading dev data...')
    dev_pair, label_counter = read_raw_sentences(dev_file, counter=label_counter)
    test_pair, label_counter = read_raw_sentences(test_file, counter=label_counter)
    dev_token  = [each_dev_pair[0] for each_dev_pair in dev_pair]
    test_token  = [each_test_pair[0] for each_test_pair in test_pair]

    print(f'Loaded {len(dev_pair)} examples from {dev_file}')
    print('Loading glove vectors...')
    embed, token_vocab = load_embeddings(embed_file)
    label_vocab = Vocab(label_counter)

    print('Preprocessing data...')
    # training data token to index
    train_samples = [sequence_to_id(sent[0], token_vocab) for sent in train_pair]
    train_labels = [sequence_to_id(sent[1], label_vocab) for sent in train_pair]
    train_predicate = [sent[2] for sent in train_pair]

    # dev data token to index
    dev_samples = [sequence_to_id(sent[0], token_vocab) for sent in dev_pair]
    dev_mask = [len(sent[0]) for sent in dev_pair]
    dev_labels = [sequence_to_id(sent[1], label_vocab) for sent in dev_pair]
    dev_predicate = [sent[2] for sent in dev_pair]

    # test data tok-> inx
    test_samples = [sequence_to_id(sent[0], token_vocab) for sent in test_pair]
    test_mask = [len(sent[0]) for sent in test_pair]
    test_labels = [sequence_to_id(sent[1], label_vocab) for sent in test_pair]
    test_predicate = [sent[2] for sent in test_pair]

    # Build binary bigram transition matrix
    constraint_mat = torch.zeros((len(label_vocab), len(label_vocab)))

    def update_constraint_mat(labels, matrix):
        for label in labels:
            last_label = None
            for label_id in label:
                if last_label is not None:
                    matrix[last_label, label_id] = 1
                last_label = label_id
        return matrix

    constraint_mat = update_constraint_mat(train_labels, constraint_mat)
    constraint_mat = update_constraint_mat(dev_labels, constraint_mat)
    constraint_mat = update_constraint_mat(test_labels, constraint_mat)


    def mask_samples(samples, labels):
        """
        Takes a list of samples and pads/truncates them to max_len, returning the mask as well
        samples: n x seq_len (list)
        labels: n x seq_len (list)
        """
        mask = [len(sent) for sent in samples]
        ### mask the training sample length
        for i in range(len(samples)):
            if len(samples[i]) < max_len:
                # samples[i] = samples[i] + [0]*(max_len - len(samples[i]))
                samples[i] = samples[i] + [token_vocab.stoi['<pad>']]*(max_len - len(samples[i]))
                mask[i] = [1]*mask[i] + [0]*(max_len - mask[i])
                labels[i] = labels[i] + [label_vocab.stoi['O']]*(max_len - len(labels[i]))
            else:
                samples[i] = samples[i][0:max_len]
                mask[i] = [1]*max_len
                labels[i] = labels[i][0:max_len]
        return samples, mask, labels

    train_samples, train_mask, train_labels = mask_samples(train_samples, train_labels)
    dev_samples, dev_mask, dev_labels = mask_samples(dev_samples, dev_labels)
    test_samples, test_mask, test_labels = mask_samples(test_samples, test_labels)
    
    ### transfer list to numpy
    train_samples_np = np.array(train_samples)
    train_mask_np = np.array(train_mask)
    train_labels_np = np.array(train_labels)
    train_predicate_np = np.array(train_predicate)
    dev_samples_np = np.array(dev_samples)
    dev_mask_np = np.array(dev_mask)
    dev_labels_np = np.array(dev_labels)
    dev_predicate_np = np.array(dev_predicate)
    test_samples_np = np.array(test_samples)
    test_mask_np = np.array(test_mask)
    test_labels_np = np.array(test_labels)
    test_predicate_np = np.array(test_predicate)
    emb_np = np.array(embed)
    print('Done preprocessing')

    print('Sample Data')
    print(f'Sample')
    print(f'\tshape: {train_samples_np.shape}')
    print(f'\tex: {train_samples_np[0]}')
    print('\t'+' '.join([token_vocab.itos[i] for i in train_samples_np[0]]))
    print(f'Label')
    print(f'\tshape: {train_labels_np.shape}')
    print(f'\tex: {train_labels_np[0]}')
    print('\t'+' '.join([label_vocab.itos[i] for i in train_labels_np[0]]))
    print('Predicate')
    print(f'\tshape: {train_predicate_np.shape}')
    print(f'\tex: {train_predicate_np[0]}')

  
    return (train_samples_np, train_mask_np, train_labels_np, train_predicate_np),\
            (dev_samples_np, dev_mask_np, dev_labels_np, dev_predicate_np, dev_token),\
            (test_samples_np, test_mask_np, test_labels_np, test_predicate_np, test_token), emb_np, token_vocab, label_vocab, constraint_mat

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









