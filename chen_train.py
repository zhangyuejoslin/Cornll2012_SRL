import torch
from tqdm import tqdm
from data_helper.data_reader import *
from model.SRL_model import SRL_Model
from config.global_config import CONFIG

train, dev, emb, vocab, labels = data_preprocesing('data/BIO-formatted-sample/train.txt',
                                    'data/BIO-formatted-sample/dev.txt',
                                    '/home/hlr/shared/data/glove6B/glove.6B.50d.txt',20)

print('first train sample data:', train[0][0])
print('first train sample mask:', train[1][0])
print('first train sample label:', train[2][0])
print('first train sample predicate position:', train[3][0])
print('----------------------------------------------------')
print('first dev sample data:', dev[0][0])
print('first dev sample mask:', dev[1][0])
print('first dev sample label:', dev[2][0])
print('first dev sample predicate position:', dev[3][0])
print('----------------------------------------------------')
print('embeding shape:', emb.shape)
print('vocab list length:', len(vocab))
print('label_list:', labels)

#### load config
cfg = CONFIG()

#### load model
model = SRL_Model(emb, len(labels), is_test=False)
print(model)

