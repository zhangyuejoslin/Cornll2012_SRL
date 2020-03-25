# global_config = {
#     'max_len':30,
# }



class CONFIG(object):
    """docstring for CONFIG"""
    def __init__(self):
        super(CONFIG, self).__init__()
        self.use_which_gpu = 5 ## GPU device id, like 0, 1, 2,...
        self.train_loc ='data/BIO-formatted/conll2012.train.txt'
        self.dev_loc =  'data/BIO-formatted/conll2012.devel.txt'
        self.test_loc = 'data/BIO-formatted/conll2012.test.txt'
        # self.glove_embedding_loc = '/home/hlr/shared/data/glove6B/glove.6B.50d.txt' # glove storing location
        self.glove_embedding_loc = '/tank/space/chen_zheng/data/glove6B/glove.6B.50d.txt' # glove storing location
        self.glove_word_dim = 50  # glove pretraining embedding dim
        self.learning_rate = 0.1   # Initial learning rate.
        self.epochs  = 10  # Number of epochs to train.
        self.num_lstm_layers = 2 # number of layers for lstm
        self.lstm_hidden_dim = 50 # lstm hidden dimension
        # self.hidden1 = 200  # Number of units in hidden layer 1.
        self.dropout = 0.1  # Dropout rate (1 - keep probability).
        self.weight_decay = 0.   # Weight for L2 loss on embedding matrix.
        self.early_stopping = 20 # Tolerance for early stopping (# of epochs).
        self.batch_size = 512
        self.gpu_node = 1 # which gpu you want to use;
        self.max_len = 20
        self.beam_search_top_k = 5
        self.model_store_dir = 'checkpoint'
        self.model_store_file = '4.pt'



