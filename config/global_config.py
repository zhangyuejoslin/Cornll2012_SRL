global_config = {
    'max_len':30,
}



class CONFIG(object):
    """docstring for CONFIG"""
    def __init__(self):
        super(CONFIG, self).__init__()
        
        self.bert_srl = '/home/hlr/shared/data/chenzheng/data/gcn/bert-base-srl-2019.06.17.tar.gz'
        self.data_loc = '/home/hlr/shared/data/chenzheng/data/gcn/data'
        self.immediate_store = '/home/hlr/shared/data/chenzheng/data/gcn/immediate_store_back' # the place to store the graph, adj matirx, degree matrix
        self.checkpoint_loc = '/home/hlr/shared/data/chenzheng/data/gcn/checkpoint'
        self.glove_embedding_loc = '/home/hlr/shared/data/glove6B/glove.6B.300d.txt' # glove storing location
        self.glove_word_dim = 300  # glove pretraining embedding dim
        self.learning_rate = 0.1   # Initial learning rate.
        self.epochs  = 200  # Number of epochs to train.
        self.num_lstm_layers = 3 # number of layers for lstm
        self.lstm_hidden_dim = 100 # lstm hidden dimension
        self.hidden1 = 200  # Number of units in hidden layer 1.
        self.dropout = 0.5  # Dropout rate (1 - keep probability).
        self.weight_decay = 0.   # Weight for L2 loss on embedding matrix.
        self.early_stopping = 20 # Tolerance for early stopping (# of epochs).
        self.batch_size = 1000
        self.gpu_node = 1 # which gpu you want to use;
        self.max_len = 30



