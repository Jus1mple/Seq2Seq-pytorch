# -*- coding:utf-8 -*-
# all configs
# author: Matthew


from enum import Enum
from vocab import myVocab, END, START, PAD




PASSWORD_DATASET_PATH = "./data"
MODEL_SAVE_PATH = "./model"
RNN_MODEL_SAVE_PATH = MODEL_SAVE_PATH + "/" + "rnn"
SEQ2SEQ_MODEL_SAVE_PATH = MODEL_SAVE_PATH + "/" + "seq2seq"


def get_printable():
    """get ASCII 95 printable codes"""
    import string
    return string.digits + string.punctuation + string.ascii_letters


printable_list = get_printable()


class RNN_MODEL_NAME(Enum):
    LSTM = 0
    GRU = 1


class CONFIG:
    def __init__(self):
        self.name = "CONFIG"


class DataLoader_CONFIG(CONFIG):
    def __init__(self) -> None:
        self.name = "DataLoader"
        self.filename = "train"
        self.batch_size = 10
        self.num_steps = -1
        self.use_random_iter = False
        self.max_tokens = -1
        self.printable_list = list(get_printable()) # printable 94 ascii characters
        self.padding_value = PAD


class RNN_CONFIG(CONFIG):
    def __init__(self, 
                vocab : myVocab, 
                bidirectional = False, 
                num_hiddens = 256, 
                num_layers = 3, 
                embedding = False, 
                embedding_dim = 20, 
                dropout = 0.5, 
                num_epoches = 5, 
                device = "GPU", 
                lr = 0.005, 
                train_name = "train"
            ) -> None:
        self.name = RNN_MODEL_NAME.LSTM # default LSTM Model
        self.bidirectional = bidirectional # default not use bidirectional model
        self.vocab_size = len(vocab) # default vocab's size
        self.num_inputs = self.vocab_size
        self.num_hiddens = num_hiddens # default the hidden layer cells' number is 256
        self.num_layers = num_layers # default use 3 layers RNN
        self.embedding = embedding # default(`True`) use embedding layer
        self.embedding_dim = embedding_dim # default embedding_dim is 20
        self.vocab = vocab # default vocab
        self.dropout = dropout # default dropout == 0.0
        # trainer config
        # self.train_name = "train.txt"
        self.num_epoches = num_epoches
        self.device = device
        self.lr = lr
        self.train_name = train_name
        self.svaing = True
        
    def update_vocab(self, new_vocab : myVocab):
        self.vocab = new_vocab
        self.vocab_size = len(new_vocab)
        self.num_inputs = len(new_vocab)


class RNN_GUESS_CONFIG(CONFIG):
    def __init__(self, vocab : myVocab) -> None:
        self.guessnumber = 1000000 # generated number
        self.vocab = vocab
        self.endsymbol = vocab[END] # end symbol
        self.bos = vocab[START]
        self.threshold = 1e-15 # the threshold of generated password's probability 
        self.max_length = 31 # generated password's maximum length
        self.min_length = 6 # generated password's minimum length
        self.pad = vocab[PAD]
        self.device = "GPU"
        self.guesser_name = "RNN_Guess"


class Encoder_CONFIG(CONFIG):
    def __init__(self, 
                vocab : myVocab, 
                embedding_size = 45, 
                num_hiddens = 256, 
                num_layers = 4, 
                dropout = 0.5, 
                bidirectional = False
            ) -> None:
        self.vocab_size = len(vocab)
        self.embedding_size = embedding_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional


class Decoder_CONFIG(CONFIG):
    def __init__(self, 
                vocab:myVocab, 
                embedding_size = 45, 
                num_hiddens = 256, 
                num_layers = 4, 
                dropout = 0.5,
                bidirectional = False,
            ) -> None:
        self.vocab_size = len(vocab)
        self.embedding_size = embedding_size
        self.output_size = len(vocab)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional



class Seq2Seq_CONFIG(CONFIG):
    def __init__(self, vocab : myVocab) -> None:
        self.vocab = vocab
        self.teacher_forcing_ratio = 0.5
        self.device = "GPU"
        self.num_inputs = len(vocab)
        self.encoder_embedding_size = 45
        self.decoder_embedding_size = 45
        self.output_size = len(vocab)
        self.num_hiddens = 256
        self.encoder_dropout = 0.5
        self.decoder_dropout = 0.5
        self.num_layers = 4
        self.bidirectional = False
        
        # Training Config
        self.num_epoches = 5
        self.lr = 0.005
        self.train_name = "train"
        self.saving = True
    
    def get_Encoder_Config(self) -> Encoder_CONFIG:
        """get corresponding Encoder Config"""
        return Encoder_CONFIG(
                vocab = self.vocab, 
                embedding_size = self.encoder_embedding_size, 
                num_hiddens = self.num_hiddens, 
                num_layers = self.num_layers, 
                dropout = self.encoder_dropout, 
                bidirectional = self.bidirectional
            )
    
    def get_Decoder_Config(self) -> Decoder_CONFIG:
        """get corresponding Decoder Config"""
        return Decoder_CONFIG(
                vocab = self.vocab, 
                embedding_size = self.decoder_embedding_size, 
                num_hiddens = self.num_hiddens, 
                num_layers = self.num_layers, 
                dropout = self.decoder_dropout,
                bidirectional = self.bidirectional
            )


class Seq2Seq_GUESS_CONFIG(CONFIG):
    def __init__(self,
                    vocab : myVocab,
                    guessnumber = 10000,
                    threshold = 1e-15,
                    minL = 6,
                    maxL = 31,
                    device = "GPU",
                    
                ):
        super(Seq2Seq_GUESS_CONFIG, self).__init__()
        self.guesser_name = "Seq2Seq_Guess"
        self.vocab = vocab
        self.eos = vocab[END]
        self.bos = vocab[START]
        self.pad = vocab[PAD]
        self.min_length = minL
        self.max_length = maxL
        self.device = device
        self.threshold = threshold
        self.guessnumber = guessnumber
        self.device = device


class AdditiveAttention_CONFIG(CONFIG):
    """AdditiveAttention Module Config"""
    def __init__(self,
                    key_size,
                    query_size, 
                    num_hiddens, 
                    dropout,
                ):
        super(AdditiveAttention_CONFIG, self).__init__()
        self.key_size = key_size
        self.query_size = query_size
        self.num_hiddens = num_hiddens
        self.dropout = dropout


class DotProductionAttention_CONFIG(CONFIG):
    """dot production Module Config"""
    def __init__(self, dropout):
        super(DotProductionAttention_CONFIG, self).__init__()
        self.dropout = dropout


class MultiHeadAttention_CONFIG(CONFIG):
    """Multi-Head Attention Config"""
    def __init__(self,
                    key_size,
                    query_szie,
                    value_size, 
                    num_hiddens, 
                    num_heads,
                    dropout,
                    bias = False
                ):
        super(MultiHeadAttention_CONFIG, self).__init__()
        self.key_size = key_size
        self.query_size = query_szie
        self.value_size = value_size
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
    
    def get_dotproductattention_config(self):
        return DotProductionAttention_CONFIG(self.dropout)


class PositionalEncoding_CONFIG(CONFIG):
    def __init__(self, num_hiddens, dropout, max_len = 1000):
        super(PositionalEncoding_CONFIG, self).__init__()
        self.num_hiddens = num_hiddens
        self.dropout_ = dropout
        self.max_len = max_len



class FFN_CONFIG(CONFIG):
    def __init__(self,
                    num_inputs, 
                    num_hiddens, 
                    num_outputs,
                    
                ):
        super(FFN_CONFIG, self).__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs


class AddNorm_CONFIG(CONFIG):
    def __init__(self, normalized_shape, dropout):
        super(AddNorm_CONFIG, self).__init__()
        self.normalized_shape = normalized_shape
        self.dropout = dropout


class EncoderBlock_CONFIG(CONFIG):
    def __init__(self, 
                    key_size, 
                    query_size, 
                    value_size, 
                    num_hiddens, 
                    norm_shape, 
                    ffn_num_inputs,
                    ffn_num_hiddens, 
                    num_heads,
                    dropout,
                    use_bias = False
                ):
        super(EncoderBlock_CONFIG, self).__init__()
        self.key_size = key_size
        self.query_size = query_size
        self.value_size = value_size
        self.num_hiddens = num_hiddens
        self.norm_shape = norm_shape
        self.ffn_num_inputs = ffn_num_inputs
        self.ffn_num_hiddens = ffn_num_hiddens
        self.num_heads = num_heads
        self.dropout_ = dropout
        self.use_bias = use_bias
    
    def get_multiattention_config(self):
        return MultiHeadAttention_CONFIG(self.key_size, self.query_size, self.value_size, self.num_hiddens, self.num_heads, self.dropout_, bias = self.use_bias)

    def get_addnorm_config(self):
        return AddNorm_CONFIG(self.norm_shape, self.dropout_)
    
    def get_ffn_config(self):
        return FFN_CONFIG(self.ffn_num_inputs, self.ffn_num_hiddens, self.num_hiddens)


class TransformerEncoder_CONFIG(CONFIG):
    def __init__(self, 
                    vocab_size, 
                    key_size, 
                    query_size, 
                    value_size,
                    num_hiddens,
                    norm_shape, 
                    ffn_num_input,
                    ffn_num_hiddens, 
                    num_heads,
                    num_layers, 
                    dropout,
                    use_bias = False
                ):
        super(TransformerEncoder_CONFIG, self).__init__()
        self.vocab_size = vocab_size
        self.key_size = key_size
        self.query_size = query_size
        self.value_size = value_size
        self.num_hiddens = num_hiddens
        self.norm_shape = norm_shape
        self.ffn_num_input = ffn_num_input
        self.ffn_num_hiddens = ffn_num_hiddens
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_ = dropout
        self.use_bias = use_bias
    
    def get_encoderblock_config(self):
        return EncoderBlock_CONFIG(self.key_size, self.query_size, self.value_size, self.num_hiddens, self.norm_shape, self.ffn_num_input, self.ffn_num_hiddens, self.num_heads, self.dropout_, self.use_bias)

    def get_positionalencoding_config(self):
        return PositionalEncoding_CONFIG(self.num_hiddens, self.dropout_)