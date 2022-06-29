# -*- coding:utf-8 -*-
# Vocab Class
# author: Matthew



UNK = '<unk>'
START = '<bos>'
END = '<eos>'
SPACE = '<sos>'
PAD = '<pad>'
MASK = '<mask>'

class myVocab:
    """Vocabulary for text."""
    def __init__(self, tokens = None, min_freq = 0, reserved_tokens = None):
        from utils import count_corpus
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        token_counter = count_corpus(tokens)
        self._token_freq = sorted(token_counter.items(), key = lambda x : x[1], reverse = True)
        self.idx_to_token = [START, END, PAD] + reserved_tokens
        self.token_to_idx = {token : idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freq:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    
    def __len__(self):
        return len(self.idx_to_token)


    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.pad)
        return [self.__getitem__(token) for token in tokens]


    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[indice] for indice in indices]
    
    # @property
    # def unk(self):
    #     # index of unkonwn token
    #     return 0
    
    
    @property
    def start(self):
        # index of start token
        return 0
    
    
    @property
    def end(self):
        # index of end token
        return 1
    
    
    @property
    def pad(self):
        # index of padding token
        return 2
    
    @property
    def token_freqs(self):
        # token frequenceis
        return self._token_freq
    
    