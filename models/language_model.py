# import random
# import torch
from torch import nn

# from .rater import Rater
from .textgen import TextRNN
# from .attn import Attn
# from .review_encoder import ReviewRNNEncoder, ReviewTransEncoder


class LanguageModel(nn.Module):
    '''
    Inputs:

    Outputs:
    '''

    def __init__(
        self,
        d_hidden,
        n_words,
        d_word_ebd,
        n_layers,
        dropout,
        rnn_type
    ):
        super(LanguageModel, self).__init__()

        self.word_ebd = nn.Embedding(n_words, d_word_ebd)

        self.textgen = TextRNN(d_hidden, d_word_ebd, n_words, n_layers, dropout, rnn_type)

        self.textgen.word_ebd = self.word_ebd

    def load_pretrained_word_ebd(self, weight):
        self.word_ebd.weight = nn.Parameter(weight)

    def forward(self, user_var, item_var, word_var, word_mem, tf_rate=1):
        return None, self.textgen(None, word_var, tf_rate=1)

    def review(self, *args, **kargs):
        return self(*args, **kargs)[1]

    def _to_text_inp(self, user_var, item_var):
        return None, None
