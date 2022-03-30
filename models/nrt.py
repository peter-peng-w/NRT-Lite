''' Replication of Neural Rating Regression with Abstractivee Tips Generation for Recommendation '''
import torch
from torch import nn

from .rater import Rater
from .textgen import TextRNN
from .mlp import MLP

from .attr_dict import AttrDict


class NRT(nn.Module):
    '''
    Inputs:
        data:
            AttrDict
            users
            items
            scores
            words
            mask

    Outputs:
    '''

    def __init__(
        self,
        n_users,        # number of users
        n_items,        # number of items
        d_ebd,          # dimension of embedding
        r_mlp_sizes,    # MLP layer sizes for rating predict
        t_hidden_size,  #
        t_n_words,      # number of words in review text
        t_d_word_ebd,   # dimension of word embedding
        t_n_layers,     #
        t_dropout,      #
        t_rnn_type,     #
    ):
        super(NRT, self).__init__()

        # user and item embedding
        self.user_ebd = nn.Embedding(n_users, d_ebd)
        self.item_ebd = nn.Embedding(n_items, d_ebd)

        # initialized from a uniform distribution between [−0.1, 0.1]
        self.user_ebd.weight = nn.Parameter(torch.rand_like(self.user_ebd.weight) * 0.2 - 0.1)
        self.item_ebd.weight = nn.Parameter(torch.rand_like(self.item_ebd.weight) * 0.2 - 0.1)

        # No.1 rate prediction -> mlp
        self.rater = Rater(d_ebd * 2, r_mlp_sizes, act='sigmoid')

        # No.2 review word prediction -> mlp
        self.ui_2_wd = MLP(d_ebd * 2, [d_ebd * 2, d_ebd * 2], act='sigmoid')
        self.wd = nn.Linear(d_ebd * 2, t_n_words)

        self.word_ebd = nn.Embedding(t_n_words, t_d_word_ebd)

        self.to_text = nn.Linear(d_ebd * 4 + 5, t_hidden_size)
        self.textgen = TextRNN(t_hidden_size, t_d_word_ebd, t_n_words, t_n_layers, t_dropout, t_rnn_type)
        self.textgen.word_ebd = self.word_ebd

    def load_pretrained_word_ebd(self, weight):
        self.word_ebd.weight = nn.Parameter(weight)

    def get_input(self, data):
        # construct the input hidden vector of the TextRNN
        user_var, item_var = data.user, data.item
        # concat user embedding and item embedding
        user_vct = self.user_ebd(user_var)
        item_vct = self.item_ebd(item_var)
        ui_var = torch.cat([user_vct, item_vct], dim=1)
        # predict the rating
        ratings = self.rater(ui_var)
        batch_size = ratings.size(0)
        # clip the predicted rating and convert it to one-hot encoding
        rating_idx = ratings.round().long()
        rating_idx[rating_idx < 1] = 1
        rating_idx[rating_idx > 5] = 5
        rating_idx[rating_idx > 0] -= 1
        rating_idx = rating_idx.unsqueeze(-1)
        rating_vct = torch.zeros([batch_size, 5], device=ratings.device).scatter_(1, rating_idx, 1.)
        # review words prediction representation
        wd_vct = self.ui_2_wd(ui_var)
        # concat 3 types of context to generate the initial hidden vector
        t_var = torch.cat([ui_var, rating_vct, wd_vct], dim=1)
        t_init_hidden = self.to_text(t_var).tanh().unsqueeze(0)

        return AttrDict(
            user_vct=user_vct,
            item_vct=item_vct,
            wd_vct=wd_vct,
            ui_t_var=t_init_hidden,
            ratings=ratings
        )

    def forward(self, data, word_var=None, tf_rate=1, mode='all'):
        input_dict = self.get_input(data)

        if mode == 'rate' or mode == 'all':
            rate_output = input_dict.ratings

        if mode == 'review' or mode == 'all':
            # print('Before feeding into text generator')
            # print('Hidden state: {}'.format(input_dict.ui_t_var))
            review_output = self.textgen(input_dict, word_var, data=data, tf_rate=tf_rate)

        if mode == 'worddict' or mode == 'all':
            # print('Before feeding into the linear layer of review generator')
            # print('Word vector: {}'.format(input_dict.wd_vct))
            # print('Word vector shape:{}'.format(input_dict.wd_vct.shape))
            wd_output = self.wd(input_dict.wd_vct).log_softmax(-1)

        if mode == 'rate':
            return rate_output
        elif mode == 'review':
            return review_output
        elif mode == 'worddict':
            return wd_output
        else:
            return rate_output, review_output, wd_output

    def rate(self, data):
        return self(data, mode='rate')

    def review(self, data, word_var, tf_rate=1):
        return self(data, word_var, tf_rate, mode='review')

    def word_dict(self, data, word_var, tf_rate=1):
        return self(data, mode='worddict')
