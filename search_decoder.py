"""
Greedy decoding is the decoding method
that we use during training when we are NOT using teacher forcing.
"""

import random
import torch
from torch import nn
from torch.nn.functional import mse_loss
# from collections import Counter

from .utils import AttrDict, gumbel_softmax

# r_c, f_c, w_c = Counter(), Counter(), Counter()


class AbstractSearchDecoder(nn.Module):
    def forward(self, batch_data):
        user_var, _ = batch_data.users, batch_data.items
        input_dict = self.model.get_input(batch_data)

        # convert ui to TextRNN init hidden
        hidden = self.model.textgen.get_init_hidden(input_dict.ui_t_var)

        start_var = torch.full(user_var.view(-1).size(), self.voc.sos_idx, dtype=torch.long, device=user_var.device)

        return self.decode(hidden, start_var, data=batch_data, input_dict=input_dict)


class SearchDecoder(AbstractSearchDecoder):
    """
    Inputs:
        user_var
        item_var

    Outputs:
        words
        probs
        hiddens
        rvw_lens
    """

    def __init__(self, model, voc, max_length=15, greedy=False, sample_length=float('inf'), topk=0):
        super().__init__()
        self.model = model
        self.voc = voc
        self.max_length = max_length
        self.greedy = greedy
        self.sample_length = sample_length
        self.topk = topk

    def set_greedy(self, greedy):
        self.greedy = greedy

    def decode(self, hidden, start_var, data=None, input_dict=None, max_length=None):
        if not max_length:
            max_length = self.max_length

        greedy = self.greedy

        words = []
        probs = []
        hiddens = []

        batch_size = start_var.size(-1)
        rvw_lens = torch.zeros(batch_size, dtype=torch.long).to(start_var.device)

        decoder_var = start_var.view(1, -1)

        for i in range(max_length):
            output_dict = self.model.textgen.decode(decoder_var, hidden, data=data, input_dict=input_dict)

            output, hidden = output_dict.output, output_dict.hidden

            # (1, batch, n_words) -> (batch, n_words)
            word_probs = output.exp().squeeze(0)

            # (batch)
            if not greedy and i < self.sample_length:
                if self.topk:
                    # not free sampling, limit it to the popular terms
                    k = self.topk if i % 2 == 0 else 5
                    word_probs, word_idxes = word_probs.topk(k)

                    word_var = torch.multinomial(word_probs, 1)
                    prob_var = word_probs.gather(-1, word_var).view(-1)
                    word_var = word_idxes.gather(-1, word_var).view(-1)
                else:
                    word_var = torch.multinomial(word_probs, 1)
                    prob_var = word_probs.gather(-1, word_var).view(-1)
                    word_var = word_var.view(-1)
            else:
                prob_var, word_var = word_probs.max(-1)

            # print('{:10s} {:6.4f} {:6.4f}'.format(self.voc[word_var[0].item()], output_dict.rate_gates[0, 0].item(), output_dict.feature_gates[0, 0].item()))

            # w_ = self.voc[word_var[0].item()]
            # r_c[w_] += output_dict.rate_gates[0, 0].item()
            # f_c[w_] += output_dict.feature_gates[0, 0].item()
            # w_c[w_] += 1

            # with open('rg.csv', 'w') as f:
            #     for w_, v in r_c.items():
            #         f.write(f'{w_}, {v / w_c[w_]}, {w_c[w_]}\n')
            # with open('fg.csv', 'w') as f:
            #     for w_, v in f_c.items():
            #         f.write(f'{w_}, {v / w_c[w_]}, {w_c[w_]}\n')

            # [(batch),...]
            words.append(word_var)
            probs.append(prob_var)
            hiddens.append(hidden)

            # verify eos
            is_eos = word_var == self.voc.eos_idx
            not_end = rvw_lens == 0

            if i != max_length - 1:
                rvw_lens[not_end * is_eos] = i + 1

                # break if whole batch end
                if (rvw_lens != 0).all():
                    break

                # next input is decoder's own current output
                # add seq dim
                decoder_var = word_var.unsqueeze(0)
            else:
                # reach max len
                rvw_lens[not_end] = max_length

        # (seq, batch)
        words = torch.stack(words, dim=0)
        probs = torch.stack(probs, dim=0)

        return AttrDict(
            words=words,
            probs=probs,
            hiddens=hiddens,
            rvw_lens=rvw_lens
        )


class BeamSearchDecoder(AbstractSearchDecoder):
    """
    Inputs:
        user_var
        item_var

    Outputs:
        words
        probs: return logp
        hiddens
        rvw_lens
    """

    def __init__(self, model, voc, max_length=15, beam_width=5, mode='best'):
        super(BeamSearchDecoder, self).__init__()
        self.model = model
        self.voc = voc
        self.beam_width = beam_width
        self.mode = mode
        self.max_length = max_length

    class BeamSearchNode:
        def __init__(self, hidden, idx, value=None, previousNode=None, logp=0, depth=0):
            self.prevnode = previousNode
            self.hidden = hidden
            self.value = value
            self.idx = idx
            self.logp = logp
            self.depth = depth

        def eval(self):
            # for now, simply choose the one with maximum average
            return self.logp / float(self.depth)

    def decode(self, hidden, start_var, data=None, input_dict=None, max_length=None):
        sos, eos = self.voc.sos_idx, self.voc.eos_idx

        # Number of sentence to generate
        endnodes = []

        # Start with the start of the sentence token
        root_idx = start_var
        root = self.BeamSearchNode(hidden, root_idx)
        leaf = [root]

        for dep in range(self.max_length):
            candidates = []

            for prevnode in leaf:
                decoder_input = prevnode.idx.view(1, 1)

                # Forward pass through decoder
                # decode for one step using decoder

                output_dict = self.model.textgen.decode(decoder_input, prevnode.hidden, data=data, input_dict=input_dict)

                output, decoder_hidden = output_dict.output, output_dict.hidden

                output = nn.functional.log_softmax(output, dim=2)

                values, indexes = output.topk(self.beam_width)

                for i in range(self.beam_width):
                    idx = indexes[0][0][i]
                    value = values[0][0][i]

                    node = self.BeamSearchNode(decoder_hidden, idx, value, prevnode, value + prevnode.logp, dep + 1)

                    candidates.append(node)

            candidates.sort(key=lambda n: n.logp, reverse=True)

            leaf = []
            for candiate in candidates[:self.beam_width]:
                if candiate.idx == eos:
                    endnodes.append(candiate)
                else:
                    leaf.append(candiate)

            # sentecnes don't need to be beam_width exactly, here just for simplicity
            if len(endnodes) >= self.beam_width:
                endnodes = endnodes[:self.beam_width]
                break

        # arrive max length before having enough results
        if len(endnodes) < self.beam_width:
            endnodes = endnodes + leaf[:self.beam_width - len(endnodes)]

        # choose the max/random from the results
        if self.mode == 'all':
            return [self.trace_tokens(n, sos) for n in endnodes]

        if self.mode == 'random':
            endnode = random.choice(endnodes)
        else:
            endnode = max(endnodes, key=lambda n: n.eval())

        tokens, probs = self.trace_tokens(endnode, sos)
        lengths = [tokens.size(0)]

        return AttrDict(
            words=tokens.unsqueeze(1),
            probs=probs.unsqueeze(1),
            rvw_lens=lengths
        )

    def trace_tokens(self, endnode, sos):
        tokens = []
        scores = []
        while endnode.idx != sos:
            tokens.append(endnode.idx)
            scores.append(endnode.value)
            endnode = endnode.prevnode

        tokens.reverse()
        scores.reverse()

        tokens = torch.stack(tokens)
        scores = torch.stack(scores)

        return tokens, scores


class DebiasSearchDecoder(nn.Module):
    """
    Inputs:
        user_var
        item_var

    Outputs:
        words
        probs
        hiddens
        rvw_lens
    """

    def __init__(self, model, voc, global_model, max_length=15, n_debias=5):
        super().__init__()
        self.model = model
        self.global_model = global_model
        self.voc = voc
        self.max_length = max_length

        self.global_lambda = 0.4
        self.n_debias = n_debias

    def decode(self, hidden, start_var, data=None, input_dict=None, max_length=None):
        if not max_length:
            max_length = self.max_length

        words = []
        probs = []
        hiddens = []

        batch_size = start_var.size(-1)
        rvw_lens = torch.zeros(batch_size, dtype=torch.long).to(start_var.device)

        decoder_var = start_var.view(1, -1)

        global_hidden = None

        for i in range(max_length):
            output_dict = self.model.textgen.decode(decoder_var, hidden, data=data, input_dict=input_dict)

            output, hidden = output_dict.output, output_dict.hidden

            global_output, global_hidden = self.global_model.textgen.decode(decoder_var, global_hidden)

            # (1, batch, n_words) -> (batch, n_words)
            word_probs = output.squeeze(0).log_softmax(-1)

            # (batch)
            if i < self.n_debias:
                global_word_probs = global_output.squeeze(0).log_softmax(-1)

                word_probs = word_probs - self.global_lambda * global_word_probs

            prob_var, word_var = word_probs.max(-1)

            # [(batch),...]
            words.append(word_var)
            probs.append(prob_var)
            hiddens.append(hidden)

            # verify eos
            is_eos = word_var == self.voc.eos_idx
            not_end = rvw_lens == 0

            if i != max_length - 1:
                rvw_lens[not_end * is_eos] = i + 1

                # break if whole batch end
                if (rvw_lens != 0).all():
                    break

                # next input is decoder's own current output
                # add seq dim
                decoder_var = word_var.unsqueeze(0)
            else:
                # reach max len
                rvw_lens[not_end] = max_length

        # (seq, batch)
        words = torch.stack(words, dim=0)
        probs = torch.stack(probs, dim=0)

        return AttrDict(
            words=words,
            probs=probs,
            hiddens=hiddens,
            rvw_lens=rvw_lens
        )


class GumbelDecoder(AbstractSearchDecoder):
    """
    Inputs:
        user_var
        item_var

    Outputs:
        words
        probs
        hiddens
        rvw_lens
    """

    def __init__(self, model, voc, max_length=15, temperature=0.5):
        super().__init__()
        self.model = model
        self.voc = voc
        self.max_length = max_length

        self.temperature = temperature

    def decode(self, hidden, start_var, data=None, input_dict=None, max_length=None):
        if not max_length:
            max_length = self.max_length

        word_dists = []

        batch_size = start_var.size(-1)
        rvw_lens = torch.zeros(batch_size, dtype=torch.long).to(start_var.device)

        decoder_var = start_var.view(1, -1)

        for i in range(max_length):
            output_dict = self.model.textgen.decode(decoder_var, hidden, data=data, input_dict=input_dict)

            output, hidden = output_dict.output, output_dict.hidden

            # (1, batch, n_words) -> (batch, n_words)
            log_probs = output.log_softmax(-1).squeeze(0)

            # (batch, n_words)
            word_dist = gumbel_softmax(log_probs, self.temperature, ST=True)
            _, word_var = word_dist.max(-1)

            # [(batch),...]
            word_dists.append(word_dist)

            # verify eos
            is_eos = word_var == self.voc.eos_idx
            not_end = rvw_lens == 0

            if i != max_length - 1:
                rvw_lens[not_end * is_eos] = i + 1

                # break if whole batch end
                if (rvw_lens != 0).all():
                    break

                # next input is decoder's own current output
                # add seq dim
                decoder_var = word_var.unsqueeze(0)
            else:
                # reach max len
                rvw_lens[not_end] = max_length

        # (seq, batch, n_words)
        word_dists = torch.stack(word_dists, dim=0)

        return AttrDict(
            words=word_dists,   # words distribution
            rvw_lens=rvw_lens
        )
