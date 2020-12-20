import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from utils.data_reader import amazon_dataset_iters
import torch.utils.tensorboard as tb
from os import path
from models.nrt import NRT
import utils.constants as constants
from utils.loss import mask_nll_loss, review_loss
import math
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import bleu_score


def test_review_bleu(gts_data, generate_data, vocab, bleu_totals, length):
    type_wights = [
        [1., 0, 0, 0],
        [.5, .5, 0, 0],
        [1 / 3, 1 / 3, 1 / 3, 0],
        [.25, .25, .25, .25]
    ]

    sf = bleu_score.SmoothingFunction()

    # batch first
    gts_idx = torch.transpose(gts_data, 0, 1)
    _, generate_idx = generate_data.max(2)
    generate_idx = torch.transpose(generate_idx, 0, 1)

    gts_sentence = []
    gene_sentence = []
    # detokenize the sentence
    for token_ids in gts_idx:
        current = [vocab.itos[id] for id in token_ids.detach().numpy()]
        gts_sentence.append(current)
    for token_ids in generate_idx:
        current = [vocab.itos[id] for id in token_ids.detach().numpy()]
        gene_sentence.append(current)
    # compute bleu score
    assert len(gts_sentence) == len(gene_sentence)
    for i in range(len(gts_sentence)):
        length += 1
        for j in range(4):
            refs = gts_sentence[i]
            sample = gene_sentence[i]
            weights = type_wights[j]
            bleu_totals[j] += bleu_score.sentence_bleu(refs, sample, smoothing_function=sf.method1, weights=weights)

    return bleu_totals, length


def train(args):
    # Load logger
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Loading the dataset
    dataset_folder = './data/Musical_Instruments_5/'
    text_vocab, tips_vocab, train_iter, val_iter, test_iter = (
        amazon_dataset_iters(dataset_folder)
    )

    # Count user and item number
    items_count = int(max([i.item.max().cpu().data.numpy() for i in train_iter] + [i.item.max().cpu().data.numpy() for i in test_iter]))
    users_count = int(max([i.user.max().cpu().data.numpy() for i in train_iter] + [i.user.max().cpu().data.numpy() for i in test_iter]))
    vocab_size = len(text_vocab.itos)

    # Load model
    model = NRT(
            users_count + 2,
            items_count + 2,
            constants.EBD_SIZE,
            constants.RATER_MLP_SIZES,
            constants.HIDDEN_DIM,
            vocab_size,
            constants.WORD_LF_NUM,
            constants.TG_HIDDEN_LAYERS,
            constants.DROPOUT_RATE,
            constants.RNN_TYPE,
        )

    model.to(device)

    alpha = constants.RR_LOSS_WEIGHT
    beta = constants.WG_LOSS_WEIGHT
    gamma = constants.TG_LOSS_WEIGHT

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=constants.REG_WEIGHT)
    # TODO: Add learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 40], gamma=0.2)

    global_step = 0
    pad_idx = tips_vocab.stoi['<pad>']
    for epoch in range(args.num_epoch):
        # Training Procedure
        model.train()
        idx = 0
        cumu_loss = 0.0
        cumu_rate_loss = 0.0
        cumu_tips_loss = 0.0
        cumu_word_loss = 0.0
        for train_batch in train_iter:
            tips = train_batch.tips
            pad_mask = ~(tips[1:] == pad_idx)
            pad_mask.to(device)
            rate_output, tips_output, wd_output = model(train_batch, tips[:-1])
            # compute loss for each task
            rate_loss = F.mse_loss(rate_output, train_batch.rating)
            tips_loss = mask_nll_loss(tips_output.output, tips[1:], pad_mask)
            word_loss = (- train_batch.text * wd_output).sum(-1).mean()
            # multi-task loss
            loss = alpha * rate_loss + beta * word_loss + gamma * tips_loss

            # Add loss statistics
            cumu_loss += loss.item()
            cumu_rate_loss += rate_loss.item()
            cumu_tips_loss += tips_loss.item()
            cumu_word_loss += word_loss.item()
            idx += 1
            if train_logger is not None and idx % 4 == 0:
                train_logger.add_scalar('rate_loss', cumu_rate_loss/4, global_step)
                train_logger.add_scalar('tips_loss', cumu_tips_loss/4, global_step)
                train_logger.add_scalar('word_loss', cumu_word_loss/4, global_step)
                print(
                    '[%d, %5d] loss: %.3f'
                    % (epoch + 1, idx + 1, cumu_loss / 4)
                )
                # Add writer to record the traning loss
                train_logger.add_scalar('loss', cumu_loss / 4, global_step=global_step)
                cumu_loss = 0.0
                cumu_rate_loss = 0.0
                cumu_tips_loss = 0.0
                cumu_word_loss = 0.0

            optimizer.zero_grad()
            loss.backward()
            # TODO: add gradient clip here
            optimizer.step()
            global_step += 1

        print('Finished Training at epoch {} ... Start Validating\n'.format(epoch + 1))

        # Validation Procedure
        model.eval()
        idx = 0
        rmse_loss = 0.0
        num_ratings = 0
        num_reviews = 0
        bleu_totals = [0.] * 4
        for valid_batch in val_iter:
            tips = valid_batch.tips
            rate_output, tips_output, wd_output = model(valid_batch, tips[:-1], tf_rate=0)
            # compute rmse
            rmse_loss += F.mse_loss(rate_output, valid_batch.rating, reduction='sum').item()
            num_ratings += valid_batch.rating.shape[0]
            # compute bleu
            bleu_totals, num_reviews = test_review_bleu(tips[1:], tips_output, tips_vocab, bleu_totals, num_reviews)

        rmse = math.sqrt(rmse_loss / num_ratings)
        bleu_totals = [bleu_total / num_reviews for bleu_total in bleu_totals]
        if valid_logger is not None:
            valid_logger.add_scalar('RMSE', rmse, global_step)
            valid_logger.add_scalar('BLEU-1', bleu_totals[0], global_step)
            valid_logger.add_scalar('BLEU-2', bleu_totals[1], global_step)
            valid_logger.add_scalar('BLEU-3', bleu_totals[2], global_step)
            valid_logger.add_scalar('BLEU-4', bleu_totals[3], global_step)
            print(
                '[%d] rating rmse: %.3f'
                % (epoch + 1, rmse)
            )
            print(
                '[%d] rating BLEU-1: %.3f'
                % (epoch + 1, bleu_totals[0])
            )
            print(
                '[%d] rating BLEU-2: %.3f'
                % (epoch + 1, bleu_totals[1])
            )
            print(
                '[%d] rating BLEU-3: %.3f'
                % (epoch + 1, bleu_totals[2])
            )
            print(
                '[%d] rating BLEU-4: %.3f'
                % (epoch + 1, bleu_totals[3])
            )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='./logging', help='The path of the logging dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-sf', '--save_model_freq', type=int, default=2, help='Frequency of saving model, per epoch')
    parser.add_argument('-s', '--save_dir', type=str, default='./exp', help='The path of experiment model dir')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size for traning')

    args = parser.parse_args()
    train(args)
