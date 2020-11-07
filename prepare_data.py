import sys, os
import numpy as np
import gzip
import json
import string
import time
import datetime
import argparse
import config
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

stop_words_filePath = "./data/stopwords_en.txt"
MIN_SUM_LEN = 5


def load_stop_words():
    # load all the stop words into the dict
    stop_words = {}
    with open(stop_words_filePath) as f:
        for line in f:
            line = line.strip('\n')
            stop_words[line] = 1
    return stop_words


def load_amazon():
    # Can have other dataset as new instances and load them into amazon data
    amazon_music = './data/Musical_Instruments_5.json'
    amazon_data = amazon_music

    size_corpus = 10000
    stop_words = load_stop_words()

    dic = {}
    i2w = {}
    w2i = {}
    i2user = {}    # new user idx convert to original user_id
    user2i = {}    # original user_id convert to new user idx
    i2item = {}    # new item idx convert to original item_id
    item2i = {}    # original item_id convert to new item idx
    user2w = {}
    item2w = {}
    x_raw = []
    timestamps = []
    w2df = {}

    # Using this translator to remove all the punctuation in a string
    translator = str.maketrans('', '', string.punctuation)

    with open(amazon_data) as f:
        for line in f:
            try:
                # split out the content from each line
                line = line.strip('\n')
                json_sections = json.loads(line)
                user_id = json_sections["reviewerID"]
                item_id = json_sections["asin"]
                review = json_sections["reviewText"].lower()
                rating = json_sections["overall"]    # rating in a range [1.0, 5,0]
                summary = json_sections["summary"].lower()
                unix_time = json_sections["unixReviewTime"]
                raw_time = json_sections["reviewTime"]    # raw time in a format of '[month] [day], [year]'

                # Convert the user_id and item_id to unique continuous id start from 0
                if user_id not in user2i:
                    user2i[user_id] = len(i2user)
                    i2user[user2i[user_id]] = user_id
                if item_id not in item2i:
                    item2i[item_id] = len(i2item)
                    i2item[item2i[item_id]] = item_id

                # If the summary sentence is too short, then extract a sentence from review text
                if len(summary.split()) < MIN_SUM_LEN:
                    review_sentences = sent_tokenize(review)
                    for sentence in review_sentences:
                        if len(sentence.split()) >= MIN_SUM_LEN:
                            summary = sentence
                            break
                
                # ?
                if len(summary.split()) < MIN_SUM_LEN:
                    continue

                review = review.translate(translator)
                terms = review.split()

            except KeyError:
                print('Wrong Key: {}'.format(line))




if __name__ == "__main__":
    pass
