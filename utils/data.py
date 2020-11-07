# -*- coding: utf-8 -*-

import re


# Lowercase and remove non-letter characters
def normalize_str(s):
    s = s.lower()
    # give a leading & ending spaces to punctuations
    s = re.sub(r'([.!?,\'])', r' \1 ', s)
    # purge unrecognized token with space
    s = re.sub(r'[^a-z.!?,\']+', r' ', s)
    # squeeze multiple spaces
    s = re.sub(r'([ ]+)', r' ', s)
    # remove extra leading & ending space
    return s.strip()


def trim_unk_data(pairs, voc):
    # Filter out pairs with trimmed words
    keep_pairs = []

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]

        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


def binary_mask(matrix, *maskvalues):
    maskvalues = set(maskvalues)

    mask = [
        [
            0 if index in maskvalues else 1
            for index in row
        ]
        for row in matrix
    ]

    return mask
