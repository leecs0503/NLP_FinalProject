import math
import os
import argparse
import numpy as np
import json
import re


def make_vocab_questions(input_dir, output_dir):
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    vocab_set = set()
    datasets = os.listdir(input_dir)
    max_length = 0
    for dataset in datasets:
        with open(input_dir + '/' + dataset) as f:
            questions = json.load(f)['questions']
        for question in questions:
            words = SENTENCE_SPLIT_REGEX.split(question['question'].lower())
            words = [w.strip() for w in words if len(w.strip()) > 0]
            vocab_set.update(words)
            max_length = max(max_length, len(words))
    vocab_list = list(vocab_set).sort()
    vocab_list.insert(0, '<pad>')
    vocab_list.insert(1, '<unk>')

    with open(output_dir + 'vocab_questions.txt', 'w') as f:
        f.writelines([w+'\n' for w in vocab_list])


def make_vocab_answers(input_dir, n_answers, output_dir):
    pass
