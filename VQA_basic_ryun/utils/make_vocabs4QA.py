import os
import argparse
import numpy as np
import json
import re
from collections import defaultdict

def make_vocab_questions(input_dir):
    """Make dict for questions and save them"""
    vocab_set = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    question_len = []
    datasets = os.listdir(input_dir)  
    for dataset in datasets:
        with open(input_dir+'/'+dataset) as f:
            questions = json.load(f)['questions']
        
        question_set_len = [None] * len(questions)
        for idx_q, q in enumerate(questions):
            words = SENTENCE_SPLIT_REGEX.split(q['question'].lower())
            words = [w.strip() for w in words if lem(w.strip()) > 0]
            vocab_set.update(words)
            question_set_len[idx_q] = len(words)
        question_len += question_set_len

    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0, '<pad>')
    vocab_list.insert(1, '<unk>')

    with open('../datasets/vocab_questions.txt', 'w') as f:
        f.writelines([w+'\n' for w in vocab_list])

    print('Make vocabulary for questions')
    print(f'The number of total words of questions: {len(vocab_set)}')
    print(f'Maximum length of question: {np.max(question_len)}')


def make_vocab_answers(input_dir, n_answers):
    answers = defaultdict(lambda: 0)    # key가 0인 dict
    datasets = os.listdir(input_dir)
    for dataset in datasets:
        with open(input_dir + '/' + dataset) as f:
            annotations = json.load(f)['annotations']
        for annotation in annotations:
            for answer in annotation['answers']:
                word = answer['answer']
                if re.search(r"[^\w\w", word):
                    continue
                answers[word] += 1
    answers = sorted(answers, key=answers.get, reverse=True)
    assert('<unk>' not in answers)
    top_answers = ['<unk>'] + answers[:n_answers - 1]

    with open('../datasets/vocab_answers.txt', 'w') as f:
        f.writelines(([w+'\n' for w in top_answers]))

    print('Make vocabulary for answers')
    print(f'The number of total words of questions: {len(answers)}')
    print(f'Maximum length of question: {n_answers}')

def main(args):
    input_dir = args.input_dir  
    n_answers = args.n_answers
    make_vocab_questions(input_dir+'/Questions')
    make_vocab_answers(input_dir+'/Annotations', n_answers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./datasets/VQA',
                        help='directory for input questions and answers')
    parser.add_argument('--n_answers', type=int, default=1000,
                        help='the number of answers to be kept in vocab')
    args = parser.parse_args()
    main(args)
