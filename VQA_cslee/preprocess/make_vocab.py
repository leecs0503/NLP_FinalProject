import os
import argparse
import numpy as np
import json
import re

def make_quest_vocab(input_dir: str, output_dir: str, n_answers: int):
    """
        COCO VQA 데이터셋에 대해 사용된 단어
        args:
            input_dir : Annotation, Questions 디렉토리가 포함된 폴더
            output_dir: vocab 정보가 저장될 폴더
    """ 

    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

    # Make Question Vocab
    print("(Start) Make Question Vocab")
    quest_vocab = set()

    question_dir = os.path.join(input_dir, "Questions")
    file_list = os.listdir(question_dir)
    for file_name in file_list:
        file_path = os.path.join(question_dir, file_name)
        _, file_extensions = os.path.splitext(file_path)
        
        if file_extensions != '.json': continue

        print(f"read {file_path}")
        with open(file_path) as f:
            questions = json.load(f)['questions']
        for quest in questions:
            words = SENTENCE_SPLIT_REGEX.split(quest['question'].lower())
            quest_vocab.update([w.strip() for w in words if len(w.strip()) > 0])
    print("(End) Done")
    quest_vocab.add('<pad>')
    quest_vocab.add('<unk>')
    quest_vocab_file_path = os.path.join(output_dir, "vocab_questions.txt")
    with open(quest_vocab_file_path, "w") as f:
        f.writelines([f"{vocab}\n" for vocab in quest_vocab])

    print(f"Write Question Vocab Complete (Size = {len(quest_vocab)}, path = {quest_vocab_file_path})")

def make_ans_vocab(input_dir: str, output_dir: str, n_answers: int):
    # Make Answer Vocab
    print("(Start) Make Answer Vocab")
    unknown_str = '<unk>'
    answer_vocab = dict()

    answer_dir = os.path.join(input_dir, "Annotations")
    file_list = os.listdir(answer_dir)
    for file_name in file_list:
        file_path = os.path.join(answer_dir, file_name)
        _, file_extensions = os.path.splitext(file_path)
        
        if file_extensions != '.json': continue

        print(f"read {file_path}")
        with open(file_path) as f:
            annotations = json.load(f)['annotations']
        for annotation in annotations:
            for answer in annotation['answers']:
                word = answer['answer']
                if re.search(r"[^\w\s]", word):
                    continue
                if word in answer_vocab:
                    answer_vocab[word] += 1
                else:
                    answer_vocab[word] = 1
    answer_vocab[unknown_str] = 1e9
    assert unknown_str in answer_vocab
    answer_vocab = sorted(answer_vocab.items(), key = lambda item: item[1], reverse = True)
    if n_answers != 0: answer_vocab = answer_vocab[:n_answers]
    
    print("(End) Done")
    
    with open('../datasets/vocab_answers.txt', 'w') as f:
        f.writelines([key+'\n' for (key, v) in answer_vocab])
    
    print("Write Done")
    print(f"The number of total words of answer: {len(answer_vocab)}, we keep {n_answers} answers by occurency")
    print(f"The minimum occurency of top {n_answers} is {answer_vocab[n_answers-1:n_answers][0][1]}")

def main(args):
    input_dir, output_dir = args.input_dir, args.output_dir
    n_answers = args.n_answers

    make_quest_vocab(input_dir, output_dir, n_answers)
    make_ans_vocab(input_dir, output_dir, n_answers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../datasets')
    parser.add_argument('--output_dir', type=str, default='../datasets')

    parser.add_argument('--n_answers', type=int, default=1000,
                        help='the number of answers to be kept in vocab, 0 is use all vocab (not recommended)')

    args = parser.parse_args()
    main(args)  