import os
import json
import re


def make_vocab_questions(input_dir: str, output_dir: str):
    SENTENCE_SPLIT_REGEX = re.compile(r"(\W+)")
    vocab_set = set()
    datasets = os.listdir(input_dir)
    for dataset in datasets:
        input_path = os.path.join(input_dir, dataset)
        _, extension = os.path.splitext(input_path)
        if extension != ".json":
            continue
        with open(input_path) as f:
            questions = json.load(f)["questions"]
        for question in questions:
            words = SENTENCE_SPLIT_REGEX.split(question["question"].lower())
            vocab_set.update([w.strip() for w in words if len(w.strip()) > 0])
    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0, "<pad>")
    vocab_list.insert(1, "<unk>")

    output_path = os.path.join(output_dir, "vocab_questions.txt")
    with open(output_path, "w") as f:
        f.writelines([w + "\n" for w in vocab_list])


def make_vocab_answers(input_dir: str, n_answers: int, output_dir: str):
    answers = dict()
    datasets = os.listdir(input_dir)
    for dataset in datasets:
        input_path = os.path.join(input_dir, dataset)
        _, extension = os.path.splitext(input_path)
        if extension != ".json":
            continue
        print(input_path)
        with open(input_path) as f:
            annotations = json.load(f)["annotations"]
        for annotation in annotations:
            for answer in annotation["answers"]:
                word = answer["answer"]
                if re.search(r"[^\w\s]", word):
                    continue
                if word not in answers:
                    answers[word] = 0
                answers[word] += 1
        del annotations

    answers = sorted(answers.items(), key=lambda item: item[1], reverse=True)
    answers = [w for w, v in answers]
    assert "<unk>" not in answers
    top_answers = ["<unk>"] + answers[: n_answers - 1]

    output_path = os.path.join(output_dir, "vocab_answers.txt")
    with open(output_path, "w") as f:
        f.writelines([w + "\n" for w in top_answers])
