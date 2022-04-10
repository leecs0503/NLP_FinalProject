import argparse
import json
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from util.text_helper import tokenize, VocabDict


def extract_answers(q_answers, valid_answer_set):
    all_answers = [answer["answer"] for answer in q_answers]
    valid_answers = [a for a in all_answers if a in valid_answer_set]
    return all_answers, valid_answers


def vqa_processing(input_dir, valid_answer_set, data_set):
    print(f"{data_set} start")
    abs_input_path = os.path.abspath(input_dir)
    annotation_dir = os.path.join(
        input_dir, "Annotations", f"v2_mscoco_{data_set}_annotations.json"
    )
    question_dir = os.path.join(
        abs_input_path, "Questions", f"v2_OpenEnded_mscoco_{data_set}_questions.json"
    )

    load_answer = data_set in ["train2014", "val2014"]
    if load_answer:
        with open(annotation_dir) as f:
            annotations = json.load(f)["annotations"]
            qid2ann_dict = {ann["question_id"]: ann for ann in annotations}

    with open(question_dir) as f:
        questions = json.load(f)["questions"]

    dataset = [None] * len(questions)
    unk_ans_count = 0
    for idx, question in enumerate(questions):
        image_id, question_str, question_id = (
            question["image_id"],
            question["question"],
            question["question_id"],
        )
        image_name = f"COCO_{data_set}_{image_id:012d}"
        image_path = os.path.join(
            abs_input_path, "Resized_Images", data_set, image_name + ".jpg"
        )
        question_tokens = tokenize(question_str)

        iminfo = dict(
            image_name=image_name,
            image_path=image_path,
            question_id=question_id,
            question_str=question_str,
            question_tokens=question_tokens,
        )

        if load_answer:
            annotation = qid2ann_dict[question_id]
            all_answers, valid_answers = extract_answers(
                annotation["answers"], valid_answer_set
            )
            if len(valid_answers) == 0:
                valid_answers = ["<unk>"]
                unk_ans_count += 1
            iminfo["all_answers"] = all_answers
            iminfo["valid_answers"] = valid_answers
        dataset[idx] = iminfo
    print(f"total {unk_ans_count} out of {len(questions)} answers are <unk>")
    return dataset


def main(args):
    input_dir, output_dir = args.input_dir, args.output_dir

    vocab_answer_file = args.output_dir + "/vocab_answers.txt"
    answer_dict = VocabDict(vocab_answer_file)
    valid_answer_set = set(answer_dict.word_list)

    train = vqa_processing(input_dir, valid_answer_set, "train2014")
    valid = vqa_processing(input_dir, valid_answer_set, "val2014")
    test = vqa_processing(input_dir, valid_answer_set, "test2015")

    np.save(os.path.join(output_dir, "train.npy"), np.array(train))
    np.save(os.path.join(output_dir, "valid.npy"), np.array(valid))
    np.save(os.path.join(output_dir, "test.npy"), np.array(test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../datasets")
    parser.add_argument("--output_dir", type=str, default="../datasets")

    args = parser.parse_args()
    main(args)
