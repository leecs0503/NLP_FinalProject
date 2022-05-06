import json
import os
import utils.text_helper as text_helper


def extract_answers(q_answers: str, valid_answer_set: str):
    all_answers = [answer["answer"] for answer in q_answers]
    valid_answers = [answer for answer in all_answers if answer in valid_answer_set]
    return all_answers, valid_answers


def vqa_processing(
    image_dir: str,
    annotation_file: str,
    question_file: str,
    valid_answer_set: set(str),
    image_set: str,
):
    if image_set in ["train2014", "val2014"]:
        load_answer = True
        with open(annotation_file % image_set) as f:
            annotations = json.load(f)["annotations"]
            qid2ann_dict = {ann["question_id"]: ann for ann in annotations}
    else:
        load_answer = False
    with open(question_file % image_set) as f:
        questions = json.load(f)["questions"]
    coco_set_name = image_set.replace("-dev", "")
    abs_image_dir = os.path.abspath(image_dir % coco_set_name)
    image_name_template = "COCO_" + coco_set_name + "_%012d"
    unk_ans_count = 0
    dataset = []
    for n_q, q in enumerate(questions):
        if (n_q + 1) % 10000 == 0:
            print("processing %d / %d" % (n_q + 1, len(questions)))
        image_id = q["image_id"]
        question_id = q["question_id"]
        image_name = image_name_template % image_id
        image_path = os.path.join(abs_image_dir, image_name + ".jpg")
        question_str = q["question"]
        question_tokens = text_helper.tokenize(question_str)

        iminfo = dict(
            image_name=image_name,
            image_path=image_path,
            question_id=question_id,
            question_str=question_str,
            question_tokens=question_tokens,
        )

        if load_answer:
            ann = qid2ann_dict[question_id]
            all_answers, valid_answers = extract_answers(
                ann["answers"], valid_answer_set
            )
            if len(valid_answers) == 0:
                valid_answers = ["<unk>"]
                unk_ans_count += 1
            iminfo["all_answers"] = all_answers
            iminfo["valid_answers"] = valid_answers

        dataset.append(iminfo)
    return dataset
