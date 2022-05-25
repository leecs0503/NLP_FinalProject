import json
import os
import utils.text_helper as text_helper
from typing import Set, List


def extract_answers(q_answers: List[str], valid_answer_set):
    all_answers = [answer["answer"] for answer in q_answers]
    valid_answers = [answer for answer in all_answers if answer in valid_answer_set]
    return all_answers, valid_answers

def visual_grounding_processing(
    phase: str,
    image_dir: str,
    dataset_file:str,
    instances_file:str,
):
    annotation_data = json.load(open(instances_file))["annotations"]
    dataset_data = json.load(open(dataset_file))

    bbox_set = dict()
    for annotation in annotation_data:
        id = annotation['id']
        bbox = annotation['bbox']
        bbox_set[id]=bbox

    dataset = []
    for data in dataset_data:
        """
        data: 
            image_id(str): image 아이디
            split(enum): 'val'|'train'
            sentences : raw에 full sentence가 저장되어 있음
            ann_id(str): annoset id
        """
        if data["split"] != phase:
            continue
        image_id = data['image_id']
        image_path = os.path.join(image_dir, f'COCO_train2014_{image_id:012}.jpg')
        ann_id = data['ann_id']
        bbox = bbox_set[ann_id]
        sentences = data["sentences"]
        for sentence_data in sentences:
            sentence = sentence_data["raw"]
            vgInfo = dict(
                image_id=image_id,
                image_path=image_path,
                sentence=sentence,
                bbox=bbox
            )
            dataset.append(vgInfo)
    return dataset


def vqa_processing(
    image_dir: str,
    annotation_file: str,
    question_file: str,
    valid_answer_set,
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

if __name__ == '__main__':
    dataset = visual_grounding_processing(
        'val',
        './datasets/Images/train2014',
        './datasets/visual_ground/dataset.json',
        './datasets/visual_ground/instances.json'
    )
