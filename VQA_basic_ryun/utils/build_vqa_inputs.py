from msilib.schema import _Validation_records
import numpy as np
import json
import os
import argparse
import text_helper
from collections import defaultdict

def extract_answers(q_answers, valid_answer_set):
    all_answers = [answer['answer'] for answer in q_answers]
    valid_answers = [a for a in all_answers if a in valid_answer_set]   
    return all_answers, valid_answers

def vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, image_set):
    print(f'building VQA {image_set} dataset')
    if image_set in ['train2014', 'val2014']:
        load_answer = True
        with open(annotation_file % image_set) as f:
            annotations = json.load(f)['annotations']
            qid2ann_dict = {ann['question_id']: ann for ann in annotations}
    else:
        load_answer = False
    with open(question_file % image_set) as f:
        questions = json.load(f)['questions']
    coco_set_name = image_set.replace('-dev', '')
    abs_image_dir = os.path.abspath(image_dir % coco_set_name)
    image_name_template = 'COCO_'+coco_set_name+'_%012d'
    dataset = [None]*len(questions)

    unk_ans_count = 0
    for idx_q, q in enumerate(questions):
        if (idx_q + 1) % 10000 == 0:
            print(f'processing {idx_q + 1} / {len(questions)}')
        image_id = q['image_id']
        question_id = q['question_id']
        image_name = image_name_template % image_id
        image_path = os.path.join(abs_image_dir, image_name+'.jpg')
        question_str = q['question']
        question_tokens = text_helper.tokenize(question_str)
    
        img_info = dict(image_name=image_name,
                      image_path=image_path,
                      question_id=question_id,
                      question_str=question_str,
                      question_tokens=question_tokens)
    
        if load_answer:
            ann = qid2ann_dict[question_id]
            all_answers, valid_answers = extract_answers(ann['answers'], valid_answer_set)
            if len(valid_answers) == 0:
                valid_answers = ['<unk>']
                unk_ans_count += 1
            img_info['all_answers'] = all_answers
            img_info['valid_answers'] = valid_answers
            
        dataset[idx_q] = img_info
    print(f'total {unk_ans_count} out of {len(questions)} answers are <unk>')
    return dataset

def main(args):
    
    image_dir = args.input_dir+'/Resized_Images/%s/'
    annotation_file = args.input_dir+'/Annotations/v2_mscoco_%s_annotations.json'
    question_file = args.input_dir+'/Questions/v2_OpenEnded_mscoco_%s_questions.json'

    vocab_answer_file = args.output_dir+'/vocab_answers.txt'
    answer_dict = text_helper.VocabDict(vocab_answer_file)
    valid_answer_set = set(answer_dict.word_list)    
    
    train = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'train2014')
    valid = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'val2014')
    test = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'test2015')
    test_dev = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'test-dev2015')
    
    np.save(args.output_dir+'/train.npy', np.array(train))
    np.save(args.output_dir+'/valid.npy', np.array(valid))
    np.save(args.output_dir+'/train_valid.npy', np.array(train+valid))
    np.save(args.output_dir+'/test.npy', np.array(test))
    np.save(args.output_dir+'/test-dev.npy', np.array(test_dev))

# interpreter에서 직접 실행하는 경우에만 if 문 내의 코드를 돌리라는 문법
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='./datasets/VQA',
                        help='directory for inputs')

    parser.add_argument('--output_dir', type=str, default='./datasets',
                        help='directory for outputs')
    
    args = parser.parse_args()
    main(args)