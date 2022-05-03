import os
import argparse
import numpy as np
import random
from utils import text_helper
from preprocess import text_preprocess
from preprocess import image_preprocess
from preprocess import create_input


def preprocess(args):
    print("1. Preprocessing images...")

    image_input_dir = os.path.join(args.input_dir, "Images")
    image_output_dir = os.path.join(args.output_dir, "Resized_Images")
    image_size = [args.image_size, args.image_size]
    if args.skip_image:
        print("Skipped.")
    else:
        image_preprocess.resize_images(image_input_dir, image_output_dir, image_size)
        print("Done.")

    print("2. Preprocessing text...")

    question_input_dir = os.path.join(args.input_dir, "Questions")
    annotation_input_dir = os.path.join(args.input_dir, "Annotations")
    if args.skip_text:
        print("Skipped")
    else:
        text_preprocess.make_vocab_questions(question_input_dir, args.output_dir)
        text_preprocess.make_vocab_answers(
            annotation_input_dir, args.n_answers, args.output_dir
        )
        print("Done.")

    print("3. Creating VQA inputs...")

    image_dir = os.path.join(image_output_dir, "%s")
    annotation_file = os.path.join(
        annotation_input_dir, "v2_mscoco_%s_annotations.json"
    )
    question_file = os.path.join(
        question_input_dir, "v2_OpenEnded_mscoco_%s_questions.json"
    )

    vocab_answer_file = os.path.join(args.output_dir, "vocab_answers.txt")
    answer_dict = text_helper.VocabDict(vocab_answer_file)
    valid_answer_set = set(answer_dict.word_list)

    train = create_input.vqa_processing(
        image_dir, annotation_file, question_file, valid_answer_set, "train2014"
    )
    valid_original = create_input.vqa_processing(
        image_dir, annotation_file, question_file, valid_answer_set, "val2014"
    )
    test = create_input.vqa_processing(
        image_dir, annotation_file, question_file, valid_answer_set, "test2015"
    )
    test_dev = create_input.vqa_processing(
        image_dir, annotation_file, question_file, valid_answer_set, "test-dev2015"
    )

    index = list(range(len(valid_original)))
    random.shuffle(index)
    valid = [valid_original[i] for o, i in enumerate(index) if o < len(index) / 2]
    valid_test = [valid_original[i] for o, i in enumerate(index) if o >= len(index) / 2]

    np.save(os.path.join(args.vqa_output_dir, "train.npy"), np.array(train))
    np.save(os.path.join(args.vqa_output_dir, "valid.npy"), np.array(valid))
    np.save(os.path.join(args.vqa_output_dir, "valid_test.npy"), np.array(valid_test))
    np.save(os.path.join(args.vqa_output_dir, "test.npy"), np.array(test))
    np.save(os.path.join(args.vqa_output_dir, "test-dev.npy"), np.array(test_dev))

    print("Preprocessing Done.")


def get_argument() -> argparse.Namespace():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join("..", "datasets"),
        help="directory for input",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("..", "datasets"),
        help="directory for output",
    )

    parser.add_argument(
        "--image_size", type=int, default=224, help="size of images after resizing"
    )

    parser.add_argument(
        "--n_answers",
        type=int,
        default=1000,
        help="the number of answers to be kept in vocab",
    )

    parser.add_argument(
        "--vqa_output_dir",
        type=str,
        default=os.path.join("..", "datasets"),
        help="directory for outputs",
    )

    parser.add_argument(
        "--skip_image", type=bool, default=False, help="skip image preprocess"
    )

    parser.add_argument(
        "--skip_text", type=bool, default=False, help="skip text preprocess"
    )

    return parser.parse_args()


def main():
    args = get_argument()
    preprocess(args)


if __name__ == "__main__":
    main()
