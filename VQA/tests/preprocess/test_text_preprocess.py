import os
from src.preprocess import text_preprocess
from src.utils.text_helper import load_str_list


def test_make_vocab_questions():
    text_dir = os.path.join(".", "tests", "preprocess", "text_example", "question")

    text_preprocess.make_vocab_questions(text_dir, text_dir)

    x = load_str_list(os.path.join(text_dir, "vocab_questions.txt"))
    y = load_str_list(os.path.join(text_dir, "question_example_correct.txt"))

    os.remove(os.path.join(text_dir, "vocab_questions.txt"))

    x.sort()
    y.sort()

    assert x == y


def test_make_vocab_answers():
    text_dir = os.path.join(".", "tests", "preprocess", "text_example", "answer")

    text_preprocess.make_vocab_answers(text_dir, 10, text_dir)

    x = load_str_list(os.path.join(text_dir, "vocab_answers.txt"))
    y = load_str_list(os.path.join(text_dir, "answer_example_correct.txt"))

    os.remove(os.path.join(text_dir, "vocab_answers.txt"))

    x.sort()
    y.sort()

    assert x == y
