import os
from src.utils.vocab_dict import VocabDict
from src.dataloader import load_data_loader
import numpy as np
import torch


def test_load_data_loader():
    dataset_path = os.path.join(".", "tests", "dataloader", "datasets")

    # dataset/vocab_questions.txt: [ <pad>, <unk>, a, b, ab, c, d ]
    qst_vocab = VocabDict(os.path.join(dataset_path, "vocab_questions.txt"))
    # dataset/vocab_answers.txt: [<unk>, yes, no, almost]
    ans_vocab = VocabDict(os.path.join(dataset_path, "vocab_answers.txt"))

    data = [
        {
            "image_name": "test1.jpg",
            "image_path": os.path.join(dataset_path, "resize_image", "test1.jpg"),
            "question_id": "41526",
            "question_str": "a b ab a",
            "question_tokens": ["a", "b", "ab", "a"],
            "all_answers": ["xxx"],
            "valid_answers": ["<unk>"],
        },
        {
            "image_name": "test2.jpg",
            "image_path": os.path.join(dataset_path, "resize_image", "test2.jpg"),
            "question_id": "41527",
            "question_str": "ab a ab c z",
            "question_tokens": ["ab", "a", "ab", "c", "z"],
            "all_answers": ["yes"],
            "valid_answers": ["yes"],
        },
        {
            "image_name": "test2.jpg",
            "image_path": os.path.join(dataset_path, "resize_image", "test2.jpg"),
            "question_id": "41527",
            "question_str": "a",
            "question_tokens": ["a"],
            "all_answers": ["no", "xxx"],
            "valid_answers": ["no"],
        },
    ]
    data_path = os.path.join(dataset_path, "train.npy")
    np.save(data_path, np.array(data))

    data_loader = load_data_loader(
        data_path=data_path,
        qst_vocab_dict=qst_vocab,
        ans_vocab_dict=ans_vocab,
        max_qst_length=5,
        max_num_ans=2,
        batch_size=2,
        num_workers=1,
        shuffle=False,
    )
    expected_tensor = {
        "image_shape": [torch.Size([2, 3, 224, 224]), torch.Size([1, 3, 224, 224])],
        "question": [  # vocab_questions: [ <pad>, <unk>, a, b, ab, c, d ]
            torch.LongTensor(
                [
                    [2, 3, 4, 2, 0],  # question_tokens: ["a", "b", "ab", "a"]
                    [4, 2, 4, 5, 1],  # question_tokens: ["ab", "a", "ab", "c", "z"]
                ]
            ),
            torch.LongTensor(
                [
                    [2, 0, 0, 0, 0],  # question_tokens: ["a"]
                ]
            ),
        ],
        # vocab_answers.txt: [<unk>, yes, no, almost]
        "label": [torch.LongTensor([0, 1]), torch.LongTensor([2])],
        "multi_choice": [
            [torch.LongTensor([0, 1]), torch.LongTensor([-1, -1])],
            [torch.LongTensor([2]), torch.LongTensor([-1])],
        ],
    }
    for batch_index, batch_sample in enumerate(data_loader):
        image = batch_sample["image"]
        question = batch_sample["question"]
        label = batch_sample["answer_label"]
        multi_choice = batch_sample["answer_multi_choice"]

        assert image.shape == expected_tensor["image_shape"][batch_index]
        assert torch.equal(question, expected_tensor["question"][batch_index])
        assert torch.equal(label, expected_tensor["label"][batch_index])
        for choice, expected_choice in zip(
            multi_choice, expected_tensor["multi_choice"][batch_index]
        ):
            assert torch.equal(choice, expected_choice)
