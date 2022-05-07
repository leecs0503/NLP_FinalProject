from src.trainer import acc_open_ended
import torch


def test_acc_open_ended():

    pred_exp = torch.Tensor([1, 2, 3, 4, 2, 3, 4])

    answer = torch.Tensor(
        [
            [1, 3, 2, 4, 2, 1, 7],
            [2, 4, 1, 4, 2, 3, 1],
            [5, 2, 4, 4, 6, 3, 4],
            [1, 5, 2, 4, 2, 2, 5],
        ]
    )

    expected_result = 2.0 / 3 + 1.0 / 3 + 0 + 1 + 1 + 2.0 / 3 + 1.0 / 3

    Y = acc_open_ended(pred_exp, answer)

    assert abs(Y - expected_result) < 1e-6
