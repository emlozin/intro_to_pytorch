import torch


def test_mse(mse_func):
    predictions = torch.rand(10)
    target = torch.rand(10)

    loss = torch.nn.MSELoss()
    try:
        assert torch.isclose(mse_func(predictions, target), loss(predictions, target))
        print('Congratulations, you defined mean squared errors method correctly!')
    except Exception:
        print('Your method is not calculating MSE correctly. Keep trying!')


def test_indexing(tensor, actual):
    expected = tensor.reshape(-1)[::4]
    try:
        assert torch.allclose(actual, expected)
        print('Correct answer, congratulations!')
    except Exception:
        print('Incorrect answer. Keep trying!')


def test_attributes(tensor):
    try:
        print(f"Tensor size:\t{tensor.size()}")
        print(f"Tensor storage_offset:\t{tensor.storage_offset()}")
        print(f"Tensor stride:\t{tensor.stride()}")
    except Exception:
        print('Define your tensor!')