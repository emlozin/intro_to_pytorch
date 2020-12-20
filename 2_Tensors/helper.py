import torch
from torch import nn
import matplotlib.pyplot as plt


def linear_model(x, params):
    return params[0]*x + params[1]


def noise(signal):
    n = torch.zeros(signal.shape)
    torch.nn.init.normal_(n)
    return n


def mse(p, t):
    return (p - t).pow(2).mean()


def show_fit(x, p, t):
    fig, ax = plt.subplots(1, figsize=(7, 7))
    ax.plot(x.numpy(), t.numpy(), marker="o", linewidth=0)
    ax.plot(x.numpy(), p.numpy())


def test_mse(mse_func):
    predictions = torch.rand(10)
    target = torch.rand(10)

    loss = nn.MSELoss()
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
