import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms


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


def plot_prediction(axe, output, label, labels=np.arange(10)):
    colors = ['g' if element == max(output) else 'b' for element in output]
    axe.barh(labels, output, color=colors)
    axe.set_aspect(0.1)
    axe.set_yticks(labels)
    axe.set_yticklabels(labels)
    axe.set_title(f"Predicted: {label}", fontsize=18, fontweight="bold")
    axe.set_xlim(0, 1.1)


def show_filters(filters=None, image_name='../img/lena.png'):
    img = transforms.ToTensor()(PIL.Image.open(image_name)).unsqueeze(0)
    filters = get_default_filters() if not filters else filters

    fig, axes = plt.subplots(figsize=(12, 6), ncols=len(filters) + 1, nrows=1)

    img_axe = axes.ravel()[0]
    img_axe.axis('off')
    img_axe.imshow(np.transpose(img.squeeze(0).numpy(), (1, 2, 0)))
    img_axe.set_title(f"Original", fontsize=18, fontweight="bold")

    for index, key in enumerate(filters):
        axe = axes.ravel()[index+1]
        axe.axis('off')
        filter_img = filters[key](img)
        axe.imshow(np.transpose(filter_img.squeeze(0).numpy(), (1, 2, 0)), cmap='gray')
        axe.set_title(f"{key}", fontsize=18, fontweight="bold")

    plt.tight_layout()


def get_axes(no_cols, show_size):
    no_cols = no_cols
    no_rows = show_size // no_cols
    fig, axes = plt.subplots(figsize=(no_cols * 3, no_rows * 3), ncols=no_cols, nrows=no_rows)
    return axes


def get_default_filters():
    sobel_tensor_x = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).expand([1, 3, 3, 3])
    sobel_tensor_y = torch.tensor([[-1., 0, 1.], [-2., 0., 2.], [-1., 0., 1.]]).expand([1, 3, 3, 3])

    return {
        'Sobel X': (lambda img: torch.nn.functional.conv2d(img, sobel_tensor_x)),
        'Sobel Y': (lambda img: torch.nn.functional.conv2d(img, sobel_tensor_y))
    }


def plot_multiple(input_x, plots):
    fig, axes = plt.subplots(figsize=(10, 4), ncols=len(plots))
    for index, plot_name in enumerate(plots):
        axes[index].plot(input_x, plots[plot_name])
        axes[index].grid(True)
        axes[index].set_title(plot_name, fontweight="bold")
    plt.tight_layout()


def validate_show_size(images, show_size):
    if show_size > len(images):
        print(f"Maximum number of images is batch size {len(images)}! Will use it instead.")
        show_size = len(images)
    return show_size
