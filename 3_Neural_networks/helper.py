import matplotlib.pyplot as plt
import numpy as np

def plot_multiple(input_x, plots):
    # source: https://github.com/udacity/deep-learning-v2-pytorch
    fig, axes = plt.subplots(figsize=(10,4), ncols=len(plots))
    for index, plot_name in enumerate(plots):
        axes[index].plot(input_x, plots[plot_name])
        axes[index].grid(True)
        axes[index].set_title(plot_name, fontweight="bold")
    plt.tight_layout()

# Code below:
# source: https://github.com/udacity/deep-learning-v2-pytorch
def test_network(network, trainloader):
    # Grab some data
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Resize images into a 1D vector, new shape is (batch size, color channels, image pixels)
    images.resize_(64, 1, 784)
    # or images.resize_(images.shape[0], 1, 784) to automatically get batch size

    # Forward pass through the network
    img_idx = 0
    ps = network.forward(images[img_idx, :])

    img = images[img_idx]
    view_classify(img.view(1, 28, 28), ps)

    return True

def view_classify(image, output, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    output = output.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(image.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), output)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
