# source: https://github.com/udacity/deep-learning-v2-pytorch

import matplotlib.pyplot as plt
import numpy as np


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


def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
