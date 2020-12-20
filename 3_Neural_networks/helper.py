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
    view_classify(img.view(1, 28, 28), labels[0], ps)

    
def view_data(trainloader):
    image_set_size = 8
    # Grab some data
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Resize images into a 1D vector, new shape is (batch size, color channels, image pixels)
    images.resize_(64, 1, 784)

    fig, axes = plt.subplots(figsize=(12, 6), ncols=image_set_size//2, nrows=2)

    for index, axe in enumerate(axes.ravel()):
        img = images[index]
        axe.imshow(img.view(1, 28, 28).numpy().squeeze())
        axe.set_title(f"Label: {labels[index]}", fontsize=18, fontweight="bold")
    
    plt.tight_layout()
    

def view_classify(image, label, output):
    ''' Function for viewing an image and it's predicted classes.
    '''
    output = output.data.numpy().squeeze()
    colors = ['g' if element == max(output) else 'b' for element in output]

    fig, (ax1, ax2) = plt.subplots(figsize=(5, 10), ncols=2)
    ax1.imshow(image.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax1.set_title(f"Correct: {label}", fontsize=18, fontweight="bold")
    ax2.barh(np.arange(10), output, color=colors)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title(f"Predicted: {output.argmax()}", fontsize=18, fontweight="bold")
    ax2.set_xlim(0, 1.1)    
    
    plt.tight_layout()
