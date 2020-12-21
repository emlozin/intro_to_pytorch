import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms


input_size = 28 * 28
hidden_sizes = [128, 64]
output_size = 10
batch_size = 64


test_model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))


def load_data():
    # Transforms define which steps will be applied to each sample
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],
                             std=[0.5]),
    ])

    # Download and load the training data
    trainset = datasets.MNIST('../data', download=True, train=True, transform=transform)
    testset = datasets.MNIST('../data', download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader


def train_network(net=test_model, epochs=30, filename='workshop_model.pt'):

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

    (trainloader, testloader) = load_data()

    start_time = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data
            images = images.view(images.shape[0], -1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f"Epoch: {epoch + 1}\tloss:{running_loss/len(trainloader):.5f}")

    print(f"Finished Training in {time.time() - start_time}")
    torch.save(net, filename)
    print(net)
    with torch.no_grad():
        view_classify(testloader, net, 64, 8)


def test_network(filename='workshop_model.pt', show_size=18, no_cols=4):
    (_, testloader) = load_data()
    net = torch.load(filename)
    print(net)
    with torch.no_grad():
        view_classify(testloader, net, show_size, no_cols)


def view_classify(testloader, net, show_size=18, no_cols=4):
    '''
    Function for viewing an image and it's predicted classes.
    '''
    if show_size > batch_size:
        print(f"!!Warning!!\nSize greater than batch size, will use {batch_size} instead...")
        show_size = batch_size
    no_cols = no_cols
    no_rows = show_size//no_cols
    # Grab some data
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images = images.view(images.shape[0], -1)

    outputs = net(images)
    outputs = torch.exp(outputs)
    outputs = list(outputs.numpy().squeeze())

    # Resize images into a 1D vector, new shape is (batch size, color channels, image pixels)
    images.resize_(64, 1, 784)

    fig, axes = plt.subplots(figsize=(no_cols * 3, no_rows * 3), ncols=no_cols, nrows=no_rows)

    for index, axe in enumerate(axes.ravel()):
        output = outputs[index // 2]
        if not index % 2:
            img = images[index // 2]
            axe.imshow(255-img.view(1, 28, 28).numpy().squeeze(), cmap="gray")
            axe.set_title(f"Correct: {labels[index // 2]}", fontsize=18, fontweight="bold")
        else:
            colors = ['g' if element == max(output) else 'b' for element in output]
            axe.barh(np.arange(10), output, color=colors)
            axe.set_aspect(0.1)
            axe.set_yticks(np.arange(10))
            axe.set_yticklabels(np.arange(10))
            axe.set_title(f"Predicted: {output.argmax()}", fontsize=18, fontweight="bold")
            axe.set_xlim(0, 1.1)

    plt.tight_layout()


if __name__ == "__main__":
    train_network(epochs=5, filename='custom_model.pt')
    plt.show()
