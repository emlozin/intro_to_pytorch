from intro_to_pytorch import helper, data

import os
import matplotlib.pyplot as plt
import time
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from IPython.display import display, Markdown

MODEL_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'workshop_model.pt')

IMAGE_SIZE = (28, 28)
BATCH_SIZE = 64


def get_default_model():
    input_size = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    hidden_sizes = [128, 64]
    output_size = 10

    return nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[1], output_size),
                         nn.LogSoftmax(dim=1))


def load_data(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],
                             std=[0.5]),
    ])

    trainset = datasets.MNIST(data.DATA_PATH, download=True, train=True, transform=transform)
    testset = datasets.MNIST(data.DATA_PATH, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader


def train_network(network, epochs=30, filename=MODEL_FILENAME):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.003, momentum=0.9)

    (trainloader, testloader) = load_data()

    start_time = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for index, sample in enumerate(trainloader, 0):
            # get the inputs; sample is a list of [inputs, labels]
            images, labels = sample
            images = images.view(images.shape[0], -1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f"Epoch: {epoch + 1}\tloss:{running_loss / len(trainloader):.5f}")

    print(f"Finished Training in {time.time() - start_time}")
    torch.save(network, filename)
    print(network)
    with torch.no_grad():
        show_classify(testloader, network, BATCH_SIZE, 8)


def test_network(filename=MODEL_FILENAME, show_size=64, no_cols=8):
    (_, testloader) = load_data()
    model = torch.load(filename)
    print(model)
    with torch.no_grad():
        show_classify(testloader, model, show_size, no_cols)


def show_classify(testloader, model, show_size=18, no_cols=4):
    with torch.no_grad():
        dataiter = iter(testloader)
        images, labels = dataiter.next()

        show_size = helper.validate_show_size(images, show_size)
        axes = helper.get_axes(no_cols, show_size * 2)

        images = images.view(images.shape[0], -1)

        outputs = model(images)
        outputs = torch.exp(outputs)
        outputs = list(outputs.numpy().squeeze())

        images.resize_(len(images), 1, IMAGE_SIZE[0] * IMAGE_SIZE[1])

        show_output_mnist(axes, images, labels, outputs)


def show_output_mnist(axes, images, labels, outputs):
    for index, axe in enumerate(axes.ravel()):
        output = outputs[index // 2]
        if not index % 2:
            img = images[index // 2]
            axe.imshow(img.view(1, IMAGE_SIZE[0], IMAGE_SIZE[1]).numpy().squeeze())
            axe.set_title(f"Correct: {labels[index // 2]}", fontsize=18, fontweight="bold")
        else:
            label = output.argmax()
            helper.plot_prediction(axe, output, label)
    plt.tight_layout()


def show_data(trainloader, show_size=20, no_cols=4):
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    show_size = helper.validate_show_size(images, show_size)
    axes = helper.get_axes(no_cols, show_size)

    images.resize_(len(images), 1, IMAGE_SIZE[0] * IMAGE_SIZE[1])

    for index, axe in enumerate(axes.ravel()):
        img = images[index]
        axe.imshow(img.view(1, IMAGE_SIZE[0], IMAGE_SIZE[1]).numpy().squeeze())
        axe.set_title(f"Label: {labels[index]}", fontsize=18, fontweight="bold")

    plt.tight_layout()


def show_batch_of_data(x, y):
    display(Markdown(f"**targets:** {y}"))
    plt.imshow(x.transpose(1, 0).reshape(-1, x.shape[0]*28))
    plt.show()


def show_first_n_batches(dl, n):
    for i, (x, y) in enumerate(dl):
        if i >= n: break
        show_batch_of_data(x, y)


if __name__ == "__main__":
    net = get_default_model()
    train_network(net, epochs=5, filename='custom_model.pt')
