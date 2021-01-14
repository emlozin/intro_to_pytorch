import torch

from intro_to_pytorch import helper

import matplotlib.pyplot as plt
import numpy as np


IMAGE_SIZE = (32, 32, 3)
BATCH_SIZE = 64

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def show_data(trainloader, show_size=10, no_cols=5):
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    show_size = helper.validate_show_size(images, show_size)
    axes = helper.get_axes(no_cols, show_size)

    for index, axe in enumerate(axes.ravel()):
        img = images[index]
        img = img / 2 + 0.5
        axe.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        axe.axis('off')
        axe.set_title(f"Label: {CIFAR10_CLASSES[labels[index]]}", fontsize=18, fontweight="bold")

    plt.tight_layout()


def show_classify(testloader, model, show_size=18, no_cols=4):
    with torch.no_grad():
        dataiter = iter(testloader)
        images, labels = dataiter.next()

        show_size = helper.validate_show_size(images, show_size)
        axes = helper.get_axes(no_cols, show_size)

        outputs = model.forward(images)
        show_output_cifar(axes, images, labels, outputs)


def show_output_cifar(axes, images, labels, outputs):
    for index, axe in enumerate(axes.ravel()):
        output = outputs[index // 2]
        if not index % 2:
            img = images[index // 2]
            img = img / 2 + 0.5
            axe.imshow(np.transpose(img.numpy().squeeze(), (1, 2, 0)))
            axe.set_title(f"Correct: {CIFAR10_CLASSES[labels[index // 2]]}", fontsize=18, fontweight="bold")
        else:
            label = CIFAR10_CLASSES[output.argmax()]
            helper.plot_prediction(axe, output, label, labels=CIFAR10_CLASSES)
    plt.tight_layout()


def accuracy(preds, target):
    return (preds.max(-1)[1] == target).float().mean()


def train_nn(model, loss, testloader, trainloader, n_epochs, lr):
    train_losses = np.array([])
    test_losses = np.array([])
    accuracies = np.array([])

    for epoch in range(n_epochs):
        for x, y in trainloader:
            train_loss = loss(model(x), y)
            train_loss.backward()
            train_losses = np.append(train_losses, train_loss.item())

            with torch.no_grad():
                for p in model.parameters():
                    p += - lr * p.grad
                    p.grad.data.zero_()

        test_loss, acc = evaluate_nn(model, loss, testloader)
        test_losses = np.append(test_losses, test_loss)
        accuracies = np.append(accuracies, acc)

        print(
            f"Epoch: {epoch+1} \t "
            f"Training loss: {train_losses[-1]:.5f} \t "
            f"Test loss: {test_losses[-1]:.5f} \t "
            f"Test accuracy: {accuracies[-1]:.5f}"
        )

    return train_losses, test_losses, accuracies


def evaluate_nn(model, loss, testloader):
    preds = torch.tensor([])
    targets = torch.tensor([]).long()

    with torch.no_grad():
        for x, y in testloader:
            targets = torch.cat([targets, y])
            preds = torch.cat([preds, model(x)])
        test_loss = loss(preds, targets)
    return test_loss.item(), accuracy(preds, targets).item()


def plot_metrics(train_losses, test_losses, accuracies):
    fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, figsize=(15, 5))

    x = np.array(range(len(train_losses)))
    iterations_per_epoch = int(len(train_losses) / len(test_losses))
    x_val = x[iterations_per_epoch - 1:: iterations_per_epoch]
    ax0.plot(x, train_losses, label='train')
    ax0.plot(x_val, test_losses, label='test')

    ax0.legend()
    ax0.set_ylabel("Loss")
    ax0.set_xlabel("Iteration")

    ax1.set_ylabel("Accuracy")
    ax1.plot(x_val, accuracies)
    ax1.set_xlabel("Iteration")
    plt.tight_layout()
