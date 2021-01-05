import torch

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torchvision


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def view_data(trainloader, show_size=10, no_cols=5):
    no_cols = no_cols
    no_rows = show_size//no_cols

    # Grab some data
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    fig, axes = plt.subplots(figsize=(no_cols * 3, no_rows * 3), ncols=no_cols, nrows=no_rows)

    for index, axe in enumerate(axes.ravel()):
        img = images[index]
        img = img / 2 + 0.5
        axe.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        axe.axis('off')
        axe.set_title(f"Label: {classes[labels[index]]}", fontsize=18, fontweight="bold")

    plt.tight_layout()


def get_default_filters():
    sobel_tensor_x = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).expand([1, 3, 3, 3])
    sobel_tensor_y = torch.tensor([[-1., 0, 1.], [-2., 0., 2.], [-1., 0., 1.]]).expand([1, 3, 3, 3])

    return {
        'Sobel X': (lambda img: torch.nn.functional.conv2d(img, sobel_tensor_x)),
        'Sobel Y': (lambda img: torch.nn.functional.conv2d(img, sobel_tensor_y))
    }


def view_filters(filters=None, image_name='../img/lena.png'):
    img = torchvision.transforms.ToTensor()(PIL.Image.open(image_name)).unsqueeze(0)
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


def view_classify(testloader, net, show_size=12, no_cols=4):
    if show_size > 64:
        print(f"!!Warning!!\nSize greater than batch size, will use {batch_size} instead...")
        show_size = 64
    no_cols = no_cols
    no_rows = show_size // no_cols

    # Grab some data
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    outputs = net.forward(images)

    fig, axes = plt.subplots(figsize=(no_cols * 3, no_rows * 3), ncols=no_cols, nrows=no_rows)

    for index, axe in enumerate(axes.ravel()):
        output = outputs[index // 2]
        if not index % 2:
            img = images[index // 2]
            img = img / 2 + 0.5
            axe.imshow(np.transpose(img.numpy().squeeze(), (1, 2, 0)))
            axe.set_title(f"Correct: {classes[labels[index // 2]]}", fontsize=18, fontweight="bold")
        else:
            colors = ['g' if element == max(output) else 'b' for element in output]
            axe.barh(classes, output, color=colors)
            axe.set_aspect(0.1)
            axe.set_yticks(classes)
            axe.set_yticklabels(classes)
            axe.set_title(f"Predicted: {classes[output.argmax()]}", fontsize=18, fontweight="bold")
            axe.set_xlim(0, 1.1)

    plt.tight_layout()
