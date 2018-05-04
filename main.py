import torch
import torch.nn as nn
import torch.optim as optim
import helper.datahelper as datahelper
from lenet5 import LeNet5

if __name__ == '__main__':

    # hyper parameters
    channels = 1
    epochs = 10
    image_size = 32
    batch_size = 100
    classes = 10
    lr = 0.001

    # get data
    train_data_loader = datahelper.get_mnist_train_data_loader(
        image_size=image_size,
        batch_size=batch_size
    )

    test_data_loader = datahelper.get_mnist_test_data_loader(
        image_size=image_size,
        batch_size=batch_size
    )

    # instantiate model
    model = LeNet5(channels, classes, act='relu')

    # instantiate loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for (batch, labels) in train_data_loader:

            # empty the gradients of the optimizer
            optimizer.zero_grad()

            # forward pass
            output = model(batch)

            # compute loss
            loss = criterion(output, labels)

            # compute gradients using backpropagation
            loss.backward()

            # update parameters using optimizer.step()
            optimizer.step()

        # print loss
        print(
            'Epoch [%d/%d], Loss: %.4f'
            % (epoch + 1, epochs, loss.item())
        )

    correct = 0
    total = 0

    for batch, labels in test_data_loader:

        output = model(batch)

        # get the max for each instance in the batch
        _, output = torch.max(output.data, dim=1)
        total += labels.size()[0]

        correct += torch.sum(torch.eq(labels, output))

    correct = correct.item()
    accuracy = correct / total * 100
    print('Accuracy:', str(accuracy))
