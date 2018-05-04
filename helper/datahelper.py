import torchvision
import torch.utils.data as tdata
from torchvision import transforms


def get_mnist_train_data_loader(image_size=64, batch_size=100):

    # transforms
    # resize image according to image_size
    # converts PIL images to tensors
    transform = transforms.Compose(
        [transforms.Resize(image_size),
         transforms.ToTensor()]
    )

    # gets mnist train dataset
    mnist_train = torchvision.datasets.MNIST(
        '.\data',
        transform=transform,
        download=True
    )

    # instantiates a DataLoader for the mnist_train with batch_size=batch_size
    data_loader = tdata.DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True
    )

    print("MNIST train dataset loaded")

    return data_loader


def get_mnist_test_data_loader(image_size=64, batch_size=100):

    # transforms
    # resize image according to image_size
    # converts PIL images to tensors
    transform = transforms.Compose(
        [transforms.Resize(image_size),
         transforms.ToTensor()]
    )

    # gets mnist test dataset
    mnist_test = torchvision.datasets.MNIST(
        '.\data',
        train=False, transform=transform,
        download=True
    )

    # instantiates a DataLoader for the mnist_train with batch_size=batch_size
    data_loader = tdata.DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=True
    )

    print("MNIST test dataset loaded")

    return data_loader
