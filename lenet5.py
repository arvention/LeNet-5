import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(
        self,
        channels,
        class_count,
        act='relu'
    ):

        super(LeNet5, self).__init__()

        conv_layers = []
        fc_layers = []

        # Conv1 -> out_ch = 6, filter = 5x5, stride = 1
        conv_layers.append(nn.Conv2d(channels, 6, 5))

        # activation
        conv_layers.append(self.get_activation(act))

        # Pool2 -> filter = 2x2, stride = 2
        conv_layers.append(nn.MaxPool2d(2, 2))

        # Conv3 -> out_ch = 16, filter = 5x5, stride = 1
        conv_layers.append(nn.Conv2d(6, 16, 5))

        # activation
        conv_layers.append(self.get_activation(act))

        # Pool4 -> filter = 2x2, stride = 2
        conv_layers.append(nn.MaxPool2d(2, 2))

        # Conv5 -> out_ch = 120, filter = 5x5
        conv_layers.append(nn.Conv2d(16, 120, 5))

        # activation
        conv_layers.append(self.get_activation(act))

        # FC6 -> in_ch = 120, out_ch = 84
        fc_layers.append(nn.Linear(120, 84))

        # activation
        fc_layers.append(self.get_activation(act))

        # FC7 -> in_ch = 84, out_ch = class_count
        fc_layers.append(nn.Linear(84, class_count))

        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(*fc_layers)

    def get_activation(self, act='relu'):

        activation = nn.ReLU(inplace=True)
        if act == 'sigmoid':
            activation = nn.Sigmoid()
        elif act == 'tanh':
            activation = nn.Tanh()

        return activation

    def forward(self, x):
        y = self.conv(x)
        y = y.view(-1, 120)
        return self.fc(y)
