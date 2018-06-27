import torch
import torch.nn as nn
from models.relu_function import myRelu
from models.softmax_function import mySoftmax


class ConvNet(nn.Module):
    """
    A Convolutional Neural Network.
    """

    def __init__(self, n_classes=10):
        """
        Model constructor.

        Parameters
        ----------
        n_classes: int
            number of output classes.
        """

        super(ConvNet, self).__init__()

        # Initialize layers
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=(1,1))
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=(1,1))
        self.conv1_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=(1,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=(1,1))
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1))
        self.conv2_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=(1,1))
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=(1,1))
        self.conv3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=(1,1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.fc1 = nn.Linear(in_features=(128 * 4 * 4), out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=n_classes)

        self.relu = myRelu.apply
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = mySoftmax.apply

    def forward(self, x):
        """
        Forward function.

        Parameters
        ----------
        x: torch.FloatTensor
            a pytorch tensor having shape (batchsize, c, h, w).

        Returns
        -------
        o: torch.FloatTensor
            a pytorch tensor having shape (batchsize, n_classes).
        """


        # Extract Features
        h = x
        h = self.relu(self.conv1_1(h))
        h = self.relu(self.conv1_2(h))
        h = self.relu(self.conv1_3(h))
        h = self.pool1(h)
        h = self.relu(self.conv2_1(h))
        h = self.relu(self.conv2_2(h))
        h = self.relu(self.conv2_3(h))
        h = self.pool2(h)
        h = self.relu(self.conv3_1(h))
        h = self.relu(self.conv3_2(h))
        h = self.relu(self.conv3_3(h))
        h = self.pool3(h)

        # Flatten out
        h = h.view(len(x), 128 * 4 * 4)

        # Classify
        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.relu(self.fc2(h))
        h = self.softmax(self.fc3(h))

        o = h

        return o


def crossentropy_loss(y_true, y_pred):
    """
    Crossentropy loss function for classification.

    Parameters
    ----------
    y_true: torch.LongTensor
        tensor holding groundtruth labels. Has shape (batchsize,).
    y_pred: torch.FloatTensor
        tensor holding model predictions. Has shape (batchsize, n_classes).

    Returns
    -------
    ce: torch.FloatTensor
        loss function value.
    """

    ce = - torch.gather(torch.log(y_pred + 1e-5), 1, y_true.unsqueeze(1))
    return torch.mean(ce)
