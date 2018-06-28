import torch
import torch.nn as nn
from models.relu_function import myRelu
from models.softmax_function import mySoftmax
from models.dropout_function import myDropout
from models.linear_function import myLinear
from torch.nn.init import xavier_uniform

def get_linear_parameters(in_features, out_features):
    """
    Initializes parameters of a linear layer with xavier uniform.

    Parameters
    ----------
    in_features: int
        fan in of the layer.
    out_features: int
        fan out of the layer.

    Returns
    -------
    tuple
        weights: torch.nn.Parameter
            weights of the linear layer.
        bias: torch.nn.Parameter
            bias of the linear layer.
    """

    weights = nn.Parameter(xavier_uniform(torch.randn(in_features, out_features)))
    bias = nn.Parameter(torch.zeros(out_features))

    return weights, bias


class ConvNet(nn.Module):
    """
    A Convolutional Neural Network.
    """

    def __init__(self, n_classes):
        """
        Model constructor.

        Parameters
        ----------
        n_classes: int
            number of output classes.
        """

        super(ConvNet, self).__init__()

        self.n_classes = n_classes

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

        self.init_fc_layers()

        self.linear = myLinear.apply
        self.relu = myRelu.apply
        self.dropout = myDropout.apply
        self.softmax = mySoftmax.apply

    def init_fc_layers(self):

        # Create random weights
        fc1_w, fc1_b = get_linear_parameters(in_features=(128 * 4 * 4), out_features=512)
        fc2_w, fc2_b = get_linear_parameters(in_features=512, out_features=256)
        fc3_w, fc3_b = get_linear_parameters(in_features=256, out_features=self.n_classes)

        # Register
        self.register_parameter('fc1_w', fc1_w)
        self.register_parameter('fc1_b', fc1_b)
        self.register_parameter('fc2_w', fc2_w)
        self.register_parameter('fc2_b', fc2_b)
        self.register_parameter('fc3_w', fc3_w)
        self.register_parameter('fc3_b', fc3_b)

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
        h = self.dropout(h, 0.5, self.training)
        h = self.relu(self.linear(h, self.fc1_w, self.fc1_b))
        h = self.dropout(h, 0.5, self.training)
        h = self.relu(self.linear(h, self.fc2_w, self.fc2_b))
        h = self.softmax(self.linear(h, self.fc3_w, self.fc3_b), 1)

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
