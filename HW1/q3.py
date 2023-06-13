import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

BATCH_SIZE = 50
EPOCH_SIZE = 15
TRAIN_SIZE = 10


class mlp_1(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(mlp_1, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.input_size)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output


class mlp_1_sigmoid(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(mlp_1_sigmoid, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.input_size)
        hidden = self.fc1(x)
        relu = self.sigmoid(hidden)
        output = self.fc2(relu)
        return output


class mlp_2(torch.nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes):
        super(mlp_2, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size2, bias=False)
        self.fc3 = torch.nn.Linear(hidden_size2, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.input_size)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        hidden2 = self.fc2(relu)
        output = self.fc3(hidden2)
        return output


class mlp_2_sigmoid(torch.nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes):
        super(mlp_2_sigmoid, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size2, bias=False)
        self.fc3 = torch.nn.Linear(hidden_size2, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.input_size)
        hidden = self.fc1(x)
        sigmoid = self.sigmoid(hidden)
        hidden2 = self.fc2(sigmoid)
        output = self.fc3(hidden2)
        return output


class cnn_3(torch.nn.Module):
    # Layer Definition
    # https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
    def __init__(self, input_size, num_classes):
        super(cnn_3, self).__init__()
        self.input_size = input_size
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(5, 5), stride=1, padding='valid')
        self.relu2 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(7, 7), stride=1, padding='valid')
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        # https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
        self.fc1 = torch.nn.Linear(in_features=16 * 3 * 3, out_features=num_classes)

    def forward(self, x):
        hidden1 = self.conv1(x)
        relu1 = self.relu1(hidden1)
        hidden2 = self.conv2(relu1)
        relu2 = self.relu2(hidden2)
        pool1 = self.pool1(relu2)
        hidden3 = self.conv3(pool1)
        pool2 = self.pool2(hidden3)
        # Reshaping linear input
        pool2 = pool2.view(BATCH_SIZE, 16 * 3 * 3)
        output = self.fc1(pool2)
        return output


class cnn_3_sigmoid(torch.nn.Module):
    # Layer Definition
    # https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
    def __init__(self, input_size, num_classes):
        super(cnn_3_sigmoid, self).__init__()
        self.input_size = input_size
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.sigmoid1 = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(5, 5), stride=1, padding='valid')
        self.sigmoid2 = torch.nn.Sigmoid()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(7, 7), stride=1, padding='valid')
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        # https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
        self.fc1 = torch.nn.Linear(in_features=16 * 3 * 3, out_features=num_classes)

    def forward(self, x):
        hidden1 = self.conv1(x)
        sigmoid1 = self.sigmoid1(hidden1)
        hidden2 = self.conv2(sigmoid1)
        sigmoid2 = self.sigmoid2(hidden2)
        pool1 = self.pool1(sigmoid2)
        hidden3 = self.conv3(pool1)
        pool2 = self.pool2(hidden3)
        # Reshaping linear input
        pool2 = pool2.view(BATCH_SIZE, 16 * 3 * 3)
        output = self.fc1(pool2)
        return output


class cnn_4_sigmoid(torch.nn.Module):
    # Layer Definition
    # https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
    def __init__(self, input_size, num_classes):
        super(cnn_4_sigmoid, self).__init__()
        self.input_size = input_size
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.sigmoid1 = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.sigmoid2 = torch.nn.Sigmoid()
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), stride=1, padding='valid')
        self.sigmoid3 = torch.nn.Sigmoid()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv4 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=1, padding='valid')
        self.sigmoid4 = torch.nn.Sigmoid()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
        self.fc1 = torch.nn.Linear(in_features=16 * 4 * 4, out_features=num_classes)

    def forward(self, x):
        hidden1 = self.conv1(x)
        sigmoid1 = self.sigmoid1(hidden1)
        hidden2 = self.conv2(sigmoid1)
        sigmoid2 = self.sigmoid2(hidden2)
        hidden3 = self.conv3(sigmoid2)
        sigmoid3 = self.sigmoid3(hidden3)
        pool1 = self.pool1(sigmoid3)
        hidden4 = self.conv4(pool1)
        sigmoid4 = self.sigmoid4(hidden4)
        pool2 = self.pool2(sigmoid4)
        pool2 = pool2.view(50, 16 * 4 * 4)
        output = self.fc1(pool2)
        return output


class cnn_4(torch.nn.Module):
    # Layer Definition
    # https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
    def __init__(self, input_size, num_classes):
        super(cnn_4, self).__init__()
        self.input_size = input_size
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), stride=1, padding='valid')
        self.relu3 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv4 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=1, padding='valid')
        self.relu4 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
        self.fc1 = torch.nn.Linear(in_features=16 * 4 * 4, out_features=num_classes)

    def forward(self, x):
        hidden1 = self.conv1(x)
        relu1 = self.relu1(hidden1)
        hidden2 = self.conv2(relu1)
        relu2 = self.relu2(hidden2)
        hidden3 = self.conv3(relu2)
        relu3 = self.relu3(hidden3)
        pool1 = self.pool1(relu3)
        hidden4 = self.conv4(pool1)
        relu4 = self.relu4(hidden4)
        pool2 = self.pool2(relu4)
        pool2 = pool2.view(50, 16 * 4 * 4)
        output = self.fc1(pool2)
        return output


class cnn_5(torch.nn.Module):
    # Layer Definition
    # https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
    def __init__(self, input_size, num_classes):
        super(cnn_5, self).__init__()
        self.input_size = input_size
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu4 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu5 = torch.nn.ReLU()
        self.conv6 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu6 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        # https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
        self.fc1 = torch.nn.Linear(in_features=8 * 4 * 4, out_features=num_classes)

    def forward(self, x):
        hidden1 = self.conv1(x)
        relu1 = self.relu1(hidden1)
        hidden2 = self.conv2(relu1)
        relu2 = self.relu2(hidden2)
        hidden3 = self.conv3(relu2)
        relu3 = self.relu3(hidden3)
        hidden4 = self.conv4(relu3)
        relu4 = self.relu4(hidden4)
        pool1 = self.pool1(relu4)
        hidden5 = self.conv5(pool1)
        relu5 = self.relu5(hidden5)
        hidden6 = self.conv6(relu5)
        relu6 = self.relu6(hidden6)
        pool2 = self.pool2(relu6)
        # Reshaping linear input
        pool2 = pool2.view(BATCH_SIZE, 8 * 4 * 4)
        output = self.fc1(pool2)
        return output


class cnn_5_sigmoid(torch.nn.Module):
    # Layer Definition
    # https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
    def __init__(self, input_size, num_classes):
        super(cnn_5_sigmoid, self).__init__()
        self.input_size = input_size
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.sigmoid1 = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.sigmoid2 = torch.nn.Sigmoid()
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.sigmoid3 = torch.nn.Sigmoid()
        self.conv4 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.sigmoid4 = torch.nn.Sigmoid()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.sigmoid5 = torch.nn.Sigmoid()
        self.conv6 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.sigmoid6 = torch.nn.Sigmoid()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        # https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
        self.fc1 = torch.nn.Linear(in_features=8 * 4 * 4, out_features=num_classes)

    def forward(self, x):
        hidden1 = self.conv1(x)
        sigmoid1 = self.sigmoid1(hidden1)
        hidden2 = self.conv2(sigmoid1)
        sigmoid2 = self.sigmoid2(hidden2)
        hidden3 = self.conv3(sigmoid2)
        sigmoid3 = self.sigmoid3(hidden3)
        hidden4 = self.conv4(sigmoid3)
        sigmoid4 = self.sigmoid4(hidden4)
        pool1 = self.pool1(sigmoid4)
        hidden5 = self.conv5(pool1)
        sigmoid5 = self.sigmoid5(hidden5)
        hidden6 = self.conv6(sigmoid5)
        sigmoid6 = self.sigmoid6(hidden6)
        pool2 = self.pool2(sigmoid6)
        # Reshaping linear input
        pool2 = pool2.view(BATCH_SIZE, 8 * 4 * 4)
        output = self.fc1(pool2)
        return output
