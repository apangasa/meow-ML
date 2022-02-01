"""First attempt at GAN for cat meows from https://zenodo.org/record/4008297#.Yd2zZWjMJPY.
Operates on STFT of 1.5 second audio clips."""

import math

import torch
from torch import nn
from torch.nn import functional
import numpy as np

from globals import STFT_SHAPE
import preprocessing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_new_dim(input, padding=0, dilation=1, kernel=1, stride=1):
    return math.floor((input + (2 * padding) - (dilation * (kernel - 1)) - 1) / stride) + 1


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 2048)
        self.fc2 = nn.Linear(2048, 16384)
        self.fc3 = nn.Linear(16384, 66625)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, 1025, 65)
        return nn.Tanh()(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        h, w = STFT_SHAPE

        self.conv1 = nn.Conv2d(1, 6, 5)
        h = compute_new_dim(h, kernel=5)
        w = compute_new_dim(w, kernel=5)

        self.pool1 = nn.MaxPool2d(2, 2)
        h = compute_new_dim(h, kernel=2, stride=2)
        w = compute_new_dim(w, kernel=2, stride=2)

        self.conv2 = nn.Conv2d(6, 12, 5)
        h = compute_new_dim(h, kernel=5)
        w = compute_new_dim(w, kernel=5)

        self.pool2 = nn.MaxPool2d(2, 2)
        h = compute_new_dim(h, kernel=2, stride=2)
        w = compute_new_dim(w, kernel=2, stride=2)

        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12 * h * w, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(functional.relu(self.conv1(x)))
        x = self.pool2(functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.relu(self.fc3(x))
        x = functional.sigmoid(self.fc4(x))
        return x


def main():

    inputs = preprocessing.preprocess_audio()

    Disc = Discriminator()
    Gen = Generator()

    for input in inputs[:5]:
        input = input.reshape(1, 1, 1025, 65)
        input = torch.FloatTensor(input)

        output = Disc(input)
        print(output)

        noise = (torch.rand(1, 128) - 0.5) / 0.5
        sim_input = Gen(noise)
        sim_output = Disc(sim_input)
        print(sim_output)


if __name__ == '__main__':
    main()
