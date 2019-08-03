import numpy as np
from typing import List
import torch
from torch import nn


def convert_age(age: int, interval: List[int] = list(np.linspace(10, 120, 10))):
    """
    Converts age to label based on given interval.

    :param age: Age to be encoded.
    :param interval: List with age's interval.
    :return: Index of age.
    """
    for index, value in enumerate(interval):
        if age <= value:
            return index


def index_to_one_hot(label: int, N: int):
    """
    Converts index to one hot encoded tensor.
    :param label: Index to encoded.
    :param N: Number of classes.
    :return: Tensor with one hot encoded index.
    """

    assert torch.max(label).item() < N

    zeros = torch.zeros(label.size(0), N)
    try:
        one_hot = zeros.scatter_(1, label, 1)
    except RuntimeError:
        print("Your label tensor is probably one dimensional, trying to reshape it.")
        label = label.view(-1, 1)
        one_hot = zeros.scatter_(1, label, 1)
    return one_hot


def total_variation_loss(image):
    rows = (image[:, :, :, 1:] - image[:, :, :, :-1])**2
    columns = (image[:, :, 1:, :] - image[:, :, :-1:, :])**2
    loss = (rows.mean(dim=3) + columns.mean(dim=2)).mean()
    return loss
