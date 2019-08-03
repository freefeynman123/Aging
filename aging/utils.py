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


def dialation_holes(hole_mask, device='cpu'):
    b, ch, h, w = hole_mask.shape
    dilation_conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False).to(device)
    torch.nn.init.constant_(dilation_conv.weight, 1.0)
    with torch.no_grad():
        output_mask = dilation_conv(hole_mask)
    updated_holes = output_mask != 0
    return updated_holes.float()


def total_variation_loss(image, mask, device='cpu'):
    hole_mask = 1 - mask
    dilated_holes = dialation_holes(hole_mask, device)
    colomns_in_Pset = dilated_holes[:, :, :, 1:] * dilated_holes[:, :, :, :-1]
    rows_in_Pset = dilated_holes[:, :, 1:, :] * dilated_holes[:, :, :-1:, :]
    loss = torch.sum(torch.abs(colomns_in_Pset * (image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
           torch.sum(torch.abs(rows_in_Pset * (image[:, :, :1:] - image[:, :, -1:, :])))
    return loss
