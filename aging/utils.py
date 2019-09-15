import os
import shutil
from typing import List

import numpy as np
import torch

ORIGIN_DIR = 'data/UTKFace'
DESTINATION_DIR = 'data/renamed'


def rename_files():
    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)
    else:
        x = input("Do you want to rewrite the folder? Y/N").lower()
        if x == 'y':
            shutil.rmtree(DESTINATION_DIR, ignore_errors=True)
            os.makedirs(DESTINATION_DIR)
        elif x == 'n':
            print("You chose not to delete the folder")
        else:
            print("Please give Y/N response")
    for index, file in enumerate(os.listdir(ORIGIN_DIR)):
        # retrieving age and gender from file names
        try:
            age, gender, _, _ = file.split('_')
        except ValueError:
            age, gender, _ = file.split('_')
        destination_name = '.'.join(['_'.join([str(age), str(gender), str(index)]), 'jpg'])
        os.rename(os.path.join(ORIGIN_DIR, file), os.path.join(DESTINATION_DIR, destination_name))

losses = {'EG_L1_loss_full': EG_L1_loss_full,
'D_img_loss_full': D_img_loss_full,
'Ez_loss_full': Ez_loss_full,
'G_tv_loss_full': G_tv_loss_full,
'EG_loss_full': EG_loss_full }

def convert_age(age: int, interval: List[int] = list(np.linspace(10, 120, 10))) -> int:
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
        label = label.view(-1, 1)
        one_hot = zeros.scatter_(1, label, 1)
    return one_hot


def total_variation_loss(image: np.ndarray) -> float:
    """
    Calculates total variation loss which allows loss function to account for noise reduction in generated images.
    :param image: Input image to calculate loss from
    :return: Value of a loss
    """
    rows = (image[:, :, :, 1:] - image[:, :, :, :-1]) ** 2
    columns = (image[:, :, 1:, :] - image[:, :, :-1:, :]) ** 2
    loss = (rows.mean(dim=3) + columns.mean(dim=2)).mean()
    return loss
