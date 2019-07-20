from typing import List
import torch


def convert_age(age: int, interval: List[int] = list(range(10, 130, 10))):
    """
    Converts age to label based on given interval.

    :param age: Age to be encoded.
    :param interval: List with age's interval.
    :return: Index of age.
    """
    for index, value in enumerate(interval):
        if age <= value:
            return index

def index_to_one_hot(index: int, N: int):
    """
    Convets index to one hot encoded tensor.
    :param index: Index to encoded.
    :param N: Number of classes.
    :return: Tensor with one hot encoded index.
    """

    zeros = torch.zeros(index.size(0), N)
    one_hot = zeros.scatter_(1, index)
    return one_hot
