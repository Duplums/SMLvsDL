# -*- coding: utf-8 -*-
"""
Common functions to spatialy normalize the data.
"""

# Imports
import logging
import numpy as np


# Global parameters
logger = logging.getLogger("SMLvsDL")


def padd(arr, shape, fill_value=0):
    """ Apply a padding.

    Parameters
    ----------
    arr: array
        the input data.
    shape: list of int
        the desired shape.
    fill_value: int, default 0
        the value used to fill the array.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    orig_shape = arr.shape
    padding = []
    for orig_i, final_i in zip(orig_shape, shape):
        shape_i = final_i - orig_i
        half_shape_i = shape_i // 2
        if shape_i % 2 == 0:
            padding.append((half_shape_i, half_shape_i))
        else:
            padding.append((half_shape_i, half_shape_i + 1))
    for cnt in range(len(arr.shape) - len(padding)):
        padding.append((0, 0))
    return np.pad(arr, padding, mode="constant", constant_values=fill_value)


def downsample(arr, scale):
    """ Apply a downsampling.

    Parameters
    ----------
    arr: array
        the input data.
    scale: int
        the downsampling scale factor in all directions.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    slices = []
    for cnt, orig_i in enumerate(arr.shape):
        if cnt == 3:
            break
        slices.append(slice(0, orig_i, scale))
    return arr[tuple(slices)]