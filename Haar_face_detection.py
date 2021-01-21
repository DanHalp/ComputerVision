import numpy as np
import cv2 as cv


def integral_matrix(m):

    """
    Create The integral matrix of the original matrix 'm'.
    :param m: a batch of matrices of size (N, H, C)
    :return: the integral matrix for each original matrix in the batch of size (N, H, C)
    """
    assert len(m.shape) < 3, "input is a batch of matrices of size (N, H, C)"

    n, h, w = m.shape
    res = np.zeros((n, h + 1, w + 1))
    _, h, w = m.shape
    for i in range(h):
        x = i + 1
        for j in range(w):
            y = j + 1
            res[:, x, y] = res[:, x, j] + res[:, i, y] - res[:, i, j] + m[:, i, j]
    return res[:, 1:, 1:]

def haar():
    pass


