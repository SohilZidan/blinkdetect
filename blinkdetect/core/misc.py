import numpy as np


def points_in_between(eye: np.ndarray) -> np.ndarray:
    """compute the positions of the middle curve between the two eyelids

    Args:
        eye (np.ndarray): [description]

    Returns:
        np.ndarray: shape of (N, 2)
    """
    frm = 1
    to = 16
    x, y = eye[:, frm:to, 0].reshape(-1), eye[:, frm:to, 1].reshape(-1)
    eye_corners = [tuple(eye[:, 0,0:2].reshape(-1)), tuple(eye[:, 8,0:2].reshape(-1))]

    midx = [eye_corners[0][0]]
    midy = [eye_corners[0][1]]
    midx += [np.mean([x[i], x[i+8]]) for i in range(7)]
    midy += [np.mean([y[i], y[i+8]]) for i in range(7)]
    midx += [eye_corners[1][0]]
    midy += [eye_corners[1][1]]

    return np.array([midx, midy]).T