import numpy as np
import torch


def face_normals(
    V : torch.Tensor,
    F : torch.Tensor,
):

    """
    Compute face normals.

    Args:
        V (torch.FloatTensor): [V, 3]
        F (torch.LongTensor): [F, 3]

    Returns:
        (torch.FloatTensor): [F, 3]
    """

    mesh = V[F]

    vec_a = mesh[:, 0] - mesh[:, 1]
    vec_b = mesh[:, 1] - mesh[:, 2]
    if type(vec_a) == np.ndarray:
        vec_a = torch.from_numpy(vec_a)
    if type(vec_b) == np.ndarray:
        vec_b = torch.from_numpy(vec_b)
    normals = torch.cross(vec_a, vec_b)
    return normals
