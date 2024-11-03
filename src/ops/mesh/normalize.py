import torch


def normalize(
    V : torch.Tensor,
    F : torch.Tensor,
    mode : str
):
    """Normalizes a mesh.
    Args:
        V (torch.FloatTensor): Vertices of shape [V, 3]
        F (torch.LongTensor): Faces of shape [F, 3]
        mode (str): Different methods of normalization.
    Returns:
        (torch.FloatTensor, torch.LongTensor):
        - Normalized Vertices
        - Faces
    """

    if mode == 'sphere':

        V_max, _ = torch.max(V, dim=0)
        V_min, _ = torch.min(V, dim=0)
        V_center = (V_max + V_min) / 2.
        V = V - V_center

        # Find the max distance to origin
        max_dist = torch.sqrt(torch.max(torch.sum(V**2, dim=-1)))
        V_scale = 1. / max_dist
        V *= V_scale
        return V, F

    elif mode == 'aabb':

        V_min, _ = torch.min(V, dim=0)
        V = V - V_min

        max_dist = torch.max(V)
        V *= 1.0 / max_dist

        V = V * 2.0 - 1.0
        V -= (torch.max(V, dim=0)[0] + torch.min(V, dim=0)[0]) / 2
 
        return V, F

    elif mode == 'none':

        return V, F
