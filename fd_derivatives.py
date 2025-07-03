import torch
from torch import Tensor


def first_central_difference_periodic(u: Tensor, dx: float) -> Tensor:
    """
    Compute the first central finite difference of a 1D periodic tensor.
    Args:
        u (torch.Tensor): 1D tensor of function values.
        dx (float): grid spacing.
    Returns:
        torch.Tensor: 1D tensor of first derivative values.
    """
    return (torch.roll(u, -1) - torch.roll(u, 1)) / (2 * dx)


def second_central_difference_periodic(u: Tensor, dx: float) -> Tensor:
    """
    Compute the second central finite difference of a 1D periodic tensor.
    Args:
        u (torch.Tensor): 1D tensor of function values.
        dx (float): grid spacing.
    Returns:
        torch.Tensor: 1D tensor of second derivative values.
    """
    return (torch.roll(u, -1) - 2 * u + torch.roll(u, 1)) / (dx ** 2)


def third_central_difference_periodic(u: Tensor, dx: float) -> Tensor:
    """
    Compute the third central finite difference of a 1D periodic tensor.
    """
    return (
        -torch.roll(u, 2)
        + 2 * torch.roll(u, 1)
        - 2 * torch.roll(u, -1)
        + torch.roll(u, -2)
    ) / (2 * dx ** 3)


def fourth_central_difference_periodic(u: Tensor, dx: float) -> Tensor:
    """
    Compute the fourth central finite difference of a 1D periodic tensor.
    """
    return (
        torch.roll(u, 2)
        - 4 * torch.roll(u, 1)
        + 6 * u
        - 4 * torch.roll(u, -1)
        + torch.roll(u, -2)
    ) / (dx ** 4)
