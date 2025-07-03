import torch

from dballpoints import neighbor_distances


def test_neighbor_distances_1d():
    # 3 points in 1D: [0], [1], [3]
    points = torch.tensor([[0.0], [1.0], [3.0]])
    neighbors = torch.tensor([[1, 2], [0, 2], [1, 0]])
    dists = neighbor_distances(points, neighbors)
    expected = torch.tensor([[1.0, 3.0], [1.0, 2.0], [2.0, 3.0]])
    assert torch.allclose(dists, expected), f"Expected {expected}, got {dists}"


def test_neighbor_distances_2d():
    # 3 points in 2D: [0,0], [1,0], [0,1]
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    neighbors = torch.tensor([[1, 2], [0, 2], [0, 1]])
    dists = neighbor_distances(points, neighbors)
    expected = torch.tensor([[1.0, 1.0], [1.0, 2**0.5], [1.0, 2**0.5]])
    assert torch.allclose(
        dists, expected, atol=1e-6
    ), f"Expected {expected}, got {dists}"


def test_neighbor_distances_identical():
    # All points at the same location
    points = torch.zeros(3, 2)
    neighbors = torch.tensor([[1, 2], [0, 2], [0, 1]])
    dists = neighbor_distances(points, neighbors)
    expected = torch.zeros(3, 2)
    assert torch.allclose(dists, expected), f"Expected {expected}, got {dists}"
