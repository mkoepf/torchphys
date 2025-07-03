import torch

from dballpoints import sum_inverse_distances


def test_sum_inverse_distances_simple():
    # 3 points in 1D: [0], [1], [3]
    points = torch.tensor([[0.0], [1.0], [3.0]])
    # k=2 (each point's 2 nearest neighbors)
    neighbors = torch.tensor([[1, 2], [0, 2], [1, 0]])
    # Manually compute distances:
    # For [0]: to [1]=1, to [3]=3 => sum=1/1+1/3=1.333...
    # For [1]: to [0]=1, to [3]=2 => sum=1/1+1/2=1.5
    # For [3]: to [1]=2, to [0]=3 => sum=1/2+1/3=0.833...
    expected = torch.tensor([1.333333, 1.5, 0.833333])
    result = sum_inverse_distances(points, neighbors)
    assert torch.allclose(
        result, expected, atol=1e-5
    ), f"Expected {expected}, got {result}"


def test_sum_inverse_distances_identical_points():
    # All points at the same location
    points = torch.zeros(4, 2)
    neighbors = torch.tensor([[1, 2], [0, 2], [0, 1], [0, 1]])
    # All distances are zero, but should be clamped to 1e-12
    result = sum_inverse_distances(points, neighbors)
    expected = torch.full((4,), 2.0 / 1e-12)
    assert torch.allclose(
        result, expected
    ), f"Expected {expected}, got {result}"


def test_sum_inverse_distances_2d():
    # 3 points in 2D: [0,0], [1,0], [0,1]
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    neighbors = torch.tensor([[1, 2], [0, 2], [0, 1]])
    # For [0,0]: to [1,0]=1, to [0,1]=1 => sum=1+1=2
    # For [1,0]: to [0,0]=1, to [0,1]=sqrt(2) => sum=1+1/sqrt(2)
    # For [0,1]: to [0,0]=1, to [1,0]=sqrt(2) => sum=1+1/sqrt(2)
    expected = torch.tensor([2.0, 1 + 1 / 2**0.5, 1 + 1 / 2**0.5])
    result = sum_inverse_distances(points, neighbors)
    assert torch.allclose(
        result, expected, atol=1e-5
    ), f"Expected {expected}, got {result}"
