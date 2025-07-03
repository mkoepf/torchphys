import torch
import pytest
from fd_derivatives import (
    first_central_difference_periodic,
    second_central_difference_periodic,
    third_central_difference_periodic,
    fourth_central_difference_periodic,
)


def test_first_central_difference_periodic_sine():
    # f(x) = sin(x), f'(x) = cos(x)
    N = 100
    L = 2 * torch.pi
    x = torch.arange(N, dtype=torch.float64) * L / N
    dx = x[1] - x[0]
    u = torch.sin(x)
    expected = torch.cos(x)
    deriv = first_central_difference_periodic(u, dx)
    max_error = torch.max(torch.abs(deriv - expected)).item()
    assert max_error < 0.07, f"Max error too high: {max_error}"


def test_second_central_difference_periodic_sine():
    # f(x) = sin(x), f''(x) = -sin(x)
    N = 100
    L = 2 * torch.pi
    x = torch.arange(N, dtype=torch.float64) * L / N
    dx = x[1] - x[0]
    u = torch.sin(x)
    expected = -torch.sin(x)
    deriv2 = second_central_difference_periodic(u, dx)
    max_error = torch.max(torch.abs(deriv2 - expected)).item()
    assert max_error < 0.07, f"Max error too high: {max_error}"


def test_third_central_difference_periodic_sine():
    # f(x) = sin(x), f'''(x) = -cos(x)
    N = 100
    L = 2 * torch.pi
    x = torch.arange(N, dtype=torch.float64) * L / N
    dx = x[1] - x[0]
    u = torch.sin(x)
    expected = -torch.cos(x)
    deriv3 = third_central_difference_periodic(u, dx)
    max_error = torch.max(torch.abs(deriv3 - expected)).item()
    assert max_error < 0.2, f"Max error too high: {max_error}"


def test_fourth_central_difference_periodic_sine():
    # f(x) = sin(x), f^(4)(x) = sin(x)
    N = 100
    L = 2 * torch.pi
    x = torch.arange(N, dtype=torch.float64) * L / N
    dx = x[1] - x[0]
    u = torch.sin(x)
    expected = torch.sin(x)
    deriv4 = fourth_central_difference_periodic(u, dx)
    max_error = torch.max(torch.abs(deriv4 - expected)).item()
    assert max_error < 0.5, f"Max error too high: {max_error}"


def test_first_central_difference_periodic_constant():
    u = torch.ones(10)
    dx = 1.0
    deriv = first_central_difference_periodic(u, dx)
    assert torch.allclose(deriv, torch.zeros_like(u))


def test_second_central_difference_periodic_linear():
    # For a linear ramp, the periodic finite difference is only zero in the interior
    u = torch.arange(10, dtype=torch.float64)
    dx = 1.0
    deriv2 = second_central_difference_periodic(u, dx)
    # Only check interior points
    assert torch.allclose(deriv2[1:-1], torch.zeros_like(u[1:-1]))


def test_third_central_difference_periodic_quadratic():
    # f(x) = x^2, f'''(x) = 0
    u = torch.arange(10, dtype=torch.float64) ** 2
    dx = 1.0
    deriv3 = third_central_difference_periodic(u, dx)
    assert torch.allclose(deriv3[2:-2], torch.zeros_like(u[2:-2]))


def test_fourth_central_difference_periodic_cubic():
    # f(x) = x^3, f^(4)(x) = 0
    u = torch.arange(10, dtype=torch.float64) ** 3
    dx = 1.0
    deriv4 = fourth_central_difference_periodic(u, dx)
    assert torch.allclose(deriv4[2:-2], torch.zeros_like(u[2:-2]))
