# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "torch",
#     "torchdiffeq",
# ]
# ///
"""
Pendulum ODE Simulation and Visualization

This script numerically solves the equations of motion for a simple pendulum
using torchdiffeq.  It supports configuration of initial conditions and
simulation parameters via command-line arguments.  The results are visualized
as time series and phase space plots using matplotlib.

Features:
- Configurable initial angle (phi0), angular velocity (psi0), total time (T),
and number of time points (N) - Uses a dataclass for parameter management
- Plots both time evolution and phase space trajectory
"""

import argparse
import math
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torchdiffeq import odeint


@dataclass
class PendulumParams:
    """
    Container for pendulum simulation parameters.

    Attributes:
        phi0 (float): Initial angle (phi).
        psi0 (float): Initial angular velocity (psi).
        T (float): Total simulation time.
        N (int): Number of time points.
    """

    phi0: float
    psi0: float

    T: float
    N: int


def rhs_psi(phi: float) -> float:
    """
    Compute the time derivative of psi (angular velocity) for the pendulum.

    Args:
        phi (float): Current angle.
    Returns:
        float: Time derivative of psi.
    """
    return -math.sin(phi)


def rhs_phi(psi: float) -> float:
    """
    Compute the time derivative of phi (angle) for the pendulum.

    Args:
        psi (float): Current angular velocity.
    Returns:
        float: Time derivative of phi.
    """
    return psi


def solve_pendulum(params: PendulumParams) -> Tuple[Tensor, Tensor]:
    """
    Solve the pendulum ODE system for given parameters.

    Args:
        params (PendulumParams): Simulation parameters.
    Returns:
        Tuple[Tensor, Tensor]: Time array and solution tensor (shape: [N, 2]).
    """
    t: Tensor = torch.linspace(0, params.T, params.N)
    y0: Tensor = torch.tensor([params.phi0, params.psi0], dtype=torch.float64)

    def system(t: Tensor, y: Tensor) -> Tensor:
        """
        ODE system for the pendulum.
        Args:
            t (Tensor): Current time (unused).
            y (Tensor): State vector [phi, psi].
        Returns:
            Tensor: Time derivatives [dphi/dt, dpsi/dt].
        """
        phi: float = y[0].item()
        psi: float = y[1].item()
        return torch.tensor([rhs_phi(psi), rhs_psi(phi)], dtype=torch.float64)

    solution: Tensor = odeint(system, y0, t)
    return t, solution


def plot_pendulum(t: Tensor, solution: Tensor) -> None:
    """
    Plot phi and psi as functions of time.

    Args:
        t (Tensor): Time array.
        solution (Tensor): Solution tensor with columns [phi, psi].
    """
    phi: Tensor = solution[:, 0]
    psi: Tensor = solution[:, 1]
    plt.figure()
    plt.plot(t, phi, label="phi")
    plt.plot(t, psi, label="psi")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Pendulum: phi and psi over time")
    plt.show()


def plot_phase_space(solution: Tensor) -> None:
    """
    Plot the phase space trajectory (psi vs phi) of the pendulum.

    Args:
        solution (Tensor): Solution tensor with columns [phi, psi].
    """
    phi: Tensor = solution[:, 0]
    psi: Tensor = solution[:, 1]
    plt.figure()
    plt.plot(phi, psi)
    plt.xlabel("phi")
    plt.ylabel("psi")
    plt.title("Pendulum Phase Space: psi vs phi")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    """
    Parse command-line arguments and run the pendulum simulation and plots.
    Allows configuration of phi0, psi0, T, and N via command line, with
    defaults.
    """
    parser = argparse.ArgumentParser(description="Simulate a simple pendulum.")
    parser.add_argument(
        "--phi0",
        type=float,
        default=0.0,
        help="Initial angle phi0 (default: 0.0)",
    )
    parser.add_argument(
        "--psi0",
        type=float,
        default=2.0,
        help="Initial angular velocity psi0 (default: 2.0)",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=10.0,
        help="Total simulation time T (default: 10.0)",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=1000,
        help="Number of time points N (default: 1000)",
    )
    args = parser.parse_args()

    params = PendulumParams(phi0=args.phi0, psi0=args.psi0, T=args.T, N=args.N)

    t, solution = solve_pendulum(params)
    plot_pendulum(t, solution)
    plot_phase_space(solution)
