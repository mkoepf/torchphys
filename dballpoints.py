# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "torch",
# ]
# ///
import argparse
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch


def random_points_in_sphere(
    N: int, d: int, radius: float, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate N random points uniformly inside a d-dimensional sphere of given
    radius.

    Args:
        N (int): Number of points.
        d (int): Dimension (>0).
        radius (float): Sphere radius.
        device: PyTorch device (optional).

    Returns:
        Tensor of shape (N, d) with points inside the sphere.
    """
    # Generate random directions (normal distribution, then normalize)
    x = torch.randn(N, d, device=device)
    x = x / x.norm(dim=1, keepdim=True)
    # Generate random radii with correct distribution
    r = torch.rand(N, 1, device=device).pow(1.0 / d) * radius
    return x * r


def k_nearest_neighbors(points: torch.Tensor, k: int) -> torch.Tensor:
    """
    For each point, find the indices of its k nearest neighbors.
    Args:
        points (Tensor): Shape (N, d)
        k (int): Number of neighbors
    Returns:
        Tensor of shape (N, k) with neighbor indices for each point
    """
    N = points.shape[0]
    # Compute pairwise distances
    dists = torch.cdist(points, points)  # (N, N)
    # Exclude self (set diagonal to large value)
    dists.fill_diagonal_(float("inf"))
    # Get indices of k smallest distances for each point
    neighbors = torch.topk(dists, k, largest=False).indices  # (N, k)
    return neighbors


def neighbor_distances(
    points: torch.Tensor, neighbors: torch.Tensor
) -> torch.Tensor:
    """
    For each point, compute the distances to its k nearest neighbors.
    Args:
        points (Tensor): Shape (N, d)
        neighbors (Tensor): Shape (N, k) with neighbor indices
    Returns:
        Tensor of shape (N, k) with distances
    """
    neighbor_points = points[neighbors]  # (N, k, d)
    points_expanded = points.unsqueeze(1)  # (N, 1, d)
    dists = torch.norm(points_expanded - neighbor_points, dim=2)  # (N, k)
    return dists


def sum_inverse_distances(
    points: torch.Tensor, neighbors: torch.Tensor
) -> torch.Tensor:
    """
    For each point, sum 1.0 / dist(x, x_i) for all x_i in its nearest neighbors
    (with clamping).

    Args:
        points (Tensor): Shape (N, d)
        neighbors (Tensor): Shape (N, k) with neighbor indices
    Returns:
        Tensor of shape (N,) with the sum for each point
    """
    dists = neighbor_distances(points, neighbors)
    dists = torch.clamp(dists, min=1e-12)
    inv_sum = (1.0 / dists).sum(dim=1)
    return inv_sum


def sum_inverse_dists_to_boundary(
    points: torch.Tensor, radius: float
) -> torch.Tensor:
    """
    Compute the sum over all points of 1.0 / |distance to boundary|.
    Args:
        points (Tensor): Shape (N, d)
        radius (float): Sphere radius
    Returns:
        float: The sum over all points
    """
    dists = dists_to_boundary(points, radius).abs()
    dists = torch.clamp(dists, min=1e-12)  # Avoid division by zero
    return (1.0 / dists).sum()


def plot_points_and_loss_on_axes(
    axes,
    points: torch.Tensor,
    loss_history: List[float],
    fig: Optional[plt.Figure] = None,
) -> None:
    """
    Helper to plot points and loss on given axes.
    Args:
        axes: list of Axes (length 2)
        points (Tensor): Shape (N, d), d=1,2,3
        loss_history (list or Tensor): Loss values over time
        fig: matplotlib Figure (optional, needed for 3D)
    """
    N, d = points.shape
    axes[0].cla()
    if d == 1:
        axes[0].scatter(points[:, 0], torch.zeros(N), alpha=0.7)
        axes[0].set_xlabel("x")
        axes[0].set_title("Points in 1D interval")
        axes[0].set_yticks([])
    elif d == 2:
        axes[0].scatter(points[:, 0], points[:, 1], alpha=0.7)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("Points in 2D disk")
        axes[0].axis("equal")
    elif d == 3:
        if fig is not None:
            axes[0].remove()
            ax3d = fig.add_subplot(121, projection="3d")
            ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.7)
            ax3d.set_xlabel("x")
            ax3d.set_ylabel("y")
            ax3d.set_zlabel("z")
            ax3d.set_title("Points in 3D sphere")
            axes[0] = ax3d
        else:
            raise ValueError("fig must be provided for 3D plot")
    else:
        raise ValueError("Visualization only supported for d=1, 2, or 3")
    axes[1].cla()
    axes[1].plot(loss_history)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss over time")


def visualize_points_and_loss(
    points: torch.Tensor, loss_history: List[float]
) -> None:
    """
    Show a figure with two subplots: left = points, right = loss over time.
    Args:
        points (Tensor): Shape (N, d), d=1,2,3
        loss_history (list or Tensor): Loss values over time
    """
    import matplotlib.pyplot as plt

    N, d = points.shape
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_points_and_loss_on_axes(axes, points, loss_history, fig=fig)
    plt.tight_layout()
    plt.show()


def visualize_points_and_loss_live(
    fig: plt.Figure, axes, points: torch.Tensor, loss_history: List[float]
) -> None:
    """
    Update the given matplotlib figure/axes with new points and loss.
    Args:
        fig: matplotlib Figure
        axes: list of Axes (length 2)
        points (Tensor): Shape (N, d), d=1,2,3
        loss_history (list or Tensor): Loss values over time
    """
    import matplotlib.pyplot as plt

    plot_points_and_loss_on_axes(axes, points, loss_history, fig=fig)
    plt.tight_layout()
    plt.pause(0.001)


def project_onto_sphere(points: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Project points outside the sphere back onto the surface.
    Args:
        points (Tensor): Shape (N, d)
        radius (float): Sphere radius
    Returns:
        Tensor of shape (N, d) with points inside or on the sphere
    """
    norms = points.norm(dim=1, keepdim=True)
    scale = torch.clamp(radius / norms, max=1.0)
    return points * scale


def optimize_points(
    points: torch.Tensor,
    rad: float,
    k: int,
    lr: float = 0.05,
    steps: int = 200,
    verbose: bool = True,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Optimize point locations to minimize the sum of inverse distances to k
    nearest neighbors.
    Args:
        points (Tensor): Initial points, shape (N, d), requires_grad=False
        k (int): Number of neighbors
        lr (float): Learning rate
        steps (int): Number of optimization steps
        verbose (bool): Print progress
    Returns:
        Tensor: Optimized points (N, d)
        List[float]: Loss history
    """
    points = points.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([points], lr=lr)
    loss_history = []
    fig, axes = None, None
    for step in range(steps):
        optimizer.zero_grad()
        neighbors = k_nearest_neighbors(points, k)
        loss = sum_inverse_distances(points, neighbors).sum()
        loss.backward()
        optimizer.step()
        # Project points back onto/inside the sphere if needed
        with torch.no_grad():
            points.data = project_onto_sphere(points.data, radius)
        loss_history.append(loss.item())
        if verbose and (step % max(1, steps // 20) == 0 or step == steps - 1):
            print(f"Step {step+1}/{steps}, Loss: {loss.item():.6f}")

            try:
                import matplotlib.pyplot as plt

                if fig is None or axes is None:
                    plt.ion()
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                visualize_points_and_loss_live(
                    fig, axes, points.detach(), loss_history
                )
            except Exception as e:
                print(f"Visualization failed: {e}")
    if fig is not None:
        plt.ioff()
    return points.detach(), loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random points in a d-dimensional sphere"
    )
    parser.add_argument(
        "--N", type=int, default=1000, help="Number of points (default: 1000)"
    )
    parser.add_argument(
        "--d", type=int, default=2, help="Dimension (default: 2)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Sphere radius (default: 1.0)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbors (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25000,
        help="Number of optimization steps (default: 25000)",
    )
    args = parser.parse_args()

    N = args.N
    d = args.d
    radius = args.radius
    k = args.k
    lr = args.lr
    steps = args.steps

    points = random_points_in_sphere(N, d, radius)

    newpoints, lh = optimize_points(
        points, rad=radius, k=k, lr=lr, steps=steps, verbose=True
    )
    print("Optimization complete. Final points:")
    print(newpoints)
