# Physics Fun with PyTorch

A collection of physics-related numerical tools and simulations using PyTorch.

## Simulations

### Pendulum 

Numerically solve and visualize the motion of a [mathematical
pendulum](https://en.wikipedia.org/wiki/Pendulum_(mechanics))

$$
\ddot{\varphi} = -\sin\left(\varphi\right)
$$

using `torchdiffeq` and `matplotlib`.

The second-order equation is turned into a set of two first-order equations

$$
\dot{\varphi}=\psi
$$
$$
\dot{\psi}=-\sin\left(\varphi\right)
$$

which are solved using the initial conditions
$\left(\varphi,\psi\right)=\left(\varphi_0,\psi_0\right)$ using a Runge-Kutta
solver.

Run from the command line:
```bash
python pendulum.py --phi0 0.5 --psi0 1.0 --T 20 --N 2000
```

- `--phi0`: Initial angle (default: 0.0)
- `--psi0`: Initial angular velocity (default: 2.0)
- `--T`: Total simulation time (default: 10.0)
- `--N`: Number of time points (default: 1000)

As default, the initial conditions $\left(0, 2\right)$ are used, which launches 
the pendulum into a nice homoclinic orbit.

The solution is displayed in two ways, both using matplotlib:

- $\varphi(t)$ and $\psi(t)$ together in one plot
- the trajectory in phase space ($\psi$ against $\varphi$)

Solving from the (default) initial condition $\left(0, 2\right)$, as shown
above, gives you the following plots:

![plot of phi(t) and psi(t)](/images/pendulum_homoclinic_time.png)
![phase space trajectory
(phi(t),psi(t))](/images/pendulum_homoclinic_phasespace.png)

### Repellant points in a d-ball

Simulates a number of points inside a d-ball (a d-dimensional ball, i.e., a line
for $d=1$, a disk for $d=2$, a ball for $d=3$, ...) repelling each other.

The simulation simply minimizes the "energy" given by

$$
E = \sum_{\vec{p}\in P} \sum_{\vec{q} \in (\textrm{neigbors of }\vec{p})})\frac{1}{|\vec{p}-\vec{q}|}.
$$

It uses PyTorch's automatic differentiation, i.e., the standard optimization
loop, i.e.,

```python
    for step in range(steps):
        optimizer.zero_grad()
        neighbors = k_nearest_neighbors(points, k)
        loss = sum_inverse_distances(points, neighbors).sum()
        loss.backward()
        optimizer.step()
```

The points are confined to the d-ball by simply projecting any "leaver" back
onto the ball's surface.

For large numbers of points, only $k$ nearest neighbors are used.

While the optimization runs, it displays both the points' locations and the
loss history.

Here's an example snapshot from 

```bash
python dballpoints.py --N 100 --d 2 --radius 10.0 --k 50 --steps 500000 --lr 0.00002
```

![plot of point locations and loss history](/images/dball.png)

## Basic utilities

The simulations make use of some basic functionality:

### Finite Difference Derivatives

Compute first, second, third, and fourth central finite differences for 1D
**periodic domains** using PyTorch tensors.

The `fd_derivatives.py` module provides functions to compute periodic finite
differences:

- `first_central_difference_periodic(u, dx)`
- `second_central_difference_periodic(u, dx)`
- `third_central_difference_periodic(u, dx)`
- `fourth_central_difference_periodic(u, dx)`

Example:
```python
import torch
from fd_derivatives import first_central_difference_periodic

u = torch.sin(torch.linspace(0, 2 * torch.pi, 100))
dx = u[1] - u[0]
deriv = first_central_difference_periodic(u, dx)
```

---

## Requirements

- Python >= 3.12
- torch
- matplotlib
- torchdiffeq
- pytest (for testing)

Install dependencies with:

```bash
pip install torch matplotlib torchdiffeq pytest
```

or let `uv` take care of dependencies using inline metadata in the scripts, e.g.

```bash
uv run pendulum.py
```

---

## Usage

### 1. Finite Difference Derivatives

---

## Testing

You can use the test in the `test_...` files to understand what the code is 
doing.

To actually run the tests, use `pytest`.

---

## License

MIT License

