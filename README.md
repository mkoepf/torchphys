# Physics Fun with PyTorch

A collection of physics-related numerical tools and simulations using PyTorch.

## Simulations

## Pendulum 

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

## Basic utilities

The simulations make use of some basic functionality:

- **Finite Difference Derivatives**: Compute first, second, third, and fourth
central finite differences for 1D periodic tensors using PyTorch.


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

### 2. Pendulum Simulation

The `pendulum.py` script simulates a simple pendulum and visualizes its motion.

The script will display time series and phase space plots.

---

## Testing

Run tests with:

```bash
pytest test_fd_derivatives.py
```

---

## File Overview

- `fd_derivatives.py`: Periodic finite difference functions (1st–4th order)
- `pendulum.py`: Pendulum ODE simulation and visualization
- `test_fd_derivatives.py`: Pytest-based tests for finite difference functions

---

## License

MIT License

