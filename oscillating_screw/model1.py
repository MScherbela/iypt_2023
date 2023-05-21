# %%
"""
This code simulates a simplified system:
The screw is modelled as 2 point masses m1, m2.
Friction is assumed to be 0 for pivoting around a pivot point R and mu for sliding.
There are no inertia terms to account for the rolling motion.
All units are cgs: cm, g, s
"""
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import dataclasses
import jax
import chex


@chex.dataclass
class ModelParams:
    m1: float
    m2: float
    r1: float
    r2: float
    alpha: float
    mu: float = 0.3
    g: float = 981.0


def build_state_matrix_and_inhom(R, phi, R_dot, phi_dot, p: ModelParams):
    c, s = jnp.cos(phi), jnp.sin(phi)
    l = p.r2 - p.r1
    A = jnp.array(
        [
            [p.m1, 0, p.m1 * p.r1 * c, l * s],
            [0, p.m1, p.m1 * p.r1 * s, -l * c],
            [p.m2, 0, p.m2 * p.r2 * c, -l * s],
            [0, p.m2, p.m2 * p.r2 * s, l * c],
        ]
    )
    mass = jnp.array([p.m1, p.m1, p.m2, p.m2])
    radius = jnp.array([p.r1, p.r1, p.r2, p.r2])
    e_centrifugal = jnp.array([s, -c, s, -c])
    e_fric = jnp.tile(jnp.sign(R_dot), 2)
    e_fric *= jnp.array([10, 10, 5, 5])
    #e_fric = jnp.tile(jnp.tanh(R_dot / 0.1), 2)
    e_gravity = jnp.array([0, 1, 0, 1])

    b = radius * phi_dot**2 * e_centrifugal
    b -= p.g * jnp.cos(p.alpha) * p.mu * e_fric
    b -= p.g * jnp.sin(p.alpha) * e_gravity
    b *= mass
    return A, b


@jax.jit
def time_step(R, phi, R_dot, phi_dot, dt, p: ModelParams):
    A, b = build_state_matrix_and_inhom(R, phi, R_dot, phi_dot, p)
    q_acc = jnp.linalg.solve(A, b)
    R_acc = q_acc[:2]
    phi_acc = q_acc[2]
    constraint_force = q_acc[3]
    
    R_dot += R_acc * dt
    phi_dot += phi_acc * dt
    R += R_dot * dt
    phi += phi_dot * dt
    return R, phi, R_dot, phi_dot, constraint_force


params = ModelParams(m1=1.0, m2=3.0, r1=1.0, r2=3.0, mu=0.4, alpha=np.radians(25))
dt = 1e-3
t_end = 20.0
t_values = np.arange(0, t_end, dt)
n_t = len(t_values)
R_values = np.zeros((n_t, 2))
phi_values = np.zeros(n_t)
R_dot_values = np.zeros((n_t, 2))
phi_dot_values = np.zeros(n_t)

R_values[0, :] = np.array([0.0, 0.0])
phi_values[0] = np.radians(90.0)
R_dot_values[0, :] = [0.0, 0.0]

for i in range(n_t - 1):
    if i % 1000 == 0:
        print(i)
    R_values[i + 1, :], phi_values[i + 1], R_dot_values[i + 1, :], phi_dot_values[i + 1], _ = time_step(
        R_values[i, :], phi_values[i], R_dot_values[i, :], phi_dot_values[i], dt, params
    )

plt.close("all")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(t_values, R_values[:, 0])
axes[0, 1].plot(t_values, R_values[:, 1])
axes[1, 0].plot(t_values, np.degrees(phi_values))
axes[1, 1].plot(R_values[:, 0], R_values[:, 1])


for ax, label in zip([axes.flatten(), "x", "y", "phi"]):
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel("t / sec")
    ax.set_ylabel(label)
axes[1, 1].set_xlabel("x / cm")
axes[1, 1].set_xlabel("y / cm")
axes[1, 1].axis("equal")


# %%
