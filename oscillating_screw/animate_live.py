# %%
import numpy as np
import matplotlib.pyplot as plt
from simulation import get_screw_params, simulate

params = get_screw_params(
    length=5, diameter=0.8, head_diameter=3, head_thickness=0.2, mu=0.30, mu_roll=0.005, alpha=np.radians(30)
)


dt = 1e-3
t_end = 2.0
phi_init_deg = 120

t_values, R_values, phi_values, theta_values, R_dot_values, phi_dot_values, theta_dot_values = simulate(
    params, phi_init_deg, dt, t_end
)
n_t = len(t_values)
x1 = R_values + np.stack([np.sin(phi_values), -np.cos(phi_values)], axis=1) * (params.s1 - params.sc)
x2 = R_values + np.stack([np.sin(phi_values), -np.cos(phi_values)], axis=1) * (params.s2 - params.sc)


fig, ax = plt.subplots(figsize=(17, 5))
n_skip = 40
fontsize = 16

ax.plot(x1[0::n_skip, 1], x1[0::n_skip, 0], color="C0", alpha=0.3)
ax.plot(x2[0::n_skip, 1], x2[0::n_skip, 0], color="C1", alpha=0.3)
(marker1,) = plt.plot([], [], marker="o", markersize=5, color="C0")
(marker2,) = plt.plot([], [], marker="o", markersize=10, color="C1")
(connection,) = plt.plot([], [], color="k")
text = ax.text(0.02, 0.9, "", transform=ax.transAxes, fontsize=fontsize)
ax.axis("equal")
ax.invert_xaxis()
ax.set_xlabel("Position along the ramp / cm", fontsize=fontsize)
ax.set_ylabel("Position across the ramp / cm", fontsize=fontsize)
ax.tick_params(labelsize=fontsize)

for i in range(0, n_t, n_skip):
    marker1.set_data([x1[i, 1]], [x1[i, 0]])
    marker2.set_data([x2[i, 1]], [x2[i, 0]])
    connection.set_data([x1[i, 1], x2[i, 1]], [x1[i, 0], x2[i, 0]])
    text.set_text(f"t = {t_values[i]:3.2f} sec")
    plt.pause(0.002)


# %%
