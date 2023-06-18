#%%
import numpy as np
import matplotlib.pyplot as plt
from simulation import get_screw_params, simulate
from matplotlib.animation import FuncAnimation
import scipy.signal

params = get_screw_params(
    length=5, diameter=0.8, head_diameter=3, head_thickness=0.2, mu=0.3, mu_roll=5e-3, alpha=np.radians(25)
)
dt = 1e-3
t_end = 2.0
phi_init_deg = 120

n_phi, n_alpha = 7, 7
phi_init_values = np.linspace(50, 150, n_phi)
alpha_values = np.linspace(25, 40, n_alpha)

ymax = np.zeros((n_phi, n_alpha))
phi_max = np.zeros((n_phi, n_alpha))
phi_final = np.zeros((n_phi, n_alpha))
n_oscillations = np.zeros((n_phi, n_alpha))

ind_simulation = 0
for i, phi_init_deg in enumerate(phi_init_values):
    for j, alpha in enumerate(alpha_values):
        print(f"Simulation {ind_simulation}/{n_phi*n_alpha}")
        params.alpha = np.radians(alpha)
        t_values, R_values, phi_values, theta_values, R_dot_values, phi_dot_values, theta_dot_values = simulate(
            params, phi_init_deg, dt, t_end
        )
        ymax[i, j] = np.max(-R_values[:, 1])
        phi_max[i, j] = np.max(np.abs(phi_values)) / (2*np.pi)
        phi_final[i, j] = np.abs(phi_values[-1] / (2*np.pi))
        peaks = scipy.signal.find_peaks(np.abs(phi_values), height=np.radians(10))
        n_oscillations[i,j] = len(peaks[0])
        ind_simulation += 1

plt.close("all")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, values, label in zip(axes.flatten(),
                             [phi_max, phi_final, ymax, n_oscillations], 
                             ["phi_max", "phi_final", "ymax", "n_oscillations"]):
    surf = ax.pcolormesh(alpha_values, phi_init_values, values, shading="auto")
    ax.set_xlabel("alpha / deg")
    ax.set_ylabel("phi_init / deg")
    ax.set_title(label)
    fig.colorbar(surf, ax=ax)
fig.tight_layout()    



# %%
