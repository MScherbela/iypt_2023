#%%
import numpy as np
import matplotlib.pyplot as plt
from simulation import get_screw_params, simulate
from matplotlib.animation import FuncAnimation
import scipy.signal

# params = get_screw_params(
#     length=5, diameter=0.8, head_diameter=3, head_thickness=0.2, mu=0.3, mu_roll=5e-3, alpha=np.radians(25)
# )

params = get_screw_params(
    length=6, diameter=0.8, head_diameter=4, head_thickness=0.2, mu=0.1, mu_roll=5e-3, alpha=np.radians(25)
)
dt = 1e-3
t_end = 4.0
phi_init_deg = 120

n_phi, n_alpha = 30, 30
phi_init_values = np.linspace(0, 175, n_phi)
alpha_values = np.linspace(8, 18, n_alpha)

ymax = np.zeros((n_phi, n_alpha))
phi_max = np.zeros((n_phi, n_alpha))
phi_final = np.zeros((n_phi, n_alpha))
n_oscillations = np.zeros((n_phi, n_alpha))
is_growing_oscillation = np.zeros((n_phi, n_alpha), dtype=bool)

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
        if np.max(np.abs(phi_values)) > np.abs(phi_values[0]) * 1.1:
            is_growing_oscillation[i, j] = True
        phi_final[i, j] = np.abs(phi_values[-1] / (2*np.pi))
        peaks = scipy.signal.find_peaks(np.abs(phi_values), height=np.radians(10))
        n_oscillations[i,j] = len(peaks[0])
        ind_simulation += 1

# plt.close("all")
# fig, axes = plt.subplots(2, 2, figsize=(14, 8))
# for ax, values, label in zip(axes.flatten(),
#                              [phi_max, phi_final, ymax, n_oscillations], 
#                              ["phi_max", "phi_final", "ymax", "n_oscillations"]):
#     surf = ax.pcolormesh(alpha_values, phi_init_values, values, shading="auto")
#     ax.set_xlabel("alpha / deg")
#     ax.set_ylabel("phi_init / deg")
#     ax.set_title(label)
#     fig.colorbar(surf, ax=ax)
# fig.tight_layout()    


#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

save_figs = True

df = pd.read_excel("data/experiment_phi_vs_alpha_phase_diagram.xlsx", header=0)
phi_values_experiment = df["phi"].values
alpha_values_experiment = df.columns[1:].astype(float).values
alpha_values_experiment = np.degrees(alpha_values_experiment)
probabilty_oscillation = df.values[:, 1:].astype(float)

# Get critical release angles
phi_crit_theory = np.ones_like(alpha_values) * np.nan
for i, alpha in enumerate(alpha_values):
    ind_osc = np.where(is_growing_oscillation[:, i])[0]
    if len(ind_osc) == 0:
        continue
    phi_crit_theory[i] = phi_init_values[np.min(ind_osc)] - 0.5 * (phi_init_values[1] - phi_init_values[0])

phi_crit_exp = np.ones_like(alpha_values_experiment) * np.nan
for i, alpha in enumerate(alpha_values_experiment):
    ind_osc = np.where(probabilty_oscillation[:, i] > 0.7)[0]
    if len(ind_osc) == 0:
        continue
    phi_crit_exp[i] = phi_values_experiment[np.min(ind_osc)]# - 0.5 * (phi_values_experiment[1] - phi_values_experiment[0])

plt.close("all")
fig, (ax, cax) = plt.subplots(1, 2, figsize=(8, 4), width_ratios=[1, 0.03])
ax.pcolormesh(alpha_values_experiment, 
              phi_values_experiment, 
              probabilty_oscillation, 
              shading="auto",
              cmap="Blues",
              clim=[0, 1.0])
cbar = fig.colorbar(ax.collections[0], cax=cax)
ax.set_title("Probability of growing oscillations", fontsize=12)
ax.set_xlabel("Ramp angle $\\alpha$ / deg", fontsize=10)
ax.set_ylabel("Release angle $\\varphi$ / deg", fontsize=10)
ax.plot(alpha_values_experiment, phi_crit_exp, color="b", lw=4, label="$\\varphi_\\mathrm{crit}$ experiment")
ax.legend(loc='lower left')
ax.set_xlim([7.8, 18.2])
fig.tight_layout()
if save_figs:
    fig.savefig("output/phasediagram_experiment.png", dpi=600, bbox_inches="tight")



indices_is_osc = np.where(is_growing_oscillation)
indices_not_osc = np.where(~is_growing_oscillation)
ax.scatter(alpha_values[indices_not_osc[1]], phi_init_values[indices_not_osc[0]], color="orange", s=20, marker="o", alpha=0.5, facecolor="none", label="Sim: Decaying osc.")
ax.scatter(alpha_values[indices_is_osc[1]], phi_init_values[indices_is_osc[0]], color="orange", s=20, marker="o", label="Sim: Growing osc.")
ax.plot(alpha_values, phi_crit_theory, color="orange", lw=4, label="$\\varphi_\\mathrm{crit}$ simulation")
ax.legend(loc='lower left')
if save_figs:
    fig.savefig("output/phasediagram_experiment_and_theory.png", dpi=600, bbox_inches="tight")


# %%
