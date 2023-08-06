#%%
import numpy as np
import matplotlib.pyplot as plt
from simulation import get_screw_params, simulate
from matplotlib.animation import FuncAnimation
import pandas as pd

df_exp = pd.read_excel("data/long2_135deg.xlsx")
dx = (df_exp["x2"] - df_exp["x1"]).values
dy = (df_exp["y2"] - df_exp["y1"]).values
phi_exp = np.arctan2(-dx, dy)
phi_exp[phi_exp < -2] += 2 * np.pi
phi_exp = np.degrees(phi_exp)
phi_exp = phi_exp - phi_exp[0] + 120
# phi_exp = np.arctan(dy/dx)
t_exp = df_exp["t"].values * 30 / 240
t_exp += 0.2


#%%

# mu_values = np.linspace(0.111, 0.1111, 2)
# mo_roll_values = np.linspace(0.01, 0.03, 10)
# head_thickness_values = np.linspace(0.15, 0.25, 10)
alpha_values = np.linspace(0.17, 0.25, 10)

all_phi_values = []
# for mu in mu_values:
# for mu_roll in mo_roll_values:
# for head_thickness in head_thickness_values:
for alpha in alpha_values:
    params = get_screw_params(
        length=6, 
        diameter=0.8,
        head_diameter=4,
        head_thickness=0.2,
        mu=0.3,
        mu_roll=0.01,
        alpha=alpha
    )

    dt = 1e-3
    t_end = 5.0
    phi_init_deg = 150


    t_values, R_values, phi_values, theta_values, R_dot_values, phi_dot_values, theta_dot_values = simulate(
        params, phi_init_deg, dt, t_end
    )
    n_t = len(t_values)
    x1 = R_values + np.stack([np.sin(phi_values), -np.cos(phi_values)], axis=1) * (params.s1 - params.sc)
    x2 = R_values + np.stack([np.sin(phi_values), -np.cos(phi_values)], axis=1) * (params.s2 - params.sc)

    phi_values = np.degrees(phi_values)
    all_phi_values.append(phi_values)

plt.close("all")
plt.plot(t_exp, phi_exp, color='k', label="exp")

for ind_param, phi_values in enumerate(all_phi_values):
    # plt.plot(t_values, phi_values, label=f"{mu_values[ind_param]:.3f}")
    # plt.plot(t_values, phi_values, label=f"{mo_roll_values[ind_param]:.3f}")
    # plt.plot(t_values, phi_values, label=f"{head_thickness_values[ind_param]:.3f}")
    plt.plot(t_values, phi_values, label=f"{alpha_values[ind_param]:.3f}")

    
plt.legend()
# %%
