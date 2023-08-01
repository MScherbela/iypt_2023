# %%
import numpy as np
import matplotlib.pyplot as plt
from simulation import get_screw_params, simulate
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import EngFormatter, StrMethodFormatter


dt = 0.5e-3
t_end = 3
phi_init_deg = 75

params1 = get_screw_params(
    length=4, 
    diameter=0.8, 
    head_diameter=3, 
    head_thickness=0.2, 
    mu=0.1, 
    mu_roll=0.001, alpha=np.radians(12)
)

params2 = get_screw_params(
    length=6, 
    diameter=0.8, 
    head_diameter=4, 
    head_thickness=0.2, 
    mu=0.1, 
    mu_roll=0.001, alpha=np.radians(12)
)


t_values, _, phi_values1, _, _, _, _ = simulate(
    params1, phi_init_deg, dt, t_end
)
t_values, _, phi_values2, _, _, _, _ = simulate(
    params2, phi_init_deg, dt, t_end
)

phi_max = 170
#%%
plt.close("all")
fig, ax = plt.subplots(1,1, figsize=(4, 3))
ax.plot(t_values, np.degrees(phi_values1))
ax.plot(t_values, np.degrees(phi_values2))
ax.axhline(0, color='k')
ax.set_xlabel("t / s")
ax.set_ylabel("$\\varphi$ / deg", labelpad=-5)
ax.set_ylim([-170, 170])
ax.set_xlim([0, 2.6])


for sign in [-1, 1]:
    ax.fill_between(t_values, 
                    sign * np.ones_like(t_values)*phi_init_deg*1.1, 
                    sign * np.ones_like(t_values)*phi_max,
                    color='C1',
                    alpha=0.1
                    )
    ax.text(0.1,
        sign * phi_init_deg*1.7,
        "Growing osc.:\n$\\max_t |\\varphi(t)| \\geq 1.1 \\varphi_0$",
        va='center',
        ha='left',
        color='C1'
        )
ax.text(1.1,
    -50,
    "Decaying osc.:\n$\\max_t |\\varphi(t)| < 1.1 \\varphi_0$",
    va='center',
    ha='left',
    color='C0'
    )

    
ax.fill_between(t_values, 
            np.ones_like(t_values)*phi_init_deg*1.1, 
            -np.ones_like(t_values)*phi_init_deg*1.1, 
            color='C0',
            alpha=0.1
            )
ax.yaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))
fig.tight_layout()
fig.savefig("output/growing_vs_decaying_osc.png", dpi=700, bbox_inches='tight')




# for sign in [-1, 1]:
#     ax.axhline(sign * phi_init_deg)
#     ax.axhline(sign * phi_init_deg * 1.2)


# %%
