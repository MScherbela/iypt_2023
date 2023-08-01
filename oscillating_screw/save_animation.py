# %%
import numpy as np
import matplotlib.pyplot as plt
from simulation import get_screw_params, simulate
from matplotlib.animation import FuncAnimation

# alpha_in_deg = 21
params = get_screw_params(
    length=6, diameter=0.8, head_diameter=4, head_thickness=0.2, mu=0.25, mu_roll=0.01, alpha=0.21
)


dt = 1e-3
t_end = 5.0
phi_init_deg = 135

t_values, R_values, phi_values, theta_values, R_dot_values, phi_dot_values, theta_dot_values = simulate(
    params, phi_init_deg, dt, t_end
)
n_t = len(t_values)
x1 = R_values + np.stack([np.sin(phi_values), -np.cos(phi_values)], axis=1) * (params.s1 - params.sc)
x2 = R_values + np.stack([np.sin(phi_values), -np.cos(phi_values)], axis=1) * (params.s2 - params.sc)


plt.close("all")
fig, ax = plt.subplots(figsize=(17, 5))

slowdown = 5
fps = 25
n_skip = int(1 / (dt * slowdown * fps))
fontsize = 16

ax.plot(x1[0::n_skip, 1], x1[0::n_skip, 0], color="C0", alpha=0.3)
ax.plot(x2[0::n_skip, 1], x2[0::n_skip, 0], color="C1", alpha=0.3)
(marker1,) = plt.plot([], [], marker="o", markersize=10, color="C0")
(marker2,) = plt.plot([], [], marker="o", markersize=20, color="C1")
(connection,) = plt.plot([], [], color="k")
text = ax.text(0.02, 0.9, "", transform=ax.transAxes, fontsize=fontsize)

# ax.axis("equal")
ax.invert_xaxis()
ax.set_xlabel("Position along the ramp / cm", fontsize=fontsize)
ax.set_ylabel("Position across the ramp / cm", fontsize=fontsize)
ax.tick_params(labelsize=fontsize)

#%%
def run_animation(i):
    marker1.set_data([x1[i, 1]], [x1[i, 0]])
    marker2.set_data([x2[i, 1]], [x2[i, 0]])
    connection.set_data([x1[i, 1], x2[i, 1]], [x1[i, 0], x2[i, 0]])
    text.set_text(f"t = {t_values[i]:3.2f} sec")
    if i % 500 < n_skip:
        print(f"{i}/{n_t}")

print("Creating animation...")
anim = FuncAnimation(fig, run_animation, frames=np.arange(0, n_t, n_skip), interval=1/fps, repeat=False)
# print("Saving to html...")
# # html = anim.to_jshtml(fps=fps)
# with open("animation.html", "w") as f:
#     f.write(html)

print("Saving as MP4")
anim.save("animation.mp4", fps=fps)
print("Done")

# %%
