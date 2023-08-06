#%%
import numpy as np
import matplotlib.pyplot as plt

r = 20e-3
m = 30e-3
g = 9.81
h = 0.1
rho_sand = 1600
cd = 0.5
A_ratio = 0.5
A = np.pi * r**2 * A_ratio

F_compression = 0.15

E = m*g*h
v0 = np.sqrt(2*E/m)


def get_force(v):
    F_drag = 0.5 * cd * rho_sand * A * v**2
    return -(F_compression + F_drag) * np.sign(v)


t_values = [0]
x_values = [0]
v_values = [v0]


dt = 1e-3
while v_values[-1] > 1e-3:
    t_new = t_values[-1] + dt
    v_new = v_values[-1] + get_force(v_values[-1]) / m * dt
    x_new = x_values[-1] + v_values[-1] * dt
    t_values.append(t_new)
    v_values.append(v_new)
    x_values.append(x_new)

plt.close("all")

fig, (ax_x, ax_v) = plt.subplots(1, 2, figsize=(10, 5))
ax_x.plot(t_values, x_values)
ax_x.set_xlabel("Time [s]")
ax_x.set_ylabel("Position [m]")
ax_v.plot(t_values, v_values)
ax_v.set_xlabel("Time [s]")
ax_v.set_ylabel("Velocity [m/s]")


# %%
