#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.interpolate
import numpy as np

fname = '/home/mscherbela/develop/iypt_2023/arrester_bed/MeasuringData.ods'
sheet = 'Sheet3'
df_all = pd.read_excel(fname, sheet_name=sheet)

# Drop unit labels and convert all to mm
df_all.columns = [c.replace(" / cm", "").replace(" / mm", "") for c in list(df_all)]
df_all.X1 *= 10
df_all.Y1 *= 10
df_all.X0 *= 10
df_all.Y0 *= 10


experiment = '#1'
df0 = df_all[(df_all.Versuch == experiment) & (df_all.X0.notnull())]
df = df_all[(df_all.Versuch == experiment) & (df_all.X1.notnull())]
x_values = np.linspace(df0.X1.min(), df0.X1.max(), 100)
y_values = np.linspace(df0.Y1.min(), df0.Y1.max(), 100)


h0 = 20
z0 = h0-scipy.interpolate.griddata((df0.X0, df0.Y1), df0.Z0, (x_values[None, :], y_values[:, None]), method='linear')
z1 = h0-scipy.interpolate.griddata((df.X1, df.Y1), df.Z1, (x_values[None, :], y_values[:, None]), method='linear')
delta_z = z1 - z0


plt.close("all")
fig, ax = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(14, 5))

for ax, values, label in zip(ax, [z0, z1, delta_z], ["z0", "z1", "delta_z"]):
    surf = ax.plot_surface(x_values[None, :], y_values[:, None], values, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(label)
    # ax.axis("equal")



# %%
0