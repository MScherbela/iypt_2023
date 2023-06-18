#%%
import numpy as np
import jax
import jax.numpy as jnp
import chex
import matplotlib.pyplot as plt


@chex.dataclass
class ModelParams:
    I_phi: float
    I_theta: float
    M: float
    s1: float
    s2: float
    sc: float
    r1: float
    r2: float
    alpha: float
    mu: float
    mu_roll: float
    g: float = 981.0


def get_screw_params(length, diameter, head_diameter, head_thickness, density=7.85, **kwargs):
    M_screw = density * np.pi * (diameter / 2) ** 2 * length
    M_head = density * np.pi * (head_diameter / 2) ** 2 * head_thickness
    M = M_screw + M_head
    sc = (M_screw * length/2 + M_head * length) / M

    I_theta_screw = 0.5 * M_screw * (diameter / 2) ** 2
    I_theta_head = 0.5 * M_head * (head_diameter / 2) ** 2
    I_theta = I_theta_screw + I_theta_head

    I_phi_head = (1/12) * M_head * (3 * (head_diameter / 2) ** 2 + head_thickness ** 2)
    I_phi_head += (length - sc) ** 2 * M_head # Steiner
    I_phi_screw = (1/12) * M_screw * (3 * (diameter / 2) ** 2 + length ** 2)
    I_phi_screw += (length / 2 - sc) ** 2 * M_screw # Steiner
    I_phi = I_phi_screw + I_phi_head
    return ModelParams(I_phi=I_phi, I_theta=I_theta, M=M, s1=0, s2=length, sc=sc, 
                       r1=diameter/2, 
                       r2=head_diameter/2, 
                       **kwargs)
    

def get_rot_matrix(phi):
    c, s = jnp.cos(phi), jnp.sin(phi)
    return jnp.array([[c, -s], [s, c]])


def get_acceleration(R, phi, theta, R_dot, phi_dot, theta_dot, p: ModelParams):
    U = get_rot_matrix(phi)

    R_dot_moving_frame = U.T @ R_dot
    radii = jnp.array([p.r1, p.r2])
    distances = jnp.array([p.s1, p.s2])

    v_rolling = radii * theta_dot
    v_parallel = R_dot_moving_frame[0]  + (distances - p.sc) * phi_dot
    v_slip_parallel =  v_parallel - v_rolling
    v_slip_normal = R_dot_moving_frame[1]

    F_normal = p.M * p.g * jnp.cos(p.alpha) * (distances[::-1] - p.sc) / (distances[::-1] - distances)

    sign_func = lambda x: jnp.sign(x)
    F_fric_parallel = -p.mu * F_normal * sign_func(v_slip_parallel)
    F_fric_normal = -p.mu * F_normal * sign_func(v_slip_normal)
    F_fric_rolling = -p.mu_roll * F_normal * sign_func(v_parallel)

    theta_acc = -jnp.dot(F_fric_parallel, radii) / p.I_theta
    # theta_acc -= p.mu_roll * theta_dot
    phi_acc = jnp.dot(F_fric_parallel + F_fric_rolling, distances - p.sc) / p.I_phi

    F_fric_total_in_moving_frame = jnp.array([jnp.sum(F_fric_parallel + F_fric_rolling), jnp.sum(F_fric_normal)])
    R_acc = p.g * jnp.sin(p.alpha) * np.array([0, -1]) + U @ F_fric_total_in_moving_frame / p.M
    return R_acc, phi_acc, theta_acc


@jax.jit
def time_step(R, phi, theta, R_dot, phi_dot, theta_dot, dt, p: ModelParams):
    R_acc, phi_acc, theta_acc = get_acceleration(R, phi, theta, R_dot, phi_dot, theta_dot, p)

    R_dot += R_acc * dt
    phi_dot += phi_acc * dt
    theta_dot += theta_acc * dt

    R += R_dot * dt
    phi += phi_dot * dt
    theta += theta_dot * dt
    return R, phi, theta, R_dot, phi_dot, theta_dot



def simulate(params, phi_init_deg=120, dt=2e-4, t_end=2.5):
    t_values = np.arange(0, t_end, dt)
    n_t = len(t_values)
    R_values = np.zeros((n_t, 2))
    phi_values = np.zeros(n_t)
    theta_values = np.zeros(n_t)
    R_dot_values = np.zeros((n_t, 2))
    phi_dot_values = np.zeros(n_t)
    theta_dot_values = np.zeros(n_t)

    R_values[0, :] = np.array([0.0, 0.0])
    phi_values[0] = np.radians(phi_init_deg)

    for i in range(n_t - 1):
        (
            R_values[i + 1, :],
            phi_values[i + 1],
            theta_values[i + 1],
            R_dot_values[i + 1, :],
            phi_dot_values[i + 1],
            theta_dot_values[i + 1],
        ) = time_step(
            R_values[i, :],
            phi_values[i],
            theta_values[i],
            R_dot_values[i, :],
            phi_dot_values[i],
            theta_dot_values[i],
            dt,
            params,
        )
    return t_values, R_values, phi_values, theta_values, R_dot_values, phi_dot_values, theta_dot_values

