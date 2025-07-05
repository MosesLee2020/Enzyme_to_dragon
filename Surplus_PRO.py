import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Parameters ---
Xp0     = 1e6       # initial PRO cells (cells/mL)
R0_def  = 100       # default initial resource (µM)
Km_p    = 2.5       # half-saturation constant (µM)
d_p     = 0.29      # death rate (1/day)
Rcell_p = 1.5e-6    # resource per cell (µM/cell)
D       = 20        # dilution factor

def pro_resource_ode(t, y, v_max):
    Xp, Xs, R = y
    growth_p = v_max * (R / (R + Km_p)) * Xp
    dXp_dt   = growth_p - d_p * Xp
    dXs_dt   = 0.0
    dR_dt    = - growth_p * Rcell_p
    return [dXp_dt, dXs_dt, dR_dt]

def surplus(v_max, T, R0):
    """Compute net surplus after one transfer of length T and dilution D."""
    sol = solve_ivp(
        fun=lambda t, y: pro_resource_ode(t, y, v_max),
        t_span=[0, T],
        y0=[Xp0, 0.0, R0],
        method='RK45',
        t_eval=np.linspace(0, T, 500)
    )
    R_pts = sol.y[2]
    integrand = R_pts / (R_pts + Km_p)
    I = np.trapz(integrand, sol.t)
    return v_max * I - d_p * T - np.log(D)

# --- 1) Surplus vs Transfer time (v_max fixed) ---
v_fixed = 0.68
T_vals = np.linspace(0.1, 20, 100)
surplus_T = [surplus(v_fixed, T, R0_def) for T in T_vals]

plt.figure()
plt.plot(T_vals, surplus_T, marker='o')
plt.axhline(0, linestyle='--')
plt.xlabel('Transfer time T (days)')
plt.ylabel('Surplus')
plt.title(f'Surplus vs Transfer Time (v_max={v_fixed}, R0={R0_def} uM)')
plt.tight_layout()
plt.show()

# --- 2) Surplus vs v_max (T fixed) ---
T_fixed = 7
v_vals = np.arange(0.68, 1.50, 0.01)
surplus_v = [surplus(v, T_fixed, R0_def) for v in v_vals]

plt.figure()
plt.plot(v_vals, surplus_v, marker='o')
plt.axhline(0, linestyle='--')
plt.xlabel('v_max (1/day)')
plt.ylabel('Surplus')
plt.title(f'Surplus vs v_max (T={T_fixed} days, R0={R0_def} uM)')
plt.tight_layout()
plt.show()

# --- 3) Surplus vs initial R0 (v_max and T fixed) ---
R0_vals = np.linspace(0, 200, 100)
surplus_R = [surplus(v_fixed, T_fixed, R0) for R0 in R0_vals]

plt.figure()
plt.plot(R0_vals, surplus_R, marker='o')
plt.axhline(0, linestyle='--')
plt.xlabel('Initial resource R0 (µM)')
plt.ylabel('Surplus')
plt.title(f'Surplus vs Initial R0 (v_max={v_fixed}, T={T_fixed} days)')
plt.tight_layout()
plt.show()
