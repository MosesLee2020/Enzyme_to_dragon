# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 10:51:46 2025

@author: Edmee
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Fixed PRO parameters and transfer protocol ---
X_p0, X_s0, R0 = 1e6, 1.15e5, 100    # initial cell densities and resource
TRANSFER_TIME = 7                    # days per transfer
DILUTION = 20                        # dilution factor
NUM_TRANSFERS = 7                    # number of transfers

# PRO kinetics (fixed)
v_p, Km_p, r_p, d_p = 0.74, 2.5, 1.5e-6, 0.29

# Default SYN kinetics (to be swept)
v_s0, Km_s0, r_s0, d_s0 = 0.5, 5.0, 2e-6, 0.2

def dydt(t, y, params):
    """ODEs for PRO and SYN competing for resource R."""
    Xp, Xs, R = y
    v_p_, Km_p_, r_p_, d_p_, v_s_, Km_s_, r_s_, d_s_ = params
    mu_p = v_p_ * R/(R+Km_p_)
    mu_s = v_s_ * R/(R+Km_s_)
    dXp = (mu_p - d_p_) * Xp
    dXs = (mu_s - d_s_) * Xs
    dR  = -mu_p * Xp * r_p_ - mu_s * Xs * r_s_
    return [dXp, dXs, dR]

def simulate_competition(v_s, Km_s, r_s, d_s):
    """Run NUM_TRANSFERS and return final SYN/PRO ratio."""
    Xp, Xs, R = X_p0, X_s0, R0
    params = [v_p, Km_p, r_p, d_p, v_s, Km_s, r_s, d_s]
    for _ in range(NUM_TRANSFERS):
        sol = solve_ivp(lambda t,y: dydt(t, y, params),
                        [0, TRANSFER_TIME],
                        [Xp, Xs, R])
        Xp = sol.y[0, -1] / DILUTION
        Xs = sol.y[1, -1] / DILUTION
        R  = R0
    return Xs / Xp

# --- Parameter sweeps ---
# 1) SYN v_max
v_s_vals = np.linspace(0.4, 1.0, 50)
ratio_v = [simulate_competition(v, Km_s0, r_s0, d_s0) for v in v_s_vals]

# 2) SYN K_m
Km_s_vals = np.linspace(0.1, 10, 50)
ratio_Km = [simulate_competition(v_s0, Km, r_s0, d_s0) for Km in Km_s_vals]

# 3) SYN resource quota
r_s_vals = np.linspace(0.5e-6, 5e-6, 50)
ratio_r = [simulate_competition(v_s0, Km_s0, r, d_s0) for r in r_s_vals]

# 4) SYN death rate
d_s_vals = np.linspace(0.001, 0.3, 50)
ratio_d = [simulate_competition(v_s0, Km_s0, r_s0, ds) for ds in d_s_vals]

# --- Plotting results ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0,0].plot(v_s_vals, ratio_v, '-o')
axs[0,0].axhline(1, color='gray', ls='--')
axs[0,0].set(title='Effect of SYN v_max',
             xlabel='SYN v_max (1/day)',
             ylabel='X_s/X_p after 7 transfers')

axs[0,1].plot(Km_s_vals, ratio_Km, '-o')
axs[0,1].axhline(1, color='gray', ls='--')
axs[0,1].set(title='Effect of SYN K_m',
             xlabel='SYN K_m (µM)',
             ylabel='')

axs[1,0].plot(r_s_vals, ratio_r, '-o')
axs[1,0].axhline(1, color='gray', ls='--')
axs[1,0].set(title='Effect of SYN resource quota',
             xlabel='SYN r (µM/cell)',
             ylabel='')

axs[1,1].plot(d_s_vals, ratio_d, '-o')
axs[1,1].axhline(1, color='gray', ls='--')
axs[1,1].set(title='Effect of SYN death rate',
             xlabel='SYN d (1/day)',
             ylabel='')

plt.tight_layout()
plt.show()
