# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from casadi import SX
import matplotlib.pyplot as plt

# %%
T = SX.sym("T")


# %%
# Logarithm of equilibrium constant: Al-O
logK_Al = -45300 / T + 11.62

# First order interaction coefficients.
e_AlAl = 0.00 + 80.5 / T
e_AlO  = 3.21 - 9720 / T
e_OAl  = 1.90 - 5750 / T
e_OO   = 0.76 - 1750 / T

# Second order interaction coefficients.
r_AlAl  = 0.0
r_AlO   = -107 - 2.75e5 / T
r_OAl   = 0.0033 - 25 / T
r_OO    = 0.0
r_AlAlO = -0.021 -13.78 / T
r_OAlO  = 127.3 + 3.273e5 / T


# pctAl, T, aAl2O3, order = params # unpacking of parameters

# log_K = logK_Al(T)
# log_aAl2O3 = np.log10(aAl2O3)
    
# if order == 0:
#     log_fAl = 0
#     log_fO  = 0
# elif (order == 1 or order ==2):
#     log_fAl = e_AlAl(T)*pctAl + e_AlO(T)*pctO
#     log_fO  = e_OAl(T)*pctAl + e_OO(T)*pctO
#     if order == 2:
#         log_fAl += r_AlAl(T)*pctAl**2 + r_AlO(T)*pctO**2 + r_AlAlO(T)*pctAl*pctO
#         log_fO  += r_OAl(T)*pctAl**2 + r_OO(T)*pctO**2 + r_OAlO(T)*pctAl*pctO
# else:
#     valid = [0, 1, 2]
#     raise ValueError(f"order: {order} is not a valid integer, choose either {valid }")
    
# eps = 2*log_fAl + 2*np.log10(pctAl) + 3*log_fO + 3*np.log10(pctO) - log_aAl2O3 - log_K



# def calc_pure(func, pctMe, T, aInc=1, x0=1e-8, order=2):
#     params = (pctMe, T, aInc, order)
#     valid = [0, 1, 2]
#     if order in valid:
#         wO = fsolve(func, x0, args=params)
#         return wO
#     else:
#         raise ValueError(f"{order} is not a valid integer, use either: {valid}")

# def data_deox(func, T, pcts_M, a_MxOy, x0, order, bound):
#     pcts_O = []
#     ### calculate oxygen in equilibrium with thermo function
#     for pct_M in pcts_M:
#         wO = calc_pure(func, pct_M, T, a_MxOy, x0, order)
#         pcts_O.append(float(wO))
#     pcts_O = np.array(pcts_O)
#     ### Bound the [%O] in the resulting curves
#     pcts_O[pcts_O > bound] = np.NaN
    
#     return pcts_M, pcts_O

# Plot using pandas and matplotlib:
# T = 1873                            # Temperature, K
# pcts = np.logspace(-4, 1, 100)      # Al content range, [%Al]
# aAl2O3 = 1                          # Al2O3 activity

# # Plot model results
# orders = [0, 1, 2]
# linestyles = [':', '--', '-']
# labels = ['0th order', '1st order', '2nd order']
# for order, linestyle, label in zip(orders, linestyles, labels):
#     pctM, pctO = data_deox(Al_O_eqs, T, pcts, aAl2O3, x0=1e-8, order=order, bound=1e-2)
    
# # Make the plot pretty
# ax.set_xlabel('[%Al]')
# ax.set_ylabel('[%O]')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.grid(which='both', ls='--', color='lightgray')
# ax.legend(loc='upper right', ncols=3, fontsize=8)

