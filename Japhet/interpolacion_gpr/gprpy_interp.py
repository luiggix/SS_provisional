# -*- coding: utf-8 -*-
"""
Created on Tue May  2 14:19:41 2023

@author: japha
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt

# Importing functions
from gpr_funcs import showProfile
from scipy import interpolate

# Setting the environment
np.set_printoptions(suppress=True)
rng = np.random.default_rng(35)

# Reading the data from files
profile = np.load('profile20_data.npy')
samples, traces = profile.shape
x = np.load('profile20_distance.npy')
t = np.load('profile20_time.npy')

# Arbitrary removal of samples in traces
traces_remove = np.linspace(1, traces-1, 900, dtype=int)
profile_removed = np.copy(profile)
for trace in traces_remove:
    samples_remove = np.linspace(1, samples-2, 420, dtype=int)
    profile_removed[samples_remove, trace] = np.nan
    
# Visualization of original and mutilated profile
titles = ['Perfil original', 'Perfil mutilado']
perfiles = [profile, profile_removed]

fig01, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, perfil, titulo in zip(axes.flatten(), perfiles, titles):
    showProfile(perfil, x, t, title=titulo, ax=ax)
plt.show()

# Test trace for visualization of mutilation and interpolation
test = traces_remove[3]

# Visualizing the original test trace (red) and remaining data (black)
fig02, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, profile[:, test], lw=2, c='tab:red', marker='o', markersize=2.5, markerfacecolor='white')
ax.scatter(t, profile_removed[:, test], c='k', s=10, zorder=10)
plt.show()

# Generating copies of mutilated profile for interpolation
profile_interp_linear = np.copy(profile_removed)
profile_interp_nearest = np.copy(profile_removed)
profile_interp_quad = np.copy(profile_removed)
profile_interp_cubic = np.copy(profile_removed)

# Interpolating missing data, leaving out NaN values
for trace in traces_remove:
    not_nans = np.logical_not(np.isnan(profile_removed[:, trace]))
    indices = np.arange(len(profile_removed[:, trace]))
    
    profile_interp_linear[:, trace] = interpolate.interp1d(indices[not_nans], profile_removed[:, trace][not_nans], kind='linear')(indices)
    profile_interp_nearest[:, trace] = interpolate.interp1d(indices[not_nans], profile_removed[:, trace][not_nans], kind='nearest')(indices)
    profile_interp_quad[:, trace] = interpolate.interp1d(indices[not_nans], profile_removed[:, trace][not_nans], kind='quadratic')(indices)
    profile_interp_cubic[:, trace] = interpolate.interp1d(indices[not_nans], profile_removed[:, trace][not_nans], kind='cubic')(indices)

# Visualizing profiles after interpolation, per method
titles = ["'nearest'", "'linear'", "'quadratic'", "'cubic'"]
perfiles = [profile_interp_nearest, profile_interp_linear,
            profile_interp_quad, profile_interp_cubic]

fig03, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
for ax, perfil, titulo in zip(axes.flat, perfiles, titles):
    showProfile(perfil, x, t, title=titulo, ax=ax)
plt.suptitle('Perfil interpolado con método:')
plt.tight_layout()
plt.show()

# Visualizing original test trace (red) and interpolated trace (black)
fig04, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

plt.suptitle('Traza interpolada con método:')

axes[0, 0].plot(t, profile[:, test], lw=2, c='tab:red')
axes[0, 0].plot(t, profile_interp_nearest[:, test], c='k', lw=1, alpha=0.6, marker='o', markersize=3)
axes[0, 0].set_title("'nearest'")
axes[0, 0].set_ylabel('Amplitud [mV]')
axes[0, 0].set_xlabel('Distancia [m]')

axes[0, 1].plot(t, profile[:, test], lw=2, c='tab:red')
axes[0, 1].plot(t, profile_interp_linear[:, test], c='k', lw=1, alpha=0.6, marker='o', markersize=3)
axes[0, 1].set_title("'linear'")

axes[1, 0].plot(t, profile[:, test], lw=2, c='tab:red')
axes[1, 0].plot(t, profile_interp_quad[:, test], c='k', lw=1, alpha=0.6, marker='o', markersize=3)
axes[1, 0].set_title("'quadratic'")
axes[1, 0].set_ylabel('Amplitud [mV]')
axes[1, 0].set_xlabel('Distancia [m]')

axes[1, 1].plot(t, profile[:, test], lw=2, c='tab:red')
axes[1, 1].plot(t, profile_interp_cubic[:, test], c='k', lw=1, alpha=0.6, marker='o', markersize=3)
axes[1, 1].set_title("'cubic'")
axes[1, 1].set_xlabel('Distancia [m]')

plt.tight_layout()
plt.show()

# Calculating percentage of different elements per interpolation method
de_interp = [np.mean(profile != perfil)*100 for perfil in perfiles]
print(f'Porcentaje de elementos diferentes (método \'nearest\'): {round(de_interp[0], 3)}%')
print(f'Porcentaje de elementos diferentes (método \'linear\'): {round(de_interp[1], 3)}%')
print(f'Porcentaje de elementos diferentes (método \'quadratic\'): {round(de_interp[2], 3)}%')
print(f'Porcentaje de elementos diferentes (método \'cubic\'): {round(de_interp[3], 3)}%')