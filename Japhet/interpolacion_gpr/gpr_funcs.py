# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:00:36 2023

@author: japha
"""

import numpy as np
import matplotlib.pyplot as plt

def showProfile(profile, x, t, title='Perfil', ax=None):
    """
    Emulation of gprpy's showProfile() function

    Parameters
    ----------
    profile : array_like
        Array containing the profile amplitude values.
    x : array_like
        Array containing the profile distance.
    t : array_like
        Array containing the profile time-travel.
    title : str, optional
        Title for the plot. The default is 'Perfil'.
    ax : TYPE, optional
        Axis to be plotted, in case of subplotting. The default is None.

    Returns
    -------
    None.

    """
    if ax is not None:
        ax.imshow(profile, extent=(x.min(), x.max(), t.max(), t.min()), aspect='auto', cmap='gray')
        ax.set_title(title)
        ax.set_ylabel('Tiempo de viaje [ns]')
        ax.set_xlabel('Distancia [m]')
    else:
        plt.imshow(profile, extent=(x.min(), x.max(), t.max(), t.min()), aspect='auto', cmap='gray')
        plt.title(title)
        plt.ylabel('Tiempo de viaje [ns]')
        plt.xlabel('Distancia [m]')
        
    return

if __name__ == '__main__':
    
    # Reading data
    data = np.load('profile20_data.npy')
    distance = np.load('profile20_distance.npy')
    time = np.load('profile20_time.npy')
    
    showProfile(data, distance, time)