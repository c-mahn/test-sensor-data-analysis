# Import von Bibliotheken
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import math as m
import sys
import os


# Funktionen
# -----------------------------------------------------------------------------

def plot_werte(datenreihen, name=["Messwerte"]):
    """
    Diese Funktion nimmt Datenreihen und plottet diese in ein Diagramm.
    """
    for i, datenreihe in enumerate(datenreihen):
        zeit = range(len(datenreihe))
        if(i == 0):
            plt.plot(zeit, datenreihe, "o")
        else:
            plt.plot(zeit, datenreihe)
    plt.legend(name)
    plt.grid()
    plt.xlabel("")
    plt.ylabel("")
    plt.title(name[0])
    plt.show()


def plot_xy(datenreihen, name=["Messwerte"]):
    """
    Diese Funktion nimmt je zwei Datenreihen und plottet diese in Abhängigkeit
    zueinander in ein Diagramm.
    """
    for i, datenreihe in enumerate(datenreihen):
        if(i == 0):
            plt.plot(datenreihe[0], datenreihe[1], "o")
        else:
            plt.plot(datenreihe[0], datenreihe[1])
    plt.legend(name)
    plt.grid()
    plt.xlabel("Y")
    plt.ylabel("X")
    plt.title(name[0])
    plt.show()


def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B


# Klassen
# -----------------------------------------------------------------------------

# Beginn des Programms
# -----------------------------------------------------------------------------

if(__name__ == '__main__'):

    # Einlesen der Daten
    with open(os.path.join("data", "time_series_converted.txt"), "r") as f:
        data = f.readlines()
    
    # Datenbereinigung
    for i, e in enumerate(data):
        data[i] = e.split(";")
        for j, e in enumerate(data[i]):
            data[i][j] = float(e.strip())

    # Plot der Datenreihen
    datenreihen = [[], [], []]
    for i in data:
        datenreihen[0].append(i[0])
        datenreihen[1].append(i[1])
        datenreihen[2].append(i[2])
    datenreihen_ohne_zeit = datenreihen[1:3]
    plot_werte(datenreihen_ohne_zeit)

    datenreihen[1] = fill_nan(np.array(datenreihen[1]))
    datenreihen[2] = fill_nan(np.array(datenreihen[2]))
    
    x = np.array([datenreihen[0]]) 
    y = np.array([datenreihen[1]])
    slope = (len(x) * np.sum(x*y) - np.sum(x) * np.sum(y)) / (len(x)*np.sum(x*x) - np.sum(x) ** 2)
    b = (np.sum(y) - slope *np.sum(x)) / len(x)

    
