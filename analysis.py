# Import von Bibliotheken
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
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
