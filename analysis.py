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


def low_pass_filter(datenreihe, filterungsgrad):
    """
    Diese Funktion macht einen vereinfachten Low-Pass-Filter, indem die letzten
    x Sensorwerte gemittelt werden.
    """
    ausgabe = []
    for i, e in enumerate(datenreihe):
        summe = 0
        for j in range(filterungsgrad):
            ji = i-j
            if(ji <= -1):  # Wenn Wert ausserhalb der Datenreihe, dann Ersatzwert
                summe =+ float(e)
            else:
                summe =+ float(datenreihe[ji])
        ausgabe.append(summe/filterungsgrad)
    return(ausgabe)


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

    # Umwandeln in Datenreihen
    datenreihen = [[], [], []]
    for i in data:
        datenreihen[0].append(i[0])
        datenreihen[1].append(i[1])
        datenreihen[2].append(i[2])
    datenreihen_ohne_zeit = datenreihen[1:3]
    plot_werte(datenreihen_ohne_zeit, ["Sensor 1", "Sensor 2"])

    # Füllen von Lücken in den Datenreihen
    for i, e in enumerate(datenreihen):
        if(i != 0):
            datenreihen[i] = fill_nan(np.array(e))
    
    # Berechnung der linearen Regression von Sensor 1
    x = np.array(datenreihen[0])
    y = np.array(datenreihen[1])
    steigung_1 = (len(x) * np.sum(x*y) - np.sum(x) * np.sum(y)) / (len(x)*np.sum(x*x) - np.sum(x) ** 2)
    offset_1 = (np.sum(y) - steigung_1 *np.sum(x)) / len(x)
    print(f'[Sensor 1] Trend: ({steigung_1:.6f})x + ({offset_1:+.6f})')
    
    # Berechnung der linearen Regression von Sensor 2
    x = np.array(datenreihen[0])
    y = np.array(datenreihen[2])
    steigung_2 = (len(x) * np.sum(x*y) - np.sum(x) * np.sum(y)) / (len(x)*np.sum(x*x) - np.sum(x) ** 2)
    offset_2 = (np.sum(y) - steigung_1 *np.sum(x)) / len(x)
    print(f'[Sensor 2] Trend: ({steigung_2:.6f})x + ({offset_2:+.6f})')

    # Plot der linearen Regression
    # plt.scatter(x, y)
    # Hier muss noch ein Diagram erzeugt werden

    # Bereinigung des Trends beider Sensorreihen
    datenreihen_ohne_trend = [[], []]
    for i, e in enumerate(datenreihen[1]):
        datenreihen_ohne_trend[0].append(e - (steigung_1*(datenreihen[0][i])+offset_1))
    for i, e in enumerate(datenreihen[2]):
        datenreihen_ohne_trend[1].append(e - (steigung_2*(datenreihen[0][i])+offset_2))

    # Plot der vom Trend bereinigten Sensorreihen
    # Hier muss noch ein Diagram erzeugt werden

    # Low-Pass-Filterung der Sensorreihen
    datenreihen_low_pass = []
    for i, e in enumerate(datenreihen_ohne_trend):
        datenreihen_low_pass.append(low_pass_filter(e, 200))

    # Plot der low-pass Sensorreihen
    plot_werte(datenreihen_low_pass, ["Low-Pass Sensor 1", "Low-Pass Sensor 2"])
    # Hier muss noch ein Diagram erzeugt werden

    # Hoch-Pass-Filterung der Sensorreihen
    datenreihen_hoch_pass = [[], []]
    for i, e in enumerate(datenreihen_low_pass[0]):
        datenreihen_hoch_pass[0].append(datenreihen_ohne_trend[0][i] - e)
    for i, e in enumerate(datenreihen_low_pass[1]):
        datenreihen_hoch_pass[1].append(datenreihen_ohne_trend[1][i] - e)

    # Plot der hoch-pass Sensorreihen
    plot_werte(datenreihen_hoch_pass, ["Hoch-Pass Sensor 1", "Hoch-Pass Sensor 2"])
    # Hier muss noch ein Diagram erzeugt werden
