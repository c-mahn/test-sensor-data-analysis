# Import von Bibliotheken
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import math as m
import sys
import os

# Funktionen
# -----------------------------------------------------------------------------

# Haupt-Programm
# -----------------------------------------------------------------------------

if(__name__ == '__main__'):
    with open(os.path.join("data", "time_series_converted.txt"), "r") as f:
        data = f.readlines()
    for i, e in enumerate(data):
        data[i] = e.split(";")
        for j, e in enumerate(data[i]):
            data[i][j] = float(e.strip())
        print(data[i])
