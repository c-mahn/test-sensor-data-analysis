# Import von Bibliotheken
# -----------------------------------------------------------------------------

import os
import datetime

# Funktionen
# -----------------------------------------------------------------------------

def timestamp_to_seconds(timestamp):
    """
    Eingabe-Format: "YYYY-MM-DD HH:MM:SS"
    """
    timestamp = timestamp.split(" ")

    datum = timestamp[0].split("-")
    for i, e in enumerate(datum):
        datum[i] = int(e)

    uhrzeit = timestamp[1].split(":")
    for i, e in enumerate(uhrzeit):
        uhrzeit[i] = int(e)

    datum = datetime.date(datum[0], datum[1], datum[2]).toordinal()
    ausgabe = ((datum*24+uhrzeit[0])*60+uhrzeit[1])*60+uhrzeit[2]
    return(ausgabe)


# Klassen
# -----------------------------------------------------------------------------

# Beginn des Programms
# -----------------------------------------------------------------------------

if(__name__ == '__main__'):
    with open(os.path.join("data", "time_series_raw.txt"), "r") as f:
        data = f.readlines()
    for i, e in enumerate(data):
        temp = e.strip()
        temp = temp.split(" ")
        temp[0] = f"{temp[0]} {temp[1]}"
        temp.pop(1)
        temp[0] = timestamp_to_seconds(temp[0])
        data[i] = temp
    with open(os.path.join("data", "time_series_converted.txt"), "w") as f:
        for i in data:
            f.write(f'{i[0]}')
            i.pop(0)
            for j in i:
                f.write(f'; {j}')
            f.write("\n")
