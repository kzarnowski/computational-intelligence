from matplotlib import markers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


BIRDS_NUM = 16

PHI = [2.05, 2.05]
PHI = sum(PHI)  # PHI SHOULD BE > 4
CHI = 2/(PHI - 2 + sqrt(PHI**2-4*PHI))  # 0.7298

"""
1. Wymyslic kryterium stopu
2) Przerobić istniejący skrypt tak aby zamiast wariantu “bare bones”
realizował obliczenia w oparciu o wariant kanoniczny (patrz wykład).
3) Rozbudować kod funkcji celu oraz kod silnika optymalizacyjnego tak aby
możliwe było uzyskiwanie wyników dla funkcji n-wymiarowych.
"""


# MOŻNA NAPRAWIĆ ZMIENIAJĄC DATA FRAME DLA PTAKOW NA TABLICE NUMPY 3D
# TABLICA BĘDZIE ROZMIARU [BIRDS_NUM, 3, DIM], GDZIE:
# DIM - WYMIAR PRZESTRZENI
# 3 - BO POTRZEBUJEMY WSPÓŁRZĘDNYCH CURRENT, BEST_PERSONAL + WEKTOR V


def rosen(x):
    return 100 * (x[1] - x[0] * x[0])**2 + (1 - x[0])**2


def update_v(
        curr_v: np.ndarray,
        curr_pos: np.ndarray,
        personal_best_pos: np.ndarray,
        group_best_pos: np.ndarray):

    personal_trend = np.random.uniform(
        0, PHI[0])*np.ndarray(personal_best_pos - curr_pos)
    group_trend = np.random.uniform(
        0, PHI[1])*np.ndarray(group_best_pos - curr_pos)
    updated_v = CHI * (curr_v + personal_trend + group_trend)
    return updated_v


x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)

n = len(x)

xy = np.array([(xi, yi) for xi in x for yi in y])

z = np.apply_along_axis(rosen, 1, xy)

# warunek stopu jako odchylenie standardowe (pulsacyjne PSO), jeśli ptaszki są
# blisko siebie to rozrzucamy je ponownie w przestrzeni

df = pd.DataFrame(xy, columns=['x', 'y'])
df['z'] = z

# print(df.head())

bx = np.random.uniform(low=-2, high=2, size=BIRDS_NUM)
by = np.random.uniform(low=-2, high=2, size=BIRDS_NUM)

b = pd.DataFrame({'x': bx, 'y': by})

bz = np.apply_along_axis(rosen, 1, b)

b['z'] = bz
b['v'] = np.ndarray(shape=(BIRDS_NUM, 3), dtype=np.float32,
                    buffer=np.ndarray(buffer=np.zeros((1, 3)), shape=(1, 3)))
b['personal_best_x'] = b['x'].copy()
b['personal_best_y'] = b['y'].copy()


best_evo = {
    'x': [],
    'y': [],
    'z': [],
    'v': [],
}


for k in range(1000):
    best_bird = b['z'].idxmin()
    group_best_pos = [b['x'][best_bird], b['y'][best_bird]]
    for col in ['x', 'y', 'z', 'v']:
        best_evo[col].append(b[col][best_bird])

    for i in range(BIRDS_NUM):
        v = update_v()


evo = pd.DataFrame(best_evo)
print(evo)


X, Y = np.meshgrid(x, y)
Z = rosen((X, Y))
plt.contour(X, Y, Z, levels=[x**2.5 for x in range(2, 25)], cmap='plasma')
plt.scatter(evo['bx'], evo['by'], marker='o',
            color='black', alpha=0.2, linewidths=0.1)
plt.show()
