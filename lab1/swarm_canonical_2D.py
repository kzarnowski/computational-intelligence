from matplotlib import markers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

np.set_printoptions(precision=16)

BIRDS_NUM = 16
DIM = 3

PHI = [2.05, 2.05]
PHI_SUM = sum(PHI)  # PHI SHOULD BE > 4
CHI = 2/(PHI_SUM - 2 + sqrt(PHI_SUM**2-4*PHI_SUM))  # 0.7298

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
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


def rosen2d(x):
    return 100 * (x[1] - x[0] * x[0])**2 + (1 - x[0])**2


def update_v(
        curr_v: np.ndarray,
        curr_pos: np.ndarray,
        personal_best_pos: np.ndarray,
        group_best_pos: np.ndarray):

    personal_trend = np.random.uniform(
        0, PHI[0])*(personal_best_pos - curr_pos)
    group_trend = np.random.uniform(
        0, PHI[1])*(group_best_pos - curr_pos)
    updated_v = CHI * (curr_v + personal_trend + group_trend)

    # print(f"curr_v: {curr_v} : new_v: {updated_v}")
    return updated_v


birds = np.ndarray(shape=(BIRDS_NUM, 3, DIM), dtype=np.float64)
# AXES:
# 0 - birds
# 1 - parameters
# 2 - coordinates

# PARAMETERS:
# 0 - current position
# 1 - personal best position
# 2 - velocity vector


x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)

n = len(x)

xy = np.array([(xi, yi) for xi in x for yi in y])

z = np.apply_along_axis(rosen, 1, xy)

# warunek stopu jako odchylenie standardowe (pulsacyjne PSO), jeśli ptaszki są
# blisko siebie to rozrzucamy je ponownie w przestrzeni


bx = np.random.uniform(low=-2, high=2, size=BIRDS_NUM)
by = np.random.uniform(low=-2, high=2, size=BIRDS_NUM)

b = pd.DataFrame({'x': bx, 'y': by})

bz = np.apply_along_axis(rosen, 1, b)

current_pos = np.array([[bx[i], by[i], bz[i]]
                        for i in range(BIRDS_NUM)], dtype=np.float64)

personal_best = np.copy(current_pos)

# velocity = np.zeros(shape=(BIRDS_NUM, 3))
velocity = np.random.uniform(-1, 1, size=(BIRDS_NUM, 3))

birds[:, 0, :] = current_pos
birds[:, 1, :] = personal_best
birds[:, 2, :] = velocity

print(birds.shape)

print(birds[1, 2, :])
# How to access data in birds:
# birds[i, 0, :] - current position of i-th bird
# birds[i, 1, :] - best position of i-th bird
# birds[i, 2, :] - velocity of i-th bird
#
# birds[i, k, 0] - x coordinate
# birds[i, k, 1] - y coordinate
# birds[i, k, 2] - rosenbrock function value (always 0 for velocity)

best_evo = {
    'k': [],  # iteration
    'x': [],
    'y': [],
    'z': []
}

# initial best bird
best = np.argmin(birds[:, 0, 2])
group_best_pos = birds[best, 0, :]

for k in range(1000):

    for i in range(BIRDS_NUM):
        birds[i, 2, :2] = update_v(
            curr_pos=birds[i, 0, :2],
            personal_best_pos=birds[i, 1, :2],
            group_best_pos=group_best_pos[:2],
            curr_v=birds[i, 2, :2]
        )

        # update position x and y
        birds[i, 0, :2] = birds[i, 0, :2] + birds[i, 2, :2]

        # calculate new value of rosenbrock function (z)
        birds[i, 0, 2] = rosen(birds[i, 0, :2])

        # update personal best
        if birds[i, 0, 2] < birds[i, 1, 2]:
            birds[i, 1] = birds[i, 0]

    best = np.argmin(birds[:, 1, 2])
    if birds[best, 0, 2] < group_best_pos[2]:
        group_best_pos = birds[best, 0, :]
        for i, col in enumerate(['x', 'y', 'z']):
            best_evo[col].append(birds[best, 0, i])
        best_evo['k'].append(k)

    #print(best, group_best_pos)


evo = pd.DataFrame(best_evo)
print(evo)

X, Y = np.meshgrid(x, y)

print(X.shape, Y.shape)
Z = rosen2d((X, Y))

plt.contour(X, Y, Z, levels=[x**2.5 for x in range(2, 25)], cmap='plasma')


plt.scatter(evo['x'], evo['y'], marker='o',
            color='black', alpha=0.2, linewidths=0.1)
plt.show()
