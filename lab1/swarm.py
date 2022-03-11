from matplotlib import markers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BIRDS_NUM = 16


def rosen(x):
    return 100 * (x[1] - x[0] * x[0])**2 + (1 - x[0])**2


def f(x, y): return 100 * (y - x**2)**2 + (1 - x)**2


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

b = pd.DataFrame({'bx': bx, 'by': by})

bz = np.apply_along_axis(rosen, 1, b)

b['bz'] = bz

best_evo = {
    'bx': [],
    'by': [],
    'bz': []
}


for k in range(1000):
    best_bird = b['bz'].idxmin()
    best_evo['bx'].append(b['bx'][best_bird])
    best_evo['by'].append(b['by'][best_bird])
    best_evo['bz'].append(b['bz'][best_bird])
    for i in range(BIRDS_NUM):
        sigma_x = abs(b['bx'][i] - b['bx'][best_bird])
        sigma_y = abs(b['by'][i] - b['by'][best_bird])

        test_x = np.random.normal(
            loc=(b['bx'][i] + b['bx'][best_bird])/2,
            scale=sigma_x
        )

        test_y = np.random.normal(
            loc=(b['by'][i] + b['by'][best_bird])/2,
            scale=sigma_y
        )

        test_z = rosen((test_x, test_y))

        if test_z < b['bz'][i]:
            b.iloc[i] = (test_x, test_y, test_z)


evo = pd.DataFrame(best_evo)
print(evo)


X, Y = np.meshgrid(x, y)
Z = f(X, Y)
plt.contour(X, Y, Z, 200)
plt.scatter(evo['bx'], evo['by'], marker='o',
            color='red', alpha=0.2, linewidths=0.1)
plt.show()
