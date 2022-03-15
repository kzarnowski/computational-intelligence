from tokenize import group
from matplotlib import markers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

np.set_printoptions(precision=16)

BIRDS_NUM = 16
DIM = 2
PHI = [2.05, 2.05]

PHI_SUM = sum(PHI)  # PHI_SUM SHOULD BE > 4
CHI = 2/(PHI_SUM - 2 + sqrt(PHI_SUM**2-4*PHI_SUM))

FRAME = (-2, 2)


def rosen(x):
    return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


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

    return updated_v


if __name__ == '__main__':
    f = rosen

    current_positions = np.random.uniform(
        low=FRAME[0],
        high=FRAME[1],
        size=(BIRDS_NUM, DIM)
    )
    best_positions = np.copy(current_positions)
    velocity = np.zeros(shape=(BIRDS_NUM, DIM))
    current_results = np.apply_along_axis(rosen, 1, current_positions)
    best_results = current_results.copy()

    # best_evo stores coordinates + iteration number + function result
    best_evo = {
        'it': [],
        'res': [],
        'pos': []
    }

    group_best_idx = np.argmin(best_results)
    group_best_pos = best_positions[group_best_idx]
    group_best_res = best_results[group_best_idx]

    for it in range(1000):
        for i in range(BIRDS_NUM):
            # update velocity
            velocity[i] = update_v(
                curr_v=velocity[i],
                curr_pos=current_positions[i],
                personal_best_pos=best_positions[i],
                group_best_pos=group_best_pos
            )

            # update position
            current_positions[i] += velocity[i]

            # update personal best
            # if current_results[i] < best_results[i]:
            #     best_positions[i] = current_positions[i]
            #     best_results[i] = current_results[i]

        # calculate new values of the given function
        current_results = np.apply_along_axis(rosen, 1, current_positions)

        # update personal best
        mask = current_results < best_results
        best_positions[mask] = current_positions[mask]
        best_results[mask] = current_results[mask]

        # update group best position
        if np.min(best_results) < group_best_res:
            group_best_idx = np.argmin(best_results)
            group_best_pos = best_positions[group_best_idx]
            group_best_res = best_results[group_best_idx]

            # append to best evo history
            best_evo['it'].append(it)
            best_evo['res'].append(group_best_res)
            best_evo['pos'].append(group_best_pos.copy())

    # RESULTS:

    best_evo_df = pd.DataFrame(best_evo)
    print(best_evo_df)

    # CONTOUR PLOT
    if DIM == 2:
        x = np.arange(FRAME[0], FRAME[1], 0.1)
        y = np.arange(FRAME[0], FRAME[1], 0.1)
        X, Y = np.meshgrid(x, y)
        Z = rosen2d((X, Y))

        evo_x = best_evo_df['pos'].apply(lambda x: x[0])
        evo_y = best_evo_df['pos'].apply(lambda x: x[1])

        plt.contour(X, Y, Z, levels=[
                    x**2.5 for x in range(2, 25)], cmap='plasma')
        plt.scatter(evo_x, evo_y, marker='o',
                    color='black', alpha=0.2, linewidths=0.1)
        plt.show()
