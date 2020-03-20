import numpy as np
from scipy.spatial import distance_matrix


def calculate_centers(bboxes):
    centers = []
    for bbox in bboxes:
        x = (bbox[1] + bbox[3]) // 2
        y = (bbox[0] + bbox[2]) // 2
        center = [x, y]
        centers.append(center)

    return np.asarray(centers)


def calculate_cell_centers(image, step_size=30):
    x_steps = [i for i in range(0, image.shape[1], step_size)]
    y_steps = [i for i in range(0, image.shape[0], step_size)]

    cell_centers = []
    for x_step in x_steps:
        for y_step in y_steps:
            cell_centers.append([x_step, y_step])

    return np.asarray(cell_centers)


def find_cell(cell_centers, point):
    distances = distance_matrix(cell_centers, [point])

    return np.argmin(distances)