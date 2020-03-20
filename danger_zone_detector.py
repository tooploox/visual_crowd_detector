import numpy as np
from scipy.spatial import distance_matrix


def calculate_centers(bboxes):
    """
    Calculates centers of bounding boxes.

    :param bboxes: List of bounding boxes in a format (y1, x1, y2, x2).
    :return: Array of center points, one per bounding box.
    """
    centers = []
    for bbox in bboxes:
        x = (bbox[1] + bbox[3]) // 2
        y = (bbox[0] + bbox[2]) // 2
        center = [x, y]
        centers.append(center)

    return np.asarray(centers)


def calculate_cell_centers(image, step_size=30):
    """
    Divides image into cells and calculates center points of those cells.
    :param image: 2 or 3 channel image
    :param step_size: size of the cell
    :return: Array of cell centers.
    """

    x_steps = [i for i in range(0, image.shape[1], step_size)]
    y_steps = [i for i in range(0, image.shape[0], step_size)]

    cell_centers = []
    for x_step in x_steps:
        for y_step in y_steps:
            cell_centers.append([x_step, y_step])

    return np.asarray(cell_centers)


def find_cell(cell_centers, point):
    """
    Finds to which cell a given point belongs based
    on it's distance from all the cells centers.

    :param cell_centers: Array of cell centers.
    :param point: Point (x, y)
    :return:  Index of a cell to which the points belongs.
    """

    distances = distance_matrix(cell_centers, [point])

    return np.argmin(distances)
