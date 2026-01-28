import numpy as np


def get_circle_from_3_pts(p1, p2, p3):
    """Return (cx, cy, r) from 3 non-collinear points."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    x1_2y1_2 = x1 * x1 + y1 * y1
    x2_2y2_2 = x2 * x2 + y2 * y2
    x3_2y3_2 = x3 * x3 + y3 * y3

    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    cx = (x1_2y1_2 * (y2 - y3) + x2_2y2_2 * (y3 - y1) + x3_2y3_2 * (y1 - y2)) / d
    cy = (x1_2y1_2 * (x3 - x2) + x2_2y2_2 * (x1 - x3) + x3_2y3_2 * (x2 - x1)) / d
    r = np.hypot(cx - x1, cy - y1)
    return cx, cy, r


def is_points_collinear(p1, p2, p3):
    """Check if points are collinear using cross product."""
    x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
    x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
    return abs(x1 * y2 - x2 * y1) < 1e-6


def fit_circle_least_squares(points: np.ndarray):
    """Algebraic least squares circle fit:
    x^2 + y^2 + ax + by + c = 0
    center = (-a/2, -b/2), r = sqrt((a^2+b^2)/4 - c)
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    a, bcoef, c = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = -a / 2.0, -bcoef / 2.0
    r2 = cx * cx + cy * cy - c
    r = np.sqrt(max(r2, 0.0))
    return cx, cy, r


def fit_circle_RANSAC(
    points: np.ndarray, num_iter: int, thresh: float
) -> tuple[tuple[float, float, float], np.ndarray]:
    """Fit a circle to a set of noisy points using RANSAC.

    Args:
        points (np.ndarray): N x 2
        num_iter (int): Number of iterations to run the RANSAC loop for
        thresh (float): Inlier threshold in pixels (abs(distance_to_center - r))

    Raises:
        ValueError: If <3 points are supplied.
        np.linalg.LinAlgError: If fit_circle_least_squares fails in the refinement step.

    Returns:
        tuple[tuple[float, float], float], np.ndarray: ((cx, cy), r), inlier_points
    """
    if len(points) < 3:
        raise ValueError("We need at least 3 points to fit a circle.")

    # RANSAC
    best_inlier_mask = np.zeros(len(points), dtype=bool)

    for _ in range(num_iter):
        while True:
            sample_idxs = np.random.choice(
                np.arange(len(points)), size=3, replace=False
            )
            p1, p2, p3 = points[sample_idxs]
            if not is_points_collinear(p1, p2, p3):
                break

        cx, cy, r = get_circle_from_3_pts(p1, p2, p3)
        d = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
        res = np.abs(d - r)
        inlier_mask = res < thresh
        count = np.count_nonzero(inlier_mask)

        if count > np.count_nonzero(best_inlier_mask):
            best_inlier_mask = inlier_mask

    # Refinement
    inlier_points = points[best_inlier_mask]
    circle = fit_circle_least_squares(inlier_points)

    return circle, inlier_points
