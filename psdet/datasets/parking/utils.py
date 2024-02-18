"""Defines data structure."""
import math
from enum import Enum
import numpy as np
from psdet.models.detectors.util.dcl import angle_label_decode
class PointShape(Enum):
    """The point shape types used to pair two marking points into slot."""
    none = 0
    l_down = 1
    t_down = 2
    t_middle = 3
    t_up = 4
    l_up = 5


def direction_diff(direction_a, direction_b):
    """Calculate the angle between two direction."""
    diff = abs(direction_a - direction_b)
    return diff if diff < math.pi else 2*math.pi - diff


def determine_point_shape(point, vector):
    """Determine which category the point is in."""
    bridge_angle_diff = 0.09757113548987695 + 0.1384059287593468
    separator_angle_diff = 0.284967562063968 + 0.1384059287593468

    vec_direct = math.atan2(vector[1], vector[0])
    vec_direct_up = math.atan2(-vector[0], vector[1])
    vec_direct_down = math.atan2(vector[0], -vector[1])
    if point.shape < 0.5:
        if direction_diff(vec_direct, point.direction) < bridge_angle_diff:
            return PointShape.t_middle
        if direction_diff(vec_direct_up, point.direction) < separator_angle_diff:
            return PointShape.t_up
        if direction_diff(vec_direct_down, point.direction) < separator_angle_diff:
            return PointShape.t_down
    else:
        if direction_diff(vec_direct, point.direction) < bridge_angle_diff:
            return PointShape.l_down
        if direction_diff(vec_direct_up, point.direction) < separator_angle_diff:
            return PointShape.l_up
    return PointShape.none


def calc_point_squre_dist(point_a, point_b):
    """Calculate distance between two marking points."""
    distx = point_a[0] - point_b[0]
    disty = point_a[1] - point_b[1]
    return distx ** 2 + disty ** 2


def calc_point_direction_angle(angle_gt, angle_pred):
    """Calculate angle between direction in rad."""

    return direction_diff(angle_gt, angle_pred)


def match_marking_points(point_a, point_b):
    """
    Determine whether a detected point match ground truth.
    Args:
        point_a: gt
        point_b: pred
    """
    
    squared_distance_thresh = 0.000277778  # 10 pixel in 600*600 image
    dist_square = calc_point_squre_dist(point_a, point_b)
    #if min(point_a.shape[1], point_b.shape[1]) <= 2:
    return dist_square < squared_distance_thresh

def match_angle_points(point_a, point_b):
    """Determine whether a detected point angle match ground truth."""
    if len(point_a) == 184:         #pcgnn-csl
        angle = np.argmax(point_a[4:], 0)
        gt = (angle - 90) / 180 * math.pi
    elif len(point_a) == 12:        #pcgnn-dcl
        dcl_label = point_a[4:]
        angle = angle_label_decode([dcl_label], 180, 180 / 256, mode=0)  # angle 0 ~ 179
        gt = (angle - 90) / 180 * math.pi
    elif len(point_a) == 5:         #pcgnn
        gt = math.asin(point_a[3])
    elif len(point_a) == 4:         #dmpr
        gt = math.asin(math.sin(point_a[2]))
    else:                           #gnn
        return False
    if_point_match = match_marking_points(point_a, point_b)
    direction_angle_thresh = 0.174532925199432957  # 10 degree in rad
    pred = math.asin(point_b[3])
    angle_diff = calc_point_direction_angle(gt, pred)
    return (if_point_match and angle_diff < direction_angle_thresh)

def match_slots(slot_a, slot_b):
    """Determine whether a detected slot match ground truth."""
    squared_distance_thresh = 0.000277778  # 10 pixel in 600*600 image
    dist_x1 = slot_b[0] - slot_a[0]
    dist_y1 = slot_b[1] - slot_a[1]
    squared_dist1 = dist_x1**2 + dist_y1**2
    dist_x2 = slot_b[2] - slot_a[2]
    dist_y2 = slot_b[3] - slot_a[3]
    squared_dist2 = dist_x2 ** 2 + dist_y2 ** 2
    return (squared_dist1 < squared_distance_thresh
            and squared_dist2 < squared_distance_thresh)





