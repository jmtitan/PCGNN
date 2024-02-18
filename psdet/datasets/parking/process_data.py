"""Perform data augmentation and preprocessing."""
import math
import cv2 as cv
import numpy as np
from psdet.models.detectors.util.dcl import angle_label_encode

def boundary_check(centralied_marks):
    """Check situation that marking point appears too near to border."""
    for mark in centralied_marks:
        if mark[0] < -260 or mark[0] > 260 or mark[1] < -260 or mark[1] > 260:
            return False
    return True

def gaussian_label_cpu(label, num_class, u=0, sig=4.0):
    """
    转换成CSL Labels：
        用高斯窗口函数根据角度θ的周期性赋予gt labels同样的周期性，使得损失函数在计算边界处时可以做到“差值很大但loss很小”；
        并且使得其labels具有环形特征，能够反映各个θ之间的角度距离
    Args:
        label (float32):[1], theta class
        num_theta_class (int): [1], theta class num
        u (float32):[1], μ in gaussian function
        sig (float32):[1], σ in gaussian function, which is window radius for Circular Smooth Label
    Returns:
        csl_label (array): [num_theta_class], gaussian function smooth label
    """
    x = np.arange(-num_class/2, num_class/2)
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))
    index = int(num_class/2 - label)
    return np.concatenate([y_sig[index:],
                           y_sig[:index]], axis=0)

def overlap_check(centralied_marks, image_w, image_h):
    """Check situation that multiple marking points appear in same cell."""
    for i in range(len(centralied_marks) - 1):
        i_x = centralied_marks[i, 0]
        i_y = centralied_marks[i, 1]
        for j in range(i + 1, len(centralied_marks)):
            j_x = centralied_marks[j, 0]
            j_y = centralied_marks[j, 1]
            if abs(j_x - i_x) < image_w / 16 and abs(j_y - i_y) < image_h / 16:
                return False
    return True

def generalize_marks_gnn(centralied_marks, image_w, image_h):
    """Convert coordinate to [0, 1] and calculate direction label."""
    generalized_marks = []
    for mark in centralied_marks:
        xval = (mark[0] + image_w / 2) / image_w
        yval = (mark[1] + image_h / 2) / image_h
        generalized_marks.append([xval, yval])
    return np.array(generalized_marks)

def generalize_marks_pcgnn(centralied_marks, image_w, image_h):
    """Convert coordinate to [0, 1] and calculate direction label."""
    generalized_marks = []
    for mark in centralied_marks:
        xval = (mark[0] + image_w/2) / image_w
        yval = (mark[1] + image_h/2) / image_h
        x1val = (mark[2] + image_w/2) / image_w
        y1val = (mark[3] + image_h/2) / image_h
        length = math.sqrt(pow(yval - y1val, 2) + pow(xval - x1val, 2))
        direction = math.atan2(y1val-yval, x1val-xval) # -pi ~ pi
        # direction = math.atan2(mark[3] - mark[1], mark[2] - mark[0])
        generalized_marks.append([xval, yval, math.cos(direction), math.sin(direction), length])
    return np.array(generalized_marks)

def generalize_marks_dmpr(centralied_marks,  image_w, image_h):
    """Convert coordinate to [0, 1] and calculate direction label."""
    generalized_marks = []
    for mark in centralied_marks:
        xval = (mark[0] + image_w / 2) / image_w
        yval = (mark[1] + image_h / 2) / image_h
        direction = math.atan2(mark[3] - mark[1], mark[2] - mark[0])
        generalized_marks.append([xval, yval, direction, mark[4]])
    return np.array(generalized_marks)

def generalize_marks_dcl(centralied_marks, image_w, image_h):
    """Convert coordinate to [0, 1] and calculate direction label."""
    generalized_marks = []
    for mark in centralied_marks:
        xval = (mark[0] + image_w/2) / image_w
        yval = (mark[1] + image_h/2) / image_h
        x1val = (mark[2] + image_w/2) / image_w
        y1val = (mark[3] + image_h/2) / image_h
        length = math.sqrt(pow(yval - y1val, 2) + pow(xval - x1val, 2))
        theta = math.atan((y1val-yval)/(x1val-xval))
        angle = (theta * 180 / math.pi) + 90  # -90~90 -> -180~0
        if x1val > xval:
            flag = 0.0  # 决定角度朝向是一四象限还是二三象限
        else:
            flag = 1.0
        dcl_label = angle_label_encode(angle, 180, 180 / 256, mode=0)
            # direction = math.atan2(mark[3] - mark[1], mark[2] - mark[0])
        generalized_marks.append(np.hstack(([xval, yval, length, flag], dcl_label)))
    return np.array(generalized_marks).astype(np.float16)

def generalize_marks_csl(centralied_marks, image_w, image_h):
    """Convert coordinate to [0, 1] and calculate direction label."""
    generalized_marks = []
    for mark in centralied_marks:
        xval = (mark[0] + image_w/2) / image_w
        yval = (mark[1] + image_h/2) / image_h
        x1val = (mark[2] + image_w/2) / image_w
        y1val = (mark[3] + image_h/2) / image_h
        length = math.sqrt(pow(yval - y1val, 2) + pow(xval - x1val, 2))
        theta = math.atan((y1val-yval)/(x1val-xval))
        angle = (theta * 180 / math.pi) + 90 # -90~90 -> 0~180
        csl_label = gaussian_label_cpu(label=angle, num_class=180, u=0, sig=2.0)
        if x1val > xval:
            flag = 0.0 # 决定角度朝向是一四象限还是二三象限
        else:
            flag = 1.0
        generalized_marks.append(np.hstack(([xval, yval, length, flag], csl_label)))
    return np.array(generalized_marks).astype(np.float16)

def rotate_vector(vector, angle_degree):
    """Rotate a vector with given angle in degree."""
    angle_rad = math.pi * angle_degree / 180
    xval = vector[0]*math.cos(angle_rad) + vector[1]*math.sin(angle_rad)
    yval = -vector[0]*math.sin(angle_rad) + vector[1]*math.cos(angle_rad)
    return xval, yval

def rotate_centralized_marks(centralied_marks, angle_degree):
    """Rotate centralized marks with given angle in degree."""
    rotated_marks = centralied_marks.copy()
    for i in range(centralied_marks.shape[0]):
        mark = centralied_marks[i]
        rotated_marks[i, 0:2] = rotate_vector(mark[0:2], angle_degree)
        rotated_marks[i, 2:4] = rotate_vector(mark[2:4], angle_degree)
    return rotated_marks

def rotate_image(image, angle_degree):
    """Rotate image with given angle in degree."""
    rows, cols, _ = image.shape
    rotation_matrix = cv.getRotationMatrix2D((rows/2, cols/2), angle_degree, 1)
    return cv.warpAffine(image, rotation_matrix, (rows, cols))
