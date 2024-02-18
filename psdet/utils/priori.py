import os
import json
from psdet.utils.config import get_config
import numpy as np
from PIL import Image


def pass_through_third_point_test(points, slots):
    dots = []
    slots = np.array(slots)
    slots = slots[:,:2]
    idx = 0
    for i in slots:
        first_point = i[0] - 1
        second_point = i[1] - 1
        x_1 = points[first_point][0]
        y_1 = points[first_point][1]
        x_2 = points[second_point][0]
        y_2 = points[second_point][1]
        for p in points:
            x_0 = p[0]
            y_0 = p[1]
            tag = idx + 1
            if x_0 == x_1 or x_0 == x_2:
                continue
            if [tag, first_point] not in slots and [first_point, tag] not in slots \
                and [tag, second_point] not in slots and [second_point, tag] not in slots:
                continue
            vec1 = np.array([x_0 - x_1, y_0 - y_1])
            vec2 = np.array([x_2 - x_0, y_2 - y_0])
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            dots.append(np.dot(vec1, vec2))
            idx += 1
    return dots


def find_limited_distance(x, y, points, slots):
    for i in slots:
        first_point = i[0] - 1
        second_point = i[1] - 1
        x.append(abs(points[first_point][0] - points[second_point][0]))
        y.append(abs(points[first_point][1] - points[second_point][1]))


def centralize_points(file_path, points):

    img_file = file_path.replace('.json', '.jpg').replace('label', 'img')
    image = Image.open(img_file)
    image_size = image.size[0]
    points[:, 0:4] += 0.5
    points[:, 0:4] /= image_size
    return points


def inference_limit(cfg):
    x_minus = []
    y_minus = []
    thresh = []
    json_path = cfg['root_path'] + '/label/'
    for file in os.listdir(json_path):
        # print(file)
        with open(json_path + file, 'r') as f:
            str = f.read()
            data = json.loads(str)
            if np.array(data['marks']).shape[0] == 1 or len(data['slots']) == 0:
                continue
            if isinstance(data['slots'][0], int):
                data['slots'] = [data['slots']]
            points = centralize_points(json_path + file, np.array(data['marks']))
            find_limited_distance(x_minus, y_minus, points, data['slots'])
            if len(points) > 3:
                thresh.extend(pass_through_third_point_test(points, data['slots']))

    x_minus = np.array(x_minus)
    y_minus = np.array(y_minus)
    VSLOT_MAX_DIST = x_minus.max()
    HSLOT_MAX_DIST = y_minus.max()
    MAX_DIST = np.sqrt(np.square(x_minus) + np.square(y_minus)).max()
    MIN_DIST = np.sqrt(np.square(x_minus) + np.square(y_minus)).min()
    thresh_max = max(thresh)
    # print('*'*20, 'interference_slot', '*'*20)
    # print(f'VSLOT_MAX_DIST = {x_minus.max()}\n'
    #       f'HSLOT_MAX_DIST = {y_minus.max()}\n'
    #       f'MAX_DIST = {np.sqrt(np.square(x_minus)+np.square(y_minus)).max()}\n'
    #       f'MIN_DIST = {np.sqrt(np.square(x_minus)+np.square(y_minus)).min()}\n'
    #       f'thresh_max = {max(thresh)}'
    #       )
    # print('*' * 20, 'interference_slot_end', '*' * 20)
    return VSLOT_MAX_DIST, HSLOT_MAX_DIST, MAX_DIST, MIN_DIST, thresh_max

