"""Inference demo of directional point detector."""
import math
import os
import numpy as np
import cv2 as cv
import time
import torch
from thop import profile
from torchvision.transforms import ToTensor
from psdet.utils.config import get_config
from psdet.utils.common import get_logger
from psdet.models.builder import build_model

inference_time = []
def plot_points_dmpr(image, pred_points):
    """Plot marking points on the image."""
    if not pred_points:
        return
    height = image.shape[0]
    width = image.shape[1]
    # zoom = math.sqrt(height **2 + width ** 2)
    for confidence, marking_point in pred_points[0]:
        p0_x = width * marking_point[0]
        p0_y = height * marking_point[1]
        cos_val = math.cos(marking_point[2])
        sin_val = math.sin(marking_point[2])
        p1_x = p0_x + 100 * cos_val
        p1_y = p0_y + 100 * sin_val
        # p2_x = p0_x - 50*sin_val
        # p2_y = p0_y + 50*cos_val
        # p3_x = p0_x + 50*sin_val
        # p3_y = p0_y - 50*cos_val
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        # p2_x = int(round(p2_x))
        # p2_y = int(round(p2_y))
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 2)
        cv.putText(image, str(confidence), (p0_x, p0_y),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        # if marking_point.shape > 0.5:
        #     cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (0, 0, 255), 2)
        # else:
        #     p3_x = int(round(p3_x))
        #     p3_y = int(round(p3_y))
        #     cv.line(image, (p2_x, p2_y), (p3_x, p3_y), (0, 0, 255), 2)
def plot_points(image, pred_points):
    """Plot marking points on the image."""
    if not pred_points:
        return
    height = image.shape[0]
    width = image.shape[1]
    # zoom = math.sqrt(height **2 + width ** 2)
    for confidence, marking_point in pred_points[0]:
        p0_x = width * marking_point[0]
        p0_y = height * marking_point[1]
        cos_val = marking_point[2]
        sin_val = marking_point[3]
        length = marking_point[4]
        p1_x = p0_x + width * length * cos_val
        p1_y = p0_y + height * length * sin_val
        # p2_x = p0_x - 50*sin_val
        # p2_y = p0_y + 50*cos_val
        # p3_x = p0_x + 50*sin_val
        # p3_y = p0_y - 50*cos_val
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        # p2_x = int(round(p2_x))
        # p2_y = int(round(p2_y))
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 2)
        cv.putText(image, str(confidence), (p0_x, p0_y),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        # if marking_point.shape > 0.5:
        #     cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (0, 0, 255), 2)
        # else:
        #     p3_x = int(round(p3_x))
        #     p3_y = int(round(p3_y))
        #     cv.line(image, (p2_x, p2_y), (p3_x, p3_y), (0, 0, 255), 2)


def plot_slots(image, pred_points, slots):
    """Plot parking slots on the image."""
    if not pred_points or not slots:
        return
    # marking_points = list(list(zip(*pred_points))[1])
    height = image.shape[0]
    width = image.shape[1]
    for slot in slots[0]:
        point_a = slot[1][:2]
        point_b = slot[1][2:4]
        p0_x = width * point_a[0] - 0.5
        p0_y = height * point_a[1] - 0.5
        p1_x = width * point_b[0] - 0.5
        p1_y = height * point_b[1] - 0.5
        # vec = np.array([p1_x - p0_x, p1_y - p0_y])
        # vec = vec / np.linalg.norm(vec)
        # distance = calc_point_squre_dist(point_a, point_b)
        # if config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST:
        #     separating_length = config.LONG_SEPARATOR_LENGTH
        # elif config.HSLOT_MIN_DIST <= distance <= config.HSLOT_MAX_DIST:
        #     separating_length = config.SHORT_SEPARATOR_LENGTH
        # p2_x = p0_x + height * separating_length * vec[1]
        # p2_y = p0_y - width * separating_length * vec[0]
        # p3_x = p1_x + height * separating_length * vec[1]
        # p3_y = p1_y - width * separating_length * vec[0]
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        # p2_x = int(round(p2_x))
        # p2_y = int(round(p2_y))
        # p3_x = int(round(p3_x))
        # p3_y = int(round(p3_y))
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (255, 0, 0), 2)
        # cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (255, 0, 0), 2)
        # cv.line(image, (p1_x, p1_y), (p3_x, p3_y), (255, 0, 0), 2)


def preprocess_image(image):
    """Preprocess numpy image to torch tensor."""
    if image.shape[0] != 512 or image.shape[1] != 512:
        image = cv.resize(image, (512, 512))
    return torch.unsqueeze(ToTensor()(image), 0)


def detect_marking_points(detector, image, device, if_thops):
    """Given image read from opencv, return detected marking points."""
    img_dict= {}
    img_dict['image'] = preprocess_image(image).to(device)
    if if_thops:
        flops, params = profile(detector, inputs=(img_dict,))
        print("flops: %s |params: %s" % (flops, params))
    time_start = time.time()
    pred_dicts, ret_dicts = detector(img_dict)
    time_end = time.time()
    return pred_dicts['points_pred'], pred_dicts['slots_pred'], (time_end - time_start)



def detect_image(detector, device, path, if_thops, output_dir, save):
    """Demo for detecting images."""
    image_file = path
    image = cv.imread(image_file)
    pred_points, pred_slots, speed = detect_marking_points(detector, image, device, if_thops)
    print(path)
    inference_time.append((speed))
    plot_points_dmpr(image, pred_points)
    plot_slots(image, pred_points, pred_slots)
    # cv.imshow('demo', image)
    # cv.waitKey(0)
    if save:
        cv.imwrite(output_dir + path.split('/')[-1], image, [int(cv.IMWRITE_JPEG_QUALITY), 100])

def detect_image_sole(detector, device, path, if_thops):
    """Demo for detecting images."""
    image_file = path
    image = cv.imread(image_file)
    pred_points, pred_slots, speed = detect_marking_points(detector, image, device, if_thops)
    print(path)
    inference_time.append((speed))
    plot_points(image, pred_points)
    plot_slots(image, pred_points, pred_slots)
    cv.imshow('demo', image)
    cv.waitKey(0)

def inference_detector(cfg, mode):
    """Inference demo of directional point detector."""
    if_thops = False
    save = True
    path = "D:/Workshop/CV/NSPS/Datasets/bjtu-ps1/img/"
    solepath =  "D:/Workshop/CV/NSPS/Datasets/bjtu-ps1/img/bjtups_1.jpg"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger(cfg.log_dir, cfg.tag)
    torch.set_grad_enabled(False)
    model = build_model(cfg.model).to(device)
    model.load_params_from_file(filename=cfg.ckpt, logger=logger, to_cpu=False)
    model.eval()
    files = os.listdir(path)
    if mode == 'iter':
        for file in files:
            detect_image(model, device, path+file, if_thops, output_dir=str(cfg.output_dir)+'/', save=save)
        print(f'inference time:{np.array(inference_time).mean()} s')
    else:
        detect_image_sole(model, device, solepath, if_thops,)
        print(f'inference time:{np.array(inference_time).mean()} s')



if __name__ == '__main__':

    mode = 'iter'
    cfg = get_config()
    inference_detector(cfg, mode)
