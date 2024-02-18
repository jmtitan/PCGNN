import json
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms as T
from psdet.datasets.base import BaseDataset
from psdet.datasets.registry import DATASETS
from psdet.utils.precision_recall import calc_average_precision, calc_precision_recall
from .process_data import boundary_check, overlap_check, rotate_centralized_marks, rotate_image
from .process_data import generalize_marks_pcgnn, generalize_marks_dmpr, generalize_marks_dcl, generalize_marks_csl, generalize_marks_gnn
from .utils import match_marking_points, match_slots, match_angle_points

@DATASETS.register
class ParkingSlotDataset(BaseDataset):

    def __init__(self, cfg, logger=None):
        super(ParkingSlotDataset, self).__init__(cfg=cfg, logger=logger)

        assert (self.root_path.exists())
        data_dir = self.root_path / 'label'

        if cfg.mode == 'train':
            data_list = self.root_path / 'main' / 'train.txt'
        elif cfg.mode == 'val':
            data_list = self.root_path / 'main' / 'val.txt'


        assert (data_dir.exists())

        with open(data_list, 'r') as f:
            line = f.read().splitlines()
            self.json_files = [str(data_dir) + '/' + i + '.json' for i in line]
            self.json_files.sort()

        if cfg.mode == 'train':
            # data augmentation
            self.image_transform = T.Compose([T.ColorJitter(brightness=0.1,
                                                            contrast=0.1, saturation=0.1, hue=0.1), T.ToTensor()])

        else:
            self.image_transform = T.Compose([T.ToTensor()])

        if self.logger:
            self.logger.info('Loading ParkingSlot {} dataset with {} samples'.format(cfg.mode, len(self.json_files)))

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = Path(self.json_files[idx])
        # load json
        # print((json_file))
        with open(str(json_file), 'r') as f:
            data = json.load(f)

        marks = np.array(data['marks'])
        if len(marks.shape) < 2:
            marks = np.expand_dims(marks, axis=0)

        max_points = self.cfg.max_points
        num_points = marks.shape[0]
        assert max_points >= num_points

        img_file = str(self.json_files[idx]).replace('.json', '.jpg').replace('label', 'img')
        image = Image.open(img_file)
        image_w = image.size[0]
        image_h = image.size[1]
        image = image.resize((512, 512), Image.BILINEAR)

        # centralize (image size = 600 x 600)
        marks[:, [0, 2]] -= (image_w / 2 + 0.5)
        marks[:, [1, 3]] -= (image_h / 2 + 0.5)

        if self.cfg.mode == 'train+' and np.random.rand() > 0.2:
            angles = np.linspace(5, 360, 72)
            np.random.shuffle(angles)
            for angle in angles:
                rotated_marks = rotate_centralized_marks(marks, angle)
                if boundary_check(rotated_marks) and overlap_check(rotated_marks, image_w, image_h):
                    image = rotate_image(image, angle)
                    marks = rotated_marks
                    break
        if self.cfg.name == 'dmpr':
            marks = generalize_marks_dmpr(marks, image_w, image_h)
        elif self.cfg.name == 'pcgnn':
            marks = generalize_marks_pcgnn(marks, image_w, image_h)
        elif self.cfg.name == 'pcgnn-dcl':
            marks = generalize_marks_dcl(marks, image_w, image_h)
        elif self.cfg.name == 'pcgnn-csl':
            marks = generalize_marks_csl(marks, image_w, image_h)
        elif self.cfg.name == 'gnn':
            marks = generalize_marks_gnn(marks, image_w, image_h)
        image = self.image_transform(image)

        # make sample with the max num points
        marks_full = np.full((max_points, marks.shape[1]), 0.0, dtype=np.float32)
        marks_full[:num_points] = marks
        match_targets = np.full((max_points, 2), -1, dtype=np.int32)

        slots = np.array(data['slots'])
        if slots.size != 0:
            if len(slots.shape) < 2:
                slots = np.expand_dims(slots, axis=0)
            for slot in slots:
                match_targets[slot[0] - 1, 0] = slot[1] - 1
                if slot[3] == 90:
                    match_targets[slot[0] - 1, 1] = 0  # 90 degree slant
                elif slot[3] == 30:
                    match_targets[slot[0] - 1, 1] = 1
                elif slot[3] == 45:
                    match_targets[slot[0] - 1, 1] = 2

        input_dict = {
            'marks': marks_full,
            'match_targets': match_targets,
            'npoints': num_points,
            'frame_id': idx,
            'image': image
        }

        return input_dict

    def generate_prediction_dicts(self, batch_dict, pred_dicts):
        pred_list = []
        pred_slots = pred_dicts['pred_slots']
        for i, slots in enumerate(pred_slots):
            single_pred_dict = {}
            single_pred_dict['frame_id'] = batch_dict['frame_id'][i]
            single_pred_dict['slots'] = slots
            pred_list.append(single_pred_dict)
        return pred_list

    def evaluate_point_detection(self, predictions_list, ground_truths_list):
        # point
        self.logger.info('*'*5 + 'Point position metric' + '*'*5)
        precisions, recalls = calc_precision_recall(
            ground_truths_list, predictions_list, match_marking_points)
        average_precision_pos = calc_average_precision(precisions, recalls)
        self.logger.info('precesions:')
        self.logger.info(precisions[-5:])
        self.logger.info('recalls:')
        self.logger.info(recalls[-5:])
        self.logger.info('Point position detection: average_precision {}'.format(average_precision_pos))
        #angle
        self.logger.info('*'*5+'Point angle metric'+'*'*5)
        precisions, recalls = calc_precision_recall(
            ground_truths_list, predictions_list, match_angle_points)
        average_precision_angle = calc_average_precision(precisions, recalls)
        self.logger.info('precesions:')
        self.logger.info(precisions[-5:])
        self.logger.info('recalls:')
        self.logger.info(recalls[-5:])
        self.logger.info('Point angle detection: average_precision {}'.format(average_precision_angle))

        return average_precision_pos, average_precision_angle

    def evaluate_slot_detection(self, predictions_list, ground_truths_list):

        precisions, recalls = calc_precision_recall(
            ground_truths_list, predictions_list, match_slots)
        average_precision = calc_average_precision(precisions, recalls)

        self.logger.info('precesions:')
        self.logger.info(precisions[-5:])
        self.logger.info('recalls:')
        self.logger.info(recalls[-5:])
        self.logger.info('Slot detection: average_precision {}'.format(average_precision))

        return average_precision

    def eval_once(self, pred_point, gt_point, slot_pred, slot_gt):
        precisions, recalls = calc_precision_recall(
            gt_point, pred_point, match_marking_points)
        ap_pos = calc_average_precision(precisions, recalls)

        precisions, recalls = calc_precision_recall(
            gt_point, pred_point, match_angle_points)
        ap_angle = calc_average_precision(precisions, recalls)

        precisions, recalls = calc_precision_recall(
            slot_gt, slot_pred, match_slots)
        ap_slot = calc_average_precision(precisions, recalls)
        return ap_pos, ap_angle, ap_slot