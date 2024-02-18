import math
import numpy as np
import torch
from torch import nn
from .util.post_process import pass_through_third_point_dmpr, calc_point_squre_dist, pair_marking_points
from .util.post_process import get_predpoints_dmpr
from psdet.models.detectors.backbones.traditionalbones import define_detector_block, YetAnotherDarknet


class DMPR(nn.modules.Module):
    """Detector for point with direction."""

    def __init__(self, cfg):
        super(DMPR, self).__init__()
        self.cfg = cfg
        input_channel_size = cfg.input_channels
        depth_factor = cfg.depth_factor
        output_channel_size = cfg.output_channels
        self.feature_extractor = YetAnotherDarknet(input_channel_size, depth_factor)

        layers = []
        layers += define_detector_block(16 * depth_factor)
        layers += define_detector_block(16 * depth_factor)
        layers += [nn.Conv2d(32 * depth_factor, output_channel_size,
                             kernel_size=1, stride=1, padding=0, bias=False)]
        self.predict = nn.Sequential(*layers)

        self.loss_func = nn.MSELoss().cuda()

    def forward(self, data_dict):
        img = data_dict['image']
        prediction = self.predict(self.feature_extractor(img))
        # 4 represents that there are 4 value: confidence, shape, offset_x,
        # offset_y, whose range is between [0, 1].
        point_pred, angle_pred = torch.split(prediction, 4, dim=1)
        point_pred = torch.sigmoid(point_pred)
        angle_pred = torch.tanh(angle_pred)
        points_pred = torch.cat((point_pred, angle_pred), dim=1)

        data_dict['points_pred'] = points_pred
        return data_dict

    def get_targets(self, data_dict):
        marks_gt_batch = data_dict['marks']
        npoints = data_dict['npoints']
        batch_size = marks_gt_batch.size()[0]
        targets = torch.zeros(batch_size, self.cfg.output_channels,
                              self.cfg.feature_map_size,
                              self.cfg.feature_map_size).cuda()

        mask = torch.zeros_like(targets)
        mask[:, 0].fill_(1.)

        for batch_idx, marks_gt in enumerate(marks_gt_batch):
            n = npoints[batch_idx].long()
            for marking_point in marks_gt[:n]:
                x, y = marking_point[:2]
                col = math.floor(x * self.cfg.feature_map_size)
                row = math.floor(y * self.cfg.feature_map_size)
                # Confidence Regression
                targets[batch_idx, 0, row, col] = 1.
                # Makring Point Shape Regression
                targets[batch_idx, 1, row, col] = marking_point[3]  # shape
                # Offset Regression
                targets[batch_idx, 2, row, col] = x * 16 - col
                targets[batch_idx, 3, row, col] = y * 16 - row
                # Direction Regression
                direction = marking_point[2]
                targets[batch_idx, 4, row, col] = math.cos(direction)
                targets[batch_idx, 5, row, col] = math.sin(direction)

                mask[batch_idx, 1:6, row, col].fill_(1.)
        return targets, mask

    def get_training_loss(self, data_dict):
        points_pred = data_dict['points_pred']
        targets, mask = self.get_targets(data_dict)

        disp_dict = {}

        loss_all = self.loss_func(points_pred * mask, targets * mask)

        tb_dict = {
            'loss_all': loss_all.item(),
            'loss_point': 0,
            'loss_edge': 0,
            'loss_angle': 0,
        }
        return loss_all, tb_dict, disp_dict

    def post_processing(self, data_dict):
        ret_dicts = {}
        pred_dicts = {}

        points_pred = data_dict['points_pred']

        points_pred_batch = []
        slots_pred = []
        for b, marks in enumerate(points_pred):
            points_pred = get_predpoints_dmpr(marks, self.cfg.point_thresh, self.cfg.boundary_thresh)
            points_pred_batch.append(points_pred)

            slots_infer = self.inference_slots(points_pred)
            slots_tmp = []
            for (i, j) in slots_infer:
                score = min(points_pred[i][0], points_pred[j][0])
                x1, y1 = points_pred[i][1][:2]
                x2, y2 = points_pred[j][1][:2]
                tmp = (score, np.array([x1, y1, x2, y2]))
                slots_tmp.append(tmp)

            slots_pred.append(slots_tmp)

        pred_dicts['points_pred'] = points_pred_batch
        pred_dicts['slots_pred'] = slots_pred
        return pred_dicts, ret_dicts

    def inference_slots(self, marking_points):
        """Inference slots based on marking points."""
        VSLOT_MIN_DIST = 0.044771278151623496
        VSLOT_MAX_DIST = 0.1099427457599304
        HSLOT_MIN_DIST = 0.15057789144568634
        HSLOT_MAX_DIST = 0.44449496544202816
        SLOT_SUPPRESSION_DOT_PRODUCT_THRESH = 0.8

        num_detected = len(marking_points)
        slots = []
        for i in range(num_detected - 1):
            for j in range(i + 1, num_detected):
                point_i = marking_points[i]
                point_j = marking_points[j]
                # Step 1: length filtration.
                distance = calc_point_squre_dist(point_i[1], point_j[1])
                if not (VSLOT_MIN_DIST <= distance <= VSLOT_MAX_DIST
                        or HSLOT_MIN_DIST <= distance <= HSLOT_MAX_DIST):
                    continue
                # Step 2: pass through filtration.
                if pass_through_third_point_dmpr(marking_points, i, j, SLOT_SUPPRESSION_DOT_PRODUCT_THRESH):
                    continue
                result = pair_marking_points(point_i, point_j)
                if result == 1:
                    slots.append((i, j))
                elif result == -1:
                    slots.append((j, i))
        return slots

