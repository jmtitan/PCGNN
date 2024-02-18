import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .util.gcn import GCNEncoder, EdgePredictor
from .util.post_process import pass_through_third_point
from .util.post_process import get_predpoints_csl
from psdet.models.detectors.backbones.traditionalbones import define_detector_block, YetAnotherDarknet, vgg16, resnet18, resnet50
from .util.mcab import MCABlock

class PCGNNCSL(nn.modules.Module):
    """Detector for point with direction."""

    def __init__(self, cfg, inter_threshs):
        super(PCGNNCSL, self).__init__()
        self.cfg = cfg
        input_channel_size = cfg.input_channels
        depth_factor = cfg.depth_factor
        output_channel_size = cfg.output_channels

        self.point_loss_func = nn.MSELoss().cuda()
        BCE_theta = nn.BCELoss().cuda()
        self.angle_loss_func = BCE_theta
        self.infer_threshs = inter_threshs
        # self.feature_extractor = create_RepVGG_PS(input_channel_size, output_channel_size)
        if cfg.backbone == 'Darknet':
            self.feature_extractor = YetAnotherDarknet(input_channel_size, depth_factor)
        elif cfg.backbone == 'VGG16':
            self.feature_extractor = vgg16()
        elif cfg.backbone == 'resnet18':
            self.feature_extractor = resnet18()
        elif cfg.backbone == 'resnet50':
            self.feature_extractor = resnet50()
        else:
            raise ValueError('{} is not implemented!'.format(cfg.backbone))

        # point
        if cfg.use_mcab_point:
            self.point_predictor = nn.Sequential()
            for i in range(cfg.mcab_layers):
                self.point_predictor.add_module('mcab', MCABlock(32 * depth_factor, 32 * depth_factor))
        else:
            layers_points = []
            layers_points += define_detector_block(16 * depth_factor)
            layers_points += define_detector_block(16 * depth_factor)
            self.point_predictor = nn.Sequential(*layers_points)

        self.point_predictor.add_module('conv_last1', nn.Conv2d(32 * depth_factor, output_channel_size,
                                                                kernel_size=1, stride=1, padding=0, bias=False))
        # descriptor
        if cfg.use_mcab_descriptor:
            self.descriptor_map = nn.Sequential()
            for i in range(cfg.mcab_layers):
                self.descriptor_map.add_module('mcab', MCABlock(32 * depth_factor, 32 * depth_factor))
        else:
            layers_descriptor = []
            layers_descriptor += define_detector_block(16 * depth_factor)
            layers_descriptor += define_detector_block(16 * depth_factor)
            self.descriptor_map = nn.Sequential(*layers_descriptor)
        self.descriptor_map.add_module('conv_last2', nn.Conv2d(32 * depth_factor, cfg.descriptor_dim,
                                                               kernel_size=1, stride=1, padding=0, bias=False))

        if cfg.use_gnn:
            self.graph_encoder = GCNEncoder(cfg.graph_encoder)

        self.edge_predictor = EdgePredictor(cfg.edge_predictor)

        if cfg.get('slant_predictor', None):
            self.slant_predictor = EdgePredictor(cfg.slant_predictor)

        if cfg.get('vacant_predictor', None):
            self.vacant_predictor = EdgePredictor(cfg.vacant_predictor)


    def forward(self, data_dict):
        img = data_dict['image']

        features = self.feature_extractor(img)  # [b, 1024, 16, 16]

        prediction = self.point_predictor(features)  # 点检测
        points_pred = torch.sigmoid(prediction)
        data_dict['points_pred'] = points_pred

        descriptor_map = self.descriptor_map(features)  # 特征抽取

        if self.training:
            marks = data_dict['marks']
            pred_dict = self.predict_slots(descriptor_map, marks[:, :, :2])
            data_dict.update(pred_dict)
        else:
            data_dict['descriptor_map'] = descriptor_map

        return data_dict

    def predict_slots(self, descriptor_map, points):
        descriptors = self.sample_descriptors(descriptor_map, points)
        data_dict = {}
        data_dict['descriptors'] = descriptors
        data_dict['points'] = points

        if self.cfg.get('slant_predictor', None):
            pred_dict = self.slant_predictor(data_dict)
            data_dict['slant_pred'] = pred_dict['edges_pred']

        if self.cfg.get('vacant_predictor', None):
            pred_dict = self.vacant_predictor(data_dict)
            data_dict['vacant_pred'] = pred_dict['edges_pred']

        if self.cfg.use_gnn:
            data_dict = self.graph_encoder(data_dict)

        pred_dict = self.edge_predictor(data_dict)

        data_dict['edge_pred'] = pred_dict['edges_pred']
        return data_dict

    def sample_descriptors(self, descriptors, keypoints):
        """ Interpolate descriptors at keypoint locations """
        b, c, h, w = descriptors.shape
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
        # 将一个source_image，通过双线性插值的方式变换到另一个大小指定的target_image中
        descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1)
        return descriptors

    def get_targets(self, data_dict):

        marks_gt_batch = data_dict['marks']
        npoints = data_dict['npoints']
        batch_size = marks_gt_batch.size()[0]
        targets = torch.zeros(batch_size, self.cfg.output_channels,
                              self.cfg.feature_map_size,
                              self.cfg.feature_map_size).cuda()
        indice = torch.zeros(batch_size,
                              self.cfg.max_points, 2).cuda()
        mask = torch.zeros_like(targets)
        mask[:, 0].fill_(1.)

        for batch_idx, marks_gt in enumerate(marks_gt_batch):
            n = npoints[batch_idx].long()
            for marking_point in marks_gt[:n]:
                x, y = marking_point[:2]
                length, flag = marking_point[2:4]
                gaussain_csl = marking_point[4:]
                col = math.floor(x * self.cfg.feature_map_size)
                row = math.floor(y * self.cfg.feature_map_size)
                # Confidence Regression
                targets[batch_idx, 0, row, col] = 1.
                # Makring Point length Regression
                targets[batch_idx, 1, row, col] = length
                # Offset Regression
                targets[batch_idx, 2, row, col] = x * 16 - col
                targets[batch_idx, 3, row, col] = y * 16 - row
                # Direction Regression
                targets[batch_idx, 4, row, col] = flag
                targets[batch_idx, 5:, row, col] = gaussain_csl

                mask[batch_idx, 1:5, row, col].fill_(1.)
        return targets, mask

    def get_training_loss(self, data_dict):
        points_pred = data_dict['points_pred']
        targets, mask = self.get_targets(data_dict)

        disp_dict = {}
        b, c, r = torch.nonzero(targets[:, 0, :, :], as_tuple=True)
        loss_point = self.point_loss_func(points_pred[:,:5,:,:] * mask[:, :5, :, :], targets[:,:5,:,:] * mask[:, :5, :, :])
        loss_angle = self.angle_loss_func(points_pred[b,5:,c,r], targets[b,5:,c,r])

        edges_pred = data_dict['edges_pred']
        edges_target = torch.zeros_like(edges_pred)
        edges_mask = torch.zeros_like(edges_pred)

        match_targets = data_dict['match_targets']
        npoints = data_dict['npoints']

        for b in range(edges_pred.shape[0]):
            n = npoints[b].long()
            y = match_targets[b]
            m = y.shape[0]
            for i in range(n):
                t = y[i, 0]
                for j in range(n):
                    idx = i * m + j
                    edges_mask[b, 0, idx] = 1
                    if j == t:
                        edges_target[b, 0, idx] = 1

        loss_edge = F.binary_cross_entropy(edges_pred, edges_target, edges_mask)
        loss_all = self.cfg.losses.weight_point * loss_point + self.cfg.losses.weight_edge * loss_edge + self.cfg.losses.weight_angle * loss_angle

        tb_dict = {
            'loss_all': loss_all.item(),
            'loss_point': loss_point.item(),
            'loss_edge': loss_edge.item(),
            'loss_angle': loss_angle.item()
        }
        return loss_all, tb_dict, disp_dict

    def post_processing(self, data_dict):
        ret_dicts = {}
        pred_dicts = {}

        points_pred = data_dict['points_pred']
        descriptor_map = data_dict['descriptor_map']

        points_pred_batch = []
        slots_pred = []
        for b, marks in enumerate(points_pred):
            points_pred = get_predpoints_csl(marks, self.cfg.point_thresh, self.cfg.boundary_thresh)  # nms
            points_pred_batch.append(points_pred)

            if len(points_pred) > 0:
                points_np = np.concatenate([p[1].reshape(1, -1) for p in points_pred], axis=0)
            else:
                points_np = np.zeros((self.cfg.max_points, 5))

            if points_np.shape[0] < self.cfg.max_points:
                points_full = np.zeros((self.cfg.max_points, 5))
                points_full[:len(points_pred)] = points_np
            else:
                points_full = points_np

            pred_dict = self.predict_slots(descriptor_map[b].unsqueeze(0),
                                           torch.Tensor(points_full[:, :2]).unsqueeze(0).cuda())
            edges = pred_dict['edges_pred'][0]
            n = points_np.shape[0]
            m = points_full.shape[0]

            slots = []
            for i in range(n):
                for j in range(n):
                    idx = i * m + j
                    score = edges[0, idx]
                    if score > 0.5:  # slot阈值
                        x1, y1 = points_np[i, :2]
                        x2, y2 = points_np[j, :2]
                        if self.inference_slots(points_np, i, j, self.infer_threshs):
                            slot = (score, np.array([x1, y1, x2, y2]))
                            slots.append(slot)

            slots_pred.append(slots)

        pred_dicts['points_pred'] = points_pred_batch
        pred_dicts['slots_pred'] = slots_pred
        return pred_dicts, ret_dicts

    def inference_slots(self, marking_points, i, j, infer_threshs):
        """
           Inference slots based on marking points.
           params:
               marking_points: (coffidence, (x, y, direction, shape))
        """
        VSLOT_MAX_DIST, HSLOT_MAX_DIST, MAX_DIST, MIN_DIST, SLOT_SUPPRESSION_DOT_PRODUCT_THRESH = infer_threshs
        SLOT_DIRECTION_THRESH = 0.5235987711485906990 # 30 / 57.29 即30度为角度差异阈值
        point_i = marking_points[i]
        point_j = marking_points[j]
        # Step 1: length filtration.
        dis_x = abs(point_i[0] - point_j[0])
        dis_y = abs(point_i[1] - point_j[1])
        if not (dis_x <= VSLOT_MAX_DIST or dis_y <= HSLOT_MAX_DIST or MIN_DIST <= math.sqrt(dis_x**2+dis_y**2) <= MAX_DIST):
            return False
        # Step 2: pass through filtration.
        if pass_through_third_point(marking_points, i, j, SLOT_SUPPRESSION_DOT_PRODUCT_THRESH):
            return False
        # Step 3: direction filtration
        if abs(point_i[2]-point_j[2]) > SLOT_DIRECTION_THRESH:
            return False
        return True