import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .util.gcn import GCNEncoder, EdgePredictor
from .util.post_process import get_predpoints_gnn
from .backbones.traditionalbones import define_detector_block, YetAnotherDarknet, vgg16, resnet18, resnet50
from .backbones.mobilenet import _BuildMobilenetV3
from .backbones.hrnet import hrnet_w18_small_v1

class GNN(nn.modules.Module):
    """Detector for point without direction."""

    def __init__(self, cfg):
        super(GNN, self).__init__()
        self.cfg = cfg
        input_channel_size = cfg.input_channels
        depth_factor = cfg.depth_factor
        output_channel_size = cfg.output_channels

        self.point_loss_func = nn.MSELoss().cuda()

        if cfg.backbone == 'Darknet':
            self.feature_extractor = YetAnotherDarknet(input_channel_size, depth_factor)
        elif cfg.backbone == 'VGG16':
            self.feature_extractor = vgg16()
        elif cfg.backbone == 'resnet18':
            self.feature_extractor = resnet18()
        elif cfg.backbone == 'resnet50':
            self.feature_extractor = resnet50()
        elif cfg.backbone == 'mobilenet':
            self.feature_extractor = _BuildMobilenetV3()
        elif cfg.backbone == 'hrnet':
            self.feature_extractor = hrnet_w18_small_v1()
        else:
            raise ValueError('{} is not implemented!'.format(cfg.backbone))

        layers_points = []
        layers_points += define_detector_block(16 * depth_factor)
        layers_points += define_detector_block(16 * depth_factor)
        layers_points += [nn.Conv2d(32 * depth_factor, output_channel_size,
                                    kernel_size=1, stride=1, padding=0, bias=False)]
        self.point_predictor = nn.Sequential(*layers_points)

        layers_descriptor = []
        layers_descriptor += define_detector_block(16 * depth_factor)
        layers_descriptor += define_detector_block(16 * depth_factor)
        layers_descriptor += [nn.Conv2d(32 * depth_factor, cfg.descriptor_dim,
                                        kernel_size=1, stride=1, padding=0, bias=False)]
        self.descriptor_map = nn.Sequential(*layers_descriptor)

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

        points_pred = self.point_predictor(features)
        points_pred = torch.sigmoid(points_pred)
        data_dict['points_pred'] = points_pred

        descriptor_map = self.descriptor_map(features)

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
        descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1)
        return descriptors

    def get_targets_points(self, data_dict):
        points_pred = data_dict['points_pred']
        marks_gt_batch = data_dict['marks']
        npoints = data_dict['npoints']

        b, c, h, w = points_pred.shape
        targets = torch.zeros(b, c, h, w).cuda()
        mask = torch.zeros_like(targets)
        mask[:, 0].fill_(1.)

        for batch_idx, marks_gt in enumerate(marks_gt_batch):
            n = npoints[batch_idx].long()
            for marking_point in marks_gt[:n]:
                x, y = marking_point[:2]
                col = math.floor(x * w)
                row = math.floor(y * h)
                # Confidence Regression
                targets[batch_idx, 0, row, col] = 1.
                # Offset Regression
                targets[batch_idx, 1, row, col] = x * w - col
                targets[batch_idx, 2, row, col] = y * h - row

                mask[batch_idx, 1:3, row, col].fill_(1.)
        return targets, mask

    def post_processing(self, data_dict):
        ret_dicts = {}
        pred_dicts = {}

        points_pred = data_dict['points_pred']
        descriptor_map = data_dict['descriptor_map']

        points_pred_batch = []
        slots_pred = []
        for b, marks in enumerate(points_pred):
            points_pred = get_predpoints_gnn(marks, self.cfg.point_thresh, self.cfg.boundary_thresh)
            points_pred_batch.append(points_pred)

            if len(points_pred) > 0:
                points_np = np.concatenate([p[1].reshape(1, -1) for p in points_pred], axis=0)
            else:
                points_np = np.zeros((self.cfg.max_points, 2))

            if points_np.shape[0] < self.cfg.max_points:
                points_full = np.zeros((self.cfg.max_points, 2))
                points_full[:len(points_pred)] = points_np
            else:
                points_full = points_np

            pred_dict = self.predict_slots(descriptor_map[b].unsqueeze(0),
                                           torch.Tensor(points_full).unsqueeze(0).cuda())
            edges = pred_dict['edges_pred'][0]
            n = points_np.shape[0]
            m = points_full.shape[0]

            slots = []
            for i in range(n):
                for j in range(n):
                    idx = i * m + j
                    score = edges[0, idx]
                    if score > 0.5:
                        x1, y1 = points_np[i, :2]
                        x2, y2 = points_np[j, :2]
                        slot = (score, np.array([x1, y1, x2, y2]))
                        slots.append(slot)

            slots_pred.append(slots)

        pred_dicts['points_pred'] = points_pred_batch
        pred_dicts['slots_pred'] = slots_pred
        return pred_dicts, ret_dicts

    def get_training_loss(self, data_dict):
        points_pred = data_dict['points_pred']
        targets, mask = self.get_targets_points(data_dict)

        disp_dict = {}

        loss_point = self.point_loss_func(points_pred * mask, targets * mask)

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
        loss_all = self.cfg.losses.weight_point * loss_point + self.cfg.losses.weight_edge * loss_edge

        tb_dict = {
            'loss_all': loss_all.item(),
            'loss_point': loss_point.item(),
            'loss_edge': loss_edge.item(),
            'loss_angle': 0
        }
        return loss_all, tb_dict, disp_dict