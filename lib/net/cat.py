import torch
import torch.nn as nn
from lib.net.point_rcnn import PointRCNN
from lib.net.rcnn_net import RCNNNet
from lib.config import cfg
import torch
import torch.nn as nn
from lib.net.rpn import RPN
from lib.net.cascade import cascade
from lib.net.iou_layer import iou_layer
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetSAModule
from lib.rpn.proposal_target_layer import ProposalTargetLayer
import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
import lib.utils.loss_utils as loss_utils
from lib.config import cfg

import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.utils.bbox_transform import decode_bbox_target
import lib.utils.iou3d.iou3d_utils as iou3d_utils

class cat(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TRAIN'):
        super().__init__()

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)

        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features
            if cfg.RCNN.BACKBONE == 'pointnet':
                self.rcnn_net = RCNNNet(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'pointsift':
                pass
            else:
                raise NotImplementedError
        self.cascade=cascade(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
        #self.iou = iou_layer()
    def forward(self, input_data):
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)

            # rcnn inference
            if cfg.RCNN.ENABLED:
                with torch.no_grad():
                    rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                    backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']
                    rpn_scores_raw = rpn_cls[:, :, 0]
                    rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                    seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
                    pts_depth = torch.norm(backbone_xyz, p=2, dim=2)

                    # proposal layer
                    rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)
                    #print(rois.size())
                    output['rois'] = rois
                    output['roi_scores_raw'] = roi_scores_raw
                    output['seg_result'] = seg_mask
                #print(rois.shape)
                rcnn_input_info = {'rpn_xyz': backbone_xyz,
                                   'rpn_features': backbone_features.permute((0, 2, 1)),
                                   'seg_mask': seg_mask,
                                   'roi_boxes3d': rois,
                                   'pts_depth': pts_depth}
                if self.training:
                    rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']
                rcnn_output = self.rcnn_net(rcnn_input_info)
                output.update(rcnn_output)
                if cfg.TRAIN.IOU_LAYER !='split':
                    cascade_input=rcnn_input_info.copy()
                    cascade_input['pred_boxes3d_1st']=output['pred_boxes3d_1st']
                    cascade_ouput=self.cascade(cascade_input)
                    output.update(cascade_ouput)

                if cfg.TRAIN.IOU_LAYER == 'split':
                    iou_out=self.iou(rcnn_output)
                    output.update(iou_out)

            ## get new roi
        elif cfg.RCNN.ENABLED:
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError

        return output