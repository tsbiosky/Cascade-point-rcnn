import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetSAModule
from lib.rpn.proposal_target_layer import ProposalTargetLayer
from lib.net.rcnn_net import RCNNNet
import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
import lib.utils.loss_utils as loss_utils
from lib.config import cfg

import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.utils.bbox_transform import decode_bbox_target
import lib.utils.iou3d.iou3d_utils as iou3d_utils

class iou_layer(nn.Module):
    def __init__(self):
        super().__init__()



        iou_layer = []
        pre_channel = 512
        iou_layer.append(pt_utils.Conv1d(pre_channel, pre_channel, bn=cfg.RCNN.USE_BN))
        iou_layer.append(pt_utils.Conv1d(pre_channel, 1, activation=None))

        if cfg.RCNN.DP_RATIO >= 0:
            iou_layer.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.iou_layer = nn.Sequential(*iou_layer)
        self.proposal_target_layer = ProposalTargetLayer()
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """

        pre_iou=self.iou_layer(input_data['pooled_feature']).transpose(1, 2).contiguous().squeeze(dim=1)

        one = {}
        if self.training:
            reg_valid_mask = input_data['reg_valid_mask']
            fg_mask = (reg_valid_mask > 0)
            iou_label=input_data['iou_label']
            iou_loss = F.mse_loss((pre_iou[fg_mask]), iou_label[fg_mask],reduce=True)

            #print(iou_loss.item())
            one['rcnn_iou_loss']=iou_loss
        pre_iou=pre_iou/2+0.5
        ret_dict = {'pre_iou': pre_iou}
        ret_dict.update(one)

        return ret_dict