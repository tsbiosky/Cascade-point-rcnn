import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.rpn.proposal_layer import ProposalLayer
import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
import lib.utils.loss_utils as loss_utils
from lib.config import cfg
import importlib
import torch
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils


class RPN(nn.Module):
    def __init__(self, use_xyz=True, mode='TRAIN'):
        super().__init__()
        self.training_mode = (mode == 'TRAIN')

        MODEL = importlib.import_module(cfg.RPN.BACKBONE)
        self.backbone_net = MODEL.get_model(input_channels=int(cfg.RPN.USE_INTENSITY), use_xyz=use_xyz)

        # classification branch
        cls_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]+3
        for k in range(0, cfg.RPN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.CLS_FC[k], bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        if cfg.RPN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.rpn_cls_layer = nn.Sequential(*cls_layers)
        ####mlp
        '''
        mlp_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1] + 3
        for k in range(0, cfg.RPN.CLS_FC.__len__()):
            mlp_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.CLS_FC[k], bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.CLS_FC[k]
        if cfg.RPN.DP_RATIO >= 0:
            mlp_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.mlp_layers = nn.Sequential(*mlp_layers)
        '''
        # regression branch
        per_loc_bin_num = int(cfg.RPN.LOC_SCOPE / cfg.RPN.LOC_BIN_SIZE) * 2
        if cfg.RPN.LOC_XZ_FINE:
            reg_channel = per_loc_bin_num * 4 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
        else:
            reg_channel = per_loc_bin_num * 2 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
        reg_channel += 1  # reg y

        reg_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]+3
        for k in range(0, cfg.RPN.CLS_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.REG_FC[k], bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, pre_channel, bn=cfg.RPN.USE_BN))
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RPN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.rpn_reg_layer = nn.Sequential(*reg_layers)
        #### iou
        '''
        pre_channel = cfg.RPN.FP_MLPS[0][-1]+3
        iou_layer=[]
        iou_layer.append(pt_utils.Conv1d(pre_channel,pre_channel,bn=cfg.RPN.USE_BN))
        iou_layer.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        self.iou_layer = nn.Sequential(*iou_layer)
        '''
        '''
        reg_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]
        for k in range(0, cfg.RPN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.REG_FC[k], bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RPN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.rpn_reg_layer_2nd = nn.Sequential(*reg_layers)
    
        reg_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]
        for k in range(0, cfg.RPN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.REG_FC[k], bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RPN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.rpn_reg_layer_3rd = nn.Sequential(*reg_layers)
        '''
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            self.rpn_cls_loss_func = loss_utils.DiceLoss(ignore_target=-1)
        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            self.rpn_cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RPN.FOCAL_ALPHA[0],
                                                                               gamma=cfg.RPN.FOCAL_GAMMA)
        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            self.rpn_cls_loss_func = F.binary_cross_entropy
        else:
            raise NotImplementedError

        self.proposal_layer = ProposalLayer(mode=mode)
        self.init_weights()

    def init_weights(self):
        if cfg.RPN.LOSS_CLS in ['SigmoidFocalLoss']:
            pi = 0.01
            nn.init.constant_(self.rpn_cls_layer[-1].conv.bias, -np.log((1 - pi) / pi))

        nn.init.normal_(self.rpn_reg_layer[-1].conv.weight, mean=0, std=0.001)
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features
    def forward(self, input_data):
        """
        :param input_data: dict (point_cloud)
        :return:
        """
        pts_input = input_data['pts_input']
        backbone_xyz, backbone_features = self.backbone_net(pts_input)  # (B, N, 3), (B, C, N)
        B,C,N=backbone_features.shape

        input_feature=torch.cat((backbone_xyz.permute(0,2,1),backbone_features),1)
        #input_feature=self.mlp_layers(input_feature)
        rpn_cls = self.rpn_cls_layer(input_feature).transpose(1, 2).contiguous()  # (B, N, 1)
        rpn_reg = self.rpn_reg_layer(input_feature).transpose(1, 2).contiguous()  # (B, N, C)
        #predict_iou=self.iou_layer(input_feature).transpose(1, 2).contiguous()
        #print(predict_iou.shape,rpn_reg.shape,rpn_cls.shape)
        ret_dict = {'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg,
                    'backbone_xyz': backbone_xyz, 'backbone_features': backbone_features}
        '''
        rpn_scores_raw = rpn_cls[:, :, 0]
        rois, roi_scores_raw = self.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)
        rpn_features=backbone_features.permute((0, 2, 1))

        pooled_features, pooled_empty_flag = \
            roipool3d_utils.roipool3d_gpu(backbone_xyz, rpn_features, rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                          sampled_pt_num=cfg.RCNN.NUM_POINTS)
        xyz_2d, features_2d = self._break_up_pc(pooled_features)
        features_2d=features_2d.view(B,-1,C).permute((0, 2, 1))
        print(features_2d.size())
        #pooled feature B,512,512,131 (512 rois each has 512 pts)
        #print(pooled_features.size(),backbone_features.size(),rois.size())
        '''
        return ret_dict

