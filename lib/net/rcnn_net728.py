import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetSAModule
from lib.rpn.proposal_target_layer import ProposalTargetLayer
import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
import lib.utils.loss_utils as loss_utils
from lib.config import cfg

import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.utils.bbox_transform import decode_bbox_target
import lib.utils.iou3d.iou3d_utils as iou3d_utils
class RCNNNet(nn.Module):
    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules_2nd = nn.ModuleList()
        channel_in = input_channels

        if cfg.RCNN.USE_RPN_FEATURES:
            self.rcnn_input_channel = 3 + int(cfg.RCNN.USE_INTENSITY) + int(cfg.RCNN.USE_MASK) + int(cfg.RCNN.USE_DEPTH)
            self.xyz_up_layer = pt_utils.SharedMLP([self.rcnn_input_channel] + cfg.RCNN.XYZ_UP_LAYER,
                                                   bn=cfg.RCNN.USE_BN)

            #self.test_layer=pt_utils.Conv1d(133, 46, activation=None)

            #print(self.test_layer)
            c_out = cfg.RCNN.XYZ_UP_LAYER[-1]
            self.merge_down_layer = pt_utils.SharedMLP([c_out * 2, c_out], bn=cfg.RCNN.USE_BN)

            #print(self.xyz_up_layer,self.merge_down_layer)
        for k in range(cfg.RCNN.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + cfg.RCNN.SA_CONFIG.MLPS[k]

            npoint = cfg.RCNN.SA_CONFIG.NPOINTS[k] if cfg.RCNN.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=npoint,
                    radius=cfg.RCNN.SA_CONFIG.RADIUS[k],
                    nsample=cfg.RCNN.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RCNN.USE_BN
                )
            )

            channel_in = mlps[-1]
        #print(self.SA_modules,self.SA_modules_2nd)
        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        cls_layers2 = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers2.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers2.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers2.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer_2nd = nn.Sequential(*cls_layers2)
        #print(self.cls_layer_2nd)
        #self.cls_layer_2nd = nn.Linear(pre_channel, cfg.RCNN.CLS_FC[k])
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer_3rd = nn.Sequential(*cls_layers)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError

        # regression layer
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)

        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer_2nd = nn.Sequential(*reg_layers)
        #self.reg_layer_2nd=nn.Linear(pre_channel, self.n_classes)
        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer_3rd = nn.Sequential(*reg_layers)
        iou_layer = []
        pre_channel = channel_in
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
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features
    def roipooling(self,input_data):
        rpn_xyz, rpn_features = input_data['rpn_xyz'], input_data['rpn_features']
        batch_rois = input_data['roi_boxes3d']
        if self.training ==False and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            dd=-1
        else:
            dd=2
        if cfg.RCNN.USE_INTENSITY:
            pts_extra_input_list = [input_data['rpn_intensity'].unsqueeze(dim=2),
                                    input_data['seg_mask'].unsqueeze(dim=2)]
        else:
            #print(input_data['seg_mask'].shape)
            pts_extra_input_list = [input_data['seg_mask'].unsqueeze(dim=dd)]

        if cfg.RCNN.USE_DEPTH:
            pts_depth = input_data['pts_depth'] / 70.0 - 0.5
            pts_extra_input_list.append(pts_depth.unsqueeze(dim=dd))
        pts_extra_input = torch.cat(pts_extra_input_list, dim=dd)

        pts_feature = torch.cat((pts_extra_input, rpn_features), dim=dd)
        if self.training == False and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            batch_rois=torch.squeeze(batch_rois,1)

        if self.training == False and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            rpn_xyz=rpn_xyz.unsqueeze(dim=0)
            pts_feature=pts_feature.unsqueeze(dim=0)
            batch_rois=batch_rois.unsqueeze(dim=0)
        #print(rpn_xyz.shape,pts_feature.shape,batch_rois.shape)
        pooled_features, pooled_empty_flag = \
            roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                          sampled_pt_num=cfg.RCNN.NUM_POINTS)

        # canonical transformation

        batch_size = batch_rois.shape[0]
        roi_center = batch_rois[:, :, 0:3]
        #print(batch_rois.shape,roi_center.shape)
        #print(pooled_features.shape,roi_center.shape)
        pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
        for k in range(batch_size):
            pooled_features[k, :, :, 0:3] = kitti_utils.rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3],
                                                                                batch_rois[k, :, 6])

        pts_input = pooled_features.view(-1, pooled_features.shape[2], pooled_features.shape[3])
        return pts_input
    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """

        if cfg.RCNN.ROI_SAMPLE_JIT:
            if self.training:
                with torch.no_grad():
                    target_dict = self.proposal_target_layer(input_data,stage=1)

                pts_input = torch.cat((target_dict['sampled_pts'], target_dict['pts_feature']), dim=2)
                target_dict['pts_input'] = pts_input
            else:
                rpn_xyz, rpn_features = input_data['rpn_xyz'], input_data['rpn_features']
                batch_rois = input_data['roi_boxes3d']
                if cfg.RCNN.USE_INTENSITY:
                    pts_extra_input_list = [input_data['rpn_intensity'].unsqueeze(dim=2),
                                            input_data['seg_mask'].unsqueeze(dim=2)]
                else:
                    pts_extra_input_list = [input_data['seg_mask'].unsqueeze(dim=2)]

                if cfg.RCNN.USE_DEPTH:
                    pts_depth = input_data['pts_depth'] / 70.0 - 0.5
                    pts_extra_input_list.append(pts_depth.unsqueeze(dim=2))
                pts_extra_input = torch.cat(pts_extra_input_list, dim=2)

                pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)
                pooled_features, pooled_empty_flag = \
                        roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                                      sampled_pt_num=cfg.RCNN.NUM_POINTS)

                # canonical transformation
                batch_size = batch_rois.shape[0]
                roi_center = batch_rois[:, :, 0:3]
                pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
                for k in range(batch_size):
                    pooled_features[k, :, :, 0:3] = kitti_utils.rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3],
                                                                                        batch_rois[k, :, 6])

                pts_input = pooled_features.view(-1, pooled_features.shape[2], pooled_features.shape[3])
        else:
            pts_input = input_data['pts_input']
            target_dict = {}
            target_dict['pts_input'] = input_data['pts_input']
            target_dict['roi_boxes3d'] = input_data['roi_boxes3d']
            if self.training:
                #input_data['ori_roi'] = torch.cat((input_data['ori_roi'], input_data['roi_boxes3d']), 1)
                target_dict['cls_label'] = input_data['cls_label']
                target_dict['reg_valid_mask'] = input_data['reg_valid_mask'].view(-1)
                target_dict['gt_of_rois'] = input_data['gt_boxes3d_ct']
        #print(pts_input.shape)
        pts_input=pts_input.view(-1,512,128+self.rcnn_input_channel)
        xyz, features = self._break_up_pc(pts_input)
        anchor_size = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
        if cfg.RCNN.USE_RPN_FEATURES:
            xyz_input = pts_input[..., 0:self.rcnn_input_channel].transpose(1, 2).unsqueeze(dim=3)
            #xyz_input = pts_input[..., 0:self.rcnn_input_channel].transpose(1, 2)

            xyz_feature = self.xyz_up_layer(xyz_input)

            rpn_feature = pts_input[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)

            merged_feature = torch.cat((xyz_feature, rpn_feature), dim=1)
            merged_feature = self.merge_down_layer(merged_feature)
            l_xyz, l_features = [xyz], [merged_feature.squeeze(dim=3)]
        else:
            l_xyz, l_features = [xyz], [features]

        for i in range(len(self.SA_modules)):
            
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)


        batch_size=input_data['roi_boxes3d'].size(0)
        batch_size_2= pts_input.shape[0] # for loss fun
        #print(input_data['roi_boxes3d'].shape,pts_input.shape)
        rcnn_cls = self.cls_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B*64, 1 or 2)
        rcnn_reg = self.reg_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B*64, C)
        pre_iou1=self.iou_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)
        if self.training:
            roi_boxes3d=target_dict['roi_boxes3d'].view(-1,7)
            cls_label=target_dict['cls_label'].float()
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label.view(-1), reduction='none')
            cls_label_flat = cls_label.view(-1)
            cls_valid_mask = (cls_label_flat >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
            gt_boxes3d_ct = target_dict['gt_of_rois']
            reg_valid_mask = target_dict['reg_valid_mask']
            fg_mask = (reg_valid_mask > 0)
            #print(rcnn_reg.view(batch_size_2, -1)[fg_mask].shape)
            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                loss_utils.get_reg_loss(rcnn_reg.view(batch_size_2, -1)[fg_mask],
                                        gt_boxes3d_ct.view(batch_size_2, 7)[fg_mask],
                                        loc_scope=cfg.RCNN.LOC_SCOPE,
                                        loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                        num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                        anchor_size=anchor_size,
                                        get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                        loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                        get_ry_fine=True)
            rcnn_loss_reg = loss_loc + loss_angle + 3 * loss_size


        else:
            roi_boxes3d=input_data['roi_boxes3d'].view(-1,7)
            one={}
        #print(rcnn_reg.size(),roi_boxes3d.size())
        #print(roi_boxes3d.shape, rcnn_reg.shape)
        pred_boxes3d_1st = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                          anchor_size=anchor_size,
                                          loc_scope=cfg.RCNN.LOC_SCOPE,
                                          loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                          num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                          get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                          loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                          get_ry_fine=True).view(batch_size, -1, 7)

        if self.training == False and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            pred_boxes3d_1st=pred_boxes3d_1st.view(-1,7)
        if self.training:

            gt=target_dict['real_gt']
            iou_label=[]
            for i in range(batch_size_2):
                iou_label.append(iou3d_utils.boxes_iou3d_gpu(pred_boxes3d_1st.view(-1,7)[i].view(1,7), gt[i].view(1,7)))
            iou_label=torch.cat(iou_label)
            iou_loss=F.smooth_l1_loss(abs(pre_iou1),iou_label)


            one = {'rcnn_loss_cls': rcnn_loss_cls, 'rcnn_loss_reg': rcnn_loss_reg}
            del cls_label, rcnn_cls_flat, batch_loss_cls, cls_label_flat, cls_valid_mask, rcnn_loss_cls, gt_boxes3d_ct, reg_valid_mask, fg_mask

        #print(pre_iou1,iou_label)
        #print(gt[0:10],pred_boxes3d_1st[0:10],roi_boxes3d[0:10])
        input_data2 = input_data.copy()
	
        #print(input_data['roi_boxes3d'].size())
        if self.training:
            #input_data2['roi_boxes3d'] = torch.cat((pred_boxes3d_1st, input_data['ori_roi']), 1)
            input_data2['roi_boxes3d'] = torch.cat((pred_boxes3d_1st,input_data['roi_boxes3d']),1)

            with torch.no_grad():
                target_dict_2nd = self.proposal_target_layer(input_data2,stage=2)
            pts_input_2 = torch.cat((target_dict_2nd['sampled_pts'], target_dict_2nd['pts_feature']), dim=2)
            target_dict_2nd['pts_input'] = pts_input_2
            roi=target_dict_2nd['roi_boxes3d']

        else:
            input_data2['roi_boxes3d']=pred_boxes3d_1st
            #input_data2['roi_boxes3d']=torch.cat((pred_boxes3d_1st, input_data['roi_boxes3d']), 1)
            roi=pred_boxes3d_1st
            #roi=torch.cat((pred_boxes3d_1st, input_data['roi_boxes3d']), 1)
            pts_input_2=self.roipooling(input_data2)
        #print(pts_input_2.shape)
        xyz_2, features_2 = self._break_up_pc(pts_input_2)
        #print(xyz_2.size(),xyz.size(),features_2.size(),features.size())
        if cfg.RCNN.USE_RPN_FEATURES:
            xyz_input_2 = pts_input_2[..., 0:self.rcnn_input_channel].transpose(1, 2).unsqueeze(dim=3)
            xyz_feature_2 = self.xyz_up_layer(xyz_input_2)

            rpn_feature_2 = pts_input_2[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)

            merged_feature_2 = torch.cat((xyz_feature_2, rpn_feature_2), dim=1)
            merged_feature_2 = self.merge_down_layer(merged_feature_2)
            l_xyz_2, l_features_2 = [xyz_2], [merged_feature_2.squeeze(dim=3)]
        else:
            l_xyz__2, l_features_2 = [xyz_2], [features_2]
        #print(l_xyz_2[0].size(), l_xyz[0].size(), l_features_2[0].size(), l_features[0].size())
        for i in range(len(self.SA_modules)):
            li_xyz_2, li_features_2 = self.SA_modules[i](l_xyz_2[i], l_features_2[i])
            l_xyz_2.append(li_xyz_2)
            l_features_2.append(li_features_2)
        del xyz,features,l_features

        rcnn_cls_2nd = self.cls_layer_2nd(l_features_2[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B*64, 1 or 2)
        rcnn_reg_2nd = self.reg_layer_2nd(l_features_2[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B*64, C)
        pre_iou2=self.iou_layer(l_features_2[-1]).transpose(1, 2).contiguous().squeeze(dim=1)
        #loss
        if self.training:
            cls_label = target_dict_2nd['cls_label'].float()
            rcnn_cls_flat = rcnn_cls_2nd.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label.view(-1), reduction='none')
            cls_label_flat = cls_label.view(-1)
            cls_valid_mask = (cls_label_flat >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
            gt_boxes3d_ct = target_dict_2nd['gt_of_rois']
            reg_valid_mask = target_dict_2nd['reg_valid_mask']
            fg_mask = (reg_valid_mask > 0)
            #print(rcnn_reg_2nd.view(batch_size_2, -1)[fg_mask].size(0))
            if rcnn_reg_2nd.view(batch_size_2, -1)[fg_mask].size(0)==0:
                fg_mask=(reg_valid_mask <= 0)
            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                loss_utils.get_reg_loss(rcnn_reg_2nd.view(batch_size_2, -1)[fg_mask],
                                        gt_boxes3d_ct.view(batch_size_2, 7)[fg_mask],
                                        loc_scope=cfg.RCNN.LOC_SCOPE,
                                        loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                        num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                        anchor_size=anchor_size,
                                        get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                        loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                        get_ry_fine=True)
            rcnn_loss_reg = loss_loc + loss_angle + 3 * loss_size

            #two = {'rcnn_loss_cls_2nd': rcnn_loss_cls, 'rcnn_loss_reg_2nd': rcnn_loss_reg}

        else:
            two={}

        sec={'rcnn_cls_2nd': rcnn_cls_2nd, 'rcnn_reg_2nd':rcnn_reg_2nd}
        #print(input_data['roi_boxes3d'].shape,input_data2['roi_boxes3d'].shape)

        pred_boxes3d_2nd=decode_bbox_target(roi.view(-1, 7), rcnn_reg_2nd.view(-1, rcnn_reg_2nd.shape[-1]),
                           anchor_size=anchor_size,
                           loc_scope=cfg.RCNN.LOC_SCOPE,
                           loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                           num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                           get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                           loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                           get_ry_fine=True).view(batch_size, -1, 7)

        if self.training:
            '''
            gt=target_dict_2nd['real_gt']
            iou_label=[]
            for i in range(batch_size_2):
                iou_label.append(iou3d_utils.boxes_iou3d_gpu(pred_boxes3d_2nd.view(-1,7)[i].view(1,7), gt[i].view(1,7)))
            iou_label=torch.cat(iou_label)
            iou_loss=F.smooth_l1_loss(abs(pre_iou2),iou_label)
            '''
            two = {'rcnn_loss_cls_2nd': rcnn_loss_cls, 'rcnn_loss_reg_2nd': rcnn_loss_reg}
            del cls_label, rcnn_cls_flat, batch_loss_cls, cls_label_flat, cls_valid_mask, rcnn_loss_cls, gt_boxes3d_ct, reg_valid_mask, fg_mask
        input_data3 = input_data2.copy()
        #del input_data2

        if self.training:
            input_data3['roi_boxes3d'] = torch.cat((pred_boxes3d_2nd, input_data2['roi_boxes3d']), 1)
            #input_data3['roi_boxes3d'] = input_data2['gt_boxes3d']
            #input_data3['roi_boxes3d'] = pred_boxes3d_2nd
            #print(input_data3['roi_boxes3d'].shape)
            with torch.no_grad():
                target_dict_3rd = self.proposal_target_layer(input_data3, stage=3)

            pts_input_3 = torch.cat((target_dict_3rd['sampled_pts'], target_dict_3rd['pts_feature']), dim=2)
            target_dict_3rd['pts_input'] = pts_input_3
            roi=target_dict_3rd['roi_boxes3d']
        else:
            input_data3['roi_boxes3d']=pred_boxes3d_2nd
            #input_data3['roi_boxes3d']=torch.cat((pred_boxes3d_2nd, input_data2['roi_boxes3d']), 1)
            roi=pred_boxes3d_2nd
            #roi=torch.cat((pred_boxes3d_2nd, input_data2['roi_boxes3d']), 1)
            pts_input_3 = self.roipooling(input_data3)
        xyz_3, features_3 = self._break_up_pc(pts_input_3)

        if cfg.RCNN.USE_RPN_FEATURES:
            xyz_input_3 = pts_input_3[..., 0:self.rcnn_input_channel].transpose(1, 2).unsqueeze(dim=3)
            xyz_feature_3 = self.xyz_up_layer(xyz_input_3)

            rpn_feature_3 = pts_input_3[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)

            merged_feature_3 = torch.cat((xyz_feature_3, rpn_feature_3), dim=1)
            merged_feature_3 = self.merge_down_layer(merged_feature_3)
            l_xyz_3, l_features_3 = [xyz_3], [merged_feature_3.squeeze(dim=3)]
        else:
            l_xyz, l_features = [xyz_3], [features_3]

        for i in range(len(self.SA_modules)):
            li_xyz_3, li_features_3 = self.SA_modules[i](l_xyz_3[i], l_features_3[i])
            l_xyz_3.append(li_xyz_3)
            l_features_3.append(li_features_3)
        del xyz_2, features_2, l_features_2
        rcnn_cls_3rd = self.cls_layer_3rd(l_features_3[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B*64, 1 or 2)
        rcnn_reg_3rd = self.reg_layer_3rd(l_features_3[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B*64, C)
        pre_iou3=self.iou_layer(l_features_3[-1]).transpose(1, 2).contiguous().squeeze(dim=1)
        #loss
        if self.training:
            cls_label = target_dict_3rd['cls_label'].float()
            rcnn_cls_flat = rcnn_cls_3rd.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label, reduction='none')
            cls_label_flat = cls_label.view(-1)
            cls_valid_mask = (cls_label_flat >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
            gt_boxes3d_ct = target_dict_3rd['gt_of_rois']
            reg_valid_mask = target_dict_3rd['reg_valid_mask']
            fg_mask = (reg_valid_mask > 0)
            #cls_mask=(target_dict_3rd['cls_label']>0)
            #print(rcnn_reg_3rd.view(batch_size_2, -1)[cls_mask].size(0))
            #print(rcnn_reg_3rd.view(batch_size_2, -1)[fg_mask].size(0))
            if rcnn_reg_3rd.view(batch_size_2, -1)[fg_mask].size(0)==0:
                fg_mask=(reg_valid_mask <= 0)
            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                loss_utils.get_reg_loss(rcnn_reg_3rd.view(batch_size_2, -1)[fg_mask],
                                        gt_boxes3d_ct.view(batch_size_2, 7)[fg_mask],
                                        loc_scope=cfg.RCNN.LOC_SCOPE,
                                        loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                        num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                        anchor_size=anchor_size,
                                        get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                        loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                        get_ry_fine=True)
            rcnn_loss_reg = loss_loc + loss_angle + 3 * loss_size

            #three = {'rcnn_loss_cls_3rd': rcnn_loss_cls, 'rcnn_loss_reg_3rd': rcnn_loss_reg}

        else:
            three={}
        pred_boxes3d_3rd = decode_bbox_target(roi.view(-1, 7), rcnn_reg_3rd.view(-1, rcnn_reg_3rd.shape[-1]),
                                              anchor_size=anchor_size,
                                              loc_scope=cfg.RCNN.LOC_SCOPE,
                                              loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                              num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                              get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                              loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                              get_ry_fine=True).view(batch_size, -1, 7)
        if self.training:
            gt=target_dict_3rd['real_gt']
            iou_label=[]
            for i in range(batch_size_2):
                iou_label.append(iou3d_utils.boxes_iou3d_gpu(pred_boxes3d_3rd.view(-1,7)[i].view(1,7), gt[i].view(1,7)))
            iou_label=torch.cat(iou_label)
            iou_loss=F.smooth_l1_loss((pre_iou3),iou_label)
            three = {'rcnn_loss_cls_3rd': rcnn_loss_cls, 'rcnn_loss_reg_3rd': rcnn_loss_reg,'rcnn_iou_loss':iou_loss}
            del cls_label, rcnn_cls_flat, batch_loss_cls, cls_label_flat, cls_valid_mask, rcnn_loss_cls, gt_boxes3d_ct, reg_valid_mask, fg_mask
        ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg,'rcnn_cls_3rd': rcnn_cls_3rd, 'rcnn_reg_3rd': rcnn_reg_3rd
                    ,'pred_boxes3d_1st':pred_boxes3d_1st,'pred_boxes3d_2nd':pred_boxes3d_2nd,'pred_boxes3d_3rd':pred_boxes3d_3rd,
                    'pre_iou3':pre_iou3,'pre_iou2':pre_iou2,'pre_iou1':pre_iou1}
        ret_dict.update(sec)
        ret_dict.update(one)
        ret_dict.update(two)
        ret_dict.update(three)
        if self.training:
            ret_dict.update(target_dict)
        return ret_dict
