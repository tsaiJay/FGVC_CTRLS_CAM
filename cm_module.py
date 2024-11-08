import torch
import torch.nn as nn
import torch.nn.functional as F


class camModule(nn.Module):
    def __init__(self, cm_args, n_features):
        super().__init__()

        assert cm_args, 'no parameters for cam module'
        self.use_norm = cm_args.use_norm
        self.use_batchnorm = cm_args.use_batchnorm
        self.scale = cm_args.scale
        self.map_function = cm_args.map_function
        
        if self.use_batchnorm:
            self.batch_norm = nn.BatchNorm2d(num_features=n_features)

        if self.map_function == 'sigmoid':
            self.map = nn.Sigmoid()
        elif self.map_function == 'tanh':
            self.map = nn.Tanh()

    def smooth_single_map(self, cam_map):
        ''' intput (1, 14, 14)'''

        ''' normalize '''
        if self.use_norm:
            map_std = cam_map.std()
            map_mean = cam_map.mean()
            cam_map = (cam_map - map_mean) / map_std

        ''' mapping value '''
        ''' 0.25 ~ 8 '''
        smooth_map = self.map(cam_map * self.scale)
        # print(norm_map.max(), norm_map.min())
        # print(smooth_map.max(), smooth_map.min())

        return smooth_map

    def get_smooth_CAM_hint(self, feature, fc_weight, label):
        ''' feature.shape: 2048x14x14 with input=448x448'''
        assert feature.size(0) == label.size(0), '(label, feature) batch_size uncompatible!'

        batch = label.size(0)

        hint_list = []
        for b in range(batch):
            hint_feature = fc_weight[int(label[b])].view(-1, 1, 1) * feature[b]
            hint_map = torch.sum(hint_feature, dim=0, keepdim=True) / hint_feature.size(1)
            # show_hint_map(hint_map, save='map')
            smooth_hint = self.smooth_single_map(hint_map)
            # show_hint_map(smooth_hint, save='map_smooth1', show)
            hint_list.append(smooth_hint)
        output_hint = torch.stack((hint_list), dim=0)  # shape (b, 1, 14, 14)

        return output_hint

    def forward(self, logit, fc_weight, label):
        if self.use_batchnorm:
            logit = self.batch_norm(logit)

        hint_map = self.get_smooth_CAM_hint(logit, fc_weight, label)

        return hint_map



