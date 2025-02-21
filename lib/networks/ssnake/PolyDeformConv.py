import torch
from torch import nn
from .snake import Snake
from lib.utils.ssnake import snake_gcn_utils, snake_config

class SnakeOriginBlock(nn.Module):
    def __init__(self,Nn):
        super(SnakeOriginBlock, self).__init__()
        self.snake = Snake(feature_dim=64+2, state_dim=128, conv_type='dgrid',coord=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self,cnn_feature, i_it_poly, ind, ct_map=None, score_infer=None):
        """
        score_infer: [Ninst,Np,1]
        """
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        poly_can = snake_gcn_utils.img_poly_to_can_poly(i_it_poly)
        poly_can_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)  # [Ninst,C,Np]
        poly_can_feat = torch.cat([poly_can_feat, poly_can.permute(0, 2, 1)*snake_config.ro], dim=1)
        init_input = poly_can_feat
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2),init_input.device)  # get adjacent index
        i_poly_snake = self.snake(init_input, adj).permute(0, 2, 1)  # snake is used to predict delta_x(or y)
        i_poly = i_it_poly * snake_config.ro + i_poly_snake
        return i_poly


class SnakeBlock(nn.Module):
    def __init__(self,Nn,dilate=1, use_aggr_ctfeat=None, use_attentive_refine=False, use_score_in=False):
        super(SnakeBlock, self).__init__()
        self.snake = Snake(feature_dim=64, state_dim=128, conv_type='dgrid')
        in_channels = 128+2
        if use_score_in:
            in_channels += 1
        self.fuse = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True))
        self.use_aggr_ctfeat = use_aggr_ctfeat
        self.use_attentive_refine = use_attentive_refine
        self.use_score_in = use_score_in
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self,cnn_feature, i_it_poly, ind, ct_map=None, score_infer=None):
        """
        score_infer: [Ninst,Np,1]
        """
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        poly_can = snake_gcn_utils.img_poly_to_can_poly(i_it_poly)
        poly_can_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)  # [Ninst,C,Np]
        poly_can_feat = torch.cat([poly_can_feat, poly_can.permute(0, 2, 1)], dim=1)
        # init_input = poly_can_feat
        if self.use_aggr_ctfeat and ct_map is not None:
            inst_feat = snake_gcn_utils.obtain_inst_feature(ct_map, cnn_feature, i_it_poly, ind)
        else:
            center = (torch.min(i_it_poly, dim=1)[0] + torch.max(i_it_poly, dim=1)[0]) * 0.5
            inst_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, center[:, None], ind, h, w)  # [Ninst,Nfeat,1]
            inst_feat = inst_feat.squeeze(2)
        if self.use_score_in:
            init_input = snake_gcn_utils.fuse_inst_feat(self.fuse, inst_feat, poly_can_feat, res=False, score=score_infer)
        else:
            init_input = snake_gcn_utils.fuse_inst_feat(self.fuse, inst_feat, poly_can_feat, res=False)

        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2),init_input.device)  # get adjacent index
        i_poly_snake = self.snake(init_input, adj).permute(0, 2, 1)  # snake is used to predict delta_x(or y)

        if self.use_attentive_refine or score_infer is not None:
            i_poly = i_it_poly * snake_config.ro + i_poly_snake * (1-score_infer)
        else:
            i_poly = i_it_poly * snake_config.ro + i_poly_snake
        return i_poly

class ScoreBlock(nn.Module):
    def __init__(self,Nn,dilate=1, use_aggr_ctfeat=None):
        super(ScoreBlock, self).__init__()
        self.snake = Snake(feature_dim=64, state_dim=128, conv_type='dgrid', out_dim=1)
        self.fuse = nn.Sequential(
            nn.Conv1d(128+2, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()
        self.use_aggr_ctfeat = use_aggr_ctfeat
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self,cnn_feature, i_it_poly, ind, ct_map=None):
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        poly_can = snake_gcn_utils.img_poly_to_can_poly(i_it_poly)
        poly_can_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)  # [Ninst,C,Np]
        poly_can_feat = torch.cat([poly_can_feat, poly_can.permute(0, 2, 1)], dim=1)
        # init_input = poly_can_feat
        if self.use_aggr_ctfeat and ct_map is not None:
            inst_feat = snake_gcn_utils.obtain_inst_feature(ct_map, cnn_feature, i_it_poly, ind)
        else:
            center = (torch.min(i_it_poly, dim=1)[0] + torch.max(i_it_poly, dim=1)[0]) * 0.5
            inst_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, center[:, None], ind, h, w)  # [Ninst,Nfeat,1]
            inst_feat = inst_feat.squeeze(2)
        init_input = snake_gcn_utils.fuse_inst_feat(self.fuse, inst_feat, poly_can_feat, res=False)

        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2),init_input.device)  # get adjacent index
        score = self.snake(init_input, adj).permute(0, 2, 1)  # snake is used to predict delta_x(or y)
        score = self.sigmoid(score)
        return score

