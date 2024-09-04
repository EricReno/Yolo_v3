import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import Conv, ConvBlocks

class FPN(nn.Module):
    def __init__(self, feat_dims):
        super(FPN, self).__init__()

        c3, c4, c5 = feat_dims
        out_dim = feat_dims[0]

        self.top_down_layer_1 = ConvBlocks(c5, int(0.5*c5))

        # P5 -> P4
        self.reduce_layer_1 = Conv(int(0.5*c5), int(0.25*c5), k=1)
        self.top_down_layer_2 = ConvBlocks(c4+int(0.25*c5), int(0.5*c4))

        # P4 -> P3
        self.reduce_layer_2 = Conv(int(0.5*c4), int(0.25*c4), k=1)
        self.top_down_layer_3 = ConvBlocks(c3+int(0.25*c4), int(0.5*c3))

        self.out_layers = nn.ModuleList([
                Conv(int(0.5*in_dim), out_dim, k=3) for in_dim in feat_dims])
        self.out_dim = [out_dim] * 3
    
    def forward(self, features):
        c3, c4, c5 = features
        
        # p5/32
        p5 = self.top_down_layer_1(c5)

        # p4/16
        p5_up = F.interpolate(self.reduce_layer_1(p5), scale_factor=2.0)
        p4 = self.top_down_layer_2(torch.cat([c4, p5_up], dim=1))
        
        # P3/8
        p4_up = F.interpolate(self.reduce_layer_2(p4), scale_factor=2.0)
        p3 = self.top_down_layer_3(torch.cat([c3, p4_up], dim=1))

        out_feats = [p3, p4, p5] # [P3, P4, P5]

        out_feats_proj = []
        for feat, layer in zip(out_feats, self.out_layers):
            out_feats_proj.append(layer(feat))
        return out_feats_proj

def build_fpn(fpn_cfg, feat_dims):
    if fpn_cfg == 'fpn':
        fpn = FPN(feat_dims)
        feat_dims = fpn.out_dim
    return fpn, feat_dims