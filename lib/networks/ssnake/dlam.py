from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from lib.networks.dcn_v2 import DCN
from lib.utils.ssnake import snake_config
from lib.utils import net_utils

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, use_residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride
        self.use_residual = use_residual

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_residual:
            out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x # x is a list of features
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else: #用了迭代
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)


    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
        # self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            # proj = DeformConv(c, o)
            # node = DeformConv(o, o)
            proj = conv3x3(c, o)
            node = conv3x3(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class CenterPrior(nn.Module):
    def __init__(self, sigma=1, shape=(64,64)):
        super(CenterPrior, self).__init__()
        self.shape = shape
        self.p = 0.5
        guass_init = self.init_ct_prior()
        self.ct_prior = nn.Parameter(guass_init)
        # self.ct_prior = guass_init.clone().detach().requires_grad_(True)
        # self.ct_prior = torch.tensor(guass_init,requires_grad=True)

    def init_ct_prior(self):
        h, w = self.shape
        mu_h = h / 2
        sigma2_h = - h ** 2 / (4 * np.log(self.p))
        guass_h = np.exp(-(np.arange(0, h) - mu_h) ** 2 / sigma2_h)
        guass_h = torch.from_numpy(guass_h)
        guass_h = guass_h.unsqueeze(-1)
        mu_w = w / 2
        sigma2_w = - w ** 2 / (4 * np.log(self.p))
        guass_w = np.exp(-(np.arange(0, w) - mu_w) ** 2 / sigma2_w)
        guass_w = torch.from_numpy(guass_w)
        guass_w = guass_w.unsqueeze(0)
        guass = guass_h * guass_w
        # guass = guass/guass.sum() # there is no need to normalize

        return guass

    def forward(self, ct_hm):
        b,c,h,w = ct_hm.shape
        ct_prior = self.ct_prior.type_as(ct_hm)
        ct_prior = ct_prior.unsqueeze(0).unsqueeze(0)
        ct_prior = F.interpolate(ct_prior,(h,w),mode='nearest')
        output = ct_hm * ct_prior
        return output

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x)

class DLA_Backbone(nn.Module):
    def __init__(self, base_name, pretrained, down_ratio,
                 last_level, out_channel=0):
        super(DLA_Backbone,self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)  # eg. base_name='dla34'
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

    def forward(self,x):
        x = self.base(x)
        x_base = list(x) # create a new list copy
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        return x_base,y

class Head_Basic_Block(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=256):
        super(Head_Basic_Block, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
    def forward(self,x):
        x = self.fc(x)
        return x


class build_Heads_only(nn.Module):
    def __init__(self,use='small', cfg_component = None):
        super(build_Heads_only, self).__init__()
        self.cfg_component = cfg_component

        if use == 'small':
            self.SideSfpn_ctfin = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True),
            )

            if True: #not use actually
                self.SideSfpn_edgefin = nn.Sequential(
                    nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True),
                )
            self.SideSfpn_whfin = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True),
            )

        self.init_weight()

    def init_weight(self):
        self.SideSfpn_ctfin[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.SideSfpn_whfin)
        return

    def forward(self,y_fpn,out):
        assert len(list(y_fpn.shape)) == 4
        ct_hm = self.SideSfpn_ctfin(y_fpn)
        out.update({'ct_hm':ct_hm})

        WHFusefin = self.SideSfpn_whfin(y_fpn)
        out.update({'wh': WHFusefin})

        return out

class Neck_Basic_Block(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels):
        super(Neck_Basic_Block,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,3,1,3//2),
            nn.BatchNorm2d(mid_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,out_channels,1,1,0)
        )
    def forward(self,x):
        x = self.block(x)
        return x


class build_Fuse_Neck(nn.Module):
    def __init__(self):
        super(build_Fuse_Neck,self).__init__()
        self.fpn_conv = Neck_Basic_Block(64,64,128)
        self.S2_conv = Neck_Basic_Block(64,64,128)
        self.S3_conv = Neck_Basic_Block(128,64,128)
        self.S4_conv = Neck_Basic_Block(256,64,128)
        self.S5_conv = Neck_Basic_Block(512,64,256)

    def interpolate(self, x, scale):
        x = nn.functional.interpolate(input=x, scale_factor=(scale, scale), mode='bilinear', align_corners=False)
        return x
    def forward(self,x_base,y_fpn,z):
        assert isinstance(x_base,(list,tuple)) and len(x_base) == 6
        edgeS2 = 1 - z['edgeS2'][:,0,...].unsqueeze(1)
        edgeS3 = 1 - z['edgeS3'][:,0,...].unsqueeze(1)
        edgeS4 = 1 - z['edgeS4'][:,0,...].unsqueeze(1)
        edgeS5 = 1 - z['edgeS5'][:,0,...].unsqueeze(1)
        featS2 = self.interpolate(self.S2_conv(x_base[2]),scale=1) * edgeS2
        featS3 = self.interpolate(self.S3_conv(x_base[3]),scale=2) * edgeS3
        featS4 = self.interpolate(self.S4_conv(x_base[4]),scale=4) * edgeS4
        featS5 = self.interpolate(self.S5_conv(x_base[5]),scale=8) * edgeS5
        # feat = torch.concat([y_fpn,featS2,featS3,featS4,featS5])
        y_fpn = self.fpn_conv(y_fpn)
        y_fpn = y_fpn+featS2+featS3+featS4+featS5

        return y_fpn

class DoubleConv(nn.Module):
    def __init__(self,
            channels_in,
            channels_mid=None,
            channels_out=None,
            kernel_size=3,
            padding=1,
            dropout=0.1,
            stride=1):
        super().__init__()
        if channels_mid is None:
            channels_mid = channels_in
        if channels_out is None:
            channels_out = channels_mid

        self.block = nn.Sequential(
            nn.Conv2d(channels_in, channels_mid, kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(channels_mid),
            nn.ReLU(),
            nn.Dropout2d(p=dropout) if dropout else nn.Identity(),
            nn.Conv2d(channels_mid, channels_out, 1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
        )
    def forward(self,x):
        x = self.block(x)
        return x

class SingleConv(nn.Module):
    def __init__(self,
            channels_in,
            channels_mid=None,
            channels_out=None,
            kernel_size=3,
            padding=1,
            dropout=0.1,
            stride=1):
        super().__init__()
        if channels_mid is None:
            channels_mid = channels_in
        if channels_out is not None:
            channels_mid = channels_out

        self.block = nn.Sequential(
            nn.Conv2d(channels_in, channels_mid, kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(channels_mid),
            nn.ReLU(),
            nn.Dropout2d(p=dropout) if dropout else nn.Identity(),
        )
    def forward(self,x):
        x = self.block(x)
        return x



class DLAMSeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0, cfg_component=None):
        super(DLAMSeg, self).__init__()
        self.backbone = DLA_Backbone(base_name, pretrained, down_ratio,
                 last_level, out_channel)
        self.head = build_Heads_only(use='small', cfg_component = cfg_component)
        self.use_refine_head = False

    def forward(self, x):
        """
        origin forward function
        """
        x_base, y = self.backbone(x)
        y_fpn = y[-1]
        z= {} # for not use side_output
        z = self.head(y_fpn,z)

        return z, y_fpn
