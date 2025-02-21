import torch.nn as nn
from .dlam import DLAMSeg
from .evolve_ms import Evolution_ms
from lib.utils import data_utils
from lib.utils.ssnake import snake_decode, snake_gcn_utils
import torch


class Network_dlam(nn.Module):
    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir='', cfg_component=None):
        super(Network_dlam, self).__init__()
        self.backbone = DLAMSeg('dla{}'.format(num_layers), heads=heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv,
                          cfg_component=cfg_component)
        self.gcn = Evolution_ms(Np=128, cfg_component=cfg_component)
        self.cfg_component = cfg_component
        self.cfg_boxaug = self.cfg_component['box_aug'] if 'box_aug' in self.cfg_component.keys() else None
        self.use_box = self.cfg_boxaug['use'] if self.cfg_boxaug is not None else False
        init_method = 'box'

        self.init_method = init_method


    def decode_detection(self, output, h, w, ct_act):
        if self.init_method in ['box']:
            wh = output['wh']
            ct_hm = output['ct_hm']
            if ct_act == 'sigmoid':
                ct, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh, K=100)
            else:
                ct, detection = snake_decode.decode_ct_hm(ct_hm, wh, K=100)
            detection[..., :4] = data_utils.clip_to_image(detection[..., :4], h, w)
            output.update({'ct': ct, 'detection': detection})
        else:
            print('Not implemented')
            raise ValueError

        return ct, detection

    def forward(self, x, batch=None):
        output, cnn_feature = self.backbone(x)
        with torch.no_grad():
            ct, detection = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3),
                                              ct_act='sigmoid')
        if self.training:
            output = self.gcn(output, cnn_feature, batch)
        else:
            output = self.gcn(output, cnn_feature, batch)

        return output

def get_network(num_layers, heads, head_conv=256, down_ratio=4, det_dir='', cfg_component=None):
    network = Network_dlam(num_layers, heads, head_conv, down_ratio, det_dir, cfg_component)
    return network
