import torch.nn as nn
from lib.utils import net_utils
import torch
EPS=0.00001

class NetworkWrapper(nn.Module):
    def __init__(self, net, cfg):
        super(NetworkWrapper, self).__init__()
        self.cfg_component = cfg['component'] if 'component' in cfg.keys() else None
        self.cfg_loss = cfg['loss'] if 'loss' in cfg.keys() else None

        self.net = net
        self.ct_crit = net_utils.FocalLoss()
        self.wh_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.hau_crit = net_utils.HausdorffLoss()
        cfg_shead = self.cfg_component['score_head'] if 'score_head' in self.cfg_component.keys() else None
        self.score_crit = net_utils.ScoreLoss(cfg_shead)
        self.cfg_boxaug = self.cfg_component['box_aug'] if 'box_aug' in self.cfg_component.keys() else None
        self.use_box = self.cfg_boxaug['use'] if self.cfg_boxaug is not None else False
        init_method = 'box'
        self.init_method = init_method

    def forward(self, batch):
        output = self.net(batch['inp'], batch)
        # output['ct_hm']:[B,C,H,W]
        # output['wh']:[B,2,H,W]
        # output['ct']:[B,N,2]
        # output['detection']:[B,N,6](6:boxes4,scores1,clses1)
        # output['i_it_4py']:[Ninst,Np1,2](Np1=40?) Ninst is the inst num of all batches
        # output['i_it_py']:[Ninst,Np2,2](Np2=128)
        # output['i_gt_4py']:[Ninst,4,2]
        # output['i_gt_py']:[Ninst,Np2,2]
        # output['ex_pred']:[Ninst,4,2]
        # output['py_pred']:list (len=3?) of [Ninst,Np2,2]
        scalar_stats = {}
        loss = 0
        if self.cfg_loss is not None and self.cfg_loss['ct'] is not False:
            weight = self.cfg_loss['ct'] if self.cfg_loss['ct'] is not True else 1.0
            ct_loss = self.ct_crit(net_utils.sigmoid(output['ct_hm']), batch['ct_hm'] ,th=1) # center
            ct_loss = ct_loss * weight
            scalar_stats.update({'ct_loss': ct_loss})
            loss += ct_loss


        if self.cfg_loss is not None and self.cfg_loss['init'] is not False:
            if self.init_method == 'box':
                weight = self.cfg_loss['init'] if self.cfg_loss['init'] is not True else 1.0
                wh_loss = self.wh_crit(output['wh'], batch['wh'], batch['ct_ind'], batch['ct_01']) # w,h
                wh_loss = wh_loss * weight
                scalar_stats.update({'wh_loss': wh_loss})
                loss += wh_loss
            else:
                print('Not implemented.')
                raise ValueError

        cfg_shead = self.cfg_component['score_head'] if 'score_head' in self.cfg_component.keys() else None
        use_shead = cfg_shead['use'] if cfg_shead is not None else False
        save_sc_gt = cfg_shead['save_sc_gt'] if 'save_sc_gt' in cfg_shead.keys() else False
        if use_shead:
            if save_sc_gt:
                sc_gts = []
            device = output['sc_inpy'][0].device
            shead_loss=torch.tensor(0.0,device=device)
            for i in range(len(output['sc_inpy'])):
                for idx in range(output['sc_inpy'][i].shape[0]):
                    assert output['sc_inpy'][i].shape[0]>0
                    part, sc_gt = self.score_crit(output['sc_pred'][i][idx,...],
                                           output['sc_inpy'][i][idx,...],
                                           output['i_gt_py'][idx,...])
                    part = part/output['sc_pred'][i].shape[0]
                    shead_loss += part*10.0
                    if save_sc_gt:
                        sc_gts.append(sc_gt)

            scalar_stats.update({'shead_loss': shead_loss})
            loss += shead_loss
            if save_sc_gt:
                sc_gts = torch.cat(sc_gts,0)
                output.update({'sc_gt':sc_gts})

        if torch.isnan(loss):
            print('nan in trainers/ssnake')
        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

