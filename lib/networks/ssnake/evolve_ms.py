import torch.nn as nn
from lib.utils.ssnake import snake_gcn_utils, snake_config
import torch
from lib.networks.ssnake.PolyDeformConv import SnakeBlock, ScoreBlock
from lib.utils import net_utils
REFINE_METHOD = ['snake0_snake1_snake2',
                 ]

class Evolution_ms(nn.Module):
    '''
    boxtrain
    '''
    def __init__(self,Np,cfg_component=None):
        super(Evolution_ms, self).__init__()
        self.Np = Np
        Nn=9
        self.Nn=Nn
        self.iter = 2
        self.cfg_component = cfg_component
        self.use_aggr_ctfeat_heatmap = False


        # score
        self.use_shead = False
        self.use_attentive_refine = False
        self.use_score_in = False
        self.use_weight_loss = False
        if 'score_head' in cfg_component.keys():
            if cfg_component['score_head']['use']:
                self.use_shead = True
                self.shead = ScoreBlock(Nn,dilate=1)
                cfg_shead = cfg_component['score_head']
                self.use_attentive_refine = False
                self.use_score_in = False
                self.use_weight_loss = False
                self.use_score_all = False
                self.score_crit = net_utils.ScoreLoss(cfg_shead)
            self.refine_method = REFINE_METHOD[0]
        self.cfg_boxaug = cfg_component['box_aug'] if 'box_aug' in cfg_component.keys() else None
        self.use_box = self.cfg_boxaug['use'] if self.cfg_boxaug is not None else False
        init_method = 'box'
        self.init_method = init_method


        setting = {}
        basic_block = SnakeBlock
        setting.update({'dilate':1,'use_aggr_ctfeat':self.use_aggr_ctfeat_heatmap,
                        'use_attentive_refine': self.use_attentive_refine,
                        'use_score_in':self.use_score_in})

        # if not self.use_same_snake:
        self.block = basic_block(Nn, **setting)
        for i in range(self.iter):
            block = basic_block(Nn,**setting)
            self.__setattr__('block'+str(i), block)


    def prepare_training(self, output, batch, w, h, cfg_component=None):
        with torch.no_grad():
            init = snake_gcn_utils.prepare_training_myshape(output, batch,self.Np, w, h, self.cfg_boxaug) # init poly shape
            ct_01 = batch['ct_01'].type(torch.bool)
            py_gt = batch['i_gt_py']  # get ground truth
            py_gt_train = snake_gcn_utils.collect_training(py_gt, ct_01)
            output.update({'i_gt_py': py_gt_train})
            return init

    def prepare_testing(self, output, cnn_feature, scale_factor=1.0):
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init = snake_gcn_utils.prepare_testing_myshape(output,self.Np, w, h, self.cfg_boxaug, scale_factor)
        return init

    def inst_evolve_poly(self, block,cnn_feature, i_it_poly, ind, ct_map=None, score=None):
        # i_it_poly's max should be in cnn_feature h/w range
        # output i_poly's max shoule be in origin size range
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        i_poly = block(cnn_feature,  i_it_poly, ind, ct_map, score)
        return i_poly

    def score_predict(self, block, cnn_feature, i_it_poly, ind):
        # i_it_poly's max should be in cnn_feature h/w range
        # output i_poly's max shoule be in origin size range
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly[...,0:1])
        i_poly = block(cnn_feature, i_it_poly, ind)
        return i_poly


    def snake0_snake1_snake2(self, output, cnn_feature):

        init = self.prepare_testing(output, cnn_feature)
        ct_hm = output['ct_hm']
        output.update({'init_py':init['init_py'],'init_ind':init['ind']})
        py = self.inst_evolve_poly(self.block, cnn_feature, init['init_py'], init['ind'], ct_hm)
        pys = [py / snake_config.ro]

        for i in range(self.iter):
            py = py / snake_config.ro
            block = self.__getattr__('block' + str(i))
            py = self.inst_evolve_poly(block, cnn_feature, py, init['ind'], ct_hm)
            pys.append(py / snake_config.ro)
        output.update({'py': pys})
        output.update({'py_fin':pys[-1]})
        return output

    def inference_process(self,refine_method, output, cnn_feature):
        assert refine_method in REFINE_METHOD
        if refine_method == 'snake0_snake1_snake2':
            output = self.snake0_snake1_snake2(output,cnn_feature)
        else:
            print('Not Implemented.')
            raise ValueError

        if self.use_shead:
            output = self.score_inference(output, cnn_feature)

        return output
    def score_calibrate(self, output):
        ct = output['ct']
        B, Ninst_b = ct.shape[:2]
        Ninst = B * Ninst_b
        score_ct = output['detection'][0, :, 4].unsqueeze(-1)
        score_point = output['scp_fin'].mean(dim=1)
        score_new = score_ct * score_point
        score_new = score_new.squeeze(1)
        output.update({'ct_fin': score_new})
        return output

    def score_estimate(self, polys, cnn_feature):
        # polys = output['py_fin']
        Ninst = len(polys)
        py_ind = torch.zeros((Ninst), dtype=torch.long, device=polys.device)
        scores = self.score_predict(self.shead, cnn_feature, polys, py_ind)
        return scores


    def score_inference(self, output, cnn_feature):
        cfg_shead = self.cfg_component['score_head']
        use_score_calibration = cfg_shead[
            'use_score_calibration'] if 'use_score_calibration' in cfg_shead.keys() else False
        polys = output['py_fin']
        scores = self.score_estimate(polys, cnn_feature)
        output.update({'scp_fin': scores})

        if use_score_calibration:
            output = self.score_calibrate(output)

        return output


    def train_process(self, output, batch, cnn_feature):
        h,w = cnn_feature.size(2), cnn_feature.size(3)
        init = self.prepare_training(output, batch, w, h, cfg_component=self.cfg_component)
        init_pys = [init['init_py']]
        ct_hm = output['ct_hm']
        py_pred = self.inst_evolve_poly(self.block, cnn_feature, init['init_py'], batch['py_ind'], ct_hm)
        py_preds = [py_pred]
        for i in range(self.iter):
            init_py = py_pred / snake_config.ro
            init_pys.append(init_py)
            block = self.__getattr__('block' + str(i))
            py_pred = self.inst_evolve_poly(block, cnn_feature, init_py, batch['py_ind'], ct_hm)
            py_preds.append(py_pred)
        output.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py'] * snake_config.ro})
        output.update({'init_pys': init_pys})
        if self.use_shead:
            in_pys, sc_preds = [], []
            for i in range(self.iter + 2):
                if i == 0:
                    in_py = init['init_py']
                else:
                    in_py = py_preds[i - 1] / snake_config.ro
                sc_pred = self.score_predict(self.shead, cnn_feature, in_py, batch['py_ind'])
                sc_preds.append(sc_pred)
                in_pys.append(in_py * snake_config.ro)
            output.update({'sc_pred': sc_preds, 'sc_inpy': in_pys})


    def obtain_score_gt(self,in_py, py_gt):
        Ninst = in_py.size(0)
        if Ninst == 0:
            return torch.zeros_like(in_py)[:,:,:1]
        sc_gts = []
        for idx in range(Ninst):
            sc_gt = self.score_crit(None,
                                  in_py[idx, ...],
                                  py_gt[idx, ...])
            sc_gts.append(sc_gt.unsqueeze(0).unsqueeze(-1))
        sc_gts = torch.cat(sc_gts,dim=0)
        torch.clip_(sc_gts,min=0.1,max=1)
        return sc_gts



    def forward(self, output, cnn_feature, batch=None):
        if batch is not None and 'test' not in batch['meta']:
            self.train_process(output, batch, cnn_feature)

        if not self.training:
            with torch.no_grad():
                output = self.inference_process(self.refine_method, output, cnn_feature)

        return output

