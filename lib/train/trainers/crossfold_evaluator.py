import numpy as np
import os
from os import path as osp
import torch
import json



class Crossfold_Evaluator():
    def __init__(self,num_fold=4):
        self.num_fold = num_fold
        self.metrics = {}
        self.metrics_final = {}
        self.fold_metrics = []

    def get_fold_result(self,i_fold,metrics):
        for k,v in iter(metrics.items()):
            if torch.is_tensor(v):
                metrics[k] = v.cpu().numpy()

        self.metrics[str(i_fold)] = metrics

    def calc_final_result(self,cfg):
        print('start to calc final')
        metrics_tmp = self.metrics[str(0)]
        for k, v in iter(metrics_tmp.items()):
            if k == 'num_imgs':
                continue
            num_tot = 0
            val_tot = 0
            for i_fold in range(self.num_fold):
                metrics = self.metrics[str(i_fold)]
                self.fold_metrics.append(metrics)
                num_imgs = metrics['num_imgs']
                num_tot += num_imgs
                val_tot += num_imgs*metrics[k]
            val_final = val_tot/num_tot
            self.metrics_final.update({k:val_final})
        self.metrics_final.update({'num_imgs':num_tot})
        self.save_metrics(cfg)
        print(f'generate final metrics:{self.metrics_final}')
    def change_type_for_save(self, metrics_input):
        metrics = metrics_input.copy()
        for k, v in iter(metrics.items()):
            if isinstance(v, np.ndarray):
                metrics[k] = v.tolist()
        return metrics

    def save_metrics(self,cfg):
        result_dir = cfg.result_dir
        result_file = osp.join(result_dir,'accuracy.json')
        metrics_save = {}
        for idx in range(self.num_fold+1):
            if idx == self.num_fold:
                which_fold = "final"
                metrics = self.metrics_final
                metrics.update({"which_fold":which_fold})
            else:
                which_fold = str(idx)
                metrics = self.fold_metrics[idx]
                metrics.update({"which_fold": which_fold})
            metrics = self.change_type_for_save(metrics)
            metrics_save.update({which_fold:metrics})

        if not osp.exists(result_dir):
            os.makedirs(result_dir)
        print(metrics_save)
        dict_str = json.dumps(metrics_save,sort_keys=True,indent=4,separators=(',',':'))
        with open(result_file,'w') as json_file:
            json_file.write(dict_str)
        # cfg

    def save_metrics_simple(self,cfg):
        '''
        for only one fold dsb dataset
        '''
        result_dir = cfg.result_dir
        result_file = osp.join(result_dir,'accuracy.json')
        metrics_save = {}
        metrics = self.metrics[str(0)]
        metrics = self.change_type_for_save(metrics)
        metrics_save = metrics
        if not osp.exists(result_dir):
            os.makedirs(result_dir)
        print(metrics_save)
        dict_str = json.dumps(metrics_save,sort_keys=True,indent=4,separators=(',',':'))
        with open(result_file,'w') as json_file:
            json_file.write(dict_str)
        # cfg




