from lib.utils import img_utils, data_utils, net_utils
from lib.utils.ssnake import snake_config
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import cycle
import os
import os.path as osp
from lib.config import cfg
import cv2

mean = snake_config.mean
std = snake_config.std




class Visualizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ct_threshold = 0.3
        self.ori_scale = (512, 512)
        self.use_postprocess = True
        self.edge_th = 10
        self.cfg_component = cfg['component'] if 'component' in cfg.keys() else None
        try:
            use = self.cfg_component['score_head']['use']
            self.use_score_calibration = self.cfg_component['score_head']['use_score_calibration'] if use else False
        except:
            self.use_score_calibration = False

    def visualize_panoptic_mask_score_inference(self, output, batch, center_threshold = 0.00):
        '''
        for vis paper
        '''
        cfg_score = self.cfg['component']['score_head']
        polys = output['py_post'].detach().cpu().numpy() * snake_config.ro
        scores = output['scp_fin'].detach().cpu().numpy()

        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        inp = inp.cpu().numpy()
        detection = output['detection_post'].squeeze(0)
        if not self.use_score_calibration:
            sc_cts = detection[:, 4]
            sc_cts = sc_cts.detach().cpu().numpy()
        else:
            sc_cts = output['ct_fin'].detach().cpu().numpy()
        inp_ori = inp.copy()

        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)
        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        h,w = inp.shape[:2]
        Ninst_tot = len(polys)
        for i in range(Ninst_tot):
            idx = Ninst_tot - i -1
            assert idx >=0 and idx < Ninst_tot
            sc_ct = sc_cts[idx]
            if sc_ct < center_threshold:
                continue
            color = next(colors)
            poly = polys[idx]
            score = scores[idx]
            # draw mask
            mask_ = np.zeros((h,w),dtype = float)
            poly1 = np.round(poly).astype(np.int32)
            cv2.fillPoly(mask_, [poly1], 1.0)
            mask_ = mask_[:,:,np.newaxis]
            alpha = 0.5 # untransparent of color (0 means total transparent)
            inp = (1-mask_)*inp + mask_*(color[np.newaxis,np.newaxis,:]*alpha + inp_ori*(1-alpha))
            # draw score poly
            poly_draw = np.append(poly, [poly[0]], axis=0)
            score = np.append(score, [score[0]], axis=0)
            # ax.plot(poly_draw[:, 0], poly_draw[:, 1], color=color, linewidth=2, markersize=2,marker='o',zorder=1)
            # score point
            ax.scatter(poly_draw[:, 0], poly_draw[:, 1], c=score, marker='o', s=4, vmin=0,vmax=1,zorder=2)
            # ax.scatter(poly[:, 0], poly[:, 1], c=color,s=4)
            # plot box
            # x_min, y_min, x_max, y_max = box[i]
            # ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color=color,
            #         linewidth=1)
            # plot score_ct
            # sc_str = format(sc_ct * 100, '.1f') + '%'
            # pad = 4
            # x_ct,y_ct = data_utils.find_centroid(poly)
            # x_pos = max(x_ct-pad,0)
            # y_pos = max(y_ct,0)
            # darker_color = color*0.5
            # ax.text(x_pos, y_pos, sc_str, color=1-color, fontsize=10)
            # ax.text(x_pos, y_pos, sc_str, color=np.ones_like(color), fontsize=10,
            #         bbox={'facecolor':darker_color,'alpha':0,'pad':pad, 'edgecolor':None})
        ori_w,ori_h = self.ori_scale
        inp = cv2.resize(inp,(ori_h,ori_w))
        ax.imshow(inp)

        root_path = osp.join('data/result/', cfg.task, cfg.model, f'vis_score_inference_th{center_threshold}')
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        img_name = batch['meta']['img_name'][0][:-4]
        img_path = osp.join(root_path, img_name)
        # plt.show()
        plt.savefig(img_path,dpi=150)
        plt.close('all')
        # break
        print(f'Output visualized image is saved at {img_path}.')

    def visualize(self, output, batch):
        self.visualize_panoptic_mask_score_inference(output, batch, center_threshold=0.5)

class Visualizer_dataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ct_threshold = 0.3
        self.ori_scale = (512, 512)
        self.use_postprocess = True
        self.edge_th = 10
        self.cfg_component = cfg['component'] if 'component' in cfg.keys() else None
        try:
            use = self.cfg_component['score_head']['use']
            self.use_score_calibration = self.cfg_component['score_head']['use_score_calibration'] if use else False
        except:
            self.use_score_calibration = False

    def visualize_panoptic_mask_score_inference(self, output, batch, center_threshold = 0.00):
        '''
        for vis paper
        '''
        cfg_score = self.cfg['component']['score_head']
        polys = output['py_post'].detach().cpu().numpy() * snake_config.ro
        scores = output['scp_fin'].detach().cpu().numpy()

        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        inp = inp.cpu().numpy()
        detection = output['detection_post'].squeeze(0)
        if not self.use_score_calibration:
            sc_cts = detection[:, 4]
            sc_cts = sc_cts.detach().cpu().numpy()
        else:
            sc_cts = output['ct_fin'].detach().cpu().numpy()
        inp_ori = inp.copy()

        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)
        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        h,w = inp.shape[:2]
        Ninst_tot = len(polys)
        for i in range(Ninst_tot):
            idx = Ninst_tot - i -1
            assert idx >=0 and idx < Ninst_tot
            sc_ct = sc_cts[idx]
            if sc_ct < center_threshold:
                continue
            color = next(colors)
            poly = polys[idx]
            score = scores[idx]
            # draw mask
            mask_ = np.zeros((h,w),dtype = float)
            poly1 = np.round(poly).astype(np.int32)
            cv2.fillPoly(mask_, [poly1], 1.0)
            mask_ = mask_[:,:,np.newaxis]
            alpha = 0.5 # untransparent of color (0 means total transparent)
            inp = (1-mask_)*inp + mask_*(color[np.newaxis,np.newaxis,:]*alpha + inp_ori*(1-alpha))
            # draw score poly
            poly_draw = np.append(poly, [poly[0]], axis=0)
            score = np.append(score, [score[0]], axis=0)
            # ax.plot(poly_draw[:, 0], poly_draw[:, 1], color=color, linewidth=2, markersize=2,marker='o',zorder=1)
            # score point
            ax.scatter(poly_draw[:, 0], poly_draw[:, 1], c=score, marker='o', s=4, vmin=0,vmax=1,zorder=2)
            # ax.scatter(poly[:, 0], poly[:, 1], c=color,s=4)
            # plot box
            # x_min, y_min, x_max, y_max = box[i]
            # ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color=color,
            #         linewidth=1)
            # plot score_ct
            # sc_str = format(sc_ct * 100, '.1f') + '%'
            # pad = 4
            # x_ct,y_ct = data_utils.find_centroid(poly)
            # x_pos = max(x_ct-pad,0)
            # y_pos = max(y_ct,0)
            # darker_color = color*0.5
            # ax.text(x_pos, y_pos, sc_str, color=1-color, fontsize=10)
            # ax.text(x_pos, y_pos, sc_str, color=np.ones_like(color), fontsize=10,
            #         bbox={'facecolor':darker_color,'alpha':0,'pad':pad, 'edgecolor':None})
        ori_w,ori_h = self.ori_scale
        inp = cv2.resize(inp,(ori_h,ori_w))
        ax.imshow(inp)

        root_path = osp.join('data/result/', cfg.task, cfg.model, f'vis_score_inference_th{center_threshold}')
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        img_name = batch['meta']['img_name'][0][:-4]
        img_path = osp.join(root_path, img_name)
        # plt.show()
        plt.savefig(img_path,dpi=150)
        plt.close('all')
        # break


    def visualize(self, output, batch):
        self.visualize_panoptic_mask_score_inference(output, batch, center_threshold=0.5)

