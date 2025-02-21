import numpy as np
import torch
from lib.postprocessors.vcoco import utils
from lib.utils.ssnake import snake_config

class Postprocessor:
    def __init__(self, adapt_cfg):
        self.cfg = adapt_cfg
        self.post_types = self.cfg.post.post_types
        self.edge_th = 10

    def detect_edge_poly(self, poly, img_scale):
        # th_toedge = 0.02
        # th_cnt = 0.2  # use 1/4, means invalid when one edge from a square is invalid
        th_toedge = self.cfg.post.th_toedge
        th_cnt = self.cfg.post.th_cnt
        assert isinstance(img_scale, (list, tuple)) and len(img_scale) == 2
        (w_img, h_img) = img_scale
        Np, _ = poly.shape
        polyh = poly[:, 0]
        polyw = poly[:, 1]
        polyh_hmin = (polyh - 0).abs()
        polyh_hmax = (polyh - h_img).abs()
        polyw_wmin = (polyh - 0).abs()
        polyw_wmax = (polyw - w_img).abs()
        poly_diffs = []
        poly_diffs.append(polyh_hmin)
        poly_diffs.append(polyh_hmax)
        poly_diffs.append(polyw_wmin)
        poly_diffs.append(polyw_wmax)
        px_toedge_h = th_toedge * h_img
        px_toedge_w = th_toedge * w_img
        valid_hmin = (polyh_hmin < px_toedge_h)
        valid_hmax = (polyh_hmax < px_toedge_h)
        valid_wmin = (polyw_wmin < px_toedge_w)
        valid_wmax = (polyw_wmax < px_toedge_w)
        valid_total = torch.cat((valid_hmin, valid_hmax, valid_wmin, valid_wmax), dim=0)
        cnt_total = valid_total.sum()

        if cnt_total > th_cnt * Np:
            valid = False
        else:
            valid = True

        return valid
    def edge_remove(self,out_ori):
        h_img, w_img = out_ori['ct_hm'].shape[-2:]
        if 'detection_post' in out_ori:
            key_det = 'detection_post'
        else:
            key_det = 'detection'
        detection = out_ori[key_det].squeeze()
        assert len(detection.shape) == 2
        score = detection[:, 4].detach().cpu().numpy()
        if 'py_post' in out_ori:
            key_py = 'py_post'
        else:
            key_py = 'py'
        py_ori = out_ori[key_py].detach().cpu().numpy()
        t = py_ori[:, :, 1].max(axis=(1))
        b = py_ori[:, :, 1].min(axis=(1))
        l = py_ori[:, :, 0].min(axis=(1))
        r = py_ori[:, :, 0].max(axis=(1))
        input = np.stack((l, b, r, t, score), axis=1)  # [l,b,r,t]
        keep = utils.edge_remove(input,(h_img,w_img),self.edge_th)

        py_post = py_ori[keep]
        assert len(keep) > 0
        assert len(py_post.shape) == 3
        device = detection.device
        py_post = torch.tensor(py_post, device=device)
        out_ori.update({'py_post': py_post})

        detection = detection[keep].unsqueeze(0)
        out_ori.update({'detection_post': detection})

        # for vis deform
        for iter_idx in iter([0,1,2]):
            deform_p = out_ori[f'deform_iter{iter_idx}']['deform_p']
            deform_m = out_ori[f'deform_iter{iter_idx}']['deform_m']
            p_ori = out_ori[f'deform_iter{iter_idx}']['p_ori']
            deform_p_post = deform_p[keep]
            deform_m_post = deform_m[keep]
            p_ori_post = p_ori[keep]
            out_ori[f'deform_iter{iter_idx}']['deform_p_post'] = deform_p_post
            out_ori[f'deform_iter{iter_idx}']['deform_m_post'] = deform_m_post
            out_ori[f'deform_iter{iter_idx}']['p_ori_post'] = p_ori_post
        return

    def nms(self,out_ori):
        detection = out_ori['detection_post'].squeeze()
        assert len(detection.shape) == 2
        score = detection[:, 4].detach().cpu().numpy()
        py_ori = out_ori['py_post'].detach().cpu().numpy()
        t = py_ori[:, :, 1].max(axis=(1))
        b = py_ori[:, :, 1].min(axis=(1))
        l = py_ori[:, :, 0].min(axis=(1))
        r = py_ori[:, :, 0].max(axis=(1))
        input = np.stack((l,b,r,t,score),axis=1)# [l,b,r,t]
        # input = detection[:,:5].detach().cpu().numpy()
        keep = utils.nms(input,thresh=0.5)
        py_post = py_ori[keep]
        assert len(keep)>0
        assert len(py_post.shape)==3
        device = detection.device
        py_post = torch.tensor(py_post,device=device)
        out_ori.update({'py_post': py_post})

        detection = detection[keep].unsqueeze(0)
        out_ori.update({'detection_post':detection})
        return

    def ct_thresh(self, out_ori):
        ct_th = snake_config.ct_threshold
        detection = out_ori['detection_post'].squeeze()
        assert len(detection.shape) == 2
        score = detection[:, 4].detach().cpu().numpy()
        py_ori = out_ori['py_post'].detach().cpu().numpy()
        keep = (score > ct_th)
        py_post = py_ori[keep]
        assert len(keep) > 0
        assert len(py_post.shape) == 3
        device = detection.device
        py_post = torch.tensor(py_post, device=device)
        out_ori.update({'py_post': py_post})

        detection = detection[keep].unsqueeze(0)
        out_ori.update({'detection_post': detection})
        return

    def post_processing(self, out_ori):
        '''
        do post processing step
        1. remove the instances which are on the edge
        '''
        # 1
        post_required = self.cfg.post.required
        if 'py_fin' in out_ori.keys():
            py_post = out_ori['py_fin']
        else:
            py_post = out_ori['py'][-1]

        out_ori.update({'py_post': py_post})
        detection = out_ori['detection']
        out_ori.update({'detection_post': detection})

        if not post_required:
            return
        # start
        for _type,_required in iter(self.post_types.items()):
            if  _type == 'nms' and _required==True:
                self.nms(out_ori)
            if _type == 'ct_th' and _required == True:
                self.ct_thresh(out_ori)
            if _type == 'edge_remove' and _required == True:
                self.edge_remove(out_ori)


        return
