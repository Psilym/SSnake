import os
from lib.utils.ssnake import snake_coco_utils, snake_config
import cv2
import numpy as np
import math
from lib.utils import data_utils
import torch.utils.data as data
from pycocotools.coco import COCO
from lib.config import cfg
class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split,speed_eval=False,cfg=None):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        self.anns = sorted(self.coco.getImgIds())
        self.anns = np.array([ann for ann in self.anns if len(self.coco.getAnnIds(imgIds=ann, iscrowd=0))]) # filter none inst image
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self.ts_cls = {0: 0, 1: 1} # thing:0, stuff:1 # villi:0, villiu:1 # {villi:thing, villiu:stuff}
        self.biggest_villi_size = 0
        self.speed_eval = speed_eval
        if cfg is not None and 'train_dataset' in cfg:
            self.scale_range = list(cfg['train_dataset']['scale_range'])
        else:
            self.scale_range = [0.7, 1.1]
        self.cfg_component = cfg['component'] if 'component' in cfg.keys() else None
        init_method = 'box'
        self.init_method = init_method

        print(f'Augment Scale Range is set to {self.scale_range}')

    def process_info(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=0)
        anno = self.coco.loadAnns(ann_ids)
        img_name = self.coco.loadImgs(int(img_id))[0]['file_name']
        path = os.path.join(self.data_root, img_name)
        return anno, path, img_id, img_name

    def read_original_data(self, anno, path):
        img = cv2.imread(path)
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]
        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno]
        return img, instance_polys, cls_ids

    def transform_original_data(self, instance_polys, flipped_h, flipped_v, height, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]

            if flipped_h:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1 #along x or w dimension
                    polys_.append(poly.copy())
                polys = polys_
            if flipped_v:
                polys_ = []
                for poly in polys:
                    poly[:, 1] = height - np.array(poly[:, 1]) - 1 #along y or h dimension
                    polys_.append(poly.copy())
                polys = polys_

            polys = snake_coco_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys, cls_ids, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        polys = instance_polys
        # polys = snake_coco_utils.get_valid_polys_my(polys) # get exterior
        polys, cls_ids = snake_coco_utils.filter_tiny_polys(polys, cls_ids)
        polys = snake_coco_utils.get_cw_polys(polys)
        polys, cls_ids = snake_coco_utils.filter_outside_polys(polys, cls_ids, bound_size=(output_h,output_w))
        return polys, cls_ids

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = snake_coco_utils.get_extreme_points(instance)
            extreme_points.append(points)
        return extreme_points

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, reg, ct_cls, ct_ind, ct_coord):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id) # center_class

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32) # center position
        ct_float = ct.copy()
        ct = np.round(ct).astype(np.float32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius) # update ct_hm

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0]) # center_index: the index transformed from coordinate
        ct_coord.append([ct[0],ct[1]])
        reg.append((ct_float - ct).tolist()) # error due to round operation

        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    def prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, h, w):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        img_init_poly = snake_coco_utils.get_init(box)
        img_init_poly = snake_coco_utils.uniformsample(img_init_poly, snake_config.init_poly_num)
        can_init_poly = snake_coco_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = extreme_point
        can_gt_poly = snake_coco_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        i_it_4pys.append(img_init_poly) # img_init_poly(4 mid points)
        c_it_4pys.append(can_init_poly) # canonical_init_poly(4 mid points)
        i_gt_4pys.append(img_gt_poly) # extreme_point
        c_gt_4pys.append(can_gt_poly)

    def prepare_evolution(self, poly, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        octagon = snake_coco_utils.get_octagon(extreme_point)
        img_init_poly = snake_coco_utils.uniformsample(octagon, snake_config.poly_num)
        can_init_poly = snake_coco_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = snake_coco_utils.uniformsample(poly, len(poly) * snake_config.gt_poly_num)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1)) # align index
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)] # align between init(from octagon) and gt(from poly)
        can_gt_poly = snake_coco_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def get_thing_anno(self,instance_polys,extreme_points,cls_ids):
        instance_polys_th = [instance_polys[idx] for idx in range(len(cls_ids)) if self.ts_cls[cls_ids[idx]] == 0]
        extreme_points_th = [extreme_points[idx] for idx in range(len(cls_ids)) if self.ts_cls[cls_ids[idx]] == 0]
        cls_ids_th = [cls_ids[idx] for idx in range(len(cls_ids)) if self.ts_cls[cls_ids[idx]] == 0]

        return instance_polys_th, extreme_points_th, cls_ids_th

    def check_void(self,instance_polys, extreme_points):
        num_void_ins = [len(ins) == 0 for ins in iter(instance_polys)]
        num_void_ex = [len(ex) == 0 for ex in iter(extreme_points)]
        return [num_void_ins,num_void_ex]

    def prepare_sem_segmentation(self,instance_polys, cls_ids,img_out_hw):
        # use the instance polygon ann and translate the stuff class to semantic binary map
        output_h, output_w = img_out_hw[2:]

        mask_ = np.zeros((output_h,output_w), dtype=np.uint8)
        for idx in range(len(cls_ids)):
            poly = np.round(instance_polys[idx]).astype(np.int32) # the poly is already list
            cv2.fillPoly(mask_, poly, 1)
        return mask_

    def prepare_edge_segmentation(self,instance_polys, cls_ids,img_out_hw):
        # use the instance polygon ann and translate the stuff class to semantic binary map
        output_h, output_w = img_out_hw[2:]

        mask_ = np.zeros((output_h,output_w), dtype=np.uint8)
        for idx in range(len(cls_ids)):
            poly = np.round(instance_polys[idx]).astype(np.int32) # the poly is already list
            # cv2.fillPoly(mask_, poly, 1)
            cv2.drawContours(mask_, [poly], -1, 1, thickness=1)
        return mask_

    def get_valid_mask(self, height, width, trans_output, inp_out_hw, mask_):
        output_w, output_h = inp_out_hw[2:]
        assert output_w == output_h
        sem_masks = np.zeros((1, output_h, output_w), dtype=np.uint8)
        sem_masks[0, :, :] = mask_
        return sem_masks
    def check_poly(self,polys,cls_ids):
        flag=True
        for idx in range(len(cls_ids)):
            poly = np.round(polys[idx]).astype(np.int32)  # the poly is already list
            if len(list(poly.shape)) < 3:
                flag=False
        return flag

    def __getitem__(self, index):
        # note ->x/w
        #     |y/h
        #     \/
        # poly [x,y]
        # .shape = [h,w]
        ann = self.anns[index]

        anno, path, img_id, img_name = self.process_info(ann)
        img, instance_polys, cls_ids = self.read_original_data(anno, path)
        height, width = img.shape[0], img.shape[1]
        inp, instance_polys, trans_in, trans_out, in_out_hw, orig_img, center, scale = \
            snake_coco_utils.augment_standard(img,instance_polys,
                                              scale_range = self.scale_range,split = self.split)
        del img
        ret = {'inp': inp}
        meta = {'img_id': img_id, 'ann': ann, 'img_name': img_name}
        meta.update({'center':center,'scale':scale})
        if self.speed_eval==True:
            meta.update({'speed_eval':self.speed_eval})
            ret.update({'meta': meta})
            return ret
        instance_polys, cls_ids = self.get_valid_polys(instance_polys, cls_ids, in_out_hw) # the invalid poly will be [] (void array)
        extreme_points = self.get_extreme_points(instance_polys)
        instance_polys_th, extreme_points_th, cls_ids_th = self.get_thing_anno(instance_polys,extreme_points,cls_ids)

        # detection
        output_h, output_w = in_out_hw[2:]
        ct_hm = np.zeros([cfg.heads.ct_hm, output_h, output_w], dtype=np.float32) # center_heatmap

        wh = [] # the w and h
        reg = [] # error due to round operation
        ct_cls = [] # center_class
        ct_ind = [] # center_index: the index transformed from coordinate
        ct_coord = [] # center_coord: the coord of centers

        # init
        i_it_4pys = [] # img_init_poly(4 mid points)
        c_it_4pys = [] # canonical_init_poly(4 mid points)
        i_gt_4pys = [] # extreme points of gt
        c_gt_4pys = [] # canonical extreme points of gt

        # evolution
        i_it_pys = [] # img_init_poly (use octagon)
        c_it_pys = [] # canonical_init_poly (use octagon)
        i_gt_pys = [] # img_gt_poly (origin poly) (align to init_poly)
        c_gt_pys = [] # canonical_gt_poly

        for i in range(len(cls_ids_th)):
            cls_id = cls_ids_th[i]
            poly = instance_polys_th[i]
            extreme_point = extreme_points_th[i]

            x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
            x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
            bbox = [x_min, y_min, x_max, y_max]
            h, w = y_max - y_min + 1, x_max - x_min + 1
            if h <= 1 or w <= 1:
                continue
            # obtain ct_hm,cls_id,wh,reg,ct_cls,ct_ind
            decode_box = self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, reg, ct_cls, ct_ind, ct_coord)
            # obtain i_it_4pys,c_it_4pys,i_gt_4pys,c_gt_4pys
            self.prepare_init(decode_box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
            # obtain i_it_pys,c_it_pys,i_gt_pys,c_gt_pys
            self.prepare_evolution(poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)

        # self.calc_biggest_villi(Scale_Map)
        detection = {'ct_hm': ct_hm, 'wh': wh, 'reg': reg, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_coord': ct_coord}
        init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}

        ret.update(detection)
        ret.update(init)
        ret.update(evolution)

        ct_num = len(ct_ind)
        meta.update({'ct_num': ct_num})
        ret.update({'meta':meta})

        # edge segmentation for instance # not use actually
        edges_th = self.prepare_edge_segmentation(instance_polys_th, cls_ids_th, in_out_hw)
        edges_th = self.get_valid_mask(height, width, trans_out, in_out_hw, edges_th)
        ret.update({'edge_hm': edges_th})


        return ret

    def __len__(self):
        # return 10
        return len(self.anns)

