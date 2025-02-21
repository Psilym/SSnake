import os
import json
import numpy as np
from lib.utils.ssnake import snake_config, snake_eval_utils
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils import data_utils
import termcolor

class Evaluator:
    def __init__(self, result_dir,adapt_cfg,stage):
        self.results = []
        self.img_ids = []
        self.aps = []
        self.sem_ious = []
        self.sem_th_ious = []
        self.edge_th_ious = []

        self.sc_accuracy = 0
        self.coco_stats = 0
        self.metrics = {}
        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        args = DatasetCatalog.get(adapt_cfg.val.dataset)
        self.ann_file = args['ann_file']
        self.data_root = args['data_root']
        self.coco = coco.COCO(self.ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.stage = stage
        if self.stage == 'test':
            self.dilation_ratio  = 0.03

        try:
            use_shead = adapt_cfg['component']['score_head']['use']
            if use_shead:
                use_score_calibration = adapt_cfg['component']['score_head']['use_score_calibration']
            else:
                use_score_calibration = False
        except:
            use_score_calibration = False
        self.use_score_calibration = use_score_calibration
        self.cfg_component = adapt_cfg['component'] if 'component' in adapt_cfg.keys() else None
        self.cfg_boxaug = self.cfg_component['box_aug'] if 'box_aug' in self.cfg_component.keys() else None
        self.use_box = self.cfg_boxaug['use'] if self.cfg_boxaug is not None else False
        init_method = 'box'

        self.init_method = init_method


    def evaluate(self, output, batch):
        detection = output['detection_post']
        if self.use_score_calibration:
            score = output['ct_fin'].detach().cpu().numpy()
            label = np.zeros_like(score).astype(int)
        else:
            if self.init_method in ['box','poly']:
                if len(detection[0].shape) < 2:
                    print(termcolor.colored('No valid detection results. Pass this image.', 'red'))
                    return
                assert len(detection[0].shape) == 2
                score = detection[0, :, 4].detach().cpu().numpy()
                label = detection[0, :, 5].detach().cpu().numpy().astype(int)
            else:
                score = detection['score'][0,:,0].detach().cpu().numpy()
                label = detection['class'][0,:,0].detach().cpu().numpy()

        py = output['py_post'].detach().cpu().numpy() * snake_config.down_ratio

        if len(py) == 0:
            return

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']
        py_coco = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]
        rles = snake_eval_utils.coco_poly_to_rle(py_coco, ori_h, ori_w)

        coco_dets = []
        for i in range(len(rles)):
            detection = {
                'image_id': img_id,
                'category_id': self.contiguous_category_id_to_json_id[label[i]],
                'segmentation': rles[i],
                'score': float('{:.4f}'.format(score[i]))
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)


    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_dets, 'segm')
        coco_eval.params.imgIds = self.img_ids
        coco_eval.params.catIds = self.contiguous_category_id_to_json_id[0] # get the cat_id of villi
        coco_eval.evaluate() # do gt dt match # gt and dt is 1to1 match
        coco_eval.accumulate() # calc pr curve
        coco_eval.summarize() # calc area under curve
        self.results = []
        self.coco_stats = coco_eval.stats
        self.aps.append(coco_eval.stats[0])

        print(f'ap:{coco_eval.stats[0]}')
        self.metrics.update({'num_imgs': len(self.img_ids),
                             'coco_stats': self.coco_stats})

        return {'ap': coco_eval.stats[0]}


Evaluator = Evaluator
