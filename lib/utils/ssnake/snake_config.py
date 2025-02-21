import numpy as np
from lib.config import cfg


mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.28863828, 0.27408164, 0.27809835],
               dtype=np.float32).reshape(1, 1, 3)
data_rng = np.random.RandomState(123)
eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                   dtype=np.float32)
eig_vec = np.array([
    [-0.58752847, -0.69563484, 0.41340352],
    [-0.5832747, 0.00994535, -0.81221408],
    [-0.56089297, 0.71832671, 0.41158938]
], dtype=np.float32)

down_ratio = 4

ro = 4

init_poly_num = 40
poly_num = 128
gt_poly_num = 128
adj_num = 4

scale = np.array([512, 512])
input_w, input_h = (512, 512)
# scale_range = np.arange(0.6, 1.4, 0.1)

box_center = False
center_scope = False

init = 'quadrangle'

# spline_num = 10
#
train_pred_box = False
# box_iou = 0.7
# confidence = 0.1
# train_pred_box_only = True

# train_pred_ex = False
# train_nearest_gt = True

ct_score = cfg.ct_score

segm_or_bbox = 'segm'

ct_threshold = cfg.test.ct_threshold

gauss_M = 7
gauss_sigma = 2
curvs_minimum = 0.1
curvs_maximum = 3
# ensemble refine
max_refine_iters = 0
refine_score_th = 1
resample_method = 'none' #'none','uniform'
ensemble_method = 'select_path' #'instance', 'select_path'
dis_specif_quantile = 0.1
big_diff_th = 0.4
origin_scale = 1.0
