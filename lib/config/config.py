import torch.cuda

from .yacs import CfgNode as CN
import argparse
import os
import torch
import termcolor
import torch.distributed as dist

cfg = CN()

# model
cfg.model = 'hello'
cfg.model_dir = 'data/model'

# network
cfg.network = 'dla_34'

# network heads
cfg.heads = CN()

# network components
cfg.component = CN()
# cfg.component.box_aug = CN()
# cfg.component.loss = CN()

# loss
cfg.loss = CN()

# train_dataset
cfg.train_dataset = CN()
cfg.train_dataset.scale_range = (0.7, 1.1)

# task
cfg.task = ''

# gpus
cfg.gpus = [0]

# if load the pretrained network
cfg.resume = True


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 140
cfg.train.num_workers = 8

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.scheduler = ''
cfg.train.milestones = [80, 120, 200, 240]
cfg.train.gamma = 0.5

cfg.train.batch_size_per_gpu = 4
cfg.train.batch_size = 4
cfg.train.fold = [1,0,2,3]

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1

# recorder
cfg.record_dir = 'data/record'

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False

cfg.save_ep = 5
cfg.eval_ep = 5

cfg.use_gt_det = False

# save the root
cfg.model_dir_root = cfg.model_dir
cfg.result_dir_root = cfg.result_dir
cfg.record_dir_root = cfg.record_dir

# -----------------------------------------------------------------------------
# post process
# -----------------------------------------------------------------------------
cfg.post = CN()
cfg.post.required = False
cfg.post.th_toedge = 0.02
cfg.post.th_cnt = 0.2
cfg.post_types = []

# -----------------------------------------------------------------------------
# snake
# -----------------------------------------------------------------------------
cfg.ct_score = 0.05
cfg.demo_path = ''


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    cfg.det_dir = os.path.join(cfg.model_dir, cfg.task, args.det)

    # assign the network head conv
    cfg.head_conv = 64 if 'res' in cfg.network else 256

    cfg.model_dir = os.path.join(cfg.model_dir, cfg.task, cfg.model)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.model)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.model)


def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    return cfg


def init_parallel_training(cfg):
    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    return

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--img_path', type=str, default='data/example.jpg')
parser.add_argument('--ckpt_path', type=str, default='data/ckpt.pth')
parser.add_argument('--det', type=str, default='')
parser.add_argument('-f', type=str, default='')
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
init_parallel_training(cfg)
print(" ")