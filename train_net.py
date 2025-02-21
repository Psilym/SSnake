from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
from lib.postprocessors import make_postprocessor
from lib.utils.net_utils import EarlyStopping
import os
import torch.multiprocessing
from lib.train.trainers.crossfold_evaluator import Crossfold_Evaluator
import random
import numpy as np

def seed_torch(seed=1234):
	random.seed(seed) # Python random
	os.environ['PYTHONHASHSEED'] = str(seed) # In order to disable hash randomization and make the experiment reproducible
	np.random.seed(seed) # numpy random
	torch.manual_seed(seed) # torch cpu random
	torch.cuda.manual_seed(seed) # torch gpu random
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.deterministic = True

seed_torch(1234)

def train(cfg, network):
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)
    postprocessor = make_postprocessor(cfg)
    earlystoper = EarlyStopping(optimizer,scheduler,recorder,cfg.model_dir,patience=100)
    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)
    # set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0\
                or (epoch + 1) == cfg.train.epoch:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

        if (epoch + 1) % cfg.eval_ep == 0:
            trainer.val(epoch, val_loader, evaluator, recorder, postprocessor=postprocessor)
            earlystoper(recorder.loss_stats['loss'].median,network,epoch)

    return network

def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg,stage='test')
    epoch = load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    postprocessor = make_postprocessor(cfg)
    trainer.val(epoch, val_loader, evaluator, postprocessor=postprocessor)
    metrics = evaluator.metrics
    return metrics

def adapt_config(cfg_ori,i_fold):
    cfg = cfg_ori.clone()
    cfg_ori.freeze() # first clone, then freeze
    cfg.merge_from_list(['train.dataset', cfg_ori.train.dataset+f'_fold{i_fold}'])
    cfg.merge_from_list(['val.dataset', cfg_ori.val.dataset+f'_fold{i_fold}'])
    cfg.merge_from_list(['test.dataset', cfg_ori.test.dataset+f'_fold{i_fold}'])
    cfg.merge_from_list(['model', cfg_ori.model + f'_fold{i_fold}'])# modify these params
    cfg.merge_from_list(['model_dir', cfg_ori.model_dir + f'_fold{i_fold}'])
    cfg.merge_from_list(['record_dir', cfg_ori.record_dir + f'_fold{i_fold}'])
    cfg.merge_from_list(['result_dir', cfg_ori.result_dir + f'_fold{i_fold}'])
    cfg_ori.defrost()
    cfg.i_fold = i_fold
    return cfg

def save_config_file(config_path,result_dir):
    import shutil as sh
    import os.path as osp
    # result_dir
    dst_path = osp.join(result_dir,'config_file.yaml')
    os.system(f"mkdir -p {result_dir}")
    sh.copy(config_path,dst_path)
    print('Copy config file to result directory.')
    return

def main_vcoco():
    num_fold = 4
    if args.test:
        cfg.resume = True
        cf_evaluator = Crossfold_Evaluator(num_fold=num_fold)
        fold_list = list([0,1,2,3])
        fold_list = fold_list[:]
        for i_fold in iter(fold_list):
            network = make_network(cfg)
            cfg_fold = adapt_config(cfg, i_fold)
            save_config_file(args.cfg_file, cfg.result_dir)
            print(f'Start fold{i_fold}...')
            metrics = test(cfg_fold, network)
            cf_evaluator.get_fold_result(i_fold,metrics)
            del network
        cf_evaluator.calc_final_result(cfg)
        return

    else:
        fold_list = cfg.train.folds
        for i_fold in iter(fold_list):
            network = make_network(cfg)
            cfg_fold = adapt_config(cfg,i_fold)
            save_config_file(args.cfg_file,cfg.result_dir)
            print(f'Start fold{i_fold}...')
            train(cfg_fold, network)
            del network
        return

def main():
    dataset = cfg.dataset if 'dataset' in cfg.keys() else 'vcoco'
    if dataset == 'vcoco':
        main_vcoco()
    else:
        print("Not implement this dataset.")
        raise ValueError


if __name__ == "__main__":
    main()
