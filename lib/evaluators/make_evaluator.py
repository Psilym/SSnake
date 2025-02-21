import imp
import os
from lib.datasets.dataset_catalog import DatasetCatalog


def _evaluator_factory(cfg,stage):
    task = cfg.task
    assert stage in ['val','test']
    if stage=='val':
        dataset = cfg.val.dataset
    else:
        dataset = cfg.test.dataset
    data_source = DatasetCatalog.get(dataset)['id']
    module = '.'.join(['lib.evaluators', data_source, task])
    path = os.path.join('lib/evaluators', data_source, task+'.py')
    evaluator = imp.load_source(module, path).Evaluator(cfg.result_dir,cfg,stage)
    return evaluator


def make_evaluator(cfg,stage='val'):
    if cfg.skip_eval:
        return None
    else:
        return _evaluator_factory(cfg,stage)
